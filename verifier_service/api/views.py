# api/views.py
import os
import re
import json
import hashlib
import requests
from typing import Any, Dict, List, Tuple, Optional

from django.conf import settings
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

# lokale
from .utils import extract_pdf_text, clean_phone_number, parse_resume_date
from .serializers import ResumePayloadSer
from .models import EnrichmentCapture

# ───────────────────────── google-genai (SDK i RI) ─────────────────────────
# pip install google-genai
from google import genai
from google.genai import types

# Model default – merr nga env ose përdor fallback
GENAI_MODEL_DEFAULT = os.getenv("GENAI_MODEL", "gemini-2.5-flash")


# ───────────────────────── HELPERS TË PËRBASHKËTA ─────────────────────────

def _resp_to_text(resp) -> str:
    """
    Nxjerr tekst nga përgjigja e google-genai në mënyrë të qëndrueshme.
    Mbështet: resp.text, resp.output_text, dhe candidates[*].content.parts[*].text
    """
    raw = getattr(resp, "text", None) or getattr(resp, "output_text", None)
    if raw:
        return str(raw)
    try:
        out = []
        for c in getattr(resp, "candidates", []) or []:
            content = getattr(c, "content", None)
            if not content:
                continue
            for p in getattr(content, "parts", []) or []:
                t = getattr(p, "text", None)
                if t:
                    out.append(t)
        return "".join(out)
    except Exception:
        return ""


def _strip_md_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    return t


def _json_relaxed(s: str):
    """
    Provo json.loads; në dështim, izolo bllokun e parë {…} më të jashtëm.
    """
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(s[i:j + 1])
    except Exception:
        pass
    raise json.JSONDecodeError("Could not parse JSON", s, 0)


def _normalize_section_rows(rows: Any, date_keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """
    Kthen vetëm rreshta që janë dict; normalizon datat me parse_resume_date.
    Ignoron elementët që s'janë dict (p.sh. string 'N/A', '—', etj).
    """
    if not isinstance(rows, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            # skip çdo element jo-dict
            continue
        # normalizo datat
        for k in date_keys:
            if row.get(k):
                row[k] = parse_resume_date(row[k]) or row[k]
        cleaned.append(row)
    return cleaned


def _merge_rows_by_key(base_rows: Any, patch_rows: Any, key_fields: List[str]) -> List[Dict[str, Any]]:
    """
    Bashkon rreshta me çelësa unikë bazuar në key_fields.
    Preferon të ruajë vlerat ekzistuese; plotëson vetëm fushat bosh.
    """
    base = base_rows if isinstance(base_rows, list) else []
    patch = patch_rows if isinstance(patch_rows, list) else []

    # Lejo vetëm dict
    base = [r for r in base if isinstance(r, dict)]
    patch = [r for r in patch if isinstance(r, dict)]

    def _k(d: Dict[str, Any]) -> str:
        return "|".join((str(d.get(f) or "").strip().lower()) for f in key_fields)

    idx = {_k(r): r for r in base}
    for pr in patch:
        kk = _k(pr)
        target = idx.get(kk)
        if not target:
            base.append(pr)
            idx[kk] = pr
            continue
        # plotëso boshët
        for dk in (
            "start", "end", "issue_date", "expiry_date", "date",
            "graduated", "program_type", "status", "city", "state", "country",
            "institution", "degree", "number", "board", "specialty"
        ):
            v = pr.get(dk)
            if v and not target.get(dk):
                target[dk] = v
    return base


def _has_any_dates(rows: Any, keys: Tuple[str, ...] = ("start", "end", "issue_date", "expiry_date", "date")) -> bool:
    rows = rows if isinstance(rows, list) else []
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in keys:
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                return True
            if v is not None and not isinstance(v, str):
                return True
    return False


def _save_capture(npi: str, name_hint: str, payload: dict, quality: dict,
                  status: str = "ready", pass_stats: Optional[dict] = None):
    return EnrichmentCapture.objects.create(
        npi=str(npi),
        name_hint=name_hint or "",
        status=status,
        payload=payload or {},
        quality=quality or {},
        pass_stats=pass_stats or {},
    )


def _enrich_fake_payload(npi_seed: Dict[str, Any], npi: str) -> Dict[str, Any]:
    """
    Payload demo kur ENRICH_FAKE=true (për zhvillim).
    """
    return {
        "informations": {
            "legalfirstname": npi_seed.get("legalfirstname") or "John",
            "legallastname": npi_seed.get("legallastname") or "Doe",
            "npinumber": re.sub(r"\D", "", str(npi))[:10],
            "phone": clean_phone_number(npi_seed.get("phone") or "+1 (781) 555-0100"),
            "address": npi_seed.get("address"),
            "address2": npi_seed.get("address2"),
            "city": npi_seed.get("city"),
            "stateprovince": npi_seed.get("stateprovince"),
            "zipcode": npi_seed.get("zipcode"),
            "specialization": npi_seed.get("specialization") or "M.D.",
        },
        "education": [],
        "medical_education": [],
        "graduate_school": [],
        "internship": [],
        "residency": [],
        "fellowship": [],
        "board_certifications": [],
        "medical_licences": [],
        "dea_registration": [],
        "other_exams": [],
        "professional_reference": [],
        "quality": {"completeness_score": 0.7, "confidence_score": 0.7},
    }


# ─────────────────────────────── VIEWS ───────────────────────────────

# ───────────────────────── VIEWS ─────────────────────────

@api_view(["GET"])
def health(request):
    return Response({"ok": True})


@api_view(["GET"])
def npi_lookup(request):
    """
    Lookup bazik nga NPI Registry (JSON zyrtar). Kthehet si rezultat minimal bazë.
    """
    npi = request.GET.get("npi")
    if not npi:
        return Response({"detail": "npi required"}, status=400)

    url = f"https://npiregistry.cms.hhs.gov/api/?number={npi}&version=2.1"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return Response({"detail": "npi upstream error"}, status=502)
        data = r.json()
        result = (data.get("results") or [])
        if not result:
            return Response({"detail": "not found", "result": {}}, status=404)

        top = result[0]
        basic = top.get("basic", {}) or {}
        addresses = top.get("addresses") or []
        addr = addresses[0] if addresses else {}

        out = {
            "legalfirstname": basic.get("first_name"),
            "legallastname": basic.get("last_name"),
            "legalmiddlename": basic.get("middle_name"),
            "npinumber": top.get("number"),
            "phone": clean_phone_number(addr.get("telephone_number")),
            "address": addr.get("address_1"),
            "address2": addr.get("address_2"),
            "city": addr.get("city"),
            "stateprovince": addr.get("state"),
            "zipcode": addr.get("postal_code"),
            "specialization": (basic.get("credential") or "")
        }
        return Response({"detail": "ok", "result": out})
    except Exception as e:
        return Response({"detail": f"npi failed: {e}"}, status=500)


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def parse_resume(request):
    """
    Parsim CV PDF → JSON sipas skemës. Opsionalisht merge me NPI (?npi=...).
    Përdor google-genai PA tools, ndaj lejohet response_mime_type="application/json".
    """
    f = request.FILES.get("file")
    if not f:
        return Response({"detail": "file missing"}, status=400)
    if not f.name.lower().endswith(".pdf"):
        return Response({"detail": "pdf only"}, status=400)
    if f.size > 10 * 1024 * 1024:
        return Response({"detail": "max 10MB"}, status=400)

    # seed opsional nga NPI
    base_info: Dict[str, Any] = {}
    npi = request.POST.get("npi") or request.query_params.get("npi")
    if npi:
        try:
            base = npi_lookup(request._request)
            if getattr(base, "status_code", 500) == 200:
                base_info = base.data.get("result") or {}
        except Exception:
            pass

    # ekstrakto tekst nga PDF
    try:
        text = extract_pdf_text(f)
        if not text.strip():
            return Response({"detail": "no text"}, status=400)
    except Exception as e:
        return Response({"detail": f"extract failed: {e}"}, status=500)

    # dev fake
    if getattr(settings, "ENRICH_FAKE", False):
        payload = {
            "informations": {
                **base_info,
                "phone": clean_phone_number(base_info.get("phone") or "+1 (212) 555-0100"),
            },
            "medical_education": [{
                "institution": "Demo Medical School", "degree": "MD",
                "city": "Boston", "state": "MA", "country": "US",
                "start": "2008-01-01", "end": "2012-05-01", "graduated": "True"
            }],
            "residency": [{
                "institution": "Demo Hospital", "specialty": "Internal Medicine",
                "city": "Boston", "state": "MA", "country": "US",
                "start": "2012-07-01", "end": "2015-06-30", "program_type": "Categorical"
            }],
            "medical_licences": [{
                "state": "MA", "number": "MA12345", "issue_date": "2015-08-01",
                "expiry_date": "2027-08-01", "status": "ACTIVE", "is_primary": "True"
            }],
            "quality": {"completeness_score": 0.78, "confidence_score": 0.8}
        }
        ser = ResumePayloadSer(data=payload); ser.is_valid(raise_exception=True)
        return Response({"detail": "ok", "parsed": ser.data, "quality": payload.get("quality")})

    api_key = os.environ.get("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
    if not api_key:
        return Response({"detail": "GEMINI_API_KEY missing"}, status=500)

    try:
        client = genai.Client(api_key=api_key)
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
        prompt = """
You are an information extraction engine. Extract ONLY the requested fields strictly.
Return VALID JSON only. Omit fields you are not confident about.
Normalize phones to +1XXXXXXXXXX and dates to YYYY-MM-DD when possible (else MM/YYYY).
Allowed sections and fields (use only these exact keys):
- informations: legalfirstname, legallastname, legalmiddlename, npinumber, phone, address, address2, city, stateprovince, zipcode, specialization, skills, total_experience_years
- education/medical_education/graduate_school: institution, degree, city, state, country, start, end, graduated
- internship/residency/fellowship: institution, specialty, city, state, country, start, end, program_type
- board_certifications: board, specialty, issue_date, expiry_date, status, certificate_id
- medical_licences: state, number, issue_date, expiry_date, status, is_primary
- dea_registration: dea_number, state, issue_date, expiry_date, status, schedules
- other_exams: exam_name, score, date, passed, details
- professional_reference: name, title, institution, phone, email, relationship
If unsure, omit.
""".strip() + "\n\n" + text[:15000]

        resp = client.models.generate_content(
            model=GENAI_MODEL_DEFAULT,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=cfg,
        )

        raw = _strip_md_fences(_resp_to_text(resp))
        if not raw:
            return Response({"detail": "parse failed: empty model response"}, status=502)

        data = _json_relaxed(raw)

        # normalizime
        info = (data.get("informations") or {})
        if "phone" in info:
            info["phone"] = clean_phone_number(info["phone"])
        if "npinumber" in info:
            info["npinumber"] = re.sub(r"\D", "", str(info["npinumber"]))[:10]

        # filtro dict-ët dhe normalizo datat për seksionet
        for section in (
            "education", "medical_education", "graduate_school", "internship",
            "residency", "fellowship", "board_certifications", "medical_licences",
            "dea_registration", "other_exams", "professional_reference"
        ):
            data[section] = _normalize_section_rows(
                data.get(section),
                date_keys=("issue_date", "expiry_date", "start", "end", "date")
            )

        data["informations"] = {
            **(info or {}),
            **{k: v for k, v in (base_info or {}).items() if v}
        }
        ser = ResumePayloadSer(data=data); ser.is_valid(raise_exception=True)
        quality = data.get("quality") or {"completeness_score": 0.8, "confidence_score": 0.75}
        return Response({"detail": "ok", "parsed": ser.data, "quality": quality})

    except json.JSONDecodeError as e:
        return Response({"detail": f"parse json failed: {e}"}, status=502)
    except Exception as e:
        return Response({"detail": f"parse failed: {e}"}, status=500)


@api_view(["GET"])
def enrich_by_npi(request):
    """
    DB-first: nëse ka EnrichmentCapture të freskët për këtë NPI → kthe direkt nga DB.
    Nëse only_cache=1 → mos bëj thirrje të jashtme (kthe hit/stale/miss).
    Përndryshe, vazhdo me Pass1/Pass2/Pass3 dhe ruaj rezultatin në DB (nëse SAVE_CAPTURE=True).
    """
    from google import genai
    from google.genai import types
    from datetime import datetime, timedelta, timezone

    npi = request.GET.get("npi")
    name_hint = request.GET.get("name") or ""
    if not npi:
        return Response({"detail": "npi required"}, status=400)

    # ── DB-FIRST guard ───────────────────────────────────────────────────
    only_cache = (request.GET.get("only_cache") or "0") == "1"
    try:
        max_age = int(request.GET.get("max_age") or str(7 * 24 * 3600))  # default: 7 ditë
    except ValueError:
        max_age = 7 * 24 * 3600

    def _is_fresh_dt(dt, ttl=max_age):
        try:
            return (datetime.now(timezone.utc) - dt) <= timedelta(seconds=ttl)
        except Exception:
            return False

    last = (
        EnrichmentCapture.objects
        .filter(npi=str(npi))
        .only("job_id", "created_at", "payload", "quality")
        .order_by("-created_at")
        .first()
    )

    if last and _is_fresh_dt(last.created_at, ttl=max_age):
        # Cache HIT → kthe direkt nga DB, pa thirrje të jashtme
        return Response({
            "detail": "ok",
            "parsed": last.payload or {},
            "quality": last.quality or {},
            "cache_status": "hit",
            "job_id": str(last.job_id),
            "created_at": last.created_at.isoformat(),
        })

    if only_cache:
        # Kërkohet rreptësisht cache → mos godit burime të jashtme
        return Response({
            "detail": "ok",
            "parsed": (last.payload if last else {}) or {},
            "quality": (last.quality if last else {}) or {},
            "cache_status": "stale" if last else "miss",
            "job_id": str(last.job_id) if last else None,
            "created_at": last.created_at.isoformat() if last else None,
        })
    # ─────────────────────────────────────────────────────────────────────

    # seed nga npi_lookup
    npi_seed: Dict[str, Any] = {}
    try:
        base = npi_lookup(request._request)
        if getattr(base, "status_code", 500) == 200:
            npi_seed = base.data.get("result") or {}
            if not name_hint:
                fn = (npi_seed.get("legalfirstname") or "").strip()
                ln = (npi_seed.get("legallastname") or "").strip()
                name_hint = (f"{fn} {ln}").strip()
    except Exception:
        pass

    # dev fake
    if getattr(settings, "ENRICH_FAKE", False):
        payload = _enrich_fake_payload(npi_seed, npi)
        if getattr(settings, "SAVE_CAPTURE", False):
            _save_capture(npi, name_hint, payload, payload.get("quality"), status="OK_FAKE")
        return Response({
            "detail": "ok",
            "parsed": payload,
            "quality": payload["quality"],
            "cache_status": "fresh",
        })

    # ── genai client & tools ────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
    if not api_key:
        return Response({"detail": "GEMINI_API_KEY missing"}, status=500)

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GENAI_MODEL", GENAI_MODEL_DEFAULT)
    tools = [types.Tool(google_search=types.GoogleSearch())]

    # PASS 1
    cfg_pass1 = types.GenerateContentConfig(tools=tools, temperature=0.25)
    prompt_pass1 = f"""
You are a web-grounded extraction agent. Using Google Search, find public info
for physician with NPI = {npi} (name hint: {name_hint or "unknown"}).

Return JSON ONLY with this exact schema:
{{
  "informations": {{
    "legalfirstname":"", "legallastname":"", "legalmiddlename":"", "npinumber":"",
    "phone":"", "address":"", "address2":"", "city":"", "stateprovince":"", "zipcode":"",
    "mailaddress":"", "mailaddress2":"", "mailcity":"", "mailingstateprovince":"", "mailingzipcode":"",
    "role":"", "gender":"", "specialization":"", "skills":"", "total_experience_years":""
  }},
  "education":[], "medical_education":[], "graduate_school":[],
  "internship":[], "residency":[], "fellowship":[],
  "board_certifications":[], "medical_licences":[], "dea_registration":[],
  "other_exams":[], "professional_reference":[],
  "quality":{{"completeness_score":0,"confidence_score":0}}
}}

Be aggressive filling fields. Prefer authoritative sources. JSON only.
""".strip()

    resp1 = client.models.generate_content(
        model=model_name,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_pass1)])],
        config=cfg_pass1,
    )
    raw1 = _strip_md_fences(_resp_to_text(resp1))
    if not raw1:
        return Response({"detail": "enrich failed: empty model response"}, status=502)

    try:
        data = _json_relaxed(raw1)
    except json.JSONDecodeError as e:
        return Response({"detail": f"json decode failed (pass1): {e}"}, status=502)

    # Normalizime bazë – info
    info = (data.get("informations") or {})
    if "phone" in info:
        info["phone"] = clean_phone_number(info["phone"])
    info["npinumber"] = re.sub(r"\D", "", str(info.get("npinumber") or npi))[:10]
    data["informations"] = {**{k: v for k, v in (npi_seed or {}).items() if v}, **(info or {})}

    # Normalizo seksionet (dict-only + datat)
    for section in (
        "education", "medical_education", "graduate_school", "internship",
        "residency", "fellowship", "board_certifications", "medical_licences",
        "dea_registration", "other_exams", "professional_reference"
    ):
        data[section] = _normalize_section_rows(
            data.get(section),
            date_keys=("issue_date", "expiry_date", "start", "end", "date")
        )

    # PASS 2 – vetëm date/status nëse mungojnë
    need_repair = {
        "education":            not _has_any_dates(data.get("education")),
        "medical_education":    not _has_any_dates(data.get("medical_education")),
        "graduate_school":      not _has_any_dates(data.get("graduate_school")),
        "internship":           not _has_any_dates(data.get("internship")),
        "residency":            not _has_any_dates(data.get("residency")),
        "fellowship":           not _has_any_dates(data.get("fellowship")),
        "board_certifications": not _has_any_dates(data.get("board_certifications")),
        "medical_licences":     not _has_any_dates(data.get("medical_licences")),
    }

    if any(need_repair.values()):
        def _mk_hint_list(section_name: str, rows: Any, keys: List[str]) -> str:
            out = []
            rows = rows if isinstance(rows, list) else []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                parts = []
                for k in keys:
                    v = (r.get(k) or "").strip()
                    if v:
                        parts.append(f"{k}={v}")
                if parts:
                    out.append("{" + ", ".join(parts) + "}")
            return f'"{section_name}": [{", ".join(out)}]'

        hints = []
        hints.append(_mk_hint_list("education", data.get("education"), ["institution", "degree", "city", "state"]))
        hints.append(_mk_hint_list("medical_education", data.get("medical_education"), ["institution", "degree", "city", "state"]))
        hints.append(_mk_hint_list("graduate_school", data.get("graduate_school"), ["institution", "degree", "city", "state"]))
        hints.append(_mk_hint_list("internship", data.get("internship"), ["institution", "specialty", "city", "state", "program_type"]))
        hints.append(_mk_hint_list("residency", data.get("residency"), ["institution", "specialty", "city", "state", "program_type"]))
        hints.append(_mk_hint_list("fellowship", data.get("fellowship"), ["institution", "specialty", "city", "state", "program_type"]))
        hints.append(_mk_hint_list("board_certifications", data.get("board_certifications"), ["board", "specialty", "status"]))
        hints.append(_mk_hint_list("medical_licences", data.get("medical_licences"), ["state", "number", "status"]))
        hint_block = ",\n".join([h for h in hints if h])

        cfg_pass2 = types.GenerateContentConfig(tools=tools, temperature=0.15)
        prompt_pass2 = f"""
Using Google Search, fetch ONLY missing date/status fields for NPI={npi} (name hint: {name_hint or "unknown"}).
Context:
{{
{hint_block}
}}
Return JSON only with same section names, including fields:
- start, end, graduated, program_type, issue_date, expiry_date, status, number.
Normalize dates to YYYY-MM or YYYY.
""".strip()

        resp2 = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_pass2)])],
            config=cfg_pass2,
        )
        raw2 = _strip_md_fences(_resp_to_text(resp2))
        if raw2:
            try:
                patch = _json_relaxed(raw2)

                data["education"] = _merge_rows_by_key(
                    data.get("education") or [], patch.get("education") or [],
                    ["institution", "degree", "city", "state"]
                )
                data["medical_education"] = _merge_rows_by_key(
                    data.get("medical_education") or [], patch.get("medical_education") or [],
                    ["institution", "degree", "city", "state"]
                )
                data["graduate_school"] = _merge_rows_by_key(
                    data.get("graduate_school") or [], patch.get("graduate_school") or [],
                    ["institution", "degree", "city", "state"]
                )
                data["internship"] = _merge_rows_by_key(
                    data.get("internship") or [], patch.get("internship") or [],
                    ["institution", "specialty", "city", "state", "program_type"]
                )
                data["residency"] = _merge_rows_by_key(
                    data.get("residency") or [], patch.get("residency") or [],
                    ["institution", "specialty", "city", "state", "program_type"]
                )
                data["fellowship"] = _merge_rows_by_key(
                    data.get("fellowship") or [], patch.get("fellowship") or [],
                    ["institution", "specialty", "city", "state", "program_type"]
                )
                data["board_certifications"] = _merge_rows_by_key(
                    data.get("board_certifications") or [], patch.get("board_certifications") or [],
                    ["board", "specialty", "status"]
                )
                data["medical_licences"] = _merge_rows_by_key(
                    data.get("medical_licences") or [], patch.get("medical_licences") or [],
                    ["state", "number", "status"]
                )

                # normalizo datat pasi u merge-uan
                for section in (
                    "education", "medical_education", "graduate_school", "internship",
                    "residency", "fellowship", "board_certifications", "medical_licences"
                ):
                    data[section] = _normalize_section_rows(
                        data.get(section),
                        date_keys=("issue_date", "expiry_date", "start", "end", "date", "graduated")
                    )
            except Exception:
                pass

    # PASS 3 – deep fill education/graduate nëse duhen
    def _education_incomplete(rows: Any) -> bool:
        rows = rows if isinstance(rows, list) else []
        if not rows:
            return True
        for r in rows:
            if not isinstance(r, dict):
                return True
            if not (r.get("institution") or r.get("degree")):
                return True
            if not (r.get("city") or r.get("state") or r.get("start") or r.get("end")):
                return True
        return False

    if _education_incomplete(data.get("education")) or \
       _education_incomplete(data.get("graduate_school")) or \
       _education_incomplete(data.get("medical_education")):

        edu_hint = []
        for sec in ("education", "graduate_school", "medical_education"):
            rows = data.get(sec) or []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                parts = []
                for k in ("institution", "degree", "city", "state", "country"):
                    v = (r.get(k) or "").strip()
                    if v:
                        parts.append(f"{k}={v}")
                if parts:
                    edu_hint.append("{" + ", ".join(parts) + "}")
        edu_context = ", ".join(edu_hint)

        cfg_pass3 = types.GenerateContentConfig(tools=tools, temperature=0.3)
        prompt_pass3 = f"""
Deep-fill EDUCATION/GRADUATE for NPI={npi} (name: {name_hint or "unknown"}).
Prefer .edu domains and PDF CVs. Given: [{edu_context}]
Return JSON only with sections education, medical_education, graduate_school
and try to populate institution, degree, city, state, country, start, end, graduated.
""".strip()

        resp3 = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_pass3)])],
            config=cfg_pass3,
        )
        raw3 = _strip_md_fences(_resp_to_text(resp3))
        if raw3:
            try:
                patch3 = _json_relaxed(raw3)
                data["education"] = _merge_rows_by_key(
                    data.get("education") or [], patch3.get("education") or [],
                    ["institution", "degree", "city", "state"]
                )
                data["medical_education"] = _merge_rows_by_key(
                    data.get("medical_education") or [], patch3.get("medical_education") or [],
                    ["institution", "degree", "city", "state"]
                )
                data["graduate_school"] = _merge_rows_by_key(
                    data.get("graduate_school") or [], patch3.get("graduate_school") or [],
                    ["institution", "degree", "city", "state"]
                )

                for section in ("education", "medical_education", "graduate_school"):
                    data[section] = _normalize_section_rows(
                        data.get(section),
                        date_keys=("start", "end", "graduated")
                    )
            except Exception:
                pass

    # Validim & RUAN (nëse SAVE_CAPTURE=True)
    try:
        ser = ResumePayloadSer(data=data)
        ser.is_valid(raise_exception=True)

        quality = data.get("quality") or {"completeness_score": 0.7, "confidence_score": 0.9}
        for k in ("completeness_score", "confidence_score"):
            v = quality.get(k)
            if isinstance(v, (int, float)) and v > 1:
                quality[k] = round(float(v) / 100.0, 3)

        if getattr(settings, "SAVE_CAPTURE", False):
            _save_capture(npi, name_hint, ser.data, quality, status="OK")

        return Response({
            "detail": "ok",
            "parsed": ser.data,
            "quality": quality,
            "cache_status": "fresh",
        })

    except Exception:
        # edhe në error validimi — ruaj çfarë ke
        if getattr(settings, "SAVE_CAPTURE", False):
            _save_capture(npi, name_hint, data, data.get("quality") or {}, status="PARTIAL")
        return Response({
            "detail": "ok (partial-validated)",
            "parsed": data,
            "quality": data.get("quality") or {},
            "cache_status": "fresh",
        })


@api_view(["GET"])
def list_captures(request):
    """
    Liston deri në 50 ruajtjet e fundit. Filtrim opsional me ?npi=...
    """
    npi = request.GET.get("npi")
    qs = EnrichmentCapture.objects.all()
    if npi:
        qs = qs.filter(npi=str(npi))
    qs = qs.order_by("-created_at")[:50]

    out = [{
        "job_id": str(r.job_id),
        "npi": r.npi,
        "name_hint": r.name_hint,
        "status": r.status,
        "quality": r.quality,
        "created_at": r.created_at.isoformat(),
    } for r in qs]
    return Response({"results": out})


@api_view(["GET"])
def get_capture(request, job_id):
    """
    Merr një ruajtje me job_id (uuid).
    """
    try:
        r = EnrichmentCapture.objects.get(job_id=job_id)
    except EnrichmentCapture.DoesNotExist:
        return Response({"detail": "not found"}, status=404)

    return Response({
        "job_id": str(r.job_id),
        "npi": r.npi,
        "name_hint": r.name_hint,
        "status": r.status,
        "quality": r.quality,
        "payload": r.payload,
        "created_at": r.created_at.isoformat(),
        "updated_at": r.updated_at.isoformat(),
    })
