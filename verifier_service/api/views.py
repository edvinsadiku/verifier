# api/views.py
import os
import re
import json
import hashlib
import requests
from datetime import date
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


def _normalize_section_rows(rows: Any, date_keys: Tuple[str, ...], min_fields: int = 2) -> List[Dict[str, Any]]:
    
    if not isinstance(rows, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            # skip çdo element jo-dict
            continue

        # normalize zip -> zipcode
        if row.get("zip") and not row.get("zipcode"):
            row["zipcode"] = row.get("zip")

        # normalizo datat
        for k in date_keys:
            if row.get(k):
                row[k] = parse_resume_date(row[k]) or row[k]

        # 🔴 FILTER: rreshtat me më pak se min_fields fusha jo-bosh i heqim
        non_empty_keys = [k for k, v in row.items() if v not in (None, "", [], {})]
        if len(non_empty_keys) < min_fields:
            # p.sh. row = {"graduated": "2002-01-01"} → SKIP
            continue

        cleaned.append(row)
    return cleaned


def _year_from_value(value: Any) -> Optional[int]:
    if value in (None, "", [], {}):
        return None

    current_year = date.today().year
    if isinstance(value, int):
        year = value
    else:
        match = re.search(r"\b(19\d{2}|20\d{2})\b", str(value))
        if not match:
            return None
        year = int(match.group(1))

    if 1950 <= year <= current_year:
        return year
    return None


def _coerce_experience_years(value: Any) -> Optional[int]:
    try:
        years = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    if 0 <= years <= 60:
        return years
    return None


def _add_experience_candidate(candidates: List[Dict[str, Any]], year: Optional[int], source: str):
    if year is None:
        return
    candidates.append({"year": year, "source": source})


def _clean_public_email(value: Any) -> str:
    email = str(value or "").strip().lower()
    if not email:
        return ""
    match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", email, re.IGNORECASE)
    if not match:
        return ""
    return match.group(0)[:254]


def _normalize_contact_email(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    info = payload.get("informations")
    if not isinstance(info, dict):
        info = {}
        payload["informations"] = info
    email = _clean_public_email(info.get("emergencycontactemail") or info.get("email"))
    info.pop("email", None)
    if email:
        info["emergencycontactemail"] = email
    return payload


def _find_public_contact_email(
    *,
    client,
    model_name: str,
    npi: str,
    name_hint: str,
    npi_seed: Dict[str, Any],
) -> str:
    try:
        tools = [types.Tool(google_search=types.GoogleSearch())]
        cfg = types.GenerateContentConfig(
            tools=tools,
            temperature=0.1,
        )
        prompt = f"""
Find one public professional/provider/practice email for this medical provider.

- NPI: {npi}
- Name hint: "{name_hint or "unknown"}"
- Official NPI registry context: {json.dumps(npi_seed or {}, ensure_ascii=False)}

Rules:
- Return ONLY strict JSON.
- Output format: {{"email":"...","source_hint":"..."}}
- Use only a public professional/provider/practice email from hospital, clinic, practice, faculty, or official professional profile pages.
- Do NOT guess an email.
- Do NOT return placeholder emails, personal emails, or unrelated directory emails.
- If no reliable public email is found, return {{"email":"","source_hint":""}}.
"""
        resp = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=cfg,
        )
        raw = _strip_md_fences(_resp_to_text(resp))
        if not raw:
            return ""
        data = _json_relaxed(raw)
        if not isinstance(data, dict):
            return ""
        return _clean_public_email(data.get("email"))
    except Exception:
        return ""


def _normalize_experience_years(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic fallback for total_experience_years.
    Gemini can omit this field even when it returns dated licences/training rows,
    so compute it from the earliest reliable clinical/professional year.
    """
    if not isinstance(payload, dict):
        return payload

    info = payload.get("informations")
    if not isinstance(info, dict):
        info = {}
        payload["informations"] = info

    existing_years = _coerce_experience_years(info.get("total_experience_years"))
    if existing_years:
        info["total_experience_years"] = existing_years
        meta = info.get("_meta") if isinstance(info.get("_meta"), dict) else {}
        meta.setdefault("experience_source", "model provided total_experience_years")
        meta.setdefault("experience_confidence", "medium")
        info["_meta"] = meta
        return payload

    candidates: List[Dict[str, Any]] = []

    for row in payload.get("medical_licences") or []:
        if not isinstance(row, dict):
            continue
        _add_experience_candidate(
            candidates,
            _year_from_value(row.get("issue_date")),
            "medical_licences.issue_date",
        )

    for section in ("internship", "residency", "fellowship"):
        for row in payload.get(section) or []:
            if not isinstance(row, dict):
                continue
            _add_experience_candidate(
                candidates,
                _year_from_value(row.get("start") or row.get("end")),
                f"{section}.start_or_end",
            )

    for section in ("medical_education", "graduate_school", "education"):
        for row in payload.get(section) or []:
            if not isinstance(row, dict):
                continue
            _add_experience_candidate(
                candidates,
                _year_from_value(row.get("graduated") or row.get("end")),
                f"{section}.graduated_or_end",
            )

    if not candidates:
        info.pop("total_experience_years", None)
        return payload

    earliest = min(candidates, key=lambda item: item["year"])
    years = max(0, date.today().year - earliest["year"])
    if years > 60:
        return payload

    info["total_experience_years"] = years
    meta = info.get("_meta") if isinstance(info.get("_meta"), dict) else {}
    meta.update(
        {
            "experience_basis_year": earliest["year"],
            "experience_source": earliest["source"],
            "experience_confidence": "medium",
            "experience_method": "deterministic_fallback",
        }
    )
    info["_meta"] = meta
    return payload


def _merge_rows_by_key(base_rows: Any, patch_rows: Any, key_fields: List[str]) -> List[Dict[str, Any]]:

    base = base_rows if isinstance(base_rows, list) else []
    patch = patch_rows if isinstance(patch_rows, list) else []

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


def _normalize_clinical_preferences(value: Any) -> str:
    if not value:
        return ""

    if isinstance(value, list):
        items = [str(v).strip() for v in value if isinstance(v, (str, int, float))]
    elif isinstance(value, str):
        items = [v.strip() for v in value.split(",")]
    else:
        return ""

    # dedupe, keep order, drop empties
    seen = set()
    out: List[str] = []
    for v in items:
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return ", ".join(out)


def _normalize_digital_passport(payload: Any) -> List[Dict[str, str]]:
    if not payload:
        return []

    items = payload if isinstance(payload, list) else [payload]

    def _split_list(value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            raw = []
            for v in value:
                raw.extend(str(v).split(","))
        else:
            raw = str(value).split(",")
        out: List[str] = []
        seen = set()
        for v in raw:
            s = v.strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def _to_str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, list):
            # join list values with comma
            return ", ".join([str(x).strip() for x in v if str(x).strip()])
        return str(v).strip()

    fields = (
        "hospital_affiliation",
        "licence_number",
        "consultation_hours",
        "available_locations",
        "state",
        "zipcode",
        "address",
        "address2",
        "city",
    )

    out: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row: Dict[str, str] = {}

        # allow "zip" fallback
        if not item.get("zipcode") and item.get("zip"):
            item = {**item, "zipcode": item.get("zip")}

        for f in fields:
            v = item.get(f)
            if f == "available_locations":
                v = _normalize_clinical_preferences(v)
            else:
                v = _to_str(v)
            if v:
                row[f] = v

        if not row:
            continue

        # Split multi-affiliations / licenses into separate rows
        affs = _split_list(row.get("hospital_affiliation"))
        lics = _split_list(row.get("licence_number"))
        if len(affs) <= 1 and len(lics) <= 1:
            out.append(row)
            continue

        n = max(len(affs), len(lics))
        for i in range(n):
            new_row = dict(row)
            if affs:
                new_row["hospital_affiliation"] = affs[i] if i < len(affs) else affs[-1]
            if lics:
                new_row["licence_number"] = lics[i] if i < len(lics) else lics[-1]
            out.append(new_row)
    return out


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
    Payload demo
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
        "preferences": {
            "clinicalpreferences": "General Surgery, Internal Medicine",
        },
        "digital_passport": [{
            "hospital_affiliation": "Demo Hospital",
            "licence_number": "HOSP-12345",
            "consultation_hours": "Mon-Fri 9am-5pm",
            "available_locations": "Boston, MA",
            "state": "MA",
            "zipcode": "02118",
            "address": "100 Main St",
            "city": "Boston",
        }],
        "quality": {"completeness_score": 0.7, "confidence_score": 0.7},
    }


# ─────────────────────────────── VIEWS ───────────────────────────────

@api_view(["GET"])
def health(request):
    return Response({"ok": True})


@api_view(["GET"])
def npi_lookup(request):


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
    Parsim CV PDF → JSON sipas skemës.
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


    try:
        text = extract_pdf_text(f)
        if not text.strip():
            return Response({"detail": "no text"}, status=400)
    except Exception as e:
        return Response({"detail": f"extract failed: {e}"}, status=500)


    if getattr(settings, "ENRICH_FAKE", False):
        payload = {
            "informations": {
                **base_info,
                "phone": clean_phone_number(base_info.get("phone") or "+1 (212) 555-0100"),
            },
            "preferences": {
                "clinicalpreferences": "General Surgery, Internal Medicine",
            },
            "digital_passport": [{
                "hospital_affiliation": "Demo Hospital",
                "licence_number": "HOSP-12345",
                "consultation_hours": "Mon-Fri 9am-5pm",
                "available_locations": "Boston, MA",
                "state": "MA",
                "zipcode": "02118",
                "address": "100 Main St",
                "city": "Boston",
            }],
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
        payload = _normalize_experience_years(payload)
        ser = ResumePayloadSer(data=payload)
        ser.is_valid(raise_exception=True)
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
- informations: legalfirstname, legallastname, legalmiddlename, npinumber, phone, emergencycontactemail, address, address2, city, stateprovince, zipcode, specialization, skills, total_experience_years
- preferences: clinicalpreferences
- digital_passport: LIST of objects with fields hospital_affiliation, licence_number, consultation_hours, available_locations, state, zipcode, address, address2, city
- education/medical_education/graduate_school: institution, degree, address, address2, city, state, zipcode, country, start, end, graduated
- internship/residency/fellowship: institution, specialty, address, address2, city, state, zipcode, country, start, end, program_type
- board_certifications: board, specialty, issue_date, expiry_date, status, certificate_id
- medical_licences: state, number, issue_date, expiry_date, status, is_primary
- dea_registration: dea_number, state, issue_date, expiry_date, status, schedules
- other_exams: exam_name, score, date, passed, details
- professional_reference: name, title, institution, phone, email, relationship
For preferences.clinicalpreferences, return a comma-separated string of clinical capabilities.
For digital_passport.available_locations, return a comma-separated list of locations.
If an entry has an institution, try to provide its full address (address, city, state, zipcode, country, and address2 if available).
For informations.emergencycontactemail, use only a public professional/provider/practice contact email found in the source text. Do not invent; omit if not present.
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
        info["emergencycontactemail"] = _clean_public_email(
            info.get("emergencycontactemail") or info.get("email")
        )
        info.pop("email", None)
        if "npinumber" in info:
            info["npinumber"] = re.sub(r"\D", "", str(info["npinumber"]))[:10]

        # filtro dict-ët dhe normalizo datat për seksionet
        for section in (
            "education", "medical_education", "graduate_school", "internship",
            "residency", "fellowship", "board_certifications", "medical_licences",
            "dea_registration", "other_exams", "professional_reference"
        ):
            min_fields = 1 if section == "internship" else 2
            data[section] = _normalize_section_rows(
                data.get(section),
                date_keys=("issue_date", "expiry_date", "start", "end", "date"),
                min_fields=min_fields,
            )

        data["informations"] = {
            **(info or {}),
            **{k: v for k, v in (base_info or {}).items() if v}
        }
        data = _normalize_contact_email(data)

        prefs = data.get("preferences") or {}
        cp = _normalize_clinical_preferences(prefs.get("clinicalpreferences") or data.get("clinicalpreferences"))
        if cp:
            prefs["clinicalpreferences"] = cp
        data["preferences"] = prefs

        dp_raw = data.get("digital_passport") or data.get("DigitalPassport") or {}
        if not dp_raw:
            dp_raw = {k: data.get(k) for k in (
                "hospital_affiliation",
                "licence_number",
                "consultation_hours",
                "available_locations",
                "state",
                "zipcode",
                "address",
                "address2",
                "city",
            ) if data.get(k)}
        dp = _normalize_digital_passport(dp_raw)
        if dp:
            data["digital_passport"] = dp

        data = _normalize_experience_years(data)
        ser = ResumePayloadSer(data=data)
        ser.is_valid(raise_exception=True)
        quality = data.get("quality") or {"completeness_score": 0.8, "confidence_score": 0.75}
        return Response({"detail": "ok", "parsed": ser.data, "quality": quality})

    except json.JSONDecodeError as e:
        return Response({"detail": f"parse json failed: {e}"}, status=502)
    except Exception as e:
        return Response({"detail": f"parse failed: {e}"}, status=500)


@api_view(["GET"])
def enrich_by_npi(request):

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

        return Response({
            "detail": "ok",
            "parsed": last.payload or {},
            "quality": last.quality or {},
            "cache_status": "hit",
            "job_id": str(last.job_id),
            "created_at": last.created_at.isoformat(),
        })

    if only_cache:

        return Response({
            "detail": "ok",
            "parsed": (last.payload if last else {}) or {},
            "quality": (last.quality if last else {}) or {},
            "cache_status": "stale" if last else "miss",
            "job_id": str(last.job_id) if last else None,
            "created_at": last.created_at.isoformat() if last else None,
        })
    # ─────────────────────────────────────────────────────────────────────

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

    # ── genai client & tools (ONE-SHOT PROMPT) ────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
    if not api_key:
        return Response({"detail": "GEMINI_API_KEY missing"}, status=500)

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GENAI_MODEL", GENAI_MODEL_DEFAULT)
    tools = [types.Tool(google_search=types.GoogleSearch())]

    try:
        npi_seed_json = json.dumps(npi_seed, ensure_ascii=False)
    except Exception:
        npi_seed_json = "{}"

    cfg = types.GenerateContentConfig(
        tools=tools,
        temperature=0.25, 
    )

    
    prompt = f"""
You are a web-grounded medical provider data extraction agent.

Use Google Search (NPPES, hospital profiles, CVs, .edu domains, PDFs, etc.)
to build a structured public profile for the provider with:

- NPI: {npi}
- Name hint: "{name_hint or "unknown"}"

You are also given context directly from the official NPI registry:
NPI_REGISTRY_CONTEXT = {npi_seed_json}

Your task is to return STRICT JSON (no markdown, no extra text) with
this top-level structure and semantics:

TOP-LEVEL KEYS (all lowercase):

- "informations": SINGLE OBJECT with fields:
    legalfirstname, legallastname, legalmiddlename,
    npinumber, phone, address, address2, city, stateprovince, zipcode,
    mailaddress, mailaddress2, mailcity, mailingstateprovince, mailingzipcode,
    role, gender, specialization, skills, total_experience_years, emergencycontactemail

- "preferences": SINGLE OBJECT with fields:
    clinicalpreferences

- "digital_passport": LIST of objects with fields:
    hospital_affiliation, licence_number, consultation_hours, available_locations,
    state, zipcode, address, address2, city

- "education": LIST of objects
- "medical_education": LIST of objects
- "graduate_school": LIST of objects
  For each entry in these 3 lists use only these fields:
    institution, degree, address, address2, city, state, zipcode, country, start, end, graduated

- "internship", "residency", "fellowship": LIST of objects.
  For each entry use only these fields:
    institution, specialty, address, address2, city, state, zipcode, country, start, end, program_type
  Note: Internship may be listed as "preliminary year", "rotating internship", or "PGY-1".
  Note: Internship may be listed as "preliminary year", "rotating internship", or "PGY-1".

- "board_certifications": LIST of objects with fields:
    board, specialty, issue_date, expiry_date, status, certificate_id

- "medical_licences": LIST of objects with fields:
    state, number, issue_date, expiry_date, status, is_primary

- "dea_registration": LIST of objects with fields:
    dea_number, state, issue_date, expiry_date, status, schedules

- "other_exams": LIST of objects with fields:
    exam_name, score, date, passed, details

- "professional_reference": LIST of objects with fields:
    name, title, institution, phone, email, relationship

- "quality": SINGLE OBJECT with fields:
    completeness_score, confidence_score
  Both completeness_score and confidence_score MUST be between 0 and 1.

IMPORTANT NORMALIZATION RULES:

1) PHONE:
   - Normalize phones to the E.164-like format "+1XXXXXXXXXX" when possible.
   - If you cannot normalize, leave phone empty.

1b) EMAIL:
   - informations.emergencycontactemail must be a public professional/provider/practice contact email.
   - Search provider profile, practice, hospital affiliation, and public professional pages.
   - Do NOT return guessed emails, private emails, or unrelated directory emails.
   - If no reliable public email is found, leave emergencycontactemail empty or omit it.

2) DATES:
   - Normalize dates to "YYYY-MM-DD" when exact, or "YYYY-MM" / "YYYY" if only partial.
   - For start/end, prefer YYYY-MM or YYYY-MM-DD.
   - For issue_date / expiry_date, do the same.
   - If date is unknown or ambiguous, omit the field entirely.

3) GROUPING (NO DUPLICATE HALF-ENTRIES):
   - Each object in education/medical_education/graduate_school/internship/residency/fellowship
     MUST represent ONE logical program (one degree or one training program).
   - NEVER split a single program into two separate objects.
     Example of BAD behavior (do NOT do this):
       - Object A: {{ "institution": "Howard University", "degree": "MD" }}
       - Object B: {{ "graduated": "2002-01-01" }}
     Instead you MUST MERGE these into ONE object:
       - {{ "institution": "Howard University", "degree": "MD", "graduated": "2002-01-01" }}

   - If you find additional info for the same program later (same institution AND same degree,
     or clearly the same program), MERGE it into the SAME object, do NOT create a new one.

4) INSTITUTION ADDRESS COMPLETENESS (STRICT):
   - For ANY entry that has an institution (education/medical_education/graduate_school/internship/residency/fellowship),
     you MUST attempt to find the institution's full address using Google Search.
   - Each emitted entry with institution MUST include at least:
     address, city, state, zipcode (and country if available).
   - If you cannot verify the address, OMIT THE ENTIRE ENTRY (do not return a partial row).
   - Do NOT guess.

5) MINIMAL COMPLETENESS FOR ROWS:
   - For any entry in education/medical_education/graduate_school:
       Include the object ONLY if you know at least TWO of these:
         institution, degree, city, state, start, end, graduated.
       If you only know 1 small fact (e.g., just "graduated"), DO NOT emit that entry.

   - For internship/residency/fellowship:
       Include the object ONLY if you know at least TWO of:
         institution, specialty, city, state, start, end, program_type.

   - For medical_licences / board_certifications:
       Include the object ONLY if you know at least TWO of:
         state/board, number/specialty/status, or issue_date/expiry_date.

   - In general, AVOID creating "half-empty" rows with a single trivial field.

6) INFORMATIONS MERGE:
   - Use NPI_REGISTRY_CONTEXT as authoritative for npinumber and base address/phone.
   - You MAY refine phone, gender, specialization, and skills if you find better data.
   - npinumber MUST be the 10-digit NPI for this provider.
   - You may enrich specialization with more detailed specialties.

7) EXPERIENCE YEARS:
   - informations.total_experience_years must be an integer number of years, not text.
   - Only set it when there is public evidence for a reliable start year.
   - Reliable start-year evidence includes: first medical license issue year, first attending/clinical role,
     first hospital affiliation, residency/fellowship start year, or medical school graduation year if that is
     the only available professional start evidence.
   - Calculate it as current calendar year minus the earliest reliable clinical/professional year.
   - Do NOT infer experience from age, seniority words, biography tone, or NPI creation date alone.
   - If the evidence is weak or ambiguous, omit total_experience_years or leave it empty.
   - Put the evidence explanation in informations._meta using:
       experience_basis_year, experience_source, experience_confidence
     where experience_confidence is "high", "medium", or "low".

8) QUALITY:
   - Set completeness_score to your estimate of how complete the overall profile is (0 to 1).
   - Set confidence_score to your overall confidence in the extracted data (0 to 1).

9) OUTPUT:
   - RETURN ONLY VALID JSON. NO markdown code fences. NO comments. NO extra keys.
   - All top-level keys must exist (you can use [] for empty lists).
10) CLINICAL PREFERENCES:
   - preferences.clinicalpreferences MUST be a comma-separated string of clinical capabilities.
   - If you are unsure, leave it empty.
11) DIGITAL PASSPORT:
   - Each object represents ONE hospital affiliation.
   - digital_passport.available_locations MUST be a comma-separated string of locations.
   - If you are unsure, leave the list empty.
"""

    data: Dict[str, Any] = {}

    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=cfg,
        )
        raw = _strip_md_fences(_resp_to_text(resp))
        if not raw:
            return Response({"detail": "enrich failed: empty model response"}, status=502)

        data = _json_relaxed(raw)

        # ── Normalizime bazë për 'informations' ─────────────────────────
        info = (data.get("informations") or {}) if isinstance(data, dict) else {}
        if "phone" in info:
            info["phone"] = clean_phone_number(info["phone"])
        info["emergencycontactemail"] = _clean_public_email(
            info.get("emergencycontactemail") or info.get("email")
        )
        info.pop("email", None)
        # siguro npinumber 10-shifror
        info["npinumber"] = re.sub(r"\D", "", str(info.get("npinumber") or npi))[:10]

        # merge me seed nga NPI registry (seed → fallback, info nga Gemini → mund të override)
        data["informations"] = {
            **{k: v for k, v in (npi_seed or {}).items() if v},
            **(info or {}),
        }
        data = _normalize_contact_email(data)
        info = (data.get("informations") or {}) if isinstance(data, dict) else {}
        if isinstance(info, dict) and not _clean_public_email(info.get("emergencycontactemail")):
            fallback_email = _find_public_contact_email(
                client=client,
                model_name=model_name,
                npi=npi,
                name_hint=name_hint,
                npi_seed=npi_seed,
            )
            if fallback_email:
                info["emergencycontactemail"] = fallback_email
                data["informations"] = info

        prefs = data.get("preferences") or {}
        cp = _normalize_clinical_preferences(prefs.get("clinicalpreferences") or data.get("clinicalpreferences"))
        if cp:
            prefs["clinicalpreferences"] = cp
        data["preferences"] = prefs

        dp_raw = data.get("digital_passport") or data.get("DigitalPassport") or {}
        if not dp_raw:
            dp_raw = {k: data.get(k) for k in (
                "hospital_affiliation",
                "licence_number",
                "consultation_hours",
                "available_locations",
                "state",
                "zipcode",
                "address",
                "address2",
                "city",
            ) if data.get(k)}
        dp = _normalize_digital_passport(dp_raw)
        if dp:
            data["digital_passport"] = dp

        # ── Normalizo seksionet me helper-in e ri (_normalize_section_rows) ──
        for section in (
            "education", "medical_education", "graduate_school", "internship",
            "residency", "fellowship", "board_certifications", "medical_licences",
            "dea_registration", "other_exams", "professional_reference",
        ):
            min_fields = 1 if section == "internship" else 2
            data[section] = _normalize_section_rows(
                data.get(section),
                date_keys=("issue_date", "expiry_date", "start", "end", "date", "graduated"),
                min_fields=min_fields,
            )

        data = _normalize_experience_years(data)

        # Siguro që 'quality' ekziston dhe është në [0,1]
        quality = data.get("quality") or {"completeness_score": 0.7, "confidence_score": 0.9}
        for k in ("completeness_score", "confidence_score"):
            v = quality.get(k)
            if isinstance(v, (int, float)):
                if v > 1:
                    quality[k] = round(float(v) / 100.0, 3)
                else:
                    quality[k] = float(v)
            else:
                quality[k] = 0.7 if k == "completeness_score" else 0.9
        data["quality"] = quality

        # ── Validim & RUAN (e njëjta logjikë si më parë) ─────────────────
        try:
            ser = ResumePayloadSer(data=data)
            ser.is_valid(raise_exception=True)

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
                _save_capture(npi, name_hint, data, quality, status="PARTIAL")
            return Response({
                "detail": "ok (partial-validated)",
                "parsed": data,
                "quality": quality,
                "cache_status": "fresh",
            })

    except json.JSONDecodeError as e:
        return Response({"detail": f"json decode failed: {e}"}, status=502)
    except Exception as e:

        if getattr(settings, "SAVE_CAPTURE", False):
            try:
                _save_capture(npi, name_hint, data if isinstance(data, dict) else {}, {}, status="ERROR")
            except Exception:
                pass
        return Response({"detail": f"enrich failed: {e}"}, status=500)


@api_view(["GET"])
def list_captures(request):
    
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
