import re, time, requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; VerifierBot/1.0; +https://example.com/bot)"
}

def fetch_url(url: str, timeout: int = 10) -> str:
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code != 200 or "text/html" not in (r.headers.get("Content-Type") or ""):
            return ""
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script","style","noscript","header","footer","nav","aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text[:100_000]
    except Exception:
        return ""

def harvest_sources(urls, max_pages=8) -> str:
    harvested = []
    for url in urls[:max_pages]:
        txt = fetch_url(url)
        if txt:
            harvested.append(f"[SOURCE] {url}\n{txt}\n")
        time.sleep(0.4)  # mos tepro me shpejtësi
    return "\n\n".join(harvested)[:400_000]
