import re, pdfplumber
from datetime import datetime

def extract_pdf_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            text += (p.extract_text() or "") + "\n"
    return text

def clean_phone_number(phone: str) -> str:
    if not phone: return ""
    digits = re.sub(r"\D","", str(phone))
    if len(digits)==10: return f"+1{digits}"
    if len(digits)==11 and digits.startswith("1"): return f"+{digits}"
    return f"+{digits}" if digits else ""

def parse_resume_date(value: str):
    if not value: return None
    try:
        if re.match(r"^\d{1,2}/\d{4}$", value):
            return datetime.strptime(value, "%m/%Y").date().replace(day=1).isoformat()
        if re.match(r"^\d{4}$", value):
            return datetime.strptime(value, "%Y").date().replace(month=1, day=1).isoformat()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            return value
    except:
        return None
    return None
