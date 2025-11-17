# verifier_service/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv

# -------------------------------------------------
# Paths & .env
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")   # ✅ ngarko .env nga rrënja e projektit

# -------------------------------------------------
# Core
# -------------------------------------------------
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-secret-change-me")
DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes", "on")
ALLOWED_HOSTS = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "*").split(",") if h.strip()] or ["*"]

# -------------------------------------------------
# Apps
# -------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "api",
]

# -------------------------------------------------
# Middleware
# -------------------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "api.auth.verify_hmac_middleware",  # ✅ API key / HMAC auth mid
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# -------------------------------------------------
# URLs / WSGI
# -------------------------------------------------
ROOT_URLCONF = "verifier_service.urls"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    }
]
WSGI_APPLICATION = "verifier_service.wsgi.application"

# -------------------------------------------------
# Database
#  - Nëse DB_NAME ekziston → PostgreSQL
#  - Përndryshe → SQLite (dev fallback)
# -------------------------------------------------
DB_NAME = os.getenv("DB_NAME")
if DB_NAME:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": DB_NAME,
            "USER": os.getenv("DB_USER", ""),
            "PASSWORD": os.getenv("DB_PASS", ""),
            "HOST": os.getenv("DB_HOST", "127.0.0.1"),
            "PORT": os.getenv("DB_PORT", "5432"),
            "CONN_MAX_AGE": int(os.getenv("DB_CONN_MAX_AGE", "60")),
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

# -------------------------------------------------
# Static
# -------------------------------------------------
STATIC_URL = "static/"

# -------------------------------------------------
# DRF
# -------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
    ],
}

# -------------------------------------------------
# Env flags / external keys
# -------------------------------------------------
def _to_bool(val: str, default=False):
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "on")

VERIFIER_SHARED_SECRET = os.getenv("VERIFIER_SHARED_SECRET", "")  # ✅ përdoret nga api.auth
GOOGLE_API_KEY         = os.getenv("GOOGLE_API_KEY", "")
GEMINI_API_KEY         = os.getenv("GEMINI_API_KEY", "")
GENAI_MODEL            = os.getenv("GENAI_MODEL", "gemini-2.5-flash")
ENRICH_FAKE            = _to_bool(os.getenv("ENRICH_FAKE", "false"))
SAVE_CAPTURE           = _to_bool(os.getenv("SAVE_CAPTURE", "false"))

# -------------------------------------------------
# (opsionale) Logging bazik në console për debug
# -------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
