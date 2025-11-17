from django.urls import path
from .views import health, npi_lookup, parse_resume, enrich_by_npi, list_captures, get_capture

urlpatterns = [
    path("health/", health),
    path("npi-lookup/", npi_lookup),
    path("parse-resume/", parse_resume),
    path("enrich-by-npi/", enrich_by_npi),

    # ⬇⬇ Këto dy janë për listimin/marrjen e capture-ve
    path("captures/", list_captures),
    path("captures/<uuid:job_id>/", get_capture),
]
