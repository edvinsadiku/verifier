from django.contrib import admin
from .models import EnrichmentCapture

@admin.register(EnrichmentCapture)
class EnrichmentCaptureAdmin(admin.ModelAdmin):
    list_display = ("npi", "status", "created_at", "job_id")
    list_filter  = ("status", "created_at")
    search_fields = ("npi", "name_hint", "job_id")
    readonly_fields = ("job_id", "created_at", "updated_at")
