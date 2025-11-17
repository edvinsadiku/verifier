import uuid
from django.db import models
from django.contrib.postgres.fields import JSONField  # Django 5.1: use models.JSONField

class EnrichmentCapture(models.Model):
    job_id     = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    npi        = models.CharField(max_length=10, db_index=True)
    name_hint  = models.TextField(blank=True, null=True)
    status     = models.CharField(max_length=20, default="ready")  # queued|running|ready|partial|error
    quality    = models.JSONField(null=True, blank=True)
    payload    = models.JSONField()            # gjithë JSON-i i Gemini (normalized)
    pass_stats = models.JSONField(null=True, blank=True)  # timing info, numri i queries, etj (opsionale)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["npi", "-created_at"]),
            models.Index(fields=["status"]),
        ]
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.npi} @ {self.created_at.isoformat()} ({self.status})"
