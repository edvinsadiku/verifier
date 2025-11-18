import uuid
from datetime import datetime, timedelta, timezone

from django.db import models
from django.core.validators import RegexValidator


class EnrichmentCapture(models.Model):
    STATUS_READY   = "ready"
    STATUS_OK      = "OK"
    STATUS_OK_FAKE = "OK_FAKE"
    STATUS_PARTIAL = "PARTIAL"
    STATUS_ERROR   = "error"

    STATUS_CHOICES = [
        (STATUS_READY, "Ready"),
        (STATUS_OK, "OK"),
        (STATUS_OK_FAKE, "OK (Fake)"),
        (STATUS_PARTIAL, "Partial"),
        (STATUS_ERROR, "Error"),
    ]

    job_id = models.UUIDField(
        default=uuid.uuid4, unique=True, editable=False,
        help_text="UUID i kapjes (job)."
    )

    npi = models.CharField(
        max_length=10,
        db_index=True,
        validators=[RegexValidator(regex=r"^\d{10}$", message="NPI duhet të ketë saktësisht 10 shifra.")],
        help_text="NPI (10 shifra).",
    )

    name_hint = models.TextField(
        blank=True, null=True,
        help_text="Emër/mbiemër qenësor nga lookup-i (opsionale)."
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_READY,
        db_index=True,
        help_text="Gjendja e kapjes."
    )

    quality = models.JSONField(
        null=True, blank=True,
        help_text="Metriqa e cilësisë: p.sh. completeness_score, confidence_score."
    )

    payload = models.JSONField(
        help_text="Payload i strukturuar (normalized) që kthehet nga enrichment/parsing."
    )

    pass_stats = models.JSONField(
        null=True, blank=True,
        help_text="Statistika kalimesh (timings, numri i kërkimeve, etj). Opsionale."
    )

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        # Table name do të jetë api_enrichmentcapture kur app quhet 'api'
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["npi", "-created_at"], name="idx_npi_created_desc"),
            models.Index(fields=["status"], name="idx_status"),
            models.Index(fields=["-created_at"], name="idx_created_desc"),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(npi__regex=r"^\d{10}$"),
                name="enrc_npi_10_digits",
            ),
        ]
        verbose_name = "Enrichment Capture"
        verbose_name_plural = "Enrichment Captures"

    def __str__(self) -> str:
        return f"{self.npi} @ {self.created_at.isoformat()} ({self.status})"

    def is_fresh(self, max_age_seconds: int = 7 * 24 * 3600) -> bool:
        """
        Kthen True nëse kjo kapje është brenda TTL-së (default 7 ditë).
        Përdor për cache HIT logic në view.
        """
        try:
            now = datetime.now(timezone.utc)
            return (now - self.created_at) <= timedelta(seconds=max_age_seconds)
        except Exception:
            return False
