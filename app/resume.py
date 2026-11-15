"""Download a resume from S3/MinIO and extract plain text."""

import io
import logging

import boto3
from docx import Document
from pypdf import PdfReader

from .config import get_settings

log = logging.getLogger(__name__)


def _s3():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint or None,
        region_name=s.s3_region,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
    )


def extract_text(data: bytes, key: str) -> str:
    name = key.lower()
    if name.endswith(".pdf") or data[:4] == b"%PDF":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    return data.decode("utf-8", errors="ignore").strip()


def parse_resume(storage_key: str) -> str:
    s = get_settings()
    try:
        obj = _s3().get_object(Bucket=s.s3_bucket, Key=storage_key)
        return extract_text(obj["Body"].read(), storage_key)
    except Exception as e:  # storage down in local dev — keep the pipeline moving
        if s.fake:
            log.warning("resume fetch failed (%s); returning placeholder text", e)
            return f"FAKE PARSED RESUME for {storage_key}\nSkills: Node.js, TypeScript, PostgreSQL, Redis"
        raise


def fetch_document(storage_key: str) -> str | None:
    """Best-effort text of any uploaded document (JD, question bank); None if unavailable."""
    s = get_settings()
    try:
        obj = _s3().get_object(Bucket=s.s3_bucket, Key=storage_key)
        return extract_text(obj["Body"].read(), storage_key) or None
    except Exception as e:
        log.warning("could not read %s: %s", storage_key, e)
        return None
