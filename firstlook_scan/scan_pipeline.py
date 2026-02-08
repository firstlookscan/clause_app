from __future__ import annotations

import hashlib
from typing import Optional, List, Tuple, Any, Dict

from firstlook_scan.types import (
    ScanResult, ScanMeta, DealSummary,
    DocumentResult, DocumentMeta, FileType
)

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def infer_file_type(filename: str) -> FileType:
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return FileType.PDF
    if fn.endswith(".docx"):
        return FileType.DOCX
    if fn.endswith(".txt"):
        return FileType.TXT
    return FileType.OTHER

def run_scan_canonical(
    *,
    scan_id: str,
    files: List[Tuple[str, bytes]],  # (filename, bytes)
    demo_mode: bool,
    app_version: Optional[str] = None,
    primary_model: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> ScanResult:
    settings = settings or {}

    meta = ScanMeta(
        scan_id=scan_id,
        demo_mode=demo_mode,
        app_version=app_version,
        primary_model=primary_model,
        settings=settings,
    )

    docs = []
    for filename, blob in files:
        doc_meta = DocumentMeta(
            document_id=_hash_bytes(blob),
            filename=filename,
            file_type=infer_file_type(filename),
            size_bytes=len(blob),
            source_label="uploaded" if not demo_mode else "demo_or_uploaded",
        )
        docs.append(DocumentResult(meta=doc_meta))

    deal = DealSummary(
        deal_name=settings.get("deal_name"),
        total_documents=len(docs),
    )

    return ScanResult(meta=meta, deal=deal, documents=docs)
