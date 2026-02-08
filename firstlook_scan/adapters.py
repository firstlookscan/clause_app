# firstlook_scan/adapters.py
from __future__ import annotations

from typing import Any, Dict

from firstlook_scan.types import (
    ScanResult, ScanMeta, BatchSummary,
    DocumentResult, ClauseFinding, ExecutiveSummary,
    FileStatus, RiskLevel, Evidence,
)

def _to_file_status(s: str) -> FileStatus:
    if s == "ok":
        return FileStatus.OK
    if s == "needs_ocr":
        return FileStatus.NEEDS_OCR
    return FileStatus.ERROR

def _to_risk_level(s: str) -> RiskLevel:
    if s not in ("High", "Medium", "Low"):
        # default safely
        return RiskLevel.LOW
    return RiskLevel(s)

def legacy_batch_output_to_scanresult(batch_output: Dict[str, Any], *, demo_mode: bool = False) -> ScanResult:
    md = batch_output.get("metadata", {})
    meta = ScanMeta(
        input_path=str(md.get("input_path", "")),
        scanned_at=str(md.get("scanned_at", "")),
        disclaimer=str(md.get("disclaimer", "")),
        demo_mode=demo_mode,
        settings={},
    )

    bs = batch_output.get("batch_summary", {}) or {}
    batch_summary = BatchSummary(**bs)

    contracts = []
    for r in (batch_output.get("contracts", []) or []):
        clauses = []
        for c in (r.get("clauses", []) or []):
            excerpt = c.get("excerpt", "") or ""
            why = c.get("why_this_matters", "") or ""
            # Evidence list is optional; keep the excerpt as evidence for future UI “click to source”
            evidence = [Evidence(text=excerpt)] if excerpt else []

            clauses.append(
                ClauseFinding(
                    clause_type=str(c.get("clause_type", "")),
                    risk_level=_to_risk_level(str(c.get("risk_level", "Low"))),
                    confidence=float(c.get("confidence", 0.0) or 0.0),
                    excerpt=excerpt,
                    why_this_matters=why,
                    evidence=evidence,
                )
            )

        exec_sum = r.get("executive_summary")
        executive_summary = ExecutiveSummary(**exec_sum) if isinstance(exec_sum, dict) else None

        contracts.append(
            DocumentResult(
                file_name=str(r.get("file_name", "")),
                file_path=str(r.get("file_path", "")),
                status=_to_file_status(str(r.get("status", "error"))),
                scanned_at=r.get("scanned_at"),
                preview=r.get("preview"),
                truncated=bool(r.get("truncated", False)),
                clauses=clauses,
                executive_summary=executive_summary,
                note=r.get("note"),
                error=r.get("error"),
            )
        )

    return ScanResult(metadata=meta, batch_summary=batch_summary, contracts=contracts)
