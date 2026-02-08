from __future__ import annotations

import pandas as pd
from firstlook_scan.types import ScanResult

def scanresult_to_findings_df(scan_result: ScanResult) -> pd.DataFrame:
    rows = []
    for doc in scan_result.contracts:
        if doc.status.value != "ok":
            continue
        for clause in doc.clauses:
            rows.append({
                "file_name": doc.file_name,
                "clause_type": clause.clause_type,
                "risk_level": clause.risk_level.value,
                "confidence": float(clause.confidence),
                "excerpt": clause.excerpt,
                "why_this_matters": clause.why_this_matters,
            })
    if not rows:
        return pd.DataFrame(columns=["file_name","clause_type","risk_level","confidence","excerpt","why_this_matters"])
    return pd.DataFrame(rows)

def export_findings_csv_bytes(scan_result: ScanResult) -> bytes:
    df = scanresult_to_findings_df(scan_result)
    return df.to_csv(index=False).encode("utf-8")
