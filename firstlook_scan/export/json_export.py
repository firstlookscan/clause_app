from __future__ import annotations
from firstlook_scan.types import ScanResult

def export_scanresult_json_bytes(scan_result: ScanResult) -> bytes:
    return scan_result.model_dump_json(indent=2).encode("utf-8")
