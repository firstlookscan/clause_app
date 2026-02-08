# firstlook_scan/types.py
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


SCHEMA_VERSION = "1.0.0"


class FileStatus(str, Enum):
    OK = "ok"
    NEEDS_OCR = "needs_ocr"
    ERROR = "error"


class RiskLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class ClauseFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clause_type: str
    risk_level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)

    excerpt: str
    why_this_matters: str

    # Optional future-proof fields
    evidence: List[Evidence] = Field(default_factory=list)


class ExecutiveSummary(BaseModel):
    model_config = ConfigDict(extra="allow")  # keep flexible for now

    counts_by_clause_type: Dict[str, int] = Field(default_factory=dict)
    counts_by_risk_level: Dict[str, int] = Field(default_factory=dict)
    top_risks: List[Dict[str, Any]] = Field(default_factory=list)
    guidance: Optional[str] = None


class DocumentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_name: str
    file_path: str
    status: FileStatus

    scanned_at: Optional[str] = None  # you store ISO Z strings already
    preview: Optional[str] = None
    truncated: bool = False

    clauses: List[ClauseFinding] = Field(default_factory=list)
    executive_summary: Optional[ExecutiveSummary] = None

    # Non-ok extras
    note: Optional[str] = None
    error: Optional[str] = None


class BatchSummary(BaseModel):
    model_config = ConfigDict(extra="allow")  # flexible; you may evolve

    total_files: int = 0
    ok_files: int = 0
    needs_ocr_files: int = 0
    error_files: int = 0
    counts_by_clause_type: Dict[str, int] = Field(default_factory=dict)
    counts_by_risk_level: Dict[str, int] = Field(default_factory=dict)
    top_risks_across_all_contracts: List[Dict[str, Any]] = Field(default_factory=list)
    guidance: Optional[str] = None
    needs_ocr: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class ScanMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    input_path: str
    scanned_at: str
    disclaimer: str
    demo_mode: bool = False
    settings: Dict[str, Any] = Field(default_factory=dict)


class ScanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: ScanMeta
    batch_summary: BatchSummary
    contracts: List[DocumentResult] = Field(default_factory=list)
