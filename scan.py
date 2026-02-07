import sys
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone

import pdfplumber
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# SETUP
# ----------------------------

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("ERROR: OPENAI_API_KEY not found. Put it in .env as OPENAI_API_KEY=sk-...")

client = OpenAI(api_key=api_key)

CLAUSE_TYPES = ["change_of_control", "termination", "exclusivity", "mfn", "revenue_commitment"]
MAX_EXCERPTS_PER_TYPE = 3


# ----------------------------
# UTILITIES
# ----------------------------

def trim_excerpt(excerpt: str, limit: int = 800) -> str:
    excerpt = excerpt.strip()
    if len(excerpt) <= limit:
        return excerpt
    head = excerpt[:500].rstrip()
    tail = excerpt[-250:].lstrip()
    return head + "\n...\n" + tail


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def dedupe_items(items):
    seen = set()
    out = []
    for it in items:
        key = (it.get("clause_type", ""), normalize_ws(it.get("excerpt", "")).lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ----------------------------
# RISK HEURISTICS (V0)
# ----------------------------

def score_risk(clause_type, text):
    t = text.lower()

    if clause_type == "change_of_control":
        if "terminate" in t or "termination" in t:
            return "High"
        return "Medium"

    if clause_type == "termination":
        if "for convenience" in t or "for its convenience" in t or "without cause" in t:
            return "High"
        return "Medium"

    if clause_type == "exclusivity":
        # You can add specific triggers later; keep conservative now
        return "Medium"

    if clause_type == "mfn":
        # Later: detect retroactive, broad scope, etc.
        return "Medium"

    if clause_type == "revenue_commitment":
        if "minimum" in t or "minimum annual" in t:
            return "High"
        return "Medium"

    return "Low"


# ----------------------------
# CONFIDENCE (simple + explainable)
# This is NOT legal certainty; it's "text clarity for this clause type".
# ----------------------------

def confidence_score(clause_type: str, excerpt: str) -> float:
    t = excerpt.lower()

    anchors = {
        "change_of_control": ["change of control", "change-of-control", "assignment", "assign", "merger", "acquisition"],
        "termination": ["terminate", "termination", "for convenience", "for its convenience", "without cause", "notice"],
        "exclusivity": ["exclusive", "exclusivity", "non-compete", "competitor"],
        "mfn": ["most favored", "most favoured", "most favored nation", "price no less favorable", "no less favorable"],
        "revenue_commitment": ["minimum", "minimum annual", "commit", "commitment", "volume", "annual spend"]
    }

    hits = 0
    for a in anchors.get(clause_type, []):
        if a in t:
            hits += 1

    # Map hits to a conservative confidence score
    if hits >= 3:
        return 0.9
    if hits == 2:
        return 0.75
    if hits == 1:
        return 0.6
    return 0.5


# ----------------------------
# EXPLANATIONS (V0)
# ----------------------------

def explain(clause_type, risk):
    explanations = {
        "change_of_control": {
            "High": "This clause allows termination or material changes upon acquisition, which may lead to immediate revenue loss or deal friction post-close.",
            "Medium": "This clause may introduce consent or notice requirements that can complicate closing or integration.",
            "Low": "This clause does not appear to materially restrict assignment or ownership changes."
        },
        "termination": {
            "High": "This agreement can be terminated without cause or on short notice, reducing the durability of expected revenue.",
            "Medium": "Termination rights exist under certain conditions, creating manageable churn risk.",
            "Low": "Termination appears limited to material breach with reasonable protections."
        },
        "exclusivity": {
            "High": "This clause significantly restricts future business or integration options.",
            "Medium": "This clause limits flexibility and may constrain post-acquisition growth strategies.",
            "Low": "No meaningful exclusivity restrictions were identified."
        },
        "mfn": {
            "High": "This clause may cap pricing upside or trigger broad discount obligations post-acquisition.",
            "Medium": "This clause could limit pricing flexibility in future customer negotiations.",
            "Low": "No material pricing parity obligations were identified."
        },
        "revenue_commitment": {
            "High": "This clause creates minimum spend or service obligations that could result in financial exposure if unmet.",
            "Medium": "This clause introduces commitments that may affect margins or forecasting accuracy.",
            "Low": "No binding revenue or volume commitments were identified."
        }
    }
    return explanations.get(clause_type, {}).get(risk, "")


# ----------------------------
# TEXT EXTRACTION
# ----------------------------

def extract_text_from_pdf_pdfplumber(pdf_path: Path) -> str:
    chunks = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                chunks.append(txt)
    return "\n\n".join(chunks).strip()


def extract_text_from_pdf_pypdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            chunks.append(txt)
    return "\n\n".join(chunks).strip()


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(errors="ignore")

    if suffix == ".pdf":
        text = extract_text_from_pdf_pdfplumber(file_path)
        if not text or len(text.strip()) < 50:
            text = extract_text_from_pdf_pypdf(file_path)

        if not text or len(text.strip()) < 50:
            raise SystemExit(
                "ERROR: Could not extract meaningful text from this PDF.\n"
                "It may be a scanned image PDF (needs OCR). Try a selectable-text PDF first."
            )
        return text

    raise SystemExit("Supported inputs: .txt, .pdf")


# ----------------------------
# AI CLAUSE DETECTION (Safe: quote-only + multi excerpts)
# ----------------------------

def ai_detect_clauses(text: str):
    prompt = f"""
You are assisting with M&A diligence.

TASK:
Extract ONLY clauses that are explicitly stated in the contract text below.

CLAUSE TYPES (use these exact strings):
- change_of_control
- termination
- exclusivity
- mfn
- revenue_commitment

RULES:
- Only extract clauses if you can quote the exact contract language.
- Do NOT infer, paraphrase, or summarize clause language.
- For each clause type, return up to {MAX_EXCERPTS_PER_TYPE} distinct excerpts if present.
- If a clause type is not present, do not include it.
- Return VALID JSON only (no markdown, no commentary).

OUTPUT FORMAT (JSON array):
[
  {{
    "clause_type": "change_of_control|termination|exclusivity|mfn|revenue_commitment",
    "excerpt": "exact quoted contract language"
  }}
]

CONTRACT TEXT:
\"\"\"
{text}
\"\"\"
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = resp.choices[0].message.content.strip()
    data = json.loads(content)

    # Safety: keep only valid clause types and non-empty excerpts
    cleaned = []
    for item in data if isinstance(data, list) else []:
        ct = (item.get("clause_type") or "").strip()
        ex = (item.get("excerpt") or "").strip()
        if ct in CLAUSE_TYPES and ex:
            cleaned.append({"clause_type": ct, "excerpt": ex})

    return dedupe_items(cleaned)


# ----------------------------
# EXECUTIVE SUMMARY
# ----------------------------

def build_executive_summary(clause_entries):
    by_type = {t: 0 for t in CLAUSE_TYPES}
    by_risk = {"High": 0, "Medium": 0, "Low": 0}
    for c in clause_entries:
        by_type[c["clause_type"]] += 1
        by_risk[c["risk_level"]] += 1

    # Top risks: prioritize High, then Medium
    risk_rank = {"High": 2, "Medium": 1, "Low": 0}
    top = sorted(
        clause_entries,
        key=lambda x: (risk_rank.get(x["risk_level"], 0), x.get("confidence", 0.0)),
        reverse=True
    )[:5]

    # One-paragraph guidance
    if by_risk["High"] > 0:
        guidance = "Review the High-risk clauses first; these may affect revenue durability, closing conditions, or post-close flexibility."
    elif by_risk["Medium"] > 0:
        guidance = "Review the Medium-risk clauses next; these are commonly negotiable but may still create integration or pricing friction."
    else:
        guidance = "No target clauses were flagged as material based on the current extraction."

    return {
        "counts_by_clause_type": by_type,
        "counts_by_risk_level": by_risk,
        "top_risks": top,
        "guidance": guidance
    }


# ----------------------------
# MAIN
# ----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python scan.py <file.txt|file.pdf>")
        return

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        raise SystemExit("File not found")

    text = extract_text(file_path)

    print("\n[Extraction OK] Preview:")
    print(text[:200].replace("\n", " ") + ("..." if len(text) > 200 else ""))
    print()

    detected = ai_detect_clauses(text)

    results = {
        "metadata": {
            "file_name": file_path.name,
            "scanned_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "disclaimer": "This output highlights clauses commonly flagged during M&A diligence and does not constitute legal advice."
        },
        "clauses": [],
        "executive_summary": {}
    }

    print("=== CLAUSE INTELLIGENCE REPORT (AI) ===\n")

    if not detected:
        print("No target clauses detected.\n")

    # Build clause entries with risk + explanation + confidence
    for item in detected:
        clause_type = item["clause_type"]
        excerpt_raw = item["excerpt"]
        excerpt = trim_excerpt(excerpt_raw)

        risk = score_risk(clause_type, excerpt)
        explanation = explain(clause_type, risk)
        conf = confidence_score(clause_type, excerpt)

        entry = {
            "clause_type": clause_type,
            "risk_level": risk,
            "confidence": conf,
            "excerpt": excerpt,
            "why_this_matters": explanation
        }

        results["clauses"].append(entry)

        print(f"{clause_type.upper()}:")
        print(f"  - [{risk} RISK | conf {conf:.2f}] {excerpt}")
        if explanation:
            print(f"    → {explanation}")
        print()

    results["executive_summary"] = build_executive_summary(results["clauses"])

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{file_path.stem}_analysis.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"JSON output written to: {out_path}\n")


if __name__ == "__main__":
    main()
