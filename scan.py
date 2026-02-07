import sys
import json
import os
import re
import csv
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


def safe_preview(text: str, n: int = 160) -> str:
    t = normalize_ws(text)
    return t[:n] + ("..." if len(t) > n else "")


def list_contract_files(path: Path):
    if path.is_file():
        return [path]

    if path.is_dir():
        files = []
        for ext in ("*.pdf", "*.txt", "*.docx"):
            files.extend(path.glob(ext))
        # sort for stable output
        files = sorted(files, key=lambda p: p.name.lower())
        return files

    raise SystemExit("Input path not found")


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
        return "Medium"

    if clause_type == "mfn":
        return "Medium"

    if clause_type == "revenue_commitment":
        if "minimum" in t or "minimum annual" in t:
            return "High"
        return "Medium"

    return "Low"


# ----------------------------
# CONFIDENCE (text clarity; not legal certainty)
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

    if suffix == ".docx":
        from docx import Document
        doc = Document(str(file_path))
        parts = []
        for p in doc.paragraphs:
            if p.text and p.text.strip():
                parts.append(p.text.strip())
        text = "\n".join(parts).strip()
        if not text or len(text) < 50:
            raise ValueError("DOCX did not contain enough text to analyze.")
        return text

    if suffix == ".pdf":
        text = extract_text_from_pdf_pdfplumber(file_path)
        if not text or len(text.strip()) < 50:
            text = extract_text_from_pdf_pypdf(file_path)

        if not text or len(text.strip()) < 50:
            raise ValueError(
                "Could not extract meaningful text (may be a scanned image PDF requiring OCR)."
            )
        return text

    raise ValueError("Supported inputs: .txt, .pdf")


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

    risk_rank = {"High": 2, "Medium": 1, "Low": 0}
    top = sorted(
        clause_entries,
        key=lambda x: (risk_rank.get(x["risk_level"], 0), x.get("confidence", 0.0)),
        reverse=True
    )[:5]

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
# SCAN ONE FILE
# ----------------------------

def scan_one(file_path: Path):
    started = datetime.now(timezone.utc)
    try:
        text = extract_text(file_path)
        preview = safe_preview(text)
        detected = ai_detect_clauses(text)

        clauses = []
        for item in detected:
            clause_type = item["clause_type"]
            excerpt = trim_excerpt(item["excerpt"])
            risk = score_risk(clause_type, excerpt)
            explanation = explain(clause_type, risk)
            conf = confidence_score(clause_type, excerpt)

            clauses.append({
                "clause_type": clause_type,
                "risk_level": risk,
                "confidence": conf,
                "excerpt": excerpt,
                "why_this_matters": explanation
            })

        summary = build_executive_summary(clauses)

        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "status": "ok",
            "preview": preview,
            "clauses": clauses,
            "executive_summary": summary,
            "scanned_at": started.isoformat().replace("+00:00", "Z")
        }

    except Exception as e:
        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "status": "error",
            "error": str(e),
            "scanned_at": started.isoformat().replace("+00:00", "Z")
        }


# ----------------------------
# BATCH SUMMARY
# ----------------------------

def build_batch_summary(file_results):
    total_files = len(file_results)
    ok_files = [r for r in file_results if r.get("status") == "ok"]
    error_files = [r for r in file_results if r.get("status") == "error"]

    # Aggregate counts across all files
    agg_by_type = {t: 0 for t in CLAUSE_TYPES}
    agg_by_risk = {"High": 0, "Medium": 0, "Low": 0}

    all_clauses_flat = []
    for r in ok_files:
        for c in r.get("clauses", []):
            agg_by_type[c["clause_type"]] += 1
            agg_by_risk[c["risk_level"]] += 1
            all_clauses_flat.append({
                "file_name": r["file_name"],
                **c
            })

    # Top risks across the batch
    risk_rank = {"High": 2, "Medium": 1, "Low": 0}
    top_batch = sorted(
        all_clauses_flat,
        key=lambda x: (risk_rank.get(x["risk_level"], 0), x.get("confidence", 0.0)),
        reverse=True
    )[:10]

    guidance = "Start with contracts that have High-risk termination or change-of-control clauses, then review MFN and commitment clauses for pricing/margin implications."

    return {
        "total_files": total_files,
        "ok_files": len(ok_files),
        "error_files": len(error_files),
        "counts_by_clause_type": agg_by_type,
        "counts_by_risk_level": agg_by_risk,
        "top_risks_across_all_contracts": top_batch,
        "guidance": guidance,
        "errors": [{"file_name": e["file_name"], "error": e["error"]} for e in error_files]
    }


def write_batch_csv(out_path: Path, file_results):
    # Flatten to rows: one row per clause
    rows = []
    for r in file_results:
        if r.get("status") != "ok":
            continue
        for c in r.get("clauses", []):
            rows.append({
                "file_name": r["file_name"],
                "clause_type": c["clause_type"],
                "risk_level": c["risk_level"],
                "confidence": f"{c['confidence']:.2f}",
                "excerpt": normalize_ws(c["excerpt"]),
                "why_this_matters": c["why_this_matters"]
            })

    fieldnames = ["file_name", "clause_type", "risk_level", "confidence", "excerpt", "why_this_matters"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ----------------------------
# MAIN
# ----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python scan.py <file_or_folder>")
        print("Examples:")
        print("  python scan.py samples")
        print("  python scan.py samples\\contract.pdf")
        return

    input_path = Path(sys.argv[1])
    files = list_contract_files(input_path)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Batch scan
    print(f"\nScanning {len(files)} file(s)...\n")
    results = []
    for i, fp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {fp.name}")
        res = scan_one(fp)
        results.append(res)

        if res.get("status") == "ok":
            s = res["executive_summary"]["counts_by_risk_level"]
            print(f"   OK | High:{s['High']}  Med:{s['Medium']}  Low:{s['Low']} | preview: {res.get('preview','')}\n")
        else:
            print(f"   ERROR | {res.get('error')}\n")

    batch_summary = build_batch_summary(results)

    batch_output = {
        "metadata": {
            "input_path": str(input_path),
            "scanned_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "disclaimer": "This output highlights clauses commonly flagged during M&A diligence and does not constitute legal advice."
        },
        "batch_summary": batch_summary,
        "contracts": results
    }

    json_path = out_dir / f"batch_{stamp}.json"
    json_path.write_text(json.dumps(batch_output, indent=2), encoding="utf-8")

    csv_path = out_dir / f"batch_{stamp}.csv"
    write_batch_csv(csv_path, results)

    # Print an organized “at-a-glance” summary
    print("\n=== BATCH SUMMARY ===")
    print(f"Files scanned: {batch_summary['total_files']} (ok: {batch_summary['ok_files']}, errors: {batch_summary['error_files']})")
    print("Clause counts:", batch_summary["counts_by_clause_type"])
    print("Risk counts:", batch_summary["counts_by_risk_level"])
    if batch_summary["top_risks_across_all_contracts"]:
        print("\nTop risks across all contracts:")
        for tr in batch_summary["top_risks_across_all_contracts"][:5]:
            print(f"- {tr['file_name']} | {tr['clause_type']} | {tr['risk_level']} | conf {tr['confidence']:.2f}")
    print("\nOutputs:")
    print(f"- JSON: {json_path}")
    print(f"- CSV:  {csv_path}\n")


if __name__ == "__main__":
    main()
