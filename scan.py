import sys
import json
import os
import re
import csv
import argparse
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

RISK_RANK = {"High": 2, "Medium": 1, "Low": 0}


# ----------------------------
# UTILITIES
# ----------------------------

def now_utc_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def trim_excerpt(excerpt: str, limit: int = 800) -> str:
    excerpt = (excerpt or "").strip()
    if len(excerpt) <= limit:
        return excerpt
    head = excerpt[:500].rstrip()
    tail = excerpt[-250:].lstrip()
    return head + "\n...\n" + tail


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


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


def list_contract_files(path: Path, max_files: int | None):
    if path.is_file():
        return [path]

    if path.is_dir():
        files = []
        for ext in ("*.pdf", "*.txt", "*.docx"):
            files.extend(path.glob(ext))
        files = sorted(files, key=lambda p: p.name.lower())
        if max_files is not None:
            files = files[:max_files]
        return files

    raise SystemExit("Input path not found")


# ----------------------------
# RISK HEURISTICS (V0)
# ----------------------------

def score_risk(clause_type, text):
    t = (text or "").lower()

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
    t = (excerpt or "").lower()
    anchors = {
        "change_of_control": ["change of control", "change-of-control", "assignment", "assign", "merger", "acquisition"],
        "termination": ["terminate", "termination", "for convenience", "for its convenience", "without cause", "notice"],
        "exclusivity": ["exclusive", "exclusivity", "non-compete", "competitor"],
        "mfn": ["most favored", "most favoured", "most favored nation", "price no less favorable", "no less favorable"],
        "revenue_commitment": ["minimum", "minimum annual", "commit", "commitment", "volume", "annual spend"]
    }
    hits = sum(1 for a in anchors.get(clause_type, []) if a in t)
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

def extract_text_from_pdf_pdfplumber(pdf_path: Path, max_pages: int | None) -> str:
    chunks = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = pdf.pages
        if max_pages is not None:
            pages = pages[:max_pages]
        for page in pages:
            txt = page.extract_text() or ""
            if txt.strip():
                chunks.append(txt)
    return "\n\n".join(chunks).strip()


def extract_text_from_pdf_pypdf(pdf_path: Path, max_pages: int | None) -> str:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages
    if max_pages is not None:
        pages = pages[:max_pages]
    chunks = []
    for page in pages:
        txt = page.extract_text() or ""
        if txt.strip():
            chunks.append(txt)
    return "\n\n".join(chunks).strip()


def extract_text_from_docx(docx_path: Path) -> str:
    from docx import Document
    doc = Document(str(docx_path))
    parts = []
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())
    return "\n".join(parts).strip()


def extract_text(file_path: Path, max_pages: int | None) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(errors="ignore")

    if suffix == ".docx":
        text = extract_text_from_docx(file_path)
        if not text or len(text) < 50:
            raise ValueError("DOCX did not contain enough text to analyze.")
        return text

    if suffix == ".pdf":
        text = extract_text_from_pdf_pdfplumber(file_path, max_pages=max_pages)
        if not text or len(text.strip()) < 50:
            text = extract_text_from_pdf_pypdf(file_path, max_pages=max_pages)
        if not text or len(text.strip()) < 50:
            # Treat this as scanned PDF needing OCR
            raise RuntimeError("needs_ocr")
        return text

    raise ValueError("Supported inputs: .txt, .pdf, .docx")


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
# SUMMARIES
# ----------------------------

def build_executive_summary(clause_entries):
    by_type = {t: 0 for t in CLAUSE_TYPES}
    by_risk = {"High": 0, "Medium": 0, "Low": 0}

    for c in clause_entries:
        by_type[c["clause_type"]] += 1
        by_risk[c["risk_level"]] += 1

    top = sorted(
        clause_entries,
        key=lambda x: (RISK_RANK.get(x["risk_level"], 0), x.get("confidence", 0.0)),
        reverse=True
    )[:5]

    if by_risk["High"] > 0:
        guidance = "Review High-risk clauses first; these may affect revenue durability, closing conditions, or post-close flexibility."
    elif by_risk["Medium"] > 0:
        guidance = "Review Medium-risk clauses next; these are commonly negotiable but may still create integration or pricing friction."
    else:
        guidance = "No target clauses were flagged as material based on the current extraction."

    return {
        "counts_by_clause_type": by_type,
        "counts_by_risk_level": by_risk,
        "top_risks": top,
        "guidance": guidance
    }


def build_batch_summary(file_results):
    total_files = len(file_results)
    ok_files = [r for r in file_results if r.get("status") == "ok"]
    needs_ocr_files = [r for r in file_results if r.get("status") == "needs_ocr"]
    error_files = [r for r in file_results if r.get("status") == "error"]

    agg_by_type = {t: 0 for t in CLAUSE_TYPES}
    agg_by_risk = {"High": 0, "Medium": 0, "Low": 0}

    all_clauses_flat = []
    for r in ok_files:
        for c in r.get("clauses", []):
            agg_by_type[c["clause_type"]] += 1
            agg_by_risk[c["risk_level"]] += 1
            all_clauses_flat.append({"file_name": r["file_name"], **c})

    top_batch = sorted(
        all_clauses_flat,
        key=lambda x: (RISK_RANK.get(x["risk_level"], 0), x.get("confidence", 0.0)),
        reverse=True
    )[:10]

    guidance = "Start with contracts with High-risk termination or change-of-control clauses, then review MFN and commitment clauses for pricing/margin implications."

    return {
        "total_files": total_files,
        "ok_files": len(ok_files),
        "needs_ocr_files": len(needs_ocr_files),
        "error_files": len(error_files),
        "counts_by_clause_type": agg_by_type,
        "counts_by_risk_level": agg_by_risk,
        "top_risks_across_all_contracts": top_batch,
        "guidance": guidance,
        "needs_ocr": [{"file_name": e["file_name"], "note": "Scanned PDF; needs OCR"} for e in needs_ocr_files],
        "errors": [{"file_name": e["file_name"], "error": e.get("error","")} for e in error_files]
    }


# ----------------------------
# OUTPUT WRITERS
# ----------------------------

def write_batch_csv(out_path: Path, file_results):
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


def write_batch_html_issue_first_dashboard(out_path: Path, batch_output: dict):
    batch_summary = batch_output["batch_summary"]
    contracts = batch_output["contracts"]

    flat = []
    for c in contracts:
        if c.get("status") != "ok":
            continue
        for clause in c.get("clauses", []):
            flat.append({
                "file_name": c["file_name"],
                "clause_type": clause["clause_type"],
                "risk_level": clause["risk_level"],
                "confidence": clause.get("confidence", 0.0),
                "excerpt": clause.get("excerpt", ""),
                "why": clause.get("why_this_matters", "")
            })

    grouped = {t: [] for t in CLAUSE_TYPES}
    for item in flat:
        grouped[item["clause_type"]].append(item)

    for t in CLAUSE_TYPES:
        grouped[t] = sorted(grouped[t], key=lambda x: (RISK_RANK.get(x["risk_level"], 0), x["confidence"]), reverse=True)

    top = batch_summary.get("top_risks_across_all_contracts", [])
    counts_type = batch_summary["counts_by_clause_type"]
    counts_risk = batch_summary["counts_by_risk_level"]

    # Chart data (simple arrays)
    type_labels = list(counts_type.keys())
    type_values = [counts_type[k] for k in type_labels]
    risk_labels = list(counts_risk.keys())
    risk_values = [counts_risk[k] for k in risk_labels]

    def pill(risk):
        cls = "pill high" if risk == "High" else ("pill med" if risk == "Medium" else "pill low")
        return f'<span class="{cls}">{risk}</span>'

    title = f"Clause-Intel Dashboard - {html_escape(batch_output['metadata']['scanned_at'])}"

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #ffffff; color: #111; }}
h1 {{ margin-bottom: 6px; }}
.small {{ color: #555; font-size: 13px; margin-top: 0; }}
.card {{ border: 1px solid #e5e5e5; border-radius: 14px; padding: 16px; margin: 14px 0; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.kv {{ font-size: 14px; color: #222; }}
.kv b {{ display:inline-block; min-width: 170px; color:#000; }}
.pill {{ display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; margin-right: 8px; border:1px solid #ddd; }}
.pill.high {{ border-color:#c00; }}
.pill.med {{ border-color:#c60; }}
.pill.low {{ border-color:#0a6; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ border-bottom: 1px solid #eee; padding: 10px; vertical-align: top; }}
th {{ text-align: left; color:#333; font-size: 13px; }}
details {{ background:#fafafa; border:1px solid #eee; border-radius:10px; padding: 10px; }}
summary {{ cursor:pointer; }}
code {{ white-space: pre-wrap; }}
.section-title {{ margin-top: 22px; }}
.badge {{ font-size:12px; color:#555; }}
</style>
</head>
<body>
<h1>Clause-Intel Dashboard</h1>
<p class="small">Issue-first view grouped by clause type. Includes charts. Not legal advice; excerpts are verbatim.</p>

<div class="card">
  <div class="grid">
    <div class="kv"><b>Scanned at:</b> {html_escape(batch_output["metadata"]["scanned_at"])}</div>
    <div class="kv"><b>Input:</b> {html_escape(batch_output["metadata"]["input_path"])}</div>
    <div class="kv"><b>Files scanned:</b> {batch_summary["total_files"]} (ok: {batch_summary["ok_files"]}, needs OCR: {batch_summary["needs_ocr_files"]}, errors: {batch_summary["error_files"]})</div>
    <div class="kv"><b>Guidance:</b> {html_escape(batch_summary.get("guidance",""))}</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h3 style="margin-top:0">Findings by clause type</h3>
    <canvas id="typeChart"></canvas>
  </div>
  <div class="card">
    <h3 style="margin-top:0">Findings by risk level</h3>
    <canvas id="riskChart"></canvas>
  </div>
</div>

<div class="card">
  <h3 style="margin-top:0">Top risks across all contracts</h3>
  <table>
    <thead><tr><th>Risk</th><th>Clause</th><th>Contract</th><th>Confidence</th></tr></thead>
    <tbody>
"""
    for tr in top[:10]:
        html += f"<tr><td>{pill(tr['risk_level'])}</td><td>{html_escape(tr['clause_type'])}</td><td>{html_escape(tr['file_name'])}</td><td>{tr.get('confidence',0.0):.2f}</td></tr>"

    html += """
    </tbody>
  </table>
</div>
"""

    # Clause type sections
    for ct in CLAUSE_TYPES:
        items = grouped[ct]
        html += f'<h2 class="section-title">{html_escape(ct)} <span class="badge">({len(items)} findings)</span></h2>'
        if not items:
            html += '<div class="card"><i>No findings.</i></div>'
            continue

        html += """
<div class="card">
  <table>
    <thead>
      <tr>
        <th style="width:120px">Risk</th>
        <th style="width:120px">Confidence</th>
        <th style="width:260px">Contract</th>
        <th>Details</th>
      </tr>
    </thead>
    <tbody>
"""
        for it in items:
            html += f"""
<tr>
  <td>{pill(it["risk_level"])}</td>
  <td>{it["confidence"]:.2f}</td>
  <td>{html_escape(it["file_name"])}</td>
  <td>
    <details>
      <summary>Show excerpt + why it matters</summary>
      <p><b>Why it matters:</b> {html_escape(it["why"])}</p>
      <p><b>Excerpt:</b></p>
      <code>{html_escape(it["excerpt"])}</code>
    </details>
  </td>
</tr>
"""
        html += """
    </tbody>
  </table>
</div>
"""

    # Needs OCR / Errors
    if batch_summary.get("needs_ocr"):
        html += '<h2 class="section-title">Needs OCR</h2><div class="card"><ul>'
        for n in batch_summary["needs_ocr"]:
            html += f"<li>{html_escape(n['file_name'])} — {html_escape(n['note'])}</li>"
        html += "</ul></div>"

    if batch_summary.get("errors"):
        html += '<h2 class="section-title">Errors</h2><div class="card"><ul>'
        for e in batch_summary["errors"]:
            html += f"<li>{html_escape(e['file_name'])} — {html_escape(e.get('error',''))}</li>"
        html += "</ul></div>"

    # Charts script
    html += f"""
<script>
const typeLabels = {json.dumps(type_labels)};
const typeValues = {json.dumps(type_values)};
const riskLabels = {json.dumps(risk_labels)};
const riskValues = {json.dumps(risk_values)};

new Chart(document.getElementById('typeChart'), {{
  type: 'bar',
  data: {{
    labels: typeLabels,
    datasets: [{{ label: 'Findings', data: typeValues }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ beginAtZero: true, precision: 0 }} }}
  }}
}});

new Chart(document.getElementById('riskChart'), {{
  type: 'doughnut',
  data: {{
    labels: riskLabels,
    datasets: [{{ label: 'Findings', data: riskValues }}]
  }},
  options: {{ responsive: true }}
}});
</script>

</body></html>
"""
    out_path.write_text(html, encoding="utf-8")


# ----------------------------
# SCAN ONE FILE
# ----------------------------

def scan_one(file_path: Path, max_pages: int | None, max_chars: int, graceful: bool):
    started = now_utc_z()

    try:
        text = extract_text(file_path, max_pages=max_pages)

        # Spend protection: cap chars sent to the model
        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True

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
            "truncated": truncated,
            "clauses": clauses,
            "executive_summary": summary,
            "scanned_at": started
        }

    except RuntimeError as e:
        if str(e) == "needs_ocr" and graceful:
            return {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "status": "needs_ocr",
                "note": "Scanned PDF; needs OCR to extract text.",
                "scanned_at": started
            }
        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "status": "error",
            "error": "Scanned PDF; needs OCR." if str(e) == "needs_ocr" else str(e),
            "scanned_at": started
        }

    except Exception as e:
        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "status": "error",
            "error": str(e),
            "scanned_at": started
        }


# ----------------------------
# MAIN
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Clause-Intel: scan contracts for diligence-relevant clauses.")
    p.add_argument("path", nargs="?", default=None, help="File or folder to scan (pdf/txt/docx).")
    p.add_argument("--out", default="outputs", help="Output folder (default: outputs)")
    p.add_argument("--max-files", type=int, default=None, help="Max files to scan when input is a folder")
    p.add_argument("--max-pages", type=int, default=60, help="Max pages to extract from PDFs (default: 60)")
    p.add_argument("--max-chars", type=int, default=120000, help="Max characters sent to AI per document (default: 120000)")
    p.add_argument("--demo", action="store_true", help="Run a demo scan using sample_contract.txt if present")
    p.add_argument("--strict", action="store_true", help="Stop on OCR/scanned PDF instead of continuing")
    return p.parse_args()


def main():
    args = parse_args()

    if args.demo:
        demo_path = Path("sample_contract.txt")
        if not demo_path.exists():
            raise SystemExit("Demo file not found: sample_contract.txt")
        input_path = demo_path
    else:
        if not args.path:
            print("Usage:")
            print("  python scan.py samples")
            print("  python scan.py samples\\contract.pdf")
            print("  python scan.py --demo")
            return
        input_path = Path(args.path)

    graceful = not args.strict

    files = list_contract_files(input_path, max_files=args.max_files)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"\nScanning {len(files)} file(s)...\n")

    results = []
    for i, fp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {fp.name}")
        res = scan_one(fp, max_pages=args.max_pages, max_chars=args.max_chars, graceful=graceful)
        results.append(res)

        if res.get("status") == "ok":
            s = res["executive_summary"]["counts_by_risk_level"]
            trunc_note = " | truncated" if res.get("truncated") else ""
            print(f"   OK{trunc_note} | High:{s['High']}  Med:{s['Medium']}  Low:{s['Low']} | preview: {res.get('preview','')}\n")
        elif res.get("status") == "needs_ocr":
            print("   NEEDS OCR | scanned PDF (no extractable text)\n")
            if args.strict:
                raise SystemExit("Stopped due to --strict and OCR-needed PDF.")
        else:
            print(f"   ERROR | {res.get('error')}\n")
            if args.strict:
                raise SystemExit("Stopped due to --strict.")

    batch_summary = build_batch_summary(results)

    batch_output = {
        "metadata": {
            "input_path": str(input_path),
            "scanned_at": now_utc_z(),
            "disclaimer": "This output highlights clauses commonly flagged during M&A diligence and does not constitute legal advice."
        },
        "batch_summary": batch_summary,
        "contracts": results
    }

    json_path = out_dir / f"batch_{stamp}.json"
    json_path.write_text(json.dumps(batch_output, indent=2), encoding="utf-8")

    csv_path = out_dir / f"batch_{stamp}.csv"
    write_batch_csv(csv_path, results)

    html_path = out_dir / f"batch_{stamp}.html"
    write_batch_html_issue_first_dashboard(html_path, batch_output)

    print("\n=== BATCH SUMMARY ===")
    print(f"Files scanned: {batch_summary['total_files']} (ok: {batch_summary['ok_files']}, needs OCR: {batch_summary['needs_ocr_files']}, errors: {batch_summary['error_files']})")
    print("Clause counts:", batch_summary["counts_by_clause_type"])
    print("Risk counts:", batch_summary["counts_by_risk_level"])
    if batch_summary["top_risks_across_all_contracts"]:
        print("\nTop risks across all contracts:")
        for tr in batch_summary["top_risks_across_all_contracts"][:5]:
            print(f"- {tr['file_name']} | {tr['clause_type']} | {tr['risk_level']} | conf {tr.get('confidence',0.0):.2f}")
    print("\nOutputs:")
    print(f"- JSON: {json_path}")
    print(f"- CSV:  {csv_path}")
    print(f"- HTML: {html_path}\n")

    if args.demo:
        print("Demo mode complete. Open the HTML report for the best view.")


if __name__ == "__main__":
    main()
