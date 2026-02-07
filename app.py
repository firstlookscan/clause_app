import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clause-Intel | Deal Triage", layout="wide")

ROOT = Path(__file__).parent
DEFAULT_SCAN = ROOT / "scan.py"


def run_scan_on_folder(input_dir: Path, out_dir: Path, max_files: int, max_pages: int, max_chars: int, strict: bool):
    """
    Calls your existing scan.py as a subprocess so we reuse all your logic.
    """
    cmd = [
        "python",
        str(DEFAULT_SCAN),
        str(input_dir),
        "--out",
        str(out_dir),
        "--max-files",
        str(max_files),
        "--max-pages",
        str(max_pages),
        "--max-chars",
        str(max_chars),
    ]
    if strict:
        cmd.append("--strict")

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def find_latest_batch_json(out_dir: Path) -> Path | None:
    batches = sorted(out_dir.glob("batch_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return batches[0] if batches else None


def load_batch_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_clauses(batch: dict) -> pd.DataFrame:
    rows = []
    for c in batch.get("contracts", []):
        status = c.get("status")
        if status != "ok":
            continue
        fname = c.get("file_name")
        for clause in c.get("clauses", []):
            rows.append({
                "file_name": fname,
                "clause_type": clause.get("clause_type"),
                "risk_level": clause.get("risk_level"),
                "confidence": float(clause.get("confidence", 0.0)),
                "excerpt": clause.get("excerpt", ""),
                "why_this_matters": clause.get("why_this_matters", ""),
            })
    if not rows:
        return pd.DataFrame(columns=["file_name","clause_type","risk_level","confidence","excerpt","why_this_matters"])
    return pd.DataFrame(rows)


def risk_rank(risk: str) -> int:
    return {"High": 3, "Medium": 2, "Low": 1}.get(risk, 0)


# -----------------------------
# SIDEBAR: Controls
# -----------------------------
st.sidebar.title("Deal Triage Controls")

mode = st.sidebar.radio("Mode", ["Upload & Scan", "Load Existing Batch JSON"], index=0)

max_files = st.sidebar.slider("Max files (safety)", min_value=1, max_value=200, value=25, step=1)
max_pages = st.sidebar.slider("Max PDF pages per doc (safety)", min_value=5, max_value=200, value=60, step=5)
max_chars = st.sidebar.slider("Max characters sent to AI per doc (safety)", min_value=20000, max_value=250000, value=120000, step=10000)
strict = st.sidebar.checkbox("Strict mode (stop on OCR/errors)", value=False)

st.sidebar.divider()
min_conf = st.sidebar.slider("Minimum confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
risk_filter = st.sidebar.multiselect("Risk levels", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
clause_filter = st.sidebar.multiselect(
    "Clause types",
    ["change_of_control", "termination", "exclusivity", "mfn", "revenue_commitment"],
    default=["change_of_control", "termination", "exclusivity", "mfn", "revenue_commitment"]
)

st.sidebar.divider()
st.sidebar.caption("Tip: For deal triage, start with High risk + confidence ≥ 0.75.")

# -----------------------------
# MAIN UI
# -----------------------------
st.title("Clause-Intel — Deal Triage Dashboard")
st.caption("Upload contracts, auto-extract deal-relevant clauses (quote-only), and triage risk across the set. Not legal advice.")

batch = None
batch_path = None
run_stdout = ""
run_stderr = ""

if mode == "Upload & Scan":
    uploaded = st.file_uploader(
        "Upload contract files (.pdf, .docx, .txt). You can upload multiple.",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    colA, colB = st.columns([1, 1])
    with colA:
        run_now = st.button("Run Scan", type="primary", disabled=(not uploaded))
    with colB:
        st.write("")

    if run_now and uploaded:
        with tempfile.TemporaryDirectory() as td:
            tmp_in = Path(td) / "inputs"
            tmp_out = Path(td) / "outputs"
            tmp_in.mkdir(parents=True, exist_ok=True)
            tmp_out.mkdir(parents=True, exist_ok=True)

            # Save uploads to disk
            for f in uploaded:
                out_file = tmp_in / f.name
                out_file.write_bytes(f.getbuffer())

            # Run scan.py on folder
            with st.spinner("Scanning contracts..."):
                code, run_stdout, run_stderr = run_scan_on_folder(
                    input_dir=tmp_in,
                    out_dir=tmp_out,
                    max_files=max_files,
                    max_pages=max_pages,
                    max_chars=max_chars,
                    strict=strict
                )

            latest = find_latest_batch_json(tmp_out)
            if latest and latest.exists():
                batch = load_batch_json(latest)
                # copy outputs to a user-visible persistent folder inside your repo
                persist_out = ROOT / "outputs_streamlit"
                persist_out.mkdir(exist_ok=True)
                # copy everything from tmp_out into outputs_streamlit with a timestamp subfolder
                stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
                dest = persist_out / f"run_{stamp}"
                dest.mkdir(parents=True, exist_ok=True)
                for p in tmp_out.glob("*"):
                    shutil.copy2(p, dest / p.name)

                batch_path = dest / latest.name
                st.success(f"Scan complete. Outputs saved to: {dest}")
            else:
                st.error("Scan ran but no batch JSON was produced. Check logs below.")

    if run_stdout or run_stderr:
        with st.expander("Run Logs (from scan.py)"):
            st.code(run_stdout or "", language="text")
            if run_stderr:
                st.code(run_stderr, language="text")

else:
    json_file = st.file_uploader("Upload a batch_*.json file (from outputs)", type=["json"], accept_multiple_files=False)
    if json_file:
        batch = json.loads(json_file.getvalue().decode("utf-8"))
        st.success("Batch JSON loaded.")

# -----------------------------
# DASHBOARD VIEW
# -----------------------------
if batch:
    meta = batch.get("metadata", {})
    summary = batch.get("batch_summary", {})
    df = flatten_clauses(batch)

    # Top tiles
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Files scanned", summary.get("total_files", 0))
    k2.metric("OK files", summary.get("ok_files", 0))
    k3.metric("Needs OCR", summary.get("needs_ocr_files", 0))
    k4.metric("Errors", summary.get("error_files", 0))

    st.write("")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Counts by Risk")
        risk_counts = summary.get("counts_by_risk_level", {})
        risk_df = pd.DataFrame([risk_counts]).T.reset_index()
        risk_df.columns = ["risk_level", "count"]
        st.bar_chart(risk_df.set_index("risk_level")["count"])
    with c2:
        st.subheader("Counts by Clause Type")
        type_counts = summary.get("counts_by_clause_type", {})
        type_df = pd.DataFrame([type_counts]).T.reset_index()
        type_df.columns = ["clause_type", "count"]
        st.bar_chart(type_df.set_index("clause_type")["count"])

    st.divider()

    # Filters
    if df.empty:
        st.warning("No clauses found in the batch (or everything failed/OCR-needed).")
    else:
        # Apply filters
        df_f = df.copy()
        df_f = df_f[df_f["confidence"] >= min_conf]
        df_f = df_f[df_f["risk_level"].isin(risk_filter)]
        df_f = df_f[df_f["clause_type"].isin(clause_filter)]

        # Sort triage-first
        df_f["risk_rank"] = df_f["risk_level"].apply(risk_rank)
        df_f = df_f.sort_values(by=["risk_rank", "confidence"], ascending=[False, False]).drop(columns=["risk_rank"])

        st.subheader("Top Risks (Deal Triage)")
        st.caption("Sorted by risk level then confidence. Click a row to inspect excerpt + rationale below.")
        st.dataframe(df_f[["file_name", "clause_type", "risk_level", "confidence"]], use_container_width=True, height=260)

        st.write("")
        st.subheader("Issue-First Drilldown")

        # Group by clause type
        for ct in ["termination", "change_of_control", "mfn", "revenue_commitment", "exclusivity"]:
            block = df_f[df_f["clause_type"] == ct]
            with st.expander(f"{ct} ({len(block)} findings)", expanded=(ct in ["termination", "change_of_control"])):
                if block.empty:
                    st.write("No findings under current filters.")
                else:
                    # Show each finding as a triage card
                    for _, row in block.iterrows():
                        st.markdown(f"**{row['file_name']}** — **{row['risk_level']}** (confidence {row['confidence']:.2f})")
                        st.markdown(f"- **Why it matters:** {row['why_this_matters']}")
                        st.code(row["excerpt"], language="text")
                        st.markdown("---")

        st.divider()

    # Downloads / outputs
    st.subheader("Outputs")
    st.write("These are the structured artifacts you can share or plug into a pipeline.")

    # Offer download of current batch JSON
    batch_bytes = json.dumps(batch, indent=2).encode("utf-8")
    st.download_button("Download batch JSON", data=batch_bytes, file_name="batch_output.json", mime="application/json")

    # If scan was run in this app, point to persistent outputs folder
    if batch_path:
        st.info(f"Latest run outputs folder: {batch_path.parent}")

    # Show OCR-needed/errors
    if summary.get("needs_ocr"):
        st.warning("Some PDFs appear scanned and need OCR:")
        st.write(summary["needs_ocr"])

    if summary.get("errors"):
        st.error("Errors:")
        st.write(summary["errors"])
else:
    st.info("Upload files and click **Run Scan**, or load an existing batch JSON to view triage.")
