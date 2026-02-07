import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FirstLook Scan | Deal Triage", layout="wide")

# -----------------------------
# DEMO MODE SETTINGS (web-safe)
# -----------------------------
DEMO_MODE = True  # True for hosted demo; set False for your local "power user" mode

DEMO_LIMITS = {
    "max_files": 8,                 # hard cap uploads
    "max_file_mb": 10,              # per-file size cap
    "max_pages": 25,                # per PDF
    "max_chars": 60000,             # chars sent to AI per document
    "max_scans_per_session": 3,     # prevent cost blowups
    "min_confidence_floor": 0.6,    # avoid noisy results in demo
}

# Prefer Streamlit secrets in hosted mode; fall back to env locally
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEMO_ACCESS_CODE = st.secrets.get("DEMO_ACCESS_CODE", os.getenv("DEMO_ACCESS_CODE", ""))

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not configured. Set it in Streamlit secrets or environment variables.")
    st.stop()


def require_access_code():
    """Simple password gate for demo deployments."""
    if not DEMO_MODE:
        return True

    if not DEMO_ACCESS_CODE:
        st.error("DEMO_ACCESS_CODE is not configured.")
        st.stop()

    if st.session_state.get("demo_authed"):
        return True

    st.title("FirstLook Scan — Demo Access")
    st.caption("Enter the access code to launch the demo.")
    code = st.text_input("Access code", type="password")
    if st.button("Enter", type="primary"):
        if code == DEMO_ACCESS_CODE:
            st.session_state["demo_authed"] = True
            st.rerun()
        else:
            st.error("Invalid access code.")
            st.stop()

    st.stop()


require_access_code()

ROOT = Path(__file__).parent
DEFAULT_SCAN = ROOT / "scan.py"


def run_scan_on_folder(input_dir: Path, out_dir: Path, max_files: int, max_pages: int, max_chars: int, strict: bool):
    """Calls scan.py as a subprocess so we reuse all scanning logic."""
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
        if c.get("status") != "ok":
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
        return pd.DataFrame(columns=["file_name", "clause_type", "risk_level", "confidence", "excerpt", "why_this_matters"])
    return pd.DataFrame(rows)


def risk_rank(risk: str) -> int:
    return {"High": 3, "Medium": 2, "Low": 1}.get(risk, 0)


def worst_risk_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Rows=contract, cols=clause_type, values=worst risk."""
    if df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["risk_rank"] = tmp["risk_level"].apply(risk_rank)

    grouped = tmp.groupby(["file_name", "clause_type"], as_index=False)["risk_rank"].max()
    pivot = grouped.pivot(index="file_name", columns="clause_type", values="risk_rank").fillna(0).astype(int)

    for ct in ["change_of_control", "termination", "exclusivity", "mfn", "revenue_commitment"]:
        if ct not in pivot.columns:
            pivot[ct] = 0

    pivot = pivot[["termination", "change_of_control", "mfn", "revenue_commitment", "exclusivity"]]
    pivot["total_risk_score"] = pivot.sum(axis=1)

    def label(v: int) -> str:
        if v >= 3:
            return "🟥 High"
        if v == 2:
            return "🟧 Medium"
        if v == 1:
            return "🟨 Low"
        return "⬜ —"

    labeled = pivot.applymap(label)
    labeled = labeled.sort_values(by="total_risk_score", ascending=False)
    labeled = labeled[["total_risk_score", "termination", "change_of_control", "mfn", "revenue_commitment", "exclusivity"]]
    return labeled


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Deal Triage Controls")

if DEMO_MODE:
    mode = "Upload & Scan"  # demo keeps it simple
    max_files = DEMO_LIMITS["max_files"]
    max_pages = DEMO_LIMITS["max_pages"]
    max_chars = DEMO_LIMITS["max_chars"]
    min_conf = DEMO_LIMITS["min_confidence_floor"]

    st.sidebar.info(
        f"Demo limits:\n"
        f"- max files: {max_files}\n"
        f"- max pages: {max_pages}\n"
        f"- max chars: {max_chars}\n"
        f"- scans/session: {DEMO_LIMITS['max_scans_per_session']}\n"
        f"- min confidence: {min_conf}"
    )
else:
    mode = st.sidebar.radio("Mode", ["Upload & Scan", "Load Existing Batch JSON"], index=0)
    max_files = st.sidebar.slider("Max files (safety)", 1, 200, 25, 1)
    max_pages = st.sidebar.slider("Max PDF pages per doc (safety)", 5, 200, 60, 5)
    max_chars = st.sidebar.slider("Max characters sent to AI per doc (safety)", 20000, 250000, 120000, 10000)
    min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05)

strict = st.sidebar.checkbox("Strict mode (stop on OCR/errors)", value=False)

st.sidebar.divider()
risk_filter = st.sidebar.multiselect("Risk levels", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
clause_filter = st.sidebar.multiselect(
    "Clause types",
    ["change_of_control", "termination", "exclusivity", "mfn", "revenue_commitment"],
    default=["change_of_control", "termination", "exclusivity", "mfn", "revenue_commitment"]
)
st.sidebar.caption("Tip: For deal triage, start with High risk + confidence ≥ 0.75.")

# -----------------------------
# MAIN UI
# -----------------------------
st.title("FirstLook Scan — Deal Triage Dashboard")
st.caption("Web demo: do not upload confidential documents. Quote-only extraction. Not legal advice.")

batch = None
batch_path = None
run_stdout = ""
run_stderr = ""

# A) Sample deal (recommended for sales)
c1, c2 = st.columns([1, 3])
with c1:
    load_sample = st.button("Load Sample Deal (Recommended)")
with c2:
    st.write("")

if load_sample:
    demo_dir = ROOT / "demo_data"
    if not demo_dir.exists():
        st.error("demo_data/ folder not found. Create it at the repo root and add demo contracts.")
        st.stop()

    # scan budget per session
    if DEMO_MODE:
        scans = st.session_state.get("scan_count", 0)
        if scans >= DEMO_LIMITS["max_scans_per_session"]:
            st.error("Demo scan limit reached for this session. Please refresh later.")
            st.stop()
        st.session_state["scan_count"] = scans + 1

    with tempfile.TemporaryDirectory() as td:
        tmp_out = Path(td) / "outputs"
        tmp_out.mkdir(parents=True, exist_ok=True)

        with st.spinner("Scanning sample deal..."):
            code, run_stdout, run_stderr = run_scan_on_folder(
                input_dir=demo_dir,
                out_dir=tmp_out,
                max_files=max_files,
                max_pages=max_pages,
                max_chars=max_chars,
                strict=strict
            )

        latest = find_latest_batch_json(tmp_out)
        if latest and latest.exists():
            batch = load_batch_json(latest)

            persist_out = ROOT / "outputs_streamlit"
            persist_out.mkdir(exist_ok=True)
            stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
            dest = persist_out / f"run_{stamp}"
            dest.mkdir(parents=True, exist_ok=True)
            for p in tmp_out.glob("*"):
                shutil.copy2(p, dest / p.name)

            batch_path = dest / latest.name
            st.success("Sample deal scan complete.")
        else:
            st.error("Sample scan ran but no batch JSON was produced. Check logs below.")

# B) Upload & scan
if mode == "Upload & Scan":
    uploaded = st.file_uploader(
        "Upload contract files (.pdf, .docx, .txt).",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if DEMO_MODE and uploaded:
        if len(uploaded) > DEMO_LIMITS["max_files"]:
            st.warning(f"Demo limit: only the first {DEMO_LIMITS['max_files']} files will be scanned.")
            uploaded = uploaded[:DEMO_LIMITS["max_files"]]

        too_big = []
        for f in uploaded:
            size_mb = len(f.getbuffer()) / (1024 * 1024)
            if size_mb > DEMO_LIMITS["max_file_mb"]:
                too_big.append((f.name, size_mb))
        if too_big:
            st.error("One or more files exceed the demo size limit:")
            for name, mb in too_big:
                st.write(f"- {name}: {mb:.1f} MB (limit {DEMO_LIMITS['max_file_mb']} MB)")
            st.stop()

    colA, colB = st.columns([1, 1])
    with colA:
        run_now = st.button("Run Scan", type="primary", disabled=(not uploaded))
    with colB:
        st.write("")

    if run_now and uploaded:
        # scan budget per session
        if DEMO_MODE:
            scans = st.session_state.get("scan_count", 0)
            if scans >= DEMO_LIMITS["max_scans_per_session"]:
                st.error("Demo scan limit reached for this session. Please refresh later or use the sample deal.")
                st.stop()
            st.session_state["scan_count"] = scans + 1

        with tempfile.TemporaryDirectory() as td:
            tmp_in = Path(td) / "inputs"
            tmp_out = Path(td) / "outputs"
            tmp_in.mkdir(parents=True, exist_ok=True)
            tmp_out.mkdir(parents=True, exist_ok=True)

            for f in uploaded:
                out_file = tmp_in / f.name
                out_file.write_bytes(f.getbuffer())

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

                persist_out = ROOT / "outputs_streamlit"
                persist_out.mkdir(exist_ok=True)
                stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
                dest = persist_out / f"run_{stamp}"
                dest.mkdir(parents=True, exist_ok=True)
                for p in tmp_out.glob("*"):
                    shutil.copy2(p, dest / p.name)

                batch_path = dest / latest.name
                st.success("Scan complete.")
            else:
                st.error("Scan ran but no batch JSON was produced. Check logs below.")

if not DEMO_MODE and mode == "Load Existing Batch JSON":
    json_file = st.file_uploader("Upload a batch_*.json file (from outputs)", type=["json"], accept_multiple_files=False)
    if json_file:
        batch = json.loads(json_file.getvalue().decode("utf-8"))
        st.success("Batch JSON loaded.")

if run_stdout or run_stderr:
    with st.expander("Run Logs (from scan.py)"):
        st.code(run_stdout or "", language="text")
        if run_stderr:
            st.code(run_stderr, language="text")

# -----------------------------
# DASHBOARD VIEW
# -----------------------------
if batch:
    summary = batch.get("batch_summary", {})
    df = flatten_clauses(batch)

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

    if df.empty:
        st.warning("No clauses found in the batch (or everything failed/OCR-needed).")
    else:
        df_f = df.copy()
        df_f = df_f[df_f["confidence"] >= min_conf]
        df_f = df_f[df_f["risk_level"].isin(risk_filter)]
        df_f = df_f[df_f["clause_type"].isin(clause_filter)]

        df_f["risk_rank"] = df_f["risk_level"].apply(risk_rank)
        df_f = df_f.sort_values(by=["risk_rank", "confidence"], ascending=[False, False]).drop(columns=["risk_rank"])

        st.subheader("Contract Risk Heatmap (Worst Risk by Clause Type)")
        st.caption("Each cell shows the worst risk detected for that clause type in that contract. Sorted by total risk score.")

        heat = worst_risk_per_cell(df_f)
        if heat.empty:
            st.write("No data under current filters.")
        else:
            st.dataframe(heat, use_container_width=True, height=320)

        st.subheader("Top Risks (Deal Triage)")
        st.caption("Sorted by risk level then confidence.")
        st.dataframe(df_f[["file_name", "clause_type", "risk_level", "confidence"]], use_container_width=True, height=260)

        st.write("")
        st.subheader("Issue-First Drilldown")

        for ct in ["termination", "change_of_control", "mfn", "revenue_commitment", "exclusivity"]:
            block = df_f[df_f["clause_type"] == ct]
            with st.expander(f"{ct} ({len(block)} findings)", expanded=(ct in ["termination", "change_of_control"])):
                if block.empty:
                    st.write("No findings under current filters.")
                else:
                    for _, row in block.iterrows():
                        st.markdown(f"**{row['file_name']}** — **{row['risk_level']}** (confidence {row['confidence']:.2f})")
                        st.markdown(f"- **Why it matters:** {row['why_this_matters']}")
                        st.code(row["excerpt"], language="text")
                        st.markdown("---")

    st.divider()
    st.subheader("Outputs")
    st.write("Download structured artifacts for integration or review.")

    batch_bytes = json.dumps(batch, indent=2).encode("utf-8")
    st.download_button("Download batch JSON", data=batch_bytes, file_name="batch_output.json", mime="application/json")

    if batch_path:
        st.info(f"Latest run outputs folder: {batch_path.parent}")

    if summary.get("needs_ocr"):
        st.warning("Some PDFs appear scanned and need OCR:")
        st.write(summary["needs_ocr"])

    if summary.get("errors"):
        st.error("Errors:")
        st.write(summary["errors"])
else:
    st.info("Use **Load Sample Deal** for the fastest demo, or upload a few non-confidential files and click **Run Scan**.")
