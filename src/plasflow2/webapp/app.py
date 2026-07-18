"""PlasFlow v2 — Streamlit Web Interface

Start with:
    plasflow2 serve
or directly:
    streamlit run src/plasflow2/webapp/app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_JOBS_DIR = Path(
    os.environ.get("PLASFLOW_JOBS_DIR", str(Path.home() / ".plasflow2" / "jobs"))
)

_STATUS_EMOJI = {"queued": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}
_STATUS_COLOR = {
    "queued": "#888888",
    "running": "#e67e22",
    "completed": "#1a7a4a",
    "failed": "#c0392b",
}

_REPORT_FILES = [
    ("Plasmid Report", "report_plasmid.html"),
    ("Circular Maps", "report_circular_plasmids.html"),
    ("Chromosome Report", "report_chromosome.html"),
    ("Phage Report", "report_phage.html"),
    ("Unclassified Report", "report_unclassified.html"),
]
_TSV_FILES = [
    ("All Predictions", "all_predictions.tsv"),
    ("Annotated Predictions", "annotated_predictions.tsv"),
    ("Gene Table", "genes.tsv"),
    ("Plasmid FASTA", "plasmid.fasta"),
]

# ── Streamlit page config (must be first st.* call) ───────────────────────────

st.set_page_config(
    page_title="PlasFlow v2",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* Sidebar styling */
section[data-testid="stSidebar"] { background: #1a2332; }
section[data-testid="stSidebar"] * { color: #e8edf2 !important; }
section[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: 1px solid #2d4a6e;
    text-align: left;
    font-size: 0.82rem;
    padding: 4px 8px;
}
section[data-testid="stSidebar"] .stButton button:hover { background: #2d4a6e; }
section[data-testid="stSidebar"] .stButton[data-testid="baseButton-primary"] button {
    background: #1a7a4a;
    border-color: #1a7a4a;
    color: white !important;
}
section[data-testid="stSidebar"] .stButton[data-testid="baseButton-primary"] button:hover {
    background: #15643d;
}
/* Metric cards */
[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px 16px;
}
/* Download buttons */
.stDownloadButton button {
    width: 100%;
    background: #f0f7f4;
    border: 1px solid #1a7a4a40;
    color: #1a7a4a;
    font-size: 0.82rem;
}
.stDownloadButton button:hover { background: #1a7a4a; color: white; }
/* Code / log box */
.stCode { font-size: 0.75rem; max-height: 400px; overflow-y: auto; }
/* Primary submit button */
.stForm .stButton button[kind="primaryFormSubmit"] {
    background: #1a7a4a;
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Job helpers ────────────────────────────────────────────────────────────────


def _jobs_dir() -> Path:
    d = Path(os.environ.get("PLASFLOW_JOBS_DIR", str(_DEFAULT_JOBS_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_dir(job_id: str) -> Path:
    return _jobs_dir() / job_id


def _read_meta(job_id: str) -> dict:
    p = _job_dir(job_id) / "job.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _write_meta(job_id: str, meta: dict) -> None:
    (_job_dir(job_id) / "job.json").write_text(json.dumps(meta, indent=2))


def _list_jobs() -> list[tuple[str, dict]]:
    out = []
    for d in sorted(_jobs_dir().iterdir(), reverse=True):
        if d.is_dir():
            m = _read_meta(d.name)
            if m:
                out.append((d.name, m))
    return out


def _tail_log(job_id: str, n: int = 120) -> str:
    log = _job_dir(job_id) / "pipeline.log"
    if not log.exists():
        return "(waiting for pipeline to start…)"
    try:
        lines = log.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return f"(error reading log: {e})"


def _duration(meta: dict) -> str:
    try:
        start = datetime.fromisoformat(meta["started_at"])
        end = (
            datetime.fromisoformat(meta["finished_at"]) if "finished_at" in meta else datetime.now()
        )
        s = int((end - start).total_seconds())
        return f"{s // 60}m {s % 60}s"
    except Exception:
        return "—"


# ── Background pipeline runner ─────────────────────────────────────────────────


def _run_pipeline(job_id: str, fasta_path: Path, params: dict) -> None:
    """Runs in a daemon thread — never call st.* from here."""
    results_dir = _job_dir(job_id) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = _job_dir(job_id) / "pipeline.log"

    cmd = [
        "plasflow2",
        "run",
        "-i",
        str(fasta_path),
        "-o",
        str(results_dir),
        "--threads",
        str(params.get("threads", 4)),
        "--min-length",
        str(params.get("min_length", 1000)),
    ]
    if params.get("skip_genomad"):
        cmd.append("--skip-genomad")
    if params.get("skip_plasmid_db"):
        cmd.append("--skip-plasmid-db")

    m = _read_meta(job_id)
    m.update({"status": "running", "started_at": datetime.now().isoformat(), "cmd": cmd})
    _write_meta(job_id, m)

    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        m["pid"] = proc.pid
        _write_meta(job_id, m)
        proc.wait()

    m = _read_meta(job_id)
    m.update(
        {
            "status": "completed" if proc.returncode == 0 else "failed",
            "finished_at": datetime.now().isoformat(),
            "returncode": proc.returncode,
        }
    )
    _write_meta(job_id, m)


def _submit_job(fasta_bytes: bytes, filename: str, params: dict) -> str:
    job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    jdir = _job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)

    fasta_path = jdir / filename
    fasta_path.write_bytes(fasta_bytes)

    _write_meta(
        job_id,
        {
            "job_id": job_id,
            "filename": filename,
            "params": params,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
        },
    )

    t = threading.Thread(target=_run_pipeline, args=(job_id, fasta_path, params), daemon=True)
    t.start()
    # Keep thread alive within this session
    st.session_state.setdefault("_threads", {})[job_id] = t
    return job_id


# ── Pages ──────────────────────────────────────────────────────────────────────


def _page_upload() -> None:
    st.markdown("## New Analysis Run")
    st.caption(
        "Upload assembled contigs (metaSPAdes, MEGAHIT, Flye, etc.) "
        "and PlasFlow v2 will classify, annotate, and score them."
    )

    with st.form("run_form", clear_on_submit=False):
        uploaded = st.file_uploader(
            "Contig FASTA",
            type=["fa", "fasta", "fna", "gz"],
            help="Multi-FASTA of assembled contigs. gzip accepted.",
        )

        col1, col2 = st.columns(2)
        threads = col1.number_input("CPU threads", min_value=1, max_value=128, value=4, step=1)
        min_len = col2.number_input(
            "Min contig length (bp)", min_value=100, max_value=10_000, value=1000, step=100
        )

        col3, col4 = st.columns(2)
        skip_genomad = col3.checkbox(
            "Skip geNomad",
            value=False,
            help="Skip geNomad SPM features. Faster; reduces marker XGBoost accuracy.",
        )
        skip_plasmid_db = col4.checkbox(
            "Skip PLSDB search",
            value=False,
            help="Skip minimap2 search against PLSDB. Faster; disables plasmid DB override.",
        )

        submitted = st.form_submit_button(
            "▶  Run Pipeline", type="primary", use_container_width=True
        )

    if submitted:
        if uploaded is None:
            st.error("Please upload a FASTA file before submitting.")
            return
        params = {
            "threads": int(threads),
            "min_length": int(min_len),
            "skip_genomad": skip_genomad,
            "skip_plasmid_db": skip_plasmid_db,
        }
        jid = _submit_job(uploaded.getvalue(), uploaded.name, params)
        st.session_state["current_job"] = jid
        st.rerun()

    # Show recent completed jobs on the landing page
    jobs = _list_jobs()
    completed = [(jid, m) for jid, m in jobs if m.get("status") == "completed"]
    if completed:
        st.divider()
        st.markdown("### Recent Completed Runs")
        for jid, m in completed[:5]:
            c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
            c1.write(f"**{m.get('filename', jid)}**")
            c2.write(m.get("created_at", "")[:16].replace("T", " "))
            c3.write(f"✅ {_duration(m)}")
            if c4.button("Open", key=f"open_{jid}"):
                st.session_state["current_job"] = jid
                st.rerun()


def _summary_stats(results_dir: Path) -> dict[str, int]:
    tsv = results_dir / "all_predictions.tsv"
    if not tsv.exists():
        return {}
    import csv

    counts: dict[str, int] = {}
    try:
        with open(tsv) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                lbl = row.get("label", "unknown")
                counts[lbl] = counts.get(lbl, 0) + 1
    except Exception:
        pass
    return counts


def _page_job(job_id: str) -> None:
    meta = _read_meta(job_id)
    if not meta:
        st.error(f"Job `{job_id}` not found.")
        return

    status = meta.get("status", "unknown")
    emoji = _STATUS_EMOJI.get(status, "❓")
    color = _STATUS_COLOR.get(status, "#888")
    fname = meta.get("filename", job_id)

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        f"<h2 style='margin-bottom:4px'>{emoji} {fname} "
        f"<span style='font-size:.65em;font-weight:400;color:{color}'>{status.upper()}</span></h2>",
        unsafe_allow_html=True,
    )

    params = meta.get("params", {})
    param_str = (
        f"threads={params.get('threads', '?')}  ·  "
        f"min_length={params.get('min_length', '?')} bp"
    )
    if params.get("skip_genomad"):
        param_str += "  ·  --skip-genomad"
    if params.get("skip_plasmid_db"):
        param_str += "  ·  --skip-plasmid-db"
    st.caption(param_str)

    # ── Metrics row ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", f"{emoji} {status}")
    if "started_at" in meta:
        col2.metric(
            "Started",
            datetime.fromisoformat(meta["started_at"]).strftime("%H:%M:%S"),
        )
        col3.metric("Duration", _duration(meta))
    else:
        col2.metric("Submitted", datetime.fromisoformat(meta["created_at"]).strftime("%H:%M:%S"))
        col3.metric("Duration", "—")
    col4.metric("Job ID", job_id[-12:])

    st.divider()

    results_dir = _job_dir(job_id) / "results"

    # ── Running / Queued: live log ─────────────────────────────────────────────
    if status in ("queued", "running"):
        st.markdown("#### Pipeline Log  *(auto-refreshing every 3 s)*")
        st.code(
            _tail_log(job_id),
            language=None,
        )
        time.sleep(3)
        st.rerun()

    # ── Completed ──────────────────────────────────────────────────────────────
    elif status == "completed":
        counts = _summary_stats(results_dir)
        if counts:
            total = sum(counts.values())
            mcols = st.columns(len(counts) + 1)
            mcols[0].metric("Total contigs", f"{total:,}")
            label_order = ["plasmid", "chromosome", "phage", "unclassified", "archaea"]
            sorted_counts = sorted(
                counts.items(), key=lambda x: label_order.index(x[0]) if x[0] in label_order else 99
            )
            for i, (lbl, cnt) in enumerate(sorted_counts, 1):
                mcols[i].metric(lbl.capitalize(), f"{cnt:,}")
            st.divider()

        # Reports
        st.markdown("#### HTML Reports")
        rcols = st.columns(3)
        for i, (label, fname) in enumerate(_REPORT_FILES):
            fpath = results_dir / fname
            if fpath.exists():
                data = fpath.read_bytes()
                rcols[i % 3].download_button(
                    f"📄 {label}",
                    data,
                    file_name=fname,
                    mime="text/html",
                    use_container_width=True,
                    key=f"dl_{job_id}_{fname}",
                )

        # Data files
        st.markdown("#### Data Files")
        dcols = st.columns(4)
        for i, (label, fname) in enumerate(_TSV_FILES):
            fpath = results_dir / fname
            if fpath.exists():
                data = fpath.read_bytes()
                mime = "text/tab-separated-values" if fname.endswith(".tsv") else "text/plain"
                dcols[i % 4].download_button(
                    f"📊 {label}",
                    data,
                    file_name=fname,
                    mime=mime,
                    use_container_width=True,
                    key=f"dl_{job_id}_{fname}",
                )

        # Log (collapsed)
        with st.expander("📋 Pipeline Log"):
            st.code(_tail_log(job_id, n=300), language=None)

    # ── Failed ─────────────────────────────────────────────────────────────────
    elif status == "failed":
        st.error(
            f"Pipeline exited with code {meta.get('returncode', '?')}. "
            "Check the log below for details."
        )
        st.code(_tail_log(job_id, n=300), language=None)

        # Offer retry
        if st.button("🔄 Retry this job", type="secondary"):
            fasta_candidates = (
                list(_job_dir(job_id).glob("*.fa"))
                + list(_job_dir(job_id).glob("*.fasta"))
                + list(_job_dir(job_id).glob("*.fna"))
                + list(_job_dir(job_id).glob("*.gz"))
            )
            if fasta_candidates:
                orig_params = meta.get("params", {})
                jid2 = _submit_job(
                    fasta_candidates[0].read_bytes(),
                    fasta_candidates[0].name,
                    orig_params,
                )
                st.session_state["current_job"] = jid2
                st.rerun()
            else:
                st.warning("Original FASTA not found — please start a new run.")


# ── Sidebar ────────────────────────────────────────────────────────────────────


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<div style='padding:8px 0 4px'>"
            "<span style='font-size:1.5rem'>🧬</span> "
            "<span style='font-size:1.1rem;font-weight:700;color:#e8edf2'>PlasFlow v2</span>"
            "</div>"
            "<div style='font-size:.75rem;color:#8a9bb0;margin-bottom:12px'>"
            "Plasmid classifier &amp; AMR risk scorer"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button("＋  New Run", type="primary", use_container_width=True):
            st.session_state.pop("current_job", None)
            st.rerun()

        st.markdown(
            "<div style='font-size:.7rem;color:#8a9bb0;margin:14px 0 6px;text-transform:uppercase;"
            "letter-spacing:.06em'>Recent Jobs</div>",
            unsafe_allow_html=True,
        )

        jobs = _list_jobs()
        if not jobs:
            st.caption("No jobs yet.")
        else:
            for jid, meta in jobs[:20]:
                status = meta.get("status", "?")
                emoji = _STATUS_EMOJI.get(status, "❓")
                fname = meta.get("filename", jid)
                label = f"{emoji} {fname[:26]}"
                is_active = st.session_state.get("current_job") == jid
                btn_key = f"sb_{jid}"
                # Highlight active job
                if is_active:
                    st.markdown(
                        f"<div style='background:#2d4a6e;border-radius:4px;padding:4px 8px;"
                        f"font-size:.8rem;margin-bottom:2px'>{label}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(label, key=btn_key, use_container_width=True):
                        st.session_state["current_job"] = jid
                        st.rerun()

        st.markdown(
            "<div style='position:absolute;bottom:16px;font-size:.65rem;color:#4a6080'>"
            f"Jobs: {_jobs_dir()}</div>",
            unsafe_allow_html=True,
        )


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    _sidebar()
    current = st.session_state.get("current_job")
    if current:
        _page_job(current)
    else:
        _page_upload()


main()
