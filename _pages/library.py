"""
Exercise Library page for PoseSync.

Browse pre-curated exercises by category, watch embedded YouTube tutorials,
and launch a session — or upload your own reference video.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXERCISES_PATH = Path(__file__).parent.parent / "data" / "exercises.json"
CACHE_DIR      = Path(__file__).parent.parent / "data" / "cache"

DIFFICULTY_COLOUR = {
    "Beginner":     "#10b981",
    "Intermediate": "#f59e0b",
    "Advanced":     "#ef4444",
    "Custom":       "#64748b",
}

CATEGORY_EMOJI = {
    "Exercise": "💪",
    "Yoga":     "🧘",
    "Stretch":  "🤸",
    "Custom":   "📁",
}


@st.cache_data
def load_exercises() -> list:
    with open(EXERCISES_PATH, encoding="utf-8") as f:
        return json.load(f)["exercises"]


def _yt_thumbnail(yt_id: str) -> str:
    # default.jpg is always available (even for restricted videos)
    return f"https://img.youtube.com/vi/{yt_id}/default.jpg"


def _card_html(ex: Dict) -> str:
    thumb   = _yt_thumbnail(ex["youtube_id"])
    emoji   = CATEGORY_EMOJI.get(ex["category"], "")
    muscles = " · ".join(ex["muscles"][:3])
    d_col   = DIFFICULTY_COLOUR.get(ex["difficulty"], "#64748b")
    desc    = ex["description"][:95] + "…" if len(ex["description"]) > 95 else ex["description"]

    return f"""
    <div style="
        background:#1a1a2e;
        border:1px solid #2d2d4e;
        border-radius:14px;
        overflow:hidden;
        margin-bottom:4px;
        min-height:300px;
    ">
      <div style="
        width:100%;height:140px;background:#0d0d14;overflow:hidden;
        display:flex;align-items:center;justify-content:center;
      ">
        <img src="{thumb}"
             style="width:100%;height:100%;object-fit:cover;"
             onerror="this.parentElement.innerHTML='<div style=\\'font-size:3rem;text-align:center;padding-top:40px;\\'>{emoji}</div>'"/>
      </div>
      <div style="padding:14px 16px 10px;">
        <div style="margin-bottom:6px;">
          <span style="background:{d_col};color:#fff;padding:2px 9px;
                       border-radius:12px;font-size:0.7rem;font-weight:600;">
            {ex['difficulty']}
          </span>
          &nbsp;
          <span style="background:#1e1b4b;color:#a5b4fc;padding:2px 9px;
                       border-radius:12px;font-size:0.7rem;font-weight:600;">
            {emoji} {ex['category']}
          </span>
        </div>
        <div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:8px 0 4px;">
          {emoji} {ex['name']}
        </div>
        <div style="font-size:0.76rem;color:#94a3b8;line-height:1.4;min-height:40px;">
          {desc}
        </div>
        <div style="font-size:0.71rem;color:#4f46e5;margin-top:8px;">
          🎯 {muscles}
        </div>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Video download + preprocess helpers
# ---------------------------------------------------------------------------

def _prepare_template(exercise: Dict) -> None:
    """Download the YouTube reference video and preprocess keypoints."""
    from core.video_processor import download_youtube_video, preprocess_reference

    yt_id     = exercise["youtube_id"]
    cache_dir = str(CACHE_DIR)

    st.markdown("### ⬇️ Preparing reference video…")
    dl_bar    = st.progress(0.0)
    dl_status = st.empty()

    def dl_cb(pct: float, msg: str) -> None:
        dl_bar.progress(min(pct, 1.0))
        dl_status.caption(msg)

    video_path = download_youtube_video(yt_id, cache_dir, dl_cb)

    if not video_path:
        st.error("Could not download the reference video. Check your internet connection, or upload your own video below.")
        return

    dl_status.caption("✅ Video ready. Analysing poses…")

    pre_bar    = st.progress(0.0)
    pre_status = st.empty()

    def pre_cb(pct: float, msg: str) -> None:
        pre_bar.progress(min(pct, 1.0))
        pre_status.caption(msg)

    keypoints = preprocess_reference(video_path, cache_dir, pre_cb)

    if not keypoints:
        st.error("Could not extract pose keypoints from the video.")
        return

    st.session_state.ref_video_path  = video_path
    st.session_state.ref_keypoints   = keypoints
    st.session_state.active_exercise = exercise
    st.session_state.session_done    = False
    st.session_state.page            = "session"
    st.rerun()


def _prepare_upload(uploaded_file) -> None:
    """Save uploaded file and preprocess its keypoints."""
    from core.video_processor import preprocess_reference

    cache_dir = str(CACHE_DIR)
    tmp_dir   = Path(cache_dir) / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest      = str(tmp_dir / uploaded_file.name)

    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.markdown("### ⚙️ Analysing your reference video…")
    bar    = st.progress(0.0)
    status = st.empty()

    def cb(pct: float, msg: str) -> None:
        bar.progress(min(pct, 1.0))
        status.caption(msg)

    keypoints = preprocess_reference(dest, cache_dir, cb)
    bar.progress(1.0)

    if not keypoints:
        st.error("Could not extract pose keypoints. Make sure there's a visible person in the video.")
        return

    custom_ex = {
        "id":              "custom",
        "name":            Path(uploaded_file.name).stem.replace("_", " ").title(),
        "category":        "Custom",
        "youtube_id":      "",
        "difficulty":      "Custom",
        "duration":        "—",
        "muscles":         ["Custom"],
        "key_joints":      [],
        "has_rep_counter": False,
        "description":     "User-uploaded reference video.",
        "tips":            [],
    }

    st.session_state.ref_video_path  = dest
    st.session_state.ref_keypoints   = keypoints
    st.session_state.active_exercise = custom_ex
    st.session_state.session_done    = False
    st.session_state.page            = "session"
    st.rerun()


# ---------------------------------------------------------------------------
# Detail panel
# ---------------------------------------------------------------------------

def _show_detail(ex: Dict) -> None:
    """Exercise detail: info + YouTube tutorial + start button."""
    emoji = CATEGORY_EMOJI.get(ex["category"], "")

    # Back button
    if st.button("← Back to Library", key="back_to_lib"):
        st.session_state._lib_selected = None
        st.rerun()

    st.markdown(
        f"<h2 style='margin-top:8px;'>{emoji} {ex['name']}</h2>",
        unsafe_allow_html=True,
    )

    col_info, col_video = st.columns([1, 1.4], gap="large")

    with col_info:
        st.markdown(f"**{ex['description']}**")

        if ex.get("tips"):
            st.markdown("**💡 Form tips:**")
            for tip in ex["tips"]:
                st.markdown(f"- {tip}")

        muscles = ", ".join(ex.get("muscles", []))
        st.markdown(f"**🎯 Muscles:** {muscles}")
        st.markdown(f"**⏱ Duration:** {ex.get('duration', '—')}")

        if ex.get("has_rep_counter"):
            st.info("📊 Rep counter will be enabled for this exercise!")

        st.markdown("---")
        if st.button("🚀 Start Live Session", type="primary", use_container_width=True, key="start_from_detail"):
            with st.spinner("Preparing reference video…"):
                _prepare_template(ex)

    with col_video:
        yt_id = ex.get("youtube_id", "")
        if yt_id:
            st.components.v1.html(
                f"""
                <div style="position:relative;padding-bottom:56.25%;height:0;
                            overflow:hidden;border-radius:12px;background:#000;">
                  <iframe
                    src="https://www.youtube.com/embed/{yt_id}?rel=0&modestbranding=1"
                    style="position:absolute;top:0;left:0;width:100%;height:100%;
                           border:0;border-radius:12px;"
                    allowfullscreen>
                  </iframe>
                </div>
                """,
                height=310,
            )
        else:
            st.info("No tutorial video available.")


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------

def run() -> None:
    # Init state
    if "_lib_selected" not in st.session_state:
        st.session_state._lib_selected = None

    # ── If an exercise is selected, show its detail ─────────────────────────
    if st.session_state._lib_selected:
        _show_detail(st.session_state._lib_selected)
        return

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style='margin-bottom:0;'>📚 Exercise Library</h1>
        <p style='color:#94a3b8;margin-top:4px;'>
            Pick an exercise to see the tutorial, then start your live session.
        </p>
        """,
        unsafe_allow_html=True,
    )

    exercises  = load_exercises()
    categories = ["All"] + sorted({e["category"] for e in exercises})

    cat_filter = st.radio(
        "Filter",
        categories,
        horizontal=True,
        label_visibility="collapsed",
    )

    filtered = exercises if cat_filter == "All" else [e for e in exercises if e["category"] == cat_filter]

    # ── Exercise grid ────────────────────────────────────────────────────────
    cols_per_row = 3
    rows = [filtered[i : i + cols_per_row] for i in range(0, len(filtered), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row, gap="medium")
        for col, ex in zip(cols, row):
            with col:
                st.markdown(_card_html(ex), unsafe_allow_html=True)
                if st.button("▶ Select", key=f"sel_{ex['id']}", use_container_width=True):
                    st.session_state._lib_selected = ex
                    st.rerun()

    # ── Custom upload ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📁  **Or upload your own reference video**", expanded=False):
        st.caption("Supports MP4, MOV, AVI · Max 200 MB")
        uploaded = st.file_uploader(
            "Upload reference video",
            type=["mp4", "mov", "avi"],
            label_visibility="collapsed",
        )
        if uploaded:
            if st.button("🚀 Prepare & Start Session", type="primary", use_container_width=True):
                _prepare_upload(uploaded)
