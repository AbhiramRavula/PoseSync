"""
PoseSync — Main Streamlit entry point.

Handles:
  • Page config + dark theme CSS
  • Sidebar navigation (library / session / dashboard)
  • Global session-state initialisation
  • Routing to page modules
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PoseSync — AI Pose Coach",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "**PoseSync** — compare your poses to a golden reference video using AI.",
    },
)

# ---------------------------------------------------------------------------
# Global dark theme + typography
# ---------------------------------------------------------------------------

st.html(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
      html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
      .stApp { background: #0d0d14; color: #f1f5f9; }
      [data-testid="stSidebar"] { background: #111122; border-right: 1px solid #1e1e3a; }
      .main .block-container { padding-top: 24px; padding-bottom: 40px; max-width: 1200px; }
      h1, h2, h3 { color: #f1f5f9; font-weight: 800; }
      h1 { font-size: 2rem; } h2 { font-size: 1.4rem; }
      .stButton > button {
        border-radius: 10px; font-weight: 600; font-size: 0.9rem; transition: all 0.2s ease;
      }
      .stButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 4px 16px rgba(124,58,237,0.3);
      }
      .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #6d28d9); border: none;
      }
      [data-testid="metric-container"] {
        background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 12px; padding: 12px 16px;
      }
      [data-testid="stExpander"] {
        background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 12px;
      }
      .stProgress > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #6d28d9);
      }
      [data-testid="stFileUploadDropzone"] {
        background: #1a1a2e; border: 2px dashed #2d2d4e; border-radius: 12px;
      }
      hr { border-color: #1e1e3a; }
    </style>
    """
)


# ---------------------------------------------------------------------------
# Global session-state defaults
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "page":            "library",
    "ref_video_path":  None,
    "ref_keypoints":   None,
    "active_exercise": None,
    "recording":       False,
    "session_frames":  [],
    "session_done":    False,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding:16px 0 24px;text-align:center;">
          <div style="font-size:2.6rem;">🧘</div>
          <div style="font-size:1.3rem;font-weight:800;color:#f1f5f9;margin-top:4px;">PoseSync</div>
          <div style="font-size:0.75rem;color:#4f46e5;letter-spacing:2px;">AI POSE COACH</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Navigation ──────────────────────────────────────────────────────────
    nav_labels = {
        "library":   "📚  Exercise Library",
        "session":   "🎥  Live Session",
        "dashboard": "📊  Dashboard",
    }

    options  = list(nav_labels.keys())
    labels   = list(nav_labels.values())
    curr_idx = options.index(st.session_state.page) if st.session_state.page in options else 0

    chosen = st.radio(
        "Navigation",
        labels,
        index=curr_idx,
        label_visibility="hidden",
    )

    chosen_key = options[labels.index(chosen)]
    if chosen_key != st.session_state.page:
        st.session_state.page = chosen_key
        st.rerun()

    # ── Status indicators ───────────────────────────────────────────────────
    st.divider()

    if st.session_state.active_exercise:
        ex = st.session_state.active_exercise
        cat_emoji = {"Exercise": "💪", "Yoga": "🧘", "Stretch": "🤸"}.get(ex.get("category", ""), "🎯")
        st.markdown(
            f"<div style='font-size:0.75rem;color:#64748b;'>SELECTED EXERCISE</div>"
            f"<div style='font-weight:600;color:#a78bfa;'>{cat_emoji} {ex['name']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-size:0.75rem;color:#4b5563;'>No exercise selected yet.</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.recording:
        st.markdown(
            "<div style='color:#ef4444;font-weight:600;margin-top:10px;'>🔴 Recording active</div>",
            unsafe_allow_html=True,
        )
    elif st.session_state.session_done:
        st.markdown(
            "<div style='color:#10b981;font-weight:600;margin-top:10px;'>✅ Session complete</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── About ───────────────────────────────────────────────────────────────
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown(
            """
            1. **Pick** an exercise from the library  
            2. **Watch** the tutorial video  
            3. **Start** the live session — follow along!  
            4. **View** your accuracy analytics  

            The AI compares **joint angles** (not pixel positions) so it works regardless of your distance from the camera.
            """
        )

    st.markdown(
        "<div style='font-size:0.7rem;color:#374151;text-align:center;padding-top:8px;'>"
        "Powered by MediaPipe · OpenCV · Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

page = st.session_state.page

if page == "library":
    from _pages.library import run
    run()

elif page == "session":
    from _pages.session import run
    run()

elif page == "dashboard":
    from _pages.dashboard import run
    run()

else:
    st.error(f"Unknown page: {page}")
    st.session_state.page = "library"
    st.rerun()
