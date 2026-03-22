"""
Live Session page for PoseSync.

Side-by-side reference video + webcam feed with:
  • Ghost skeleton overlay (reference pose on user's camera)
  • Real-time accuracy gauge (colour-coded)
  • Per-joint live score bars
  • Rep counter for exercises that support it
  • Session recording saved to session_state for the dashboard
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

from core.pose_comparator import RepCounter, score_frame, score_to_colour
from core.pose_extractor import PoseExtractor

CACHE_DIR   = str(Path(__file__).parent.parent / "data" / "cache")
SESSION_DIR = str(Path(__file__).parent.parent / "data" / "sessions")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults = {
        "recording":       False,
        "session_frames":  [],
        "cap":             None,
        "ref_cap":         None,
        "extractor":       None,
        "rep_counter":     None,
        "frame_index":     0,
        "session_done":    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _start_recording() -> None:
    ex = st.session_state.get("active_exercise", {})

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam. Check your camera permissions.")
        return

    ref_cap = cv2.VideoCapture(st.session_state.ref_video_path)
    if not ref_cap.isOpened():
        cap.release()
        st.error("❌ Cannot open reference video.")
        return

    st.session_state.cap            = cap
    st.session_state.ref_cap        = ref_cap
    st.session_state.extractor      = PoseExtractor(model_complexity=0)  # fast mode
    st.session_state.recording      = True
    st.session_state.session_frames = []
    st.session_state.frame_index    = 0
    st.session_state.session_done   = False

    # Rep counter (if applicable)
    if ex.get("has_rep_counter"):
        st.session_state.rep_counter = RepCounter(
            up_angle=ex.get("rep_up_angle", 160),
            down_angle=ex.get("rep_down_angle", 90),
        )
    else:
        st.session_state.rep_counter = None


def _stop_recording() -> None:
    st.session_state.recording = False

    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

    if st.session_state.ref_cap:
        st.session_state.ref_cap.release()
        st.session_state.ref_cap = None

    if st.session_state.extractor:
        st.session_state.extractor.close()
        st.session_state.extractor = None

    # Persist to session_state for dashboard
    st.session_state.session_done = True


def _resize_fit(frame: np.ndarray, max_w: int = 640, max_h: int = 480) -> np.ndarray:
    """Resize keeping aspect ratio within max bounds."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def _accuracy_gauge_html(score: float) -> str:
    colour  = score_to_colour(score)
    label   = "Excellent" if score >= 85 else "Good" if score >= 65 else "Keep going" if score >= 45 else "Adjust pose"
    return f"""
    <div style="text-align:center;padding:12px 0;">
      <div style="font-size:3.6rem;font-weight:800;color:{colour};line-height:1;">
        {score:.0f}%
      </div>
      <div style="font-size:0.9rem;color:#94a3b8;margin-top:4px;">{label}</div>
    </div>
    """


def _joint_bars_html(joint_scores: Dict[str, float]) -> str:
    rows = ""
    for name, val in joint_scores.items():
        col = score_to_colour(val)
        rows += f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
          <div style="width:110px;font-size:0.72rem;color:#94a3b8;">{name}</div>
          <div style="flex:1;background:#1e1e3a;border-radius:6px;height:8px;overflow:hidden;">
            <div style="width:{val}%;height:100%;background:{col};border-radius:6px;
                        transition:width 0.3s ease;"></div>
          </div>
          <div style="width:38px;text-align:right;font-size:0.72rem;color:{col};">{val:.0f}%</div>
        </div>
        """
    return f'<div style="padding:0 4px;">{rows}</div>'


# ---------------------------------------------------------------------------
# Live feed fragment (runs every ~50 ms)
# ---------------------------------------------------------------------------

@st.fragment(run_every=0.05)
def _live_feed_fragment() -> None:
    if not st.session_state.get("recording", False):
        return

    cap     = st.session_state.cap
    ref_cap = st.session_state.ref_cap
    ex_obj  = st.session_state.extractor

    if cap is None or ref_cap is None or ex_obj is None:
        return

    # --- Read frames ---
    ok_user, user_frame = cap.read()
    ok_ref,  ref_frame  = ref_cap.read()

    if not ok_ref:  # loop reference video
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok_ref, ref_frame = ref_cap.read()

    if not ok_user or not ok_ref:
        return

    user_frame = cv2.flip(user_frame, 1)   # mirror webcam
    user_frame = _resize_fit(user_frame)
    ref_frame  = _resize_fit(ref_frame)

    # --- Extract poses ---
    user_data = ex_obj.extract(user_frame)
    ref_data  = ex_obj.extract(ref_frame)

    # --- Draw skeletons ---
    ref_frame  = ex_obj.draw_skeleton(ref_frame,  ref_data,  color=(50, 255, 120))
    user_frame = ex_obj.draw_ghost_skeleton(user_frame, ref_data)          # ghost
    user_frame = ex_obj.draw_skeleton(user_frame, user_data, color=(255, 100, 60))

    # --- Score ---
    result   = score_frame(ref_data, user_data, ex_obj)
    overall  = result["overall"]
    joints   = result["joints"]

    # --- Rep counter ---
    reps = None
    rc   = st.session_state.rep_counter
    if rc is not None:
        ex = st.session_state.get("active_exercise", {})
        joint_key = ex.get("rep_joint", "left_elbow")
        angle_val = result.get("user_angles", {}).get(
            next((j[0] for j in
                  [("Left Elbow","left_elbow"),("Right Elbow","right_elbow"),
                   ("Left Knee","left_knee"),("Right Knee","right_knee")]
                  if j[1] == joint_key), "Left Elbow"),
            180.0,
        )
        reps = rc.update(angle_val)

    # --- Save frame data ---
    st.session_state.session_frames.append(
        {
            "ts":      time.time(),
            "overall": overall,
            "joints":  joints,
            "detected": result["detected"],
        }
    )
    st.session_state.frame_index += 1

    # --- Display ---
    img_col, score_col = st.columns([3, 1], gap="medium")

    with img_col:
        v1, v2 = st.columns(2, gap="small")
        v1.image(
            cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB),
            caption="📹 Reference",
            use_container_width=True,
        )
        v2.image(
            cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB),
            caption="🎥 You (live)",
            use_container_width=True,
        )

    with score_col:
        st.markdown(_accuracy_gauge_html(overall), unsafe_allow_html=True)

        if reps is not None:
            colour = "#7c3aed"
            st.markdown(
                f"""<div style="text-align:center;padding:6px 0 10px;">
                  <div style="font-size:2rem;font-weight:700;color:{colour};">{reps}</div>
                  <div style="font-size:0.78rem;color:#94a3b8;">reps</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown(_joint_bars_html(joints), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------

def run() -> None:
    _init_session_state()

    # Guard: need reference video
    if not st.session_state.get("ref_video_path"):
        st.info("👈 Please select an exercise from the **Exercise Library** first.")
        return

    ex = st.session_state.get("active_exercise", {})

    # ── Header ──────────────────────────────────────────────────────────────
    cat_emoji = {"Exercise": "💪", "Yoga": "🧘", "Stretch": "🤸"}.get(ex.get("category", ""), "🎯")
    st.markdown(
        f"<h1 style='margin-bottom:2px;'>{cat_emoji} {ex.get('name','Live Session')}</h1>"
        f"<p style='color:#94a3b8;margin-top:0;'>{ex.get('description','')}</p>",
        unsafe_allow_html=True,
    )

    # ── Tips expander ────────────────────────────────────────────────────────
    tips = ex.get("tips", [])
    if tips:
        with st.expander("💡 Form tips", expanded=False):
            for t in tips:
                st.markdown(f"- {t}")

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl_l, ctrl_r, ctrl_info = st.columns([1, 1, 2])

    recording = st.session_state.recording

    with ctrl_l:
        start_disabled = recording or st.session_state.session_done
        if st.button("▶ Start Session", type="primary", disabled=start_disabled, use_container_width=True):
            _start_recording()
            st.rerun()

    with ctrl_r:
        if st.button("⏹ Stop & Analyse", disabled=not recording, use_container_width=True):
            _stop_recording()
            st.session_state.page = "dashboard"
            st.rerun()

    with ctrl_info:
        if recording:
            frames = len(st.session_state.session_frames)
            elapsed = int(frames / 20)  # ~20 fps
            st.markdown(
                f"<div style='color:#10b981;'>🔴 Recording — {elapsed}s · {frames} frames</div>",
                unsafe_allow_html=True,
            )
        elif st.session_state.session_done:
            st.success("✅ Session complete! View your results in the Dashboard.")

    st.divider()

    # ── Live feed ────────────────────────────────────────────────────────────
    if recording:
        _live_feed_fragment()
    elif not recording and not st.session_state.session_done:
        st.markdown(
            """
            <div style='text-align:center;padding:60px 0;color:#4b5563;'>
              <div style='font-size:4rem;'>📷</div>
              <div style='font-size:1.1rem;margin-top:12px;'>
                Press <strong>▶ Start Session</strong> to begin.<br/>
                Make sure your webcam is on and you have enough space to move.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
