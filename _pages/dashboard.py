"""
Analytics Dashboard page for PoseSync.

Displays post-session results:
  • Hero score with letter grade
  • Accuracy timeline chart (Plotly)
  • Per-joint heatmap (horizontal bars)
  • Best / worst moment thumbnails
  • Key session stats (duration, frames, avg accuracy)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.pose_comparator import get_grade, score_to_colour

CACHE_DIR = str(Path(__file__).parent.parent / "data" / "cache")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hero_html(score: float, grade: str, colour: str, exercise_name: str) -> str:
    return f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid {colour}44;
        border-radius: 20px;
        padding: 32px 28px;
        text-align: center;
        margin-bottom: 20px;
    ">
      <div style="font-size:0.85rem;color:#94a3b8;letter-spacing:2px;text-transform:uppercase;">
        Session Results — {exercise_name}
      </div>
      <div style="font-size:5rem;font-weight:900;color:{colour};line-height:1;margin:12px 0 6px;">
        {score:.1f}%
      </div>
      <div style="
        display:inline-block;
        background:{colour}22;
        color:{colour};
        border:1.5px solid {colour};
        border-radius:8px;
        padding:4px 20px;
        font-size:1.4rem;
        font-weight:800;
        letter-spacing:3px;
      ">{grade}</div>
      <div style="font-size:0.85rem;color:#94a3b8;margin-top:12px;">
        {"Outstanding form! 🏆" if score >= 90
         else "Great effort! 💪" if score >= 75
         else "Good start, keep practising! 🎯" if score >= 55
         else "Keep going — consistency is key! 🔄"}
      </div>
    </div>
    """


def _stat_card_html(icon: str, label: str, value: str, colour: str = "#94a3b8") -> str:
    return f"""
    <div style="
        background:#1a1a2e;
        border:1px solid #2d2d4e;
        border-radius:12px;
        padding:16px;
        text-align:center;
    ">
      <div style="font-size:1.8rem;">{icon}</div>
      <div style="font-size:1.4rem;font-weight:700;color:{colour};margin:4px 0 2px;">{value}</div>
      <div style="font-size:0.75rem;color:#64748b;">{label}</div>
    </div>
    """


def _timeline_chart(frames: List[Dict]) -> Optional[go.Figure]:
    if not frames:
        return None

    # Aggregate to per-second buckets
    if len(frames) == 0:
        return None

    t0      = frames[0]["ts"]
    buckets: Dict[int, List[float]] = {}

    for f in frames:
        sec = int(f["ts"] - t0)
        buckets.setdefault(sec, []).append(f["overall"])

    seconds = sorted(buckets)
    avg_acc = [float(np.mean(buckets[s])) for s in seconds]

    # Colour each point
    colours = [score_to_colour(v) for v in avg_acc]

    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatter(
        x=seconds, y=avg_acc,
        mode="lines",
        line=dict(color="#7c3aed", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.08)",
        name="Accuracy",
        hovertemplate="Time: %{x}s<br>Accuracy: %{y:.1f}%<extra></extra>",
    ))

    # Threshold lines
    for threshold, label, lc in [(75, "Good (75%)", "#10b981"), (50, "Fair (50%)", "#f59e0b")]:
        fig.add_hline(
            y=threshold,
            line=dict(color=lc, dash="dot", width=1),
            annotation_text=label,
            annotation_font=dict(size=10, color=lc),
        )

    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 102]),
        plot_bgcolor="#0d0d14",
        paper_bgcolor="#0d0d14",
        font=dict(color="#94a3b8"),
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
        xaxis=dict(gridcolor="#1e1e3e"),
        yaxis2=dict(gridcolor="#1e1e3e"),
    )
    fig.update_xaxes(gridcolor="#1e1e3e")
    fig.update_yaxes(gridcolor="#1e1e3e")

    return fig


def _joint_breakdown_html(avg_joint_scores: Dict[str, float]) -> str:
    rows = ""
    sorted_joints = sorted(avg_joint_scores.items(), key=lambda x: x[1])
    for name, val in sorted_joints:
        col = score_to_colour(val)
        grade, _ = get_grade(val)
        rows += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
          <div style="width:130px;font-size:0.8rem;color:#94a3b8;">{name}</div>
          <div style="flex:1;background:#1e1e3a;border-radius:8px;height:12px;overflow:hidden;">
            <div style="width:{val}%;height:100%;background:{col};border-radius:8px;"></div>
          </div>
          <div style="width:44px;text-align:right;font-size:0.82rem;font-weight:700;color:{col};">
            {val:.0f}%
          </div>
          <div style="width:28px;font-size:0.75rem;color:{col};font-weight:600;">{grade}</div>
        </div>
        """
    return f'<div style="padding:0 6px;">{rows}</div>'


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------

def run() -> None:
    # ── Guard ───────────────────────────────────────────────────────────────
    frames: List[Dict] = st.session_state.get("session_frames", [])

    if not frames:
        if st.session_state.get("session_done"):
            st.warning("No frame data recorded. Please run a session first.")
        else:
            st.info("👈 Complete a **Live Session** to see your analytics here.")
        return

    ex = st.session_state.get("active_exercise", {})
    ex_name = ex.get("name", "Exercise")

    # ── Compute aggregates ─────────────────────────────────────────────────
    detected = [f for f in frames if f.get("detected", True)]
    all_scores  = [f["overall"] for f in detected] if detected else [0.0]
    avg_score   = float(np.mean(all_scores))
    best_score  = float(np.max(all_scores))
    worst_score = float(np.min(all_scores))

    # Per-joint averages
    joint_names = list(frames[0]["joints"].keys()) if frames else []
    avg_joints: Dict[str, float] = {}
    for jn in joint_names:
        vals = [f["joints"].get(jn, 0.0) for f in detected] if detected else [0.0]
        avg_joints[jn] = round(float(np.mean(vals)), 1)

    # Duration
    if len(frames) > 1:
        duration_s = int(frames[-1]["ts"] - frames[0]["ts"])
    else:
        duration_s = 0

    grade, grade_colour = get_grade(avg_score)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("<h1 style='margin-bottom:4px;'>📊 Session Analytics</h1>", unsafe_allow_html=True)

    # ── Hero score ───────────────────────────────────────────────────────────
    st.markdown(_hero_html(avg_score, grade, grade_colour, ex_name), unsafe_allow_html=True)

    # ── Stat cards ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.markdown(
        _stat_card_html("⏱", "Duration", f"{duration_s}s", "#94a3b8"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        _stat_card_html("🏆", "Best moment", f"{best_score:.0f}%", "#10b981"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        _stat_card_html("📉", "Worst moment", f"{worst_score:.0f}%", score_to_colour(worst_score)),
        unsafe_allow_html=True,
    )
    c4.markdown(
        _stat_card_html("🖼", "Frames", f"{len(frames)}", "#94a3b8"),
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── Timeline chart ───────────────────────────────────────────────────────
    st.markdown("### 📈 Accuracy Over Time")
    fig = _timeline_chart(frames)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-joint breakdown ──────────────────────────────────────────────────
    st.markdown("### 🦴 Per-Joint Breakdown")
    if avg_joints:
        weakest = min(avg_joints, key=avg_joints.get)
        strongest = max(avg_joints, key=avg_joints.get)
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(_joint_breakdown_html(avg_joints), unsafe_allow_html=True)
        with col_b:
            st.markdown(
                f"""
                <div style="background:#1a1a2e;border:1px solid #2d2d4e;
                            border-radius:12px;padding:16px;margin-top:4px;">
                  <div style="font-size:0.75rem;color:#64748b;margin-bottom:8px;">HIGHLIGHTS</div>
                  <div style="margin-bottom:10px;">
                    <div style="font-size:0.7rem;color:#10b981;">💪 Strongest</div>
                    <div style="font-size:0.9rem;font-weight:700;color:#f1f5f9;">{strongest}</div>
                    <div style="font-size:0.8rem;color:#10b981;">{avg_joints[strongest]:.0f}%</div>
                  </div>
                  <div>
                    <div style="font-size:0.7rem;color:#ef4444;">⚠️ Needs work</div>
                    <div style="font-size:0.9rem;font-weight:700;color:#f1f5f9;">{weakest}</div>
                    <div style="font-size:0.8rem;color:#ef4444;">{avg_joints[weakest]:.0f}%</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Form tips based on weak joints ──────────────────────────────────────
    if avg_joints:
        weak_joints = [j for j, s in avg_joints.items() if s < 60]
        if weak_joints:
            st.markdown("### 💡 Personalised Feedback")
            tips_map = {
                "Left Elbow":     "Try to consciously watch your left arm alignment during practice.",
                "Right Elbow":    "Focus on keeping your right elbow at the correct angle.",
                "Left Knee":      "Pay attention to your left knee — it may be tracking inward or outward.",
                "Right Knee":     "Your right knee alignment needs attention — keep it over your toes.",
                "Left Hip":       "Your left hip positioning was off — try opening or squaring it more.",
                "Right Hip":      "Your right hip needs more rotation or alignment work.",
                "Left Shoulder":  "Keep your left shoulder down and back — avoid shrugging.",
                "Right Shoulder": "Your right shoulder position drifted — focus on keeping it stable.",
            }
            for j in weak_joints:
                tip = tips_map.get(j, f"Work on your {j} positioning.")
                st.markdown(f"- **{j}**: {tip}")

    # ── Actions ──────────────────────────────────────────────────────────────
    st.divider()
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("🔄 Retry This Exercise", use_container_width=True):
            st.session_state.session_frames = []
            st.session_state.session_done   = False
            st.session_state.page           = "session"
            st.rerun()

    with b2:
        if st.button("📚 Browse Library", use_container_width=True):
            st.session_state.session_frames  = []
            st.session_state.session_done    = False
            st.session_state.ref_video_path  = None
            st.session_state.ref_keypoints   = None
            st.session_state.active_exercise = None
            st.session_state.page            = "library"
            st.rerun()

    with b3:
        # CSV export
        import io
        import csv

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["frame", "timestamp", "overall"] + joint_names)
        for i, f in enumerate(frames):
            writer.writerow(
                [i, f["ts"], f["overall"]] + [f["joints"].get(j, 0) for j in joint_names]
            )
        st.download_button(
            "⬇ Export CSV",
            data=buf.getvalue(),
            file_name="posesync_session.csv",
            mime="text/csv",
            use_container_width=True,
        )
