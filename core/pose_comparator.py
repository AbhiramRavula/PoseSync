"""
Pose comparison and accuracy scoring for PoseSync.

Uses joint-angle analysis for scale- and position-invariant comparison between
a reference pose and the user's pose.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional, Dict, List, Tuple

from core.pose_extractor import PoseExtractor

# ---------------------------------------------------------------------------
# Joint definitions — each entry: (display_name, landmark_a, vertex_b, landmark_c)
# The angle is measured at the vertex landmark.
# ---------------------------------------------------------------------------
KEY_JOINTS: List[Tuple[str, str, str, str]] = [
    ("Left Elbow",    "left_shoulder",  "left_elbow",    "left_wrist"),
    ("Right Elbow",   "right_shoulder", "right_elbow",   "right_wrist"),
    ("Left Knee",     "left_hip",       "left_knee",     "left_ankle"),
    ("Right Knee",    "right_hip",      "right_knee",    "right_ankle"),
    ("Left Hip",      "left_shoulder",  "left_hip",      "left_knee"),
    ("Right Hip",     "right_shoulder", "right_hip",     "right_knee"),
    ("Left Shoulder", "left_elbow",     "left_shoulder", "left_hip"),
    ("Right Shoulder","right_elbow",    "right_shoulder","right_hip"),
]

# Weight each joint — hips matter more for yoga / full-body movements
JOINT_WEIGHTS: Dict[str, float] = {
    "Left Elbow":     1.0,
    "Right Elbow":    1.0,
    "Left Knee":      1.2,
    "Right Knee":     1.2,
    "Left Hip":       1.3,
    "Right Hip":      1.3,
    "Left Shoulder":  1.0,
    "Right Shoulder": 1.0,
}

# Angle tolerance bucket — diff ≥ this → score 0
MAX_ANGLE_DIFF = 60.0  # degrees


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------

def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the angle (degrees) at vertex *b* formed by vectors b→a and b→c.
    Numerically stable via eps in the denominator.
    """
    ba = a - b
    bc = c - b
    norms = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norms < 1e-8:
        return 0.0
    cos_val = np.clip(np.dot(ba, bc) / norms, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _get_lm(extractor: PoseExtractor, data: Dict, name: str) -> Optional[np.ndarray]:
    return extractor.get_landmark_array(data, name)


# ---------------------------------------------------------------------------
# Per-frame scoring
# ---------------------------------------------------------------------------

def score_frame(
    ref_data: Optional[Dict],
    user_data: Optional[Dict],
    extractor: PoseExtractor,
) -> Dict:
    """
    Compute per-frame accuracy.

    Returns::

        {
            "overall": float,          # 0–100
            "joints": {name: float},   # per-joint scores 0–100
            "ref_angles": {name: float},
            "user_angles": {name: float},
            "detected": bool,
        }
    """
    empty = {
        "overall": 0.0,
        "joints": {j[0]: 0.0 for j in KEY_JOINTS},
        "ref_angles": {j[0]: 0.0 for j in KEY_JOINTS},
        "user_angles": {j[0]: 0.0 for j in KEY_JOINTS},
        "detected": False,
    }

    if ref_data is None or user_data is None:
        return empty

    joint_scores: Dict[str, float] = {}
    ref_angles:   Dict[str, float] = {}
    user_angles:  Dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for joint_name, lm_a, lm_b, lm_c in KEY_JOINTS:
        ra = _get_lm(extractor, ref_data, lm_a)
        rb = _get_lm(extractor, ref_data, lm_b)
        rc = _get_lm(extractor, ref_data, lm_c)
        ua = _get_lm(extractor, user_data, lm_a)
        ub = _get_lm(extractor, user_data, lm_b)
        uc = _get_lm(extractor, user_data, lm_c)

        if any(x is None for x in (ra, rb, rc, ua, ub, uc)):
            joint_scores[joint_name] = 0.0
            ref_angles[joint_name]   = 0.0
            user_angles[joint_name]  = 0.0
            continue

        r_ang = compute_angle(ra, rb, rc)
        u_ang = compute_angle(ua, ub, uc)

        diff  = abs(r_ang - u_ang)
        score = max(0.0, 100.0 - (diff / MAX_ANGLE_DIFF) * 100.0)

        joint_scores[joint_name] = round(score, 1)
        ref_angles[joint_name]   = round(r_ang, 1)
        user_angles[joint_name]  = round(u_ang, 1)

        w = JOINT_WEIGHTS.get(joint_name, 1.0)
        weighted_sum += score * w
        total_weight += w

    overall = round(weighted_sum / total_weight, 1) if total_weight else 0.0

    return {
        "overall":    overall,
        "joints":     joint_scores,
        "ref_angles": ref_angles,
        "user_angles":user_angles,
        "detected":   True,
    }


# ---------------------------------------------------------------------------
# Grade / colour helpers
# ---------------------------------------------------------------------------

GRADE_MAP = [
    (90, "A+", "#10b981"),
    (80, "A",  "#34d399"),
    (70, "B",  "#3b82f6"),
    (60, "C",  "#f59e0b"),
    (0,  "D",  "#ef4444"),
]


def get_grade(score: float) -> Tuple[str, str]:
    """Return (letter_grade, hex_colour) for a 0-100 accuracy score."""
    for threshold, grade, colour in GRADE_MAP:
        if score >= threshold:
            return grade, colour
    return "D", "#ef4444"


def score_to_colour(score: float) -> str:
    """Return a hex colour string based on score (green/amber/red)."""
    if score >= 75:
        return "#10b981"
    if score >= 50:
        return "#f59e0b"
    return "#ef4444"


# ---------------------------------------------------------------------------
# Rep counter
# ---------------------------------------------------------------------------

class RepCounter:
    """
    Counts exercise repetitions using a hysteresis state machine on a joint angle.

    Usage::

        counter = RepCounter(up_angle=160, down_angle=90)
        reps = counter.update(current_elbow_angle)
    """

    def __init__(
        self,
        up_angle: float = 160.0,
        down_angle: float = 90.0,
        smoothing: int = 5,
    ) -> None:
        self.up_angle   = up_angle
        self.down_angle = down_angle
        self._state     = "up"
        self.count      = 0
        self._history: deque = deque(maxlen=smoothing)

    def update(self, angle: float) -> int:
        """Feed a new angle reading; returns current rep count."""
        self._history.append(angle)
        smooth = float(np.mean(self._history))

        if self._state == "up" and smooth < self.down_angle:
            self._state = "down"
        elif self._state == "down" and smooth > self.up_angle:
            self._state = "up"
            self.count += 1

        return self.count

    def reset(self) -> None:
        self.count  = 0
        self._state = "up"
        self._history.clear()
