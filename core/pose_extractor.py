"""
MediaPipe Pose wrapper for PoseSync.

Handles landmark extraction and colour-coded skeleton visualisation.
"""

from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Dict, Any, Tuple, List


class PoseExtractor:
    """Extracts and visualises body pose landmarks using MediaPipe Pose."""

    # MediaPipe skeletal connections
    POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

    # Named landmark index map (subset of the 33 MediaPipe landmarks)
    LANDMARKS: Dict[str, int] = {
        "nose": 0,
        "left_shoulder": 11,  "right_shoulder": 12,
        "left_elbow":    13,  "right_elbow":    14,
        "left_wrist":    15,  "right_wrist":    16,
        "left_hip":      23,  "right_hip":      24,
        "left_knee":     25,  "right_knee":     26,
        "left_ankle":    27,  "right_ankle":    28,
    }

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._mp_pose = mp.solutions.pose
        self._mp_draw = mp.solutions.drawing_utils
        self._pose = self._mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True,
        )
        # Core body landmark indices that must be visible for a valid detection
        # (left/right shoulders = 11,12 ; left/right hips = 23,24)
        self._CORE_IDX = [11, 12, 23, 24, 13, 14, 25, 26]

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, frame: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Extract pose landmarks from a BGR frame.

        Returns a dict ``{landmarks: List[dict], world_landmarks}`` or
        ``None`` if no person is detected or frame is invalid.
        """
        if frame is None or frame.size == 0:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True

        if not results.pose_landmarks:
            return None

        h, w = frame.shape[:2]
        landmarks: List[Dict] = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append(
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                    "px": int(lm.x * w),
                    "py": int(lm.y * h),
                }
            )

        # Require at least half of core body landmarks to be clearly visible.
        # This prevents background noise / reflections being tracked as a person.
        core_vis = [landmarks[i]["visibility"] for i in self._CORE_IDX]
        if sum(v > 0.55 for v in core_vis) < 4:
            return None

        return {"landmarks": landmarks, "world_landmarks": results.pose_world_landmarks}

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks_data: Optional[Dict],
        color: Tuple[int, int, int] = (50, 255, 120),
        thickness: int = 2,
        dot_radius: int = 4,
    ) -> np.ndarray:
        """Draw a coloured skeleton overlay on *frame* (copy returned)."""
        if landmarks_data is None or frame is None:
            return frame

        out = frame.copy()
        landmarks = landmarks_data["landmarks"]

        for conn in self.POSE_CONNECTIONS:
            a_idx, b_idx = conn
            a, b = landmarks[a_idx], landmarks[b_idx]
            if a["visibility"] > 0.5 and b["visibility"] > 0.5:
                cv2.line(out, (a["px"], a["py"]), (b["px"], b["py"]),
                         color, thickness, cv2.LINE_AA)

        for lm in landmarks:
            if lm["visibility"] > 0.5:
                cv2.circle(out, (lm["px"], lm["py"]), dot_radius, color, -1, cv2.LINE_AA)

        return out

    def draw_ghost_skeleton(
        self,
        frame: np.ndarray,
        ref_landmarks_data: Optional[Dict],
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Overlay the reference skeleton as a semi-transparent 'ghost'
        (cyan / gold tones) on top of *frame*.
        """
        if ref_landmarks_data is None or frame is None:
            return frame

        overlay = self.draw_skeleton(
            frame,
            ref_landmarks_data,
            color=(30, 200, 255),  # amber-cyan
            thickness=3,
            dot_radius=6,
        )
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # ------------------------------------------------------------------
    # Landmark access
    # ------------------------------------------------------------------

    def get_landmark_array(
        self, landmarks_data: Optional[Dict], name: str
    ) -> Optional[np.ndarray]:
        """Return normalised (x, y, z) array for a named landmark."""
        if landmarks_data is None:
            return None
        idx = self.LANDMARKS.get(name)
        if idx is None:
            return None
        lm = landmarks_data["landmarks"][idx]
        return np.array([lm["x"], lm["y"], lm["z"]], dtype=np.float32)

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
