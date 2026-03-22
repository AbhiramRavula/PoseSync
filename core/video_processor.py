"""
Reference video preprocessing and YouTube download utilities for PoseSync.

Key features:
- Hash-based caching: same video is never re-processed.
- yt-dlp integration: download exercise templates from YouTube.
- Progress callbacks for Streamlit UI feedback.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_hash(path: str) -> str:
    """Return MD5 hex-digest of a file (for cache invalidation)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


ProgressCB = Optional[Callable[[float, str], None]]


# ---------------------------------------------------------------------------
# Reference video preprocessing
# ---------------------------------------------------------------------------

def preprocess_reference(
    video_path: str,
    cache_dir: str,
    progress_cb: ProgressCB = None,
) -> Optional[List[Optional[Dict]]]:
    """
    Extract pose keypoints for every frame in the reference video.

    Results are cached as JSON keyed by the file's MD5.
    Returns a list (one entry per frame) of landmark dicts or ``None``.
    """
    from core.pose_extractor import PoseExtractor

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # --- cache lookup ---
    try:
        fhash = _file_hash(video_path)
    except OSError:
        fhash = Path(video_path).stem

    cache_path = Path(cache_dir) / f"ref_{fhash}.json"

    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            if progress_cb:
                progress_cb(1.0, "✅ Loaded from cache — no re-processing needed!")
            return data["keypoints"]
        except Exception:
            pass  # corrupt cache → re-process

    # --- process video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if progress_cb:
            progress_cb(0.0, "❌ Cannot open video file.")
        return None

    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    keypoints: List[Optional[Dict]] = []

    with PoseExtractor(model_complexity=1) as extractor:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Resize for speed (keep AR)
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            data = extractor.extract(frame)
            entry = {"frame": idx, "landmarks": data["landmarks"] if data else None}
            keypoints.append(entry)

            idx += 1
            if progress_cb and idx % 10 == 0:
                progress_cb(
                    min(idx / total, 0.99),
                    f"⚙️  Analysing frame {idx} / {total}…",
                )

    cap.release()

    # --- save cache ---
    try:
        with open(cache_path, "w") as f:
            json.dump({"hash": fhash, "keypoints": keypoints}, f)
    except Exception:
        pass  # non-fatal

    if progress_cb:
        progress_cb(1.0, f"✅ Done! Extracted {idx} frames.")

    return keypoints


def load_reference_keypoints(video_path: str, cache_dir: str) -> Optional[List]:
    """Return cached keypoints without re-processing. Returns None if no cache."""
    try:
        fhash = _file_hash(video_path)
    except OSError:
        return None

    cache_path = Path(cache_dir) / f"ref_{fhash}.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)["keypoints"]
        except Exception:
            pass
    return None


def get_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Retrieve a single frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def get_video_fps(video_path: str) -> float:
    """Return the FPS of a video, or 30.0 as a safe default."""
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return fps if fps > 0 else 30.0


# ---------------------------------------------------------------------------
# YouTube downloader
# ---------------------------------------------------------------------------

def download_youtube_video(
    youtube_id: str,
    cache_dir: str,
    progress_cb: ProgressCB = None,
) -> Optional[str]:
    """
    Download a YouTube video with yt-dlp (max 480p, cached).

    Returns the path to the downloaded video file, or ``None`` on failure.
    The file extension is determined by what yt-dlp selects (mp4, webm, etc.).
    """
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        if progress_cb:
            progress_cb(0.0, "❌ yt-dlp not installed.")
        return None

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Check if any cached version already exists (any extension)
    existing = list(Path(cache_dir).glob(f"{youtube_id}.*"))
    existing = [f for f in existing if f.stat().st_size > 10_000]
    if existing:
        if progress_cb:
            progress_cb(1.0, "✅ Using cached video.")
        return str(existing[0])

    if progress_cb:
        progress_cb(0.05, "🌐 Connecting to YouTube…")

    def _hook(d: dict) -> None:
        if d["status"] == "downloading" and progress_cb:
            try:
                raw   = d.get("_percent_str", "0%").strip().rstrip("%")
                pct   = float(raw) / 100.0
                speed = d.get("_speed_str", "")
                progress_cb(min(pct * 0.9, 0.89), f"⬇️  Downloading {d.get('_percent_str','')} — {speed}")
            except Exception:
                pass
        elif d["status"] == "finished" and progress_cb:
            progress_cb(0.95, "🔄 Processing…")

    # Use %(ext)s so the actual extension matches the downloaded format.
    # Format priority:
    #   1. Best combined stream ≤ 480p in any container (most compatible, no ffmpeg needed)
    #   2. Absolute best available stream
    outtmpl = str(Path(cache_dir) / f"{youtube_id}.%(ext)s")

    ydl_opts = {
        # "worstvideo" is avoided; we prefer small+usable over tiny+broken.
        # bestvideo+bestaudio requires ffmpeg; fallback to single-stream formats.
        "format": (
            "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo[height<=480]+bestaudio"
            "/best[height<=480]"
            "/best"
        ),
        "outtmpl":      outtmpl,
        "quiet":        True,
        "no_warnings":  True,
        "progress_hooks": [_hook],
        # Retry on transient errors
        "retries":      3,
        "fragment_retries": 3,
    }

    try:
        import yt_dlp

        url = f"https://www.youtube.com/watch?v={youtube_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded file (extension may vary)
        downloaded = list(Path(cache_dir).glob(f"{youtube_id}.*"))
        downloaded = [f for f in downloaded if f.stat().st_size > 10_000]
        if downloaded:
            if progress_cb:
                progress_cb(1.0, f"✅ Download complete! ({downloaded[0].suffix.lstrip('.')})")
            return str(downloaded[0])

        if progress_cb:
            progress_cb(0.0, "❌ Download finished but file not found.")

    except Exception as exc:
        short = str(exc).split("\n")[0][:200]   # truncate long yt-dlp tracebacks
        if progress_cb:
            progress_cb(0.0, f"❌ Download failed: {short}")

    return None

