# PoseSync 🧘 — AI Pose Coach
A production-ready **yoga & exercise pose accuracy tracker** built with MediaPipe, OpenCV, and Streamlit.

## Features

- 📚 **Exercise Library** — 12 curated exercises (Yoga, Exercise, Stretch) with YouTube tutorial videos
- 🎥 **Live Session** — Side-by-side reference video + webcam with:
  - Ghost skeleton overlay (reference pose projected onto your camera feed)
  - Real-time accuracy gauge (0–100%)
  - Per-joint live score bars
  - Rep counter for push-ups, squats, lunges
- 📊 **Analytics Dashboard** — Post-session results:
  - Overall accuracy + letter grade (A+…D)
  - Accuracy timeline chart
  - Per-joint breakdown + personalised feedback
  - CSV export

## How It Works

The app uses **joint-angle comparison** (not raw pixel positions) — so it works regardless of your distance from the camera. 8 key joints (elbows, knees, hips, shoulders) are compared between the reference pose and your pose every frame.

## Setup

```bash
# Install dependencies (most are likely already installed)
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open **http://localhost:8501**

## Usage

1. Open the **Exercise Library** and pick an exercise (or upload your own video)
2. Watch the tutorial, then click **🚀 Start Live Session**
3. Stand in front of your webcam and follow the reference video
4. Press **⏹ Stop & Analyse** to see your results

## Tech Stack

| Component | Library |
|-----------|---------|
| Pose Estimation | MediaPipe Pose (33 landmarks) |
| Video Processing | OpenCV |
| YouTube Download | yt-dlp |
| UI & Dashboard | Streamlit + Plotly |
| Accuracy Scoring | NumPy (joint angle comparison) |

## Project Structure

```
PoseSync/
├── app.py                  # Main entry point
├── core/
│   ├── pose_extractor.py   # MediaPipe wrapper
│   ├── pose_comparator.py  # Joint angle scoring & rep counter
│   └── video_processor.py # Video preprocessing + YouTube download
├── _pages/
│   ├── library.py          # Exercise library page
│   ├── session.py          # Live session page
│   └── dashboard.py       # Analytics dashboard
└── data/
    └── exercises.json      # 12 curated exercise definitions
```
