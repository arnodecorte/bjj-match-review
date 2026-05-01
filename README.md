# BJJ Match Review

> Chess.com-style game analysis for Brazilian Jiu-Jitsu

Upload a match. Get a structured breakdown of positions, transitions, mistakes, and drilling recommendations — the same way chess.com reviews your games.

## What it does

- Detects and timestamps BJJ positions (guard, mount, back, etc.)
- Classifies techniques and transitions (berimbolo, kimura, guard pass, etc.)
- Scores positional control over time
- Flags **Brilliant moves**, **Mistakes**, and **Blunders**
- Generates a natural language match report with drilling recommendations

## Status

✅ **Phase 0 complete** — position classifier + web UI shipped.

See [ROADMAP.md](./ROADMAP.md) for the full build plan.

---

## Quick start (Phase 0 web UI)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.10+ required. YOLOv8 model weights (~6 MB) are downloaded
> automatically on first run.

### 2. Start the server

```bash
make dev
# or directly:
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 3. Upload a video

Drag-and-drop any BJJ sparring video (MP4, MOV, AVI, WebM).
The classifier samples at 2 fps, extracts keypoints with YOLOv8-pose,
and returns a timestamped position log.

Without a trained model the **heuristic classifier** (geometry-based rules)
is used automatically.  The banner in the UI will tell you which mode is active.

---

## Training the ML classifier (optional, improves accuracy)

### 1. Download the ViCoS dataset

```bash
bash scripts/download_vicos.sh
# Images + annotations saved to data/vicos/
```

Dataset: [ViCoS Lab BJJ](https://vicos.si/resources/jiujitsu/) — 120k images,
18 position classes, CC BY-NC-SA 4.0 licence.

### 2. Train

```bash
make train
# python -m classifier.train --data-dir data/vicos --epochs 50 --output models/classifier.pt
```

The server automatically loads `models/classifier.pt` on startup.
Restart the server after training to switch from heuristic to ML mode.

---

## Project layout

```
bjj-match-review/
├── classifier/
│   ├── labels.py       # 18-class ViCoS label definitions
│   ├── model.py        # PyTorch MLP classifier
│   ├── preprocess.py   # Keypoint normalisation
│   ├── inference.py    # YOLOv8-pose → position log pipeline
│   └── train.py        # ViCoS training script
├── data/
│   └── vicos.py        # Dataset loader + verifier
├── api/
│   └── server.py       # FastAPI server
├── frontend/
│   ├── index.html      # Web UI
│   ├── styles.css
│   └── app.js
├── scripts/
│   └── download_vicos.sh
├── models/             # Trained weights go here (gitignored)
├── requirements.txt
└── Makefile
```

## Classifier architecture

| Component | Detail |
|---|---|
| Pose estimation | YOLOv8-nano-pose (COCO 17-point, 2 athletes) |
| Input features | 17 kp × 2 athletes × 3 (x, y, conf) = **102 features** |
| Classifier | MLP: 256 → 128 → 64 → 18 classes |
| Fallback | Geometry-based heuristic (no weights needed) |
| Target accuracy | >85% (ViCoS 18-class, matching roadmap goal) |

## Data sources

See [DATA_SOURCES.md](./DATA_SOURCES.md) for full details on the three
anchor datasets used in Phase 0.
