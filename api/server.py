"""
FastAPI server — BJJ Match Review  (Phase 0)

Endpoints
---------
POST /api/upload          Upload a video; returns {job_id}.
GET  /api/status/{id}     Poll job progress.
GET  /api/results/{id}    Retrieve the position log (JSON).
GET  /api/video/{id}      Stream the original video.
GET  /api/meta            Server metadata (model type, labels, …).
GET  /                    Serves the web UI (frontend/index.html).

Start the server:
    python -m api.server               # production
    uvicorn api.server:app --reload    # development
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from classifier.inference import PositionInference
from classifier.labels import DISPLAY_NAMES, POSITION_COLORS, POSITION_LABELS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path("/tmp/bjj_jobs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("models/classifier.pt")

SAMPLE_FPS = 2.0          # Frames per second to classify
MAX_UPLOAD_MB = 500       # Hard limit on uploaded video size

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BJJ Match Review",
    description="Phase 0 — position classifier web UI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global inference engine (lazy-init of YOLOv8 inside the class)
# ---------------------------------------------------------------------------

_inferrer: PositionInference | None = None


def get_inferrer() -> PositionInference:
    global _inferrer
    if _inferrer is None:
        _inferrer = PositionInference(
            model_path=str(MODEL_PATH) if MODEL_PATH.exists() else None,
        )
    return _inferrer


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}

_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

def _run_inference(job_id: str, video_path: str) -> None:
    """Called in a thread-pool worker. Updates _jobs[job_id] in place."""
    _jobs[job_id]["status"] = "processing"

    def on_progress(p: float) -> None:
        _jobs[job_id]["progress"] = round(p * 100)

    try:
        inferrer = get_inferrer()
        results = inferrer.process_video(
            video_path=video_path,
            fps=SAMPLE_FPS,
            progress_callback=on_progress,
        )
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["progress"] = 100
        _jobs[job_id]["results"] = results
        _jobs[job_id]["using_heuristic"] = inferrer.using_heuristic
        logger.info("Job %s done — %d entries", job_id, len(results))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", job_id)
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/api/upload", summary="Upload a BJJ video for analysis")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> JSONResponse:
    if not (file.content_type or "").startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_path = job_dir / f"input{suffix}"

    # Stream to disk in chunks to avoid loading entire file into RAM
    with open(video_path, "wb") as fh:
        total_bytes = 0
        limit = MAX_UPLOAD_MB * 1024 * 1024
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            total_bytes += len(chunk)
            if total_bytes > limit:
                fh.close()
                video_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Video exceeds {MAX_UPLOAD_MB} MB limit.",
                )
            fh.write(chunk)

    _jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "filename": file.filename,
        "video_path": str(video_path),
    }

    # Dispatch to thread pool
    background_tasks.add_task(_executor.submit, _run_inference, job_id, str(video_path))

    logger.info("Job %s queued — %s (%.1f MB)", job_id, file.filename, total_bytes / 1e6)
    return JSONResponse({"job_id": job_id})


@app.get("/api/status/{job_id}", summary="Poll job status")
async def get_status(job_id: str) -> JSONResponse:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse({
        "status":   job["status"],
        "progress": job.get("progress", 0),
        "filename": job.get("filename"),
        "error":    job.get("error"),
    })


@app.get("/api/results/{job_id}", summary="Retrieve position log")
async def get_results(job_id: str) -> JSONResponse:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not finished (status={job['status']}).",
        )
    return JSONResponse({
        "results":          job["results"],
        "using_heuristic":  job.get("using_heuristic", True),
    })


@app.get("/api/video/{job_id}", summary="Stream original video")
async def get_video(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    video_path = Path(job["video_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing.")
    return FileResponse(str(video_path), media_type="video/mp4")


@app.get("/api/meta", summary="Server metadata")
async def get_meta() -> JSONResponse:
    inferrer = get_inferrer()
    return JSONResponse({
        "model":            "heuristic" if inferrer.using_heuristic else "mlp",
        "labels":           POSITION_LABELS,
        "display_names":    DISPLAY_NAMES,
        "colors":           POSITION_COLORS,
        "sample_fps":       SAMPLE_FPS,
    })


# ---------------------------------------------------------------------------
# Serve static frontend (must be registered last)
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Dev server entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
