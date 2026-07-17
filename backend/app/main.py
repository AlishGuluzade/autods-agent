import os
import shutil
import threading
import uuid

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent.graph import compiled_graph
from app.job_store import create_job, get_job

app = FastAPI(title="AutoDS Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/autods_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# All API routes live under /api/*. This is what lets one container serve
# both the FastAPI backend and the built React frontend on a single port
# (as required by platforms like Hugging Face Spaces) without the two ever
# colliding: the frontend's static files own "/", the API owns "/api/*".
api = APIRouter(prefix="/api")


class AnalyzeResponse(BaseModel):
    job_id: str


@api.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported in this MVP.")

    dataset_id = str(uuid.uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"dataset_id": dataset_id}


def _run_graph_in_background(job_id: str, dataset_path: str):
    initial_state = {"job_id": job_id, "dataset_path": dataset_path, "logs": [], "status": "queued"}
    try:
        compiled_graph.invoke(initial_state)
    except Exception as exc:  # keep the job store updated even on failure
        from app.job_store import update_job

        friendly_message = (
            "The agent couldn't finish analyzing this file. This usually means the dataset's "
            "structure confused the automatic target-column detection (e.g. no clear column to "
            "predict, or a column that looks like an ID/free-text field). Try a dataset with an "
            "obvious outcome column (like 'churn', 'survived', or 'price'), or add a GROQ_API_KEY "
            "so a real LLM makes the planning decisions instead of the built-in heuristics."
        )
        update_job(
            job_id,
            {
                "status": "error",
                "error": friendly_message,
                "error_detail": str(exc),
            },
        )


@api.post("/analyze/{dataset_id}", response_model=AnalyzeResponse)
async def analyze(dataset_id: str):
    dataset_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Upload it first via /api/upload.")

    job_id = str(uuid.uuid4())
    create_job(job_id, {"job_id": job_id, "status": "queued", "logs": []})

    thread = threading.Thread(target=_run_graph_in_background, args=(job_id, dataset_path), daemon=True)
    thread.start()

    return {"job_id": job_id}


@api.get("/status/{job_id}")
async def status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")

    return {
        "status": job.get("status"),
        "logs": job.get("logs", []),
        "metrics": job.get("metrics"),
        "best_model_name": job.get("best_model_name"),
        "explanation_text": job.get("explanation_text"),
        "top_features": job.get("top_features"),
        "error": job.get("error"),
        "error_detail": job.get("error_detail"),
    }


@api.get("/report/{job_id}")
async def download_report(job_id: str):
    job = get_job(job_id)
    if job is None or not job.get("report_path"):
        raise HTTPException(status_code=404, detail="Report not ready yet.")

    return FileResponse(job["report_path"], filename=f"autods_report_{job_id}.pdf", media_type="application/pdf")


@api.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(api)

# --- Single-container deployment support (e.g. Hugging Face Spaces) ---
# If a built frontend exists at this path (copied there by the root
# Dockerfile), serve it. This must be mounted AFTER the /api router above,
# since Starlette matches explicit routes before falling through to a
# mount at "/". Local dev with `npm run dev` + vite proxy doesn't need
# this at all -- it only matters for the combined single-container image.
FRONTEND_DIST = os.getenv("FRONTEND_DIST", "/app/frontend_dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")