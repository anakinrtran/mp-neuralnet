"""Workspace server: serves the block editor and runs the code it generates.

Run from the repo root:
    python -m uvicorn server.app:app --port 8080

API
  GET  /api/config                    palette and block limits
  GET  /api/runs                      run history
  POST /api/runs        {code}        start a run (409 if one is already running)
  GET  /api/runs/{id}/events          progress as Server-Sent Events (replays from the start)
  POST /api/runs/{id}/stop            stop a run
  GET  /api/runs/{id}/weights?kind=   first-layer weights ("templates" or "features")
  GET  /api/runs/{id}/decoded         decoded practice grid
  GET  /api/practice/characters.png   the practice characters, for hover previews
  GET  /api/blocks                    saved blocks (null if none yet)
  PUT  /api/blocks      {blocks}      save blocks
  GET  /api/submission                the run saved for grading (null if none)
  POST /api/submission  {run_id}      save a finished run's code and weights for grading

Environment
  MPNN_RUNS    run history and saved blocks (default: repo/runs)
  MPNN_SUBMIT  where Submit puts model.py, model.pt and submission.json (default: repo/submission).
               In PrairieLearn this is the workspace home, and those files are its gradedFiles.
"""

import asyncio
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from server.runs import RunManager  # noqa: E402
from server.weights import WeightsUnavailable, first_layer_units  # noqa: E402
from shared import config  # noqa: E402
from shared.grids import mosaic  # noqa: E402
from shared.harness import DATA_DIR  # noqa: E402
from shared.palette import CLASSES, PALETTE_HEX  # noqa: E402

RUNS_DIR = Path(os.environ.get("MPNN_RUNS", REPO / "runs"))
SUBMIT_DIR = Path(os.environ.get("MPNN_SUBMIT", REPO / "submission"))
BLOCKS_FILE = RUNS_DIR / "blocks.json"
EDITOR_DIR = REPO / "editor"

app = FastAPI(title="MP NeuralNet workspace")
manager = RunManager(RUNS_DIR)


class RunRequest(BaseModel):
    code: str


class BlocksRequest(BaseModel):
    blocks: dict


class SubmitRequest(BaseModel):
    run_id: str


def get_run(run_id: str):
    run = manager.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "No such run.")
    return run


@app.get("/api/config")
def get_config():
    return {
        "classes": CLASSES,
        "colors": [PALETTE_HEX[c] for c in CLASSES],
        "num_classes": config.NUM_CLASSES,
        "hidden": {"default": config.HIDDEN_DEFAULT, "min": config.HIDDEN_MIN, "max": config.HIDDEN_MAX},
        "epochs": {"default": config.DEFAULT_EPOCHS, "max": config.MAX_EPOCHS},
    }


@app.get("/api/runs")
def list_runs():
    return manager.history()


@app.post("/api/runs")
def start_run(req: RunRequest):
    try:
        return manager.start(req.code).summary()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str):
    run = get_run(run_id)

    async def stream():
        sent = 0
        while True:
            while sent < len(run.events):
                event = run.events[sent]
                sent += 1
                yield f"data: {json.dumps(event)}\n\n"
                if event["event"] == "end":
                    return
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    manager.stop(get_run(run_id).id)
    return {"ok": True}


@app.get("/api/runs/{run_id}/weights")
def run_weights(run_id: str, kind: str = "features"):
    if kind not in ("templates", "features"):
        raise HTTPException(400, "kind must be 'templates' or 'features'.")
    try:
        return first_layer_units(get_run(run_id).dir, kind)
    except WeightsUnavailable as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/runs/{run_id}/decoded")
def run_decoded(run_id: str):
    path = get_run(run_id).dir / "practice_pred.npy"
    if not path.exists():
        raise HTTPException(404, "This run didn't decode the practice image.")
    return {"grid": np.load(path).tolist()}


_characters_png: bytes | None = None


@app.get("/api/practice/characters.png")
def practice_characters():
    global _characters_png
    if _characters_png is None:
        buf = io.BytesIO()
        Image.fromarray(mosaic(np.load(DATA_DIR / "practice_ws1.npz")["x"])).save(buf, format="PNG")
        _characters_png = buf.getvalue()
    return Response(_characters_png, media_type="image/png")


@app.get("/api/blocks")
def get_blocks():
    return {"blocks": json.loads(BLOCKS_FILE.read_text()) if BLOCKS_FILE.exists() else None}


@app.put("/api/blocks")
def put_blocks(req: BlocksRequest):
    # write then rename, so a crash never leaves half a file
    tmp = BLOCKS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(req.blocks))
    tmp.replace(BLOCKS_FILE)
    return {"ok": True}


@app.get("/api/submission")
def get_submission():
    path = SUBMIT_DIR / "submission.json"
    return {"submission": json.loads(path.read_text()) if path.exists() else None}


@app.post("/api/submission")
def submit(req: SubmitRequest):
    run = get_run(req.run_id)
    if run.status != "done" or not (run.dir / "model.pt").exists():
        raise HTTPException(400, "Only a run that finished training can be submitted.")
    s = run.summary()
    info = {"run_id": run.id, "layers": s["layers"], "epochs": s["epochs"],
            "val_accuracy": s["val_accuracy"], "submitted": time.time()}
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("model.py", "model.pt"):
        shutil.copyfile(run.dir / name, SUBMIT_DIR / name)
    (SUBMIT_DIR / "submission.json").write_text(json.dumps(info, indent=2))
    return {"submission": info}


if EDITOR_DIR.exists():
    app.mount("/", StaticFiles(directory=EDITOR_DIR, html=True), name="editor")
