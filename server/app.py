"""Workspace server: serves the block editor and runs the code it generates.

Run from the repo root:
    python -m uvicorn server.app:app --port 8080

API
  GET  /api/config                    palette and block limits
  GET  /api/runs                      run history
  POST /api/runs        {code}        start a run (409 if one is already running)
  GET  /api/runs/{id}/events          progress as Server-Sent Events (replays from the start)
  GET  /api/runs/{id}/curve?kind=     training loss ("loss") or validation accuracy ("val") points
  POST /api/runs/{id}/stop            stop a run
  GET  /api/runs/{id}/weights?kind=   first-layer weights ("templates" or "features")
  GET  /api/runs/{id}/timelapse       first Conv layer's filters and outputs during training
  GET  /api/runs/{id}/onestep?start=  one training step on a misread validation character, taken apart
  GET  /api/runs/{id}/decoded         decoded practice grid
  GET  /api/practice/characters.png   the practice characters, for hover previews
  GET  /api/blocks                    saved blocks (null if none yet)
  PUT  /api/blocks      {blocks}      save blocks
  GET  /api/submission                the run Save & Grade will grade (null if none)
  POST /api/upload      (image bytes)  Workspace 5: encode a picture and decode it with the latest finished run

Environment
  MPNN_WORKSPACE  which workspace this is: ws1, ws2, ... (default ws1)
  MPNN_RUNS    run history and saved blocks (default: repo/runs/<workspace>)
  MPNN_SUBMIT  where each finished run's model.py, model.pt and submission.json are copied
               (default: repo/submission). In PrairieLearn this is the workspace home, and
               those files are its gradedFiles.
"""

import asyncio
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from server.onestep import one_step  # noqa: E402
from server.runs import RunManager  # noqa: E402
from server.upload import encode_and_decode  # noqa: E402
from server.weights import WeightsUnavailable, first_layer_units, timelapse  # noqa: E402
from shared import config, harness  # noqa: E402
from shared.grids import mosaic  # noqa: E402
from shared.harness import DATA_DIR, WORKSPACE  # noqa: E402
from shared.palette import CLASSES, load_palette, rgb_to_hex  # noqa: E402

RUNS_DIR = Path(os.environ.get("MPNN_RUNS", REPO / "runs" / WORKSPACE.id))
SUBMIT_DIR = Path(os.environ.get("MPNN_SUBMIT", REPO / "submission"))
BLOCKS_FILE = RUNS_DIR / "blocks.json"
EDITOR_DIR = REPO / "editor"

app = FastAPI(title="MP NeuralNet workspace")
manager = RunManager(RUNS_DIR, SUBMIT_DIR)


class RunRequest(BaseModel):
    code: str


class BlocksRequest(BaseModel):
    blocks: dict


def get_run(run_id: str):
    run = manager.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "No such run.")
    return run


@app.get("/api/config")
def get_config():
    return {
        "workspace": WORKSPACE.id,
        "title": WORKSPACE.title,
        "classes": CLASSES,
        "colors": [rgb_to_hex(c) for c in load_palette(WORKSPACE.id)],
        "num_classes": config.NUM_CLASSES,
        "hidden": {"default": config.HIDDEN_DEFAULT, "min": config.HIDDEN_MIN, "max": config.HIDDEN_MAX},
        "conv": {"default": config.CONV_DEFAULT, "min": config.CONV_MIN, "max": config.CONV_MAX},
        "epochs": {"default": WORKSPACE.default_epochs, "max": WORKSPACE.max_epochs},
        "batch_sizes": config.BATCH_SIZES,
        "batch_default": config.BATCH_SIZE,
        # shortcut: ignores MPNN_TRAIN_PER_CLASS (dev only), which would make scheduler lengths off
        "train_size": WORKSPACE.train_per_class * config.NUM_CLASSES,
        # Workspace 4's training-set size field on Load data (characters per class)
        "train_per_class": {"default": WORKSPACE.train_per_class, "min": harness.MIN_TRAIN_PER_CLASS,
                            "max": WORKSPACE.train_per_class},
        "max_patience": harness.MAX_PATIENCE,
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


@app.get("/api/runs/{run_id}/curve")
def run_curve(run_id: str, kind: str = "loss"):
    """The run's training loss ("loss") or validation accuracy ("val") as [epoch progress, value]
    points, for overlaying runs."""
    event, key = {"loss": ("step", "loss"), "val": ("val", "accuracy")}.get(kind, (None, None))
    if event is None:
        raise HTTPException(400, "kind must be 'loss' or 'val'.")
    return [[e["progress"], e[key]] for e in get_run(run_id).events if e["event"] == event]


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


@app.get("/api/runs/{run_id}/timelapse")
def run_timelapse(run_id: str):
    try:
        return timelapse(get_run(run_id).dir)
    except WeightsUnavailable as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/runs/{run_id}/onestep")
def run_onestep(run_id: str, start: int = 0):
    run = get_run(run_id)
    if run.status != "done":
        raise HTTPException(400, "Only a run that finished training can be taken apart.")
    try:
        return one_step(run.dir, start)
    except ValueError as e:
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
        Image.fromarray(mosaic(np.load(DATA_DIR / WORKSPACE.practice_file)["x"])).save(buf, format="PNG")
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


@app.post("/api/upload")
async def upload_image(request: Request):
    run = next((r for r in reversed(manager.history()) if r["status"] == "done"), None)
    if run is None:
        raise HTTPException(400, "Finish a run first: your latest finished model decodes your picture.")
    data = await request.body()
    try:
        # model inference and image work: off the event loop, so runs keep streaming
        result = await asyncio.to_thread(encode_and_decode, data, manager.runs[run["id"]].dir)
    except (ValueError, harness.StudentError) as e:
        raise HTTPException(400, str(e)) from e
    return {**result, "run_id": run["id"]}


class EditorFiles(StaticFiles):
    """The editor's files, revalidated on every load (a cheap 304 when unchanged), so a
    rebuilt image never runs next to stale JavaScript cached by the browser."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


if EDITOR_DIR.exists():
    app.mount("/", EditorFiles(directory=EDITOR_DIR, html=True), name="editor")
