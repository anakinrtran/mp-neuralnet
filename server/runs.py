"""Runs student code in a subprocess and collects its progress events.

Each run lives in RUNS_DIR/<id>/ with:
  model.py          the generated code
  model.pt          trained weights (written by the harness)
  practice_pred.npy decoded practice grid (written by the harness)
  summary.json      what the run history panel shows

Every run that finishes training is also copied to SUBMIT_DIR (model.py, model.pt,
submission.json), so PrairieLearn's Save & Grade always grades the latest finished run.
"""

import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "shared" / "harness.py"


@dataclass
class Run:
    id: str
    dir: Path
    started: float
    status: str = "running"  # running, done, error, stopped
    events: list = field(default_factory=list)
    proc: subprocess.Popen | None = None

    def summary(self) -> dict:
        found = {e["event"]: e for e in self.events}
        val = found.get("evaluate") or found.get("val")
        return {
            "id": self.id,
            "started": self.started,
            "status": self.status,
            "layers": found.get("model", {}).get("layers"),
            "settings": found.get("settings", {}).get("text"),
            "epochs": found.get("train_start", {}).get("epochs"),
            "val_accuracy": val["accuracy"] if val else None,
            "seconds": found.get("done", {}).get("seconds"),
            "error": found.get("error", {}).get("message"),
            "decoded": "decoded" in found,
        }


def crash_message(code: int, run_dir: Path) -> str:
    """Plain words for a run that died without explaining itself."""
    if code in (-9, 137):  # SIGKILL: in a workspace, the 1 GB memory limit (Stop is handled before this)
        batch = "batch_size=" in (run_dir / "model.py").read_text(encoding="utf-8")
        return ("The run ran out of memory and was stopped (the workspace has 1 GB). Conv layers with many "
                "filters before the first MaxPool use the most memory" +
                (", and so do bigger batch sizes. Try a smaller batch size or fewer filters." if batch
                 else ". Try fewer filters, or put a MaxPool earlier."))
    return f"The run crashed (exit code {code})."


class RunManager:
    def __init__(self, runs_dir: Path, submit_dir: Path):
        self.runs_dir = runs_dir
        self.submit_dir = submit_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, Run] = {}
        self.lock = threading.Lock()
        self._load_history()

    def _load_history(self):
        """Restore finished runs from disk so history survives a server restart."""
        for path in sorted(self.runs_dir.glob("*/summary.json")):
            s = json.loads(path.read_text())
            status = "stopped" if s["status"] == "running" else s["status"]
            self.runs[s["id"]] = Run(id=s["id"], dir=path.parent, started=s["started"],
                                     status=status, events=s.get("events", []))

    def active(self) -> Run | None:
        return next((r for r in self.runs.values() if r.status == "running"), None)

    def start(self, code: str) -> Run:
        with self.lock:
            if self.active():
                raise RuntimeError("A run is already in progress. Stop it first.")
            run_id = time.strftime("%Y%m%d-%H%M%S")
            while run_id in self.runs:
                run_id += "x"
            run = Run(id=run_id, dir=self.runs_dir / run_id, started=time.time())
            run.dir.mkdir(parents=True)
            (run.dir / "model.py").write_text(code, encoding="utf-8")
            run.proc = subprocess.Popen(
                [sys.executable, "-u", str(HARNESS), str(run.dir / "model.py"), "--run-dir", str(run.dir)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                cwd=run.dir,
            )
            self.runs[run_id] = run
        threading.Thread(target=self._read, args=(run,), daemon=True).start()
        return run

    def _read(self, run: Run):
        for line in run.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("@@"):
                run.events.append(json.loads(line[2:]))
            elif line.strip():
                run.events.append({"event": "output", "text": line})  # student prints, warnings
        code = run.proc.wait()
        if run.status == "running":
            run.status = "done" if code == 0 else "error"
            if code != 0 and not any(e["event"] == "error" for e in run.events):
                run.events.append({"event": "error", "message": crash_message(code, run.dir)})
        if run.status == "done" and (run.dir / "model.pt").exists():
            self._save_for_grading(run)
        run.events.append({"event": "end", "status": run.status})
        self._save(run)

    def stop(self, run_id: str):
        run = self.runs[run_id]
        if run.status == "running" and run.proc:
            run.status = "stopped"
            run.proc.kill()

    def _save_for_grading(self, run: Run):
        s = run.summary()
        info = {"run_id": run.id, "layers": s["layers"], "epochs": s["epochs"],
                "val_accuracy": s["val_accuracy"], "saved": time.time()}
        self.submit_dir.mkdir(parents=True, exist_ok=True)
        for name in ("model.py", "model.pt"):
            shutil.copyfile(run.dir / name, self.submit_dir / name)
        (self.submit_dir / "submission.json").write_text(json.dumps(info, indent=2))

    def _save(self, run: Run):
        # keep chart events so a reloaded page can redraw them; drop printed output
        events = [e for e in run.events if e["event"] != "output"]
        (run.dir / "summary.json").write_text(json.dumps({**run.summary(), "events": events}))

    def history(self) -> list[dict]:
        return [r.summary() for r in sorted(self.runs.values(), key=lambda r: r.started)]
