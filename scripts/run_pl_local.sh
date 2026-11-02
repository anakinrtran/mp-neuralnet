#!/usr/bin/env bash
# Run PrairieLearn locally with this repo's questions, workspaces and external grading,
# using the locally built images (no pulling). Then open http://localhost:3000/pl
# and preview the question from the course's Questions page.
#
#   docker build --platform linux/amd64 -f workspace-image/Dockerfile -t cs124h/mpnn-workspace:ws1 .
#   docker build --platform linux/amd64 -f grader/Dockerfile -t cs124h/mpnn-grader:ws1 .
#   scripts/run_pl_local.sh [JOBS_DIR]
#
# Works from Git Bash with Docker Desktop on Windows, and from Linux/macOS/WSL.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="${1:-$HOME/pl_ag_jobs}"
mkdir -p "$JOBS"

if command -v cygpath >/dev/null; then
  # Git Bash: give docker Windows paths, and give PrairieLearn the path Docker Desktop's
  # daemon uses for the same folder, since PrairieLearn asks the daemon to mount job dirs.
  export MSYS_NO_PATHCONV=1
  win() { cygpath -w "$1"; }
  # Start from the Windows form (C:/...): cygpath -u can give Git Bash aliases such as /tmp.
  HOST_JOBS="/run/desktop/mnt/host/$(cygpath -m "$JOBS" | sed -E 's#^([a-zA-Z]):/#\L\1/#')"
else
  win() { echo "$1"; }
  HOST_JOBS="$JOBS"
fi

# Interactive in a terminal (Ctrl-C stops it); otherwise detached:
# follow with "docker logs -f pl-local", stop with "docker stop pl-local".
TTY=$([ -t 0 ] && echo -it || echo -d)
docker run $TTY --rm --name pl-local -p 3000:3000 \
  -v "$(win "$REPO/pl-dev"):/course" \
  -v "$(win "$REPO/questions"):/course/questions" \
  -v "$(win "$REPO/pl-dev/config.json"):/PrairieLearn/config.json" \
  -v "$(win "$JOBS"):/jobs" \
  -e HOST_JOBS_DIR="$HOST_JOBS" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --add-host=host.docker.internal:172.17.0.1 \
  prairielearn/prairielearn
