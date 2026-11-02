# Student workspace for every MP NeuralNet workspace: the block editor and the server that
# runs its code. Each question picks its workspace with MPNN_WORKSPACE in workspaceOptions.environment.
# Build from the repo root:
#   docker build --platform linux/amd64 -f workspace-image/Dockerfile -t IMAGE .
FROM python:3.13-slim

COPY requirements.txt /opt/mpnn/requirements.txt
RUN pip install --no-cache-dir -r /opt/mpnn/requirements.txt

# Public data only: training/validation sets and every workspace's practice characters (no labels).
COPY data/processed/public/ /opt/mpnn/data/
COPY shared/ /opt/mpnn/shared/
COPY server/ /opt/mpnn/server/
COPY editor/ /opt/mpnn/editor/

# PrairieLearn runs workspaces as 1001:1001 and mounts the student's files at the home dir.
RUN groupadd -g 1001 student && useradd -u 1001 -g 1001 -m -d /home/student student
ENV MPNN_DATA=/opt/mpnn/data \
    MPNN_RUNS=/home/student/.mpnn \
    MPNN_SUBMIT=/home/student \
    PYTHONDONTWRITEBYTECODE=1
LABEL com.prairielearn.workspace.port=8080 \
      com.prairielearn.workspace.home=/home/student \
      com.prairielearn.workspace.rewrite-url=true

USER 1001:1001
WORKDIR /opt/mpnn
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080"]
