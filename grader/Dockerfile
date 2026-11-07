# External grader for every MP NeuralNet workspace; each question sets MPNN_WORKSPACE
# in externalGradingOptions.environment. Labels are not in the image: PrairieLearn mounts them
# from the question's tests/ directory at /grade/tests.
# Build from the repo root:
#   docker build --platform linux/amd64 -f grader/Dockerfile -t IMAGE .
FROM python:3.13-slim

COPY requirements.txt /opt/mpnn/requirements.txt
RUN pip install --no-cache-dir -r /opt/mpnn/requirements.txt

COPY data/processed/public/practice_*.npz /opt/mpnn/data/
# Hidden (Workspace 4) and held-out (unused workspace) grid characters: only ever in this image.
COPY data/processed/grader/ /opt/mpnn/data/
COPY shared/ /opt/mpnn/shared/
COPY grader/grade.py grader/predict.py /opt/mpnn/grader/

# Student code runs as this user, which can't read /grade/tests.
RUN useradd -M -s /usr/sbin/nologin ag
ENV MPNN_DATA=/opt/mpnn/data \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/mpnn
CMD ["python", "/opt/mpnn/grader/grade.py"]
