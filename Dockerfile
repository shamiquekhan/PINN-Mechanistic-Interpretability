# PINN Mechanistic Interpretability — reproducible environment
# Build:  docker build -t pinn-mi .
# Run:    docker run --gpus all -v $(pwd)/runs:/workspace/runs pinn-mi pytest tests/
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# System deps for matplotlib (Agg backend) and builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps (pinned).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source.
COPY . .

# Default: run the test suite. Override for specific stages:
#   docker run --gpus all pinn-mi python -m experiments.run_pipeline --stages 5
CMD ["python", "-m", "pytest", "tests/", "-q"]
