# Base image
FROM python:3.11-slim AS base

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Adjusted paths relative to the root context
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/corrupted_mnist/ src/corrupted_mnist/
COPY models/ models/
COPY data/ data/

WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/corrupted_mnist/evaluate.py"]
