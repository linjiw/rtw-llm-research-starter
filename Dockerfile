FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/rtw-llm
COPY . /workspace/rtw-llm

RUN python3.11 -m pip install --upgrade pip setuptools wheel && \
    python3.11 -m pip install torch --index-url https://download.pytorch.org/whl/cu124 && \
    python3.11 -m pip install -e .

CMD ["bash"]
