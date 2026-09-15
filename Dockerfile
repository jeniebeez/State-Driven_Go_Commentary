# ==============================================================================
# Dockerfile for State-Driven Go Commentary
# Python 3.12 Environment (Matching vllm_2 Conda Spec)
# ==============================================================================

FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    poppler-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency configuration
COPY requirements.txt .

# Install dependencies with PyTorch index fallback
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130 || \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, dataset, and knowledge base
COPY src/ ./src
COPY data/ ./data
COPY go_knowledge_base/ ./go_knowledge_base
COPY README.md .

# Create output directory for evaluation results
RUN mkdir -p eval-results

# Set default API endpoint (connecting to host machine's vLLM / OpenAI API service)
ENV API_URL="http://host.docker.internal:8000/v1/chat/completions"

# Default execution command
CMD ["python", "src/sgf_rag_multi-agentic-llm.py"]
