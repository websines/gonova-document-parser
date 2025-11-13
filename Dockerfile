# Multi-stage Dockerfile for Document Parser API
# Optimized for production deployment

FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy dependency files
COPY pyproject.toml requirements.txt ./
COPY .env.example .env

# Install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt && \
    uv pip install fastapi uvicorn[standard]

# Copy application code
COPY document_parser/ ./document_parser/

# Create required directories
RUN mkdir -p uploads outputs logs

# Expose API port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8080

# Activate venv and run API
CMD ["/bin/bash", "-c", "source .venv/bin/activate && uvicorn document_parser.api:app --host ${HOST} --port ${PORT} --workers 1"]
