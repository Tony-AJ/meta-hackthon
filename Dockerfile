# ── Cloud Resource Allocation – OpenEnv Environment ──────────────────────
# Single Dockerfile with MODE switch:
#   MODE=server  (default) → OpenEnv FastAPI server for HF Space
#   MODE=demo              → Gradio interactive UI

FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (v6)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "import openenv; print('openenv OK')"

# Copy project files (respects .dockerignore)
COPY . /app

# Create non-root user for security
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port (HF Spaces standard)
EXPOSE 7860

# Default mode: Unified Server (API + Gradio UI)
ENV MODE=server

# Entrypoint: Always run the unified server.
# The Gradio UI is now mounted inside the FastAPI app.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]