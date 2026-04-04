# ── Cloud Resource Allocation – OpenEnv Environment ──────────────────────
# Single Dockerfile with MODE switch:
#   MODE=server  (default) → OpenEnv FastAPI server for HF Space
#   MODE=demo              → Gradio interactive UI

FROM python:3.10-slim

WORKDIR /app

# Install dependencies (v2 - cache bust)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && echo "=== Installed packages ===" \
    && pip list | grep -i openenv \
    && python -c "import openenv; print('openenv OK:', openenv.__file__)"

# Copy project files (respects .dockerignore)
COPY . /app

# Create non-root user for security
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port (HF Spaces standard)
EXPOSE 7860

# Default mode: OpenEnv server
ENV MODE=server

# Entrypoint: switch between server and demo based on MODE
CMD if [ "$MODE" = "demo" ]; then \
      python app.py; \
    else \
      uvicorn server.app:app --host 0.0.0.0 --port 7860; \
    fi