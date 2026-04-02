# ── Cloud Resource Allocation – OpenEnv Environment ──────────────────────────
# Single Dockerfile with MODE switch:
#   MODE=server  (default) → OpenEnv FastAPI server for HF Space
#   MODE=demo              → Gradio interactive UI

FROM python:3.10-slim

WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

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