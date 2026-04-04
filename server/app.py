"""
server/app.py
=============
FastAPI application for the Cloud Resource Allocation OpenEnv environment.

Uses the ``create_fastapi_app`` helper from ``openenv.core.env_server``
to expose reset / step / state endpoints automatically.
"""

import sys
import os

# Ensure project root is on the path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app

from models import CloudAction, CloudObservation
from server.cloud_environment import CloudResourceEnvironment

import gradio as gr
from app import demo

# create_fastapi_app expects the Environment CLASS, not an instance
app = create_fastapi_app(CloudResourceEnvironment, CloudAction, CloudObservation)

# Mount the beautiful Gradio UI onto the FastAPI app so both are available on port 7860
# Mounting at "/" ensures it is the landing page for the Space.
# FastAPI's internal routes (like /reset) will take precedence.
app = gr.mount_gradio_app(app, demo, path="/")


@app.get("/health")
def health_check():
    """Environment health check."""
    return {
        "status": "online",
        "environment": "Cloud Resource Allocation",
        "mode": "Unified Server (API + UI)",
        "endpoints": {
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "metadata": "/metadata",
            "ui": "/"
        }
    }


def main():
    """Entry point for the OpenEnv server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
