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

# create_fastapi_app expects the Environment CLASS, not an instance
app = create_fastapi_app(CloudResourceEnvironment, CloudAction, CloudObservation)
