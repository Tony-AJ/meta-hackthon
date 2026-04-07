"""
client.py
=========
EnvClient subclass for the Cloud Resource Allocation environment.

Used by ``inference.py`` to connect to the environment server
(either via Docker or a remote HF Space URL).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import CloudAction, CloudObservation, CloudState


class CloudEnv(EnvClient[CloudAction, CloudObservation, CloudState]):
    """
    Client for the Cloud Resource Allocation OpenEnv environment.

    Usage (async):
        async with CloudEnv(base_url="http://localhost:7860") as client:
            result = await client.reset()
            result = await client.step(CloudAction(action_id=3))

    Usage (sync via .sync()):
        with CloudEnv(base_url="http://localhost:7860").sync() as client:
            result = client.reset()

    Usage (from Docker):
        env = await CloudEnv.from_docker_image("cloud-resource-env:latest")
    """

    def _step_payload(self, action: CloudAction) -> dict:
        """Serialise an action into the JSON payload sent to the server."""
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[CloudObservation]:
        """Parse the server response into a typed StepResult."""
        obs = CloudObservation(**payload)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> CloudState:
        """Parse the server state response into a typed CloudState."""
        return CloudState(**payload)
