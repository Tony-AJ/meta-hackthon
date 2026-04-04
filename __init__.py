"""
Cloud Resource Allocation – OpenEnv Environment
================================================

Exports the public API for this environment package:
  - CloudAction       action dataclass
  - CloudObservation  observation dataclass
  - CloudState        state dataclass
  - CloudEnv          EnvClient subclass (for inference)

Note: Requires ``openenv-core`` to be installed. When running tests or
the standalone Gradio demo, this file is not imported.
"""

try:
    from models import CloudAction, CloudObservation, CloudState
    from client import CloudEnv

    __all__ = [
        "CloudAction",
        "CloudObservation",
        "CloudState",
        "CloudEnv",
    ]
except ImportError:
    # Allow the project to be used without openenv installed
    # (e.g. for running tests, standalone Gradio demo, or DQN training)
    __all__ = []
