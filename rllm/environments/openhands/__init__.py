"""
OpenHands-compatible environment implementation.

This module provides:
- OHEnv: Environment with OpenHands-style tools for SWE-Bench training
- RuntimeClient: Abstract base for docker runtime interaction
- LocalDockerClient: Local docker client (uses extracted r2egym code)
- RemoteDockerClient: Remote docker client via HTTP API

No external r2egym dependency required.
"""

from rllm.environments.openhands.oh_env import OHEnv
from rllm.environments.openhands.runtime_client import (
    RuntimeClient,
    CommandResult,
)
from rllm.environments.openhands.r2egym.remote_docker_proxy import RemoteDockerClient

__all__ = [
    "OHEnv",
    "RuntimeClient",
    "RemoteDockerClient",
    "CommandResult",
]

