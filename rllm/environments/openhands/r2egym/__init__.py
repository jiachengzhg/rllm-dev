"""
Minimal R2E-Gym core functionality extracted for OHEnv.

This module contains the essential components from R2E-Gym needed for
docker/kubernetes runtime interaction, without the full R2E-Gym dependencies.

Key components:
- DockerRuntime: Docker/Kubernetes container management
- RemoteDockerClient: Remote docker client proxy
- HuashanDockerClient: Huashan MCP docker client proxy
- Constants: Common configuration values
"""

from rllm.environments.openhands.r2egym.docker_runtime import DockerRuntime
from rllm.environments.openhands.r2egym.constants import (
    CMD_TIMEOUT,
    DOCKER_PATH,
    DEFAULT_NAMESPACE,
)
from rllm.environments.openhands.r2egym.remote_docker_proxy import (
    RemoteDockerClient,
    RemoteContainer,
    from_remote,
)
from rllm.environments.openhands.r2egym.huashan_docker_proxy import (
    HuashanDockerClient,
    HuashanContainer,
    from_huashan,
)

__all__ = [
    "DockerRuntime",
    "RemoteDockerClient",
    "RemoteContainer",
    "from_remote",
    "HuashanDockerClient",
    "HuashanContainer",
    "from_huashan",
    "CMD_TIMEOUT",
    "DOCKER_PATH",
    "DEFAULT_NAMESPACE",
]

