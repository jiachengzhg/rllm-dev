"""
Runtime client for docker environment interaction.

This module provides an abstraction layer for docker runtime interaction,
supporting both local docker and remote docker server.

No external r2egym dependency required - uses the local extracted version.
"""

from dataclasses import dataclass


@dataclass
class CommandResult:
    """Result of a command execution in the runtime."""
    output: str
    exit_code: int
    metadata: dict | None = None

    def is_success(self) -> bool:
        return self.exit_code == 0


class RuntimeClient:
    """
    Runtime client for docker environment interaction.
    
    This client supports:
    - Local docker: Direct local docker interaction
    - Remote docker: Network-based docker interaction via HTTP API
    - Huashan: Huashan platform interaction via MCP
    - Kubernetes: Kubernetes pod interaction
    
    The client abstracts away the complexity of docker/k8s management and
    provides simple interfaces for command execution, file operations, etc.
    
    Example:
        # Local docker
        client = RuntimeClient(backend="docker")
        client.connect(ds_entry)
        result = client.run("echo hello")
        
        # Remote docker
        client = RuntimeClient(
            backend="docker",
            use_remote=True,
            remote_server_url="http://192.168.1.100:8000"
        )
        client.connect(ds_entry)
        result = client.run("echo hello")
        
        # Huashan platform
        client = RuntimeClient(
            backend="docker",
            use_huashan=True,
            huashan_server_url="https://xxx/mcp"
        )
        client.connect(ds_entry)
        result = client.run("echo hello")
    """

    def __init__(
        self,
        backend: str = "docker",
        step_timeout: int = 90,
        reward_timeout: int = 300,
        verbose: bool = True,
        use_remote: bool = False,
        remote_server_url: str | None = None,
        remote_api_key: str | None = None,
        use_huashan: bool = False,
        huashan_server_url: str | None = None,
    ):
        """
        Initialize the runtime client.
        
        Args:
            backend: Backend type, "docker" or "kubernetes".
            step_timeout: Default timeout for command execution (seconds).
            reward_timeout: Default timeout for reward computation (seconds).
            verbose: Whether to print verbose logs.
            use_remote: Whether to use remote Docker server.
            remote_server_url: URL of the remote Docker server.
            remote_api_key: Optional API key for remote server authentication.
            use_huashan: Whether to use Huashan platform via MCP.
            huashan_server_url: URL of the Huashan MCP server.
        """
        self.backend = backend
        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.verbose = verbose
        self.use_remote = use_remote
        self.remote_server_url = remote_server_url
        self.remote_api_key = remote_api_key
        self.use_huashan = use_huashan
        self.huashan_server_url = huashan_server_url
        self.runtime = None
        self._connected = False

    def connect(self, ds: dict, **kwargs) -> None:
        """
        Connect to the runtime environment.
        
        Args:
            ds: Dataset entry containing task information (docker_image, etc.)
            **kwargs: Additional arguments passed to DockerRuntime.
        """
        # Use locally extracted DockerRuntime (no external r2egym needed)
        from rllm.environments.openhands.r2egym import DockerRuntime

        self.ds = ds
        self.runtime = DockerRuntime(
            ds=ds,
            backend=self.backend,
            use_remote=self.use_remote,
            remote_server_url=self.remote_server_url,
            remote_api_key=self.remote_api_key,
            use_huashan=self.use_huashan,
            huashan_server_url=self.huashan_server_url,
            **kwargs,
        )
        self._connected = True

    def run(self, command: str, timeout: int | None = None) -> CommandResult:
        """
        Run a command in the docker container.
        
        Args:
            command: Command to execute.
            timeout: Optional timeout override (seconds).
            
        Returns:
            CommandResult with output and exit code.
        """
        if not self._connected or self.runtime is None:
            raise RuntimeError("Runtime not connected. Call connect() first.")

        if timeout is None:
            timeout = self.step_timeout

        output, error_code = self.runtime.run(command, timeout=timeout)
        
        # Parse exit code
        try:
            if isinstance(error_code, str):
                if error_code.isdigit():
                    exit_code = int(error_code)
                elif "Error" in error_code:
                    exit_code = -1
                else:
                    exit_code = 0
            else:
                exit_code = int(error_code)
        except (ValueError, AttributeError):
            exit_code = -1

        return CommandResult(output=output, exit_code=exit_code)

    def copy_to_container(self, src_path: str, dest_path: str) -> None:
        """
        Copy a file to the container.
        
        Args:
            src_path: Local source file path.
            dest_path: Destination path in the container.
        """
        if not self._connected or self.runtime is None:
            raise RuntimeError("Runtime not connected. Call connect() first.")
        self.runtime.copy_to_container(src_path, dest_path)

    def get_task_instruction(self) -> str:
        """Get the task instruction from the environment."""
        if not self._connected or self.runtime is None:
            raise RuntimeError("Runtime not connected. Call connect() first.")
        return self.runtime.get_task_instruction()

    def compute_reward(self, timeout: int | None = None) -> float:
        """
        Compute the reward for the current state.
        
        Args:
            timeout: Optional timeout override (seconds).
            
        Returns:
            Reward value (typically 0.0 or 1.0).
        """
        if not self._connected or self.runtime is None:
            raise RuntimeError("Runtime not connected. Call connect() first.")
        if timeout is None:
            timeout = self.reward_timeout
        return self.runtime._calculate_reward(timeout=timeout)

    def get_last_test_output(self) -> str | None:
        """Get the last reward test output from runtime."""
        if not self._connected or self.runtime is None:
            raise RuntimeError("Runtime not connected. Call connect() first.")
        return getattr(self.runtime, "last_test_output", None)

    def close(self) -> None:
        """Close the runtime connection."""
        if self.runtime is not None:
            self.runtime.close()
        self._connected = False

    def reset(self) -> None:
        """Reset the runtime environment."""
        if self.runtime is not None:
            self.runtime.reset()

    def get_patch(self) -> str:
        """Get the current git diff patch."""
        if not self._connected or self.runtime is None:
            raise RuntimeError("Runtime not connected. Call connect() first.")
        return self.runtime.get_patch()

