"""
Minimal DockerRuntime extracted from R2E-Gym.

This is a simplified version that provides core functionality for:
- Docker/Kubernetes container management
- Command execution
- File operations
- SWE-Bench reward calculation

Dependencies:
- docker (for local docker)
- kubernetes (optional, for k8s)
- swebench (for evaluation)
"""

import concurrent.futures
import datetime
import hashlib
import io
import json
import logging
import os
import re
import tarfile
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from rllm.environments.openhands.r2egym.constants import (
    CMD_TIMEOUT,
    DEFAULT_NAMESPACE,
    DOCKER_PATH,
    SKIP_FILES_NEW,
)

def parse_log_pytest(log: str | None) -> dict[str, str]:
    if log is None:
        return {}
    test_status_map: dict[str, str] = {}
    if "short test summary info" not in log:
        return test_status_map
    log = log.split("short test summary info")[-1].strip().split("\n")
    for line in log:
        if "PASSED" in line:
            test_name = ".".join(line.split("::")[1:])
            test_status_map[test_name] = "PASSED"
        elif "FAILED" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
            test_status_map[test_name] = "FAILED"
        elif "ERROR" in line:
            try:
                test_name = ".".join(line.split("::")[1:])
            except IndexError:
                test_name = line
            test_name = test_name.split(" - ")[0]
            test_status_map[test_name] = "ERROR"
    return test_status_map


def parse_log_fn(repo_name: str):
    return parse_log_pytest


def decolor_dict_keys(key: dict[str, str]) -> dict[str, str]:
    decolor = lambda k: re.sub(r"\u001b\[\d+m", "", k)
    return {decolor(k): v for k, v in key.items()}


def get_logger(name: str = "DockerRuntime", level: int = logging.INFO) -> logging.Logger:
    """Get a simple logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class ExecutionEnvironment(ABC):
    """Base runtime class."""
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def close(self):
        pass


class DockerRuntime(ExecutionEnvironment):
    """
    Docker runtime for interacting with docker/kubernetes environments.
    
    Supports:
    - SWE-Bench verified docker images
    - Docker and Kubernetes backends
    """

    def __init__(
        self,
        ds: dict,
        repo_path: str = "/testbed",
        alt_path: str = "/root",
        docker_image: str | None = None,
        command: str = "/bin/bash",
        logger: logging.Logger | None = None,
        backend: str = "docker",
        use_remote: bool = False,
        remote_server_url: str | None = None,
        remote_api_key: str | None = None,
        use_huashan: bool = False,
        huashan_server_url: str | None = None,
        **docker_kwargs,
    ):
        """
        Initialize DockerRuntime.
        
        Args:
            ds: Dataset entry with task information.
            repo_path: Path to repository in container.
            alt_path: Alternative path for scripts.
            docker_image: Docker image to use (inferred from ds if not provided).
            command: Command to run in container.
            logger: Optional logger instance.
            backend: "docker" or "kubernetes".
            use_remote: Whether to use remote Docker server.
            remote_server_url: URL of the remote Docker server (required if use_remote=True).
            remote_api_key: Optional API key for remote server authentication.
            use_huashan: Whether to use Huashan platform via MCP.
            huashan_server_url: URL of the Huashan MCP server (required if use_huashan=True).
        """
        assert ds, f"Dataset not provided for docker image: {docker_image}"
        assert backend in ["docker", "kubernetes"], f"Invalid backend: {backend}"
        
        self.ds = ds
        self.backend = backend
        self.use_remote = use_remote
        self.remote_server_url = remote_server_url
        self.remote_api_key = remote_api_key
        self.use_huashan = use_huashan
        self.huashan_server_url = huashan_server_url
        
        # Get docker image from dataset
        ds_image = ds.get("docker_image") or ds.get("image_name")
        if not ds_image:
            raise ValueError(f"No docker image found in ds: {ds}")
        self.docker_image = docker_image or ds_image
        
        # Check if this is a swebench image
        self.swebench_verified = "swebench" in self.docker_image
        
        # Set runtime params
        self.repo_path = repo_path
        self.alt_path = alt_path
        self.command = command
        self.repo_name = ds.get("repo") or ds.get("repo_name")
        self.docker_kwargs = docker_kwargs
        self.last_test_output: str | None = None
        
        # Setup logger
        if logger is None:
            logger_name = "KubernetesRuntime" if backend == "kubernetes" else "DockerRuntime"
            self.logger = get_logger(logger_name)
        else:
            self.logger = logger
        
        # Initialize client
        self.container = None
        self._init_client()
        
        # Generate container name
        self.container_name = self._get_container_name(self.docker_image)
        if self.backend == "kubernetes":
            self.container_name = str(uuid.uuid4())
        
        # Start container
        self.start_container(self.docker_image, command, self.container_name, **docker_kwargs)
        
        # Setup environment
        self.setup_env()
        self.logger.info(f"Environment initialized - repo: {self.repo_name}, image: {self.docker_image}")

    def _init_client(self):
        """Initialize docker, kubernetes, huashan, or remote docker client."""
        if self.backend == "docker":
            if self.use_huashan:
                # Use Huashan platform via MCP
                if not self.huashan_server_url:
                    raise ValueError(
                        "huashan_server_url is required when use_huashan=True. "
                        "Example: https://10.44.254.18/modelserving/cce-gy1/xxx/mcp"
                    )
                from rllm.environments.openhands.r2egym.huashan_docker_proxy import (
                    HuashanDockerClient,
                )
                self.client = HuashanDockerClient(
                    server_url=self.huashan_server_url,
                    timeout=300,
                )
                self.logger.info(f"Connected to Huashan MCP server: {self.huashan_server_url}")
            elif self.use_remote:
                # Use remote Docker client
                if not self.remote_server_url:
                    raise ValueError(
                        "remote_server_url is required when use_remote=True. "
                        "Example: http://192.168.1.100:8000"
                    )
                from rllm.environments.openhands.r2egym.remote_docker_proxy import (
                    RemoteDockerClient,
                )
                self.client = RemoteDockerClient(
                    server_url=self.remote_server_url,
                    api_key=self.remote_api_key,
                    timeout=120,
                )
                self.logger.info(f"Connected to remote Docker server: {self.remote_server_url}")
            else:
                # Use local Docker client
                try:
                    import docker
                    self.client = docker.from_env(timeout=120)
                except ImportError:
                    raise ImportError("docker package is required. Install with: pip install docker")
        elif self.backend == "kubernetes":
            try:
                from kubernetes import client, config
                try:
                    config.load_incluster_config()
                except Exception:
                    config.load_kube_config()
                self.client = client.CoreV1Api()
            except ImportError:
                raise ImportError("kubernetes package is required. Install with: pip install kubernetes")

    @staticmethod
    def _get_container_name(image_name: str) -> str:
        """Generate unique container name."""
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-").replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def _start_kubernetes_pod(
        self, docker_image: str, command: str, pod_name: str, **docker_kwargs
    ):
        """Start or connect to a Kubernetes pod."""
        from kubernetes import client, watch
        import kubernetes
        
        try:
            self.container = self.client.read_namespaced_pod(
                name=pod_name, namespace=DEFAULT_NAMESPACE, _request_timeout=60,
            )
            self.logger.info(f"Found existing Kubernetes pod: {pod_name}")
            return
        except client.ApiException as e:
            if e.status != 404:
                raise e

        # Create pod
        env_vars = {"PATH": DOCKER_PATH, **docker_kwargs.get("environment", {})}
        env_spec = [{"name": k, "value": str(v)} for k, v in env_vars.items()]
        pod_body = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name},
            "spec": {
                "restartPolicy": "Never",
                "containers": [{
                    "name": pod_name,
                    "image": docker_image,
                    "command": ["/bin/sh", "-c"],
                    "args": [command] if isinstance(command, str) else command,
                    "stdin": True,
                    "tty": True,
                    "env": env_spec,
                    "resources": {"requests": {"cpu": "1", "memory": "1Gi"}},
                }],
                "imagePullSecrets": [{"name": "dockerhub-pro"}],
            },
        }

        # Create with retry
        max_retries = 5
        backoff = 5
        for attempt in range(1, max_retries + 1):
            try:
                pod = self.client.create_namespaced_pod(
                    namespace=DEFAULT_NAMESPACE, body=pod_body, _request_timeout=120,
                )
                break
            except client.ApiException as e:
                if e.status in (409, 429, 500, 503) and attempt < max_retries:
                    self.logger.warning(f"Retrying pod creation ({attempt}/{max_retries})")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                else:
                    raise
        else:
            raise RuntimeError(f"Failed to create pod after {max_retries} retries")

        # Wait for pod to be running
        w = watch.Watch()
        try:
            stream = w.stream(
                self.client.list_namespaced_pod,
                namespace=DEFAULT_NAMESPACE,
                field_selector=f"metadata.name={pod_name}",
                timeout_seconds=1200,
            )
            for event in stream:
                phase = event["object"].status.phase
                if phase == "Running":
                    w.stop()
                    break
                if phase in ["Failed", "Succeeded", "Unknown"]:
                    w.stop()
                    raise RuntimeError(f"Pod entered terminal phase: {phase}")
            self.container = pod
        except Exception as e:
            self.logger.error(f"Error waiting for pod: {e}")
            raise

    def start_container(
        self, docker_image: str, command: str, ctr_name: str, **docker_kwargs
    ):
        """Start or reuse a container."""
        try:
            if self.backend == "docker":
                containers = self.client.containers.list(all=True, filters={"name": ctr_name})
                if containers:
                    self.container = containers[0]
                    if self.container.status != "running":
                        self.container.start()
                else:
                    self.container = self.client.containers.run(
                        docker_image, command, name=ctr_name,
                        detach=True, tty=True, stdin_open=True, **docker_kwargs,
                    )
            elif self.backend == "kubernetes":
                self._start_kubernetes_pod(docker_image, command, ctr_name, **docker_kwargs)
        except Exception as e:
            self.logger.error(f"Container start error: {e}")
            self.stop_container()
            raise

    def _stop_kubernetes_pod(self):
        """Stop Kubernetes pod."""
        import kubernetes
        from kubernetes import watch
        
        try:
            self.client.delete_namespaced_pod(
                name=self.container_name, namespace=DEFAULT_NAMESPACE,
                body=kubernetes.client.V1DeleteOptions(grace_period_seconds=0),
                _request_timeout=60,
            )
        except kubernetes.client.rest.ApiException as e:
            if e.status != 404:
                raise

    def stop_container(self):
        """Stop and remove container."""
        try:
            if self.container:
                if self.backend == "docker":
                    self.container.stop()
                    self.container.remove()
                elif self.backend == "kubernetes":
                    self._stop_kubernetes_pod()
        except Exception as e:
            self.logger.error(f"Container stop error: {e}")

    def setup_env(self):
        """Setup the container environment."""
        if self.swebench_verified:
            self._setup_env_swebench()
        else:
            self._setup_env_r2e()

    def _setup_env_swebench(self):
        """Setup SWE-Bench verified environment."""
        try:
            self.run("chmod +x /run_tests.sh")
            self.alt_path = "/"
            self.run("ln -s /opt/miniconda3/envs/testbed /root/.venv")
            self.run("python -m pip install chardet")
        except Exception as e:
            self.logger.error(f"Error setting up environment: {e}")

    def _setup_env_r2e(self):
        """Setup R2E-Gym environment (non-SWE-Bench)."""
        try:
            if self.repo_name == "sympy" and self.repo_path != "/sympy":
                self.repo_path = "/sympy"

            # Create symlinks for venv and python executables
            self.run(f"ln -s {self.repo_path}/.venv {self.alt_path}/.venv")
            self.run(
                f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python"
            )
            self.run(
                f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3"
            )
            self.run(
                f"find {self.repo_path}/.venv/bin -type f -executable -exec ln -sf {{}} {self.alt_path}/.local/bin/ \\;"
            )

            # Install required packages
            self.run("uv pip install chardet")

            # Clean up caches
            self.run("find . -name '*.pyc' -delete")
            self.run("find . -name '__pycache__' -exec rm -rf {} +")
            tests_root = "/r2e_tests" if self.repo_name != "sympy" else f"{self.repo_path}/r2e_tests"
            self.run(f"find {tests_root} -name '*.pyc' -delete")
            self.run(f"find {tests_root} -name '__pycache__' -exec rm -rf {{}} +")

            # Move hidden files and tests to alt_path, then create symlinks
            for skip_file in SKIP_FILES_NEW:
                self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")
            if self.repo_name != "sympy":
                self.run(f"cp -r /r2e_tests {self.alt_path}/r2e_tests")
            self.run(f"ln -s {self.alt_path}/r2e_tests {self.repo_path}/r2e_tests")
        except Exception as e:
            self.logger.error(f"Error setting up R2E-Gym environment: {e}")

    def get_task_instruction(self) -> str:
        """Get task instruction from dataset."""
        try:
            content = self.ds["problem_statement"]
            match = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL)
            if match:
                return match.group(1)
            return content
        except Exception:
            return self.ds.get("problem_statement", "")

    def _run_kubernetes(
        self, code: str, timeout: int = CMD_TIMEOUT, args: str = "", workdir: str = "",
    ) -> tuple[str, str]:
        """Execute command in Kubernetes pod."""
        from kubernetes import client
        from kubernetes.stream import stream
        
        command = ""
        if workdir:
            command += f"cd {workdir} && "
        command += f"timeout {timeout} {code} {args}"
        full_command = ["/bin/sh", "-c", command]
        
        try:
            def execute_command():
                resp = stream(
                    self.client.connect_get_namespaced_pod_exec,
                    self.container_name, DEFAULT_NAMESPACE,
                    command=full_command,
                    stderr=True, stdin=False, stdout=True, tty=False,
                    _preload_content=False,
                )
                chunks = []
                while resp.is_open():
                    resp.update(timeout=1)
                    if resp.peek_stdout():
                        chunks.append(resp.read_stdout())
                    if resp.peek_stderr():
                        chunks.append(resp.read_stderr())
                resp.close()
                return "".join(chunks), resp.returncode

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(execute_command)
                output, exit_code = future.result(timeout=timeout + 5)

            if exit_code == 124:
                return f"Command timed out (>{timeout}s)", "-1"
            if exit_code != 0:
                return output, f"Error: Exit code {exit_code}"
            
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(exit_code)
            
        except concurrent.futures.TimeoutError:
            return f"Command timed out (>{timeout}s)", "-1"
        except Exception as e:
            return f"Error: {e}", "-1"

    def run(
        self, code: str, timeout: int = CMD_TIMEOUT, args: str = "", workdir: str | None = None,
    ) -> tuple[str, str]:
        """Execute command in container."""
        exec_workdir = workdir or self.repo_path

        if self.backend == "kubernetes":
            return self._run_kubernetes(code, timeout, args, workdir=exec_workdir)

        command = f"timeout {timeout} {code} {args}"
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.container.exec_run,
                    cmd=["/bin/sh", "-c", command],
                    workdir=exec_workdir,
                    stdout=True, stderr=True,
                    environment={"PATH": DOCKER_PATH},
                )
                exec_result = future.result(timeout=timeout + 5)

            output = exec_result.output.decode("utf-8", errors="replace")
            error_code = exec_result.exit_code

            if error_code == 124:
                return f"Command timed out (>{timeout}s)", "-1"
            if error_code != 0:
                return output, f"Error: Exit code {error_code}"
            
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(error_code)

        except concurrent.futures.TimeoutError:
            return f"Command timed out (>{timeout}s)", "-1"
        except Exception as e:
            return f"Error: {e}", "-1"

    def _copy_to_container_kubernetes(self, src_path: str, dest_path: str):
        """Copy file to Kubernetes pod."""
        from kubernetes.stream import stream
        
        dest_dir = os.path.dirname(dest_path)
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(src_path, arcname=os.path.basename(dest_path))
        tar_stream.seek(0)

        exec_command = ["tar", "xmf", "-", "-C", dest_dir]
        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            self.container_name, DEFAULT_NAMESPACE,
            command=exec_command,
            stderr=True, stdin=True, stdout=True, tty=False,
            _preload_content=False,
        )
        resp.write_stdin(tar_stream.read())
        resp.close()

    def copy_to_container(self, src_path: str, dest_path: str):
        """Copy file to container."""
        if self.backend == "docker":
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(src_path, arcname=os.path.basename(dest_path))
            tar_stream.seek(0)
            self.container.put_archive(os.path.dirname(dest_path), tar_stream.read())
        else:
            self._copy_to_container_kubernetes(src_path, dest_path)

    def run_tests(self, timeout: int = 300) -> tuple[str, str]:
        """Run test script."""
        output, error_code = self.run(f"bash {self.alt_path}/run_tests.sh", timeout=timeout)
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        return output, error_code

    def get_patch(self) -> str:
        """Get git diff of current state."""
        output, _ = self.run("git add -A && git diff --cached")
        return output

    def read_file(self, rel_file_path: str) -> str:
        """Read a file inside the container."""
        if rel_file_path.startswith("/"):
            path = rel_file_path
        else:
            path = f"{self.alt_path}/{rel_file_path}"
        output, _ = self.run(f"cat {path}")
        return output

    def apply_patch(self, patch: str) -> tuple[str, str]:
        """Apply a git patch."""
        patch_path = f"/tmp/{self.container_name}_{uuid.uuid4()}.patch"
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(patch)
            temp_path = f.name
        # Ensure non-root container users can read the copied patch file.
        os.chmod(temp_path, 0o644)
        self.copy_to_container(temp_path, patch_path)
        os.unlink(temp_path)
        return self.run(f"git apply --whitespace=fix {patch_path}")

    def _calculate_reward_swebench(self, timeout: int = 300) -> float:
        """Calculate reward using SWE-Bench evaluation."""
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec
            from swebench.harness.log_parsers import MAP_REPO_TO_PARSER, get_eval_type
            from swebench.harness.grading import get_eval_tests_report, get_resolution_status
            from swebench.harness.constants import (
                APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT,
                KEY_INSTANCE_ID, FAIL_TO_PASS, PASS_TO_PASS,
                MAP_REPO_VERSION_TO_SPECS, ResolvedStatus,
            )
        except ImportError:
            raise ImportError(
                "swebench is required for reward calculation. "
                "Install with: pip install swebench"
            )

        test_spec = make_test_spec(self.ds)
        out, _ = self.run("/run_tests.sh", timeout=timeout)
        self.last_test_output = out
        
        # Parse test output
        bad_codes = [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT]
        if any(code in out for code in bad_codes):
            return 0.0

        repo = test_spec.repo
        version = test_spec.version
        log_parser = MAP_REPO_TO_PARSER[repo]
        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]
        
        content = out.split(test_cmd)[-1]
        eval_status_map = log_parser(content, test_spec)
        
        eval_ref = {
            KEY_INSTANCE_ID: test_spec.instance_id,
            FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: test_spec.PASS_TO_PASS,
        }
        report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=get_eval_type(test_spec))
        success = get_resolution_status(report) == ResolvedStatus.FULL.value
        return int(success)

    def _calculate_reward(self, timeout: int = 300) -> float:
        """Calculate reward for current state."""
        if self.swebench_verified:
            return self._calculate_reward_swebench(timeout=timeout)
        return self._calculate_reward_r2e(timeout=timeout)

    def _calculate_reward_r2e(self, timeout: int = 300) -> float:
        output, _ = self.run_tests(timeout=timeout)
        self.last_test_output = output
        parsed = parse_log_fn(self.repo_name)(output)
        parsed = decolor_dict_keys(parsed)
        try:
            expected_json = self.ds["expected_output_json"]
        except Exception:
            expected_json = self.read_file("expected_test_output.json")

        expected: dict = json.loads(expected_json)
        expected = decolor_dict_keys(expected)

        parsed = {k.split(" - ")[0]: parsed[k] for k in sorted(parsed.keys())}
        expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

        if len(parsed) != len(expected):
            return 0.0

        for key in parsed.keys():
            if not key:
                continue
            if key not in expected or parsed[key] != expected[key]:
                return 0.0
        return 1.0

    def reset(self):
        """Reset the environment."""
        self.stop_container()
        self.start_container(self.docker_image, self.command, self.container_name, **self.docker_kwargs)
        self.setup_env()

    def close(self):
        """Close the runtime."""
        self.stop_container()
        if self.backend == "docker":
            self.client.close()

