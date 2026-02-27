"""
Constants for R2E-Gym runtime.
"""

# Timeout for bash commands (in seconds)
CMD_TIMEOUT = 120

# Default namespace for Kubernetes
DEFAULT_NAMESPACE = "default"

# PATH environment variable for docker containers
DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Hidden / excluded files
SKIP_FILES = [
    "run_tests.sh",
    "syn_issue.json",
    "expected_test_output.json",
    "execution_result.json",
    "parsed_commit.json",
    "modified_files.json",
    "modified_entities.json",
    "r2e_tests",
]

SKIP_FILES_NEW = [
    "run_tests.sh",
    "r2e_tests",
]

