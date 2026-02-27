"""
OpenHands-compatible Environment for Code Agent Training.

This environment provides OpenHands-style tool interfaces for SWE-Bench training.
No external r2egym dependency required - uses locally extracted runtime code.

Dependencies:
- docker (for local docker)
- kubernetes (optional, for k8s)
- swebench (for evaluation)
- datasets (for loading R2E-Gym dataset)
"""

import json
import os
import re
import tempfile
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset

from rllm.environments.base.base_env import BaseEnv
from rllm.environments.openhands.runtime_client import RuntimeClient

# Max response length for tool outputs
MAX_EDITOR_RESPONSE_LEN_CHAR = 16000
MAX_BASH_RESPONSE_LEN_CHAR = 30000
PARAM_NAMES_THAT_SHOULD_BE_JSON = ["task_list",]
CLIPPED_NOTICE = (
    "<response clipped><NOTE>Due to the max output limit, only part of the full "
    "response has been shown to you.</NOTE>"
)

# Default R2E-Gym dataset
DEFAULT_R2E_ENV_ID = "R2E-Gym/R2E-Gym-Lite"

# OpenHands tool names
OPENHANDS_TOOLS = [
    "execute_bash",
    "str_replace_editor",
    "think",
    "finish",
    "task_tracker",
]


class OHEnv(BaseEnv):
    """
    OpenHands-compatible Environment for SWE tasks.
    
    This environment provides:
    - OpenHands-style tool interfaces (execute_bash, str_replace_editor, etc.)
    - Docker/Kubernetes backend for runtime
    - Extensible runtime client for remote execution
    
    No external r2egym package required - uses locally extracted code.
    
    Attributes:
        entry: Dataset entry containing task information.
        runtime_client: Client for docker/kubernetes runtime interaction.
        backend: Backend type ("docker" or "kubernetes").
    """

    def __init__(
        self,
        entry: dict | None = None,
        dataset: Dataset | None = None,
        idx: int | None = None,
        step_timeout: int = 90,
        reward_timeout: int = 300,
        backend: str = "docker",
        delete_image: bool = False,
        verbose: bool = True,
        runtime_client: RuntimeClient | None = None,
        use_remote: bool = False,
        remote_server_url: str | None = None,
        remote_api_key: str | None = None,
        use_huashan: bool = False,
        huashan_server_url: str | None = None,
    ):
        """
        Initialize the OpenHands environment.

        Args:
            entry: Dataset entry. If None, loads from dataset.
            dataset: Dataset to use. If None, uses default R2E-Gym dataset.
            idx: Index in dataset. If None, randomly selects.
            step_timeout: Timeout for each step in seconds.
            reward_timeout: Timeout for reward computation in seconds.
            backend: Backend type ("docker" or "kubernetes").
            delete_image: Whether to delete docker image after closing.
            verbose: Whether to print verbose logs.
            runtime_client: Optional custom runtime client for remote execution.
            use_remote: Whether to use remote Docker server.
            remote_server_url: URL of the remote Docker server (e.g., http://192.168.1.100:8000).
            remote_api_key: Optional API key for remote server authentication.
            use_huashan: Whether to use Huashan platform via MCP.
            huashan_server_url: URL of the Huashan MCP server.
        """
        # Load entry from dataset if not provided
        if entry is not None:
            self.entry = entry
            self.dataset = None
            self.idx = None
        else:
            if dataset is None:
                dataset = load_dataset(DEFAULT_R2E_ENV_ID, split="test")
            self.dataset = dataset

            if idx is None:
                idx = np.random.randint(0, len(self.dataset))
            assert 0 <= idx < len(self.dataset), "Selected index out of range"
            self.idx = idx
            self.entry = self.dataset[idx]

        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.backend = backend
        self.delete_image = delete_image
        self.verbose = verbose
        self.use_remote = use_remote
        self.remote_server_url = remote_server_url
        self.remote_api_key = remote_api_key
        self.use_huashan = use_huashan
        self.huashan_server_url = huashan_server_url
        self.total_steps = 0
        self.done = False
        self.last_test_output: str | None = None
        self._task_list: list[dict[str, Any]] = []

        # Initialize runtime client
        if runtime_client is not None:
            self.runtime_client = runtime_client
        else:
            self.runtime_client = RuntimeClient(
                backend=backend,
                step_timeout=step_timeout,
                reward_timeout=reward_timeout,
                verbose=verbose,
                use_remote=use_remote,
                remote_server_url=remote_server_url,
                remote_api_key=remote_api_key,
                use_huashan=use_huashan,
                huashan_server_url=huashan_server_url,
            )
        
        self._env_initialized = False

    def _init_env(self) -> None:
        """Initialize the docker environment."""
        if self._env_initialized:
            return
        
        # Connect runtime client
        self.runtime_client.connect(self.entry)
        
        # Setup OpenHands-style tools in the container
        self._setup_tools()
        
        self._env_initialized = True

    def _setup_tools(self) -> None:
        """
        Setup OpenHands-style file editor tool in the container.
        
        Creates a simple str_replace_editor script in the container.
        """
        # Create str_replace_editor script content
        str_replace_editor_script = '''#!/usr/bin/env python3
"""
Simple str_replace_editor implementation for OpenHands-style file editing.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

UNDO_STATE_PATH = "/tmp/str_replace_editor_undo.json"

def _load_undo_state():
    if not os.path.exists(UNDO_STATE_PATH):
        return {}
    try:
        with open(UNDO_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_undo_state(state):
    with open(UNDO_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)

def _record_undo(path):
    state = _load_undo_state()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        state[path] = {"existed": True, "content": content}
    else:
        state[path] = {"existed": False, "content": ""}
    _save_undo_state(state)

def _is_hidden(name):
    return name.startswith(".")

def _format_rel(rel_root, name):
    if rel_root == ".":
        return name
    return f"{rel_root}/{name}"

def _content_from_lines(lines):
    return "\\n".join(line.rstrip("\\n") for line in lines)

def _make_output(snippet_content, snippet_description, start_line=1):
    numbered = "\\n".join(
        [
            f"{i + start_line:6}\\t{line}"
            for i, line in enumerate(snippet_content.split("\\n"))
        ]
    )
    return (
        f"Here's the result of running `cat -n` on {snippet_description}:\\n"
        + numbered
        + "\\n"
    )

def view_dir(path, max_depth=2):
    """List non-hidden files/dirs up to max_depth using find (OpenHands style)."""
    if not os.path.isdir(path):
        return (
            f"Invalid `path` parameter: {path}. The path {path} does not exist. "
            "Please provide a valid path."
        )

    qpath = shlex.quote(path)

    hidden_cmd = f"find -L {qpath} -mindepth 1 -maxdepth 1 -name '.*'"
    hidden_proc = subprocess.run(
        hidden_cmd, shell=True, text=True, capture_output=True, check=False
    )
    hidden_stdout = hidden_proc.stdout
    hidden_count = (
        len(hidden_stdout.strip().split("\\n")) if hidden_stdout.strip() else 0
    )

    hidden_path_depth1 = shlex.quote(f"{path}/\\.*")
    hidden_path_depth2 = shlex.quote(f"{path}/*/\\.*")
    list_cmd = (
        f"find -L {qpath} -maxdepth {max_depth} -not \\( "
        f"-path {hidden_path_depth1} -o -path {hidden_path_depth2} "
        f"\\) | sort"
    )
    list_proc = subprocess.run(
        list_cmd, shell=True, text=True, capture_output=True, check=False
    )
    if list_proc.stderr.strip():
        return list_proc.stderr.strip()

    paths = list_proc.stdout.strip().split("\\n") if list_proc.stdout.strip() else []
    formatted_paths = []
    for p in paths:
        if os.path.isdir(p):
            formatted_paths.append(f"{p}/")
        else:
            formatted_paths.append(p)

    msg = (
        f"Here's the files and directories up to 2 levels deep in {path}, "
        "excluding hidden items:\\n" + "\\n".join(formatted_paths)
    )
    if hidden_count > 0:
        msg += (
            f"\\n\\n{hidden_count} hidden files/directories in this directory are "
            f"excluded. You can use 'ls -la {path}' to see them."
        )
    return msg

def view_file(path, view_range=None):
    """View file content with optional line range."""
    if os.path.isdir(path):
        if view_range:
            return (
                f"Invalid `view_range` parameter: {view_range}. "
                "The `view_range` parameter is not allowed when `path` points to "
                "a directory."
            )
        return view_dir(path, max_depth=2)
    if not os.path.exists(path):
        return (
            f"Invalid `path` parameter: {path}. The path {path} does not exist. "
            "Please provide a valid path."
        )
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    warning_message = None
    if view_range:
        try:
            parsed = json.loads(view_range)
            if not isinstance(parsed, list) or len(parsed) != 2:
                return (
                    f"Invalid `view_range` parameter: {view_range}. "
                    "It should be a list of two integers."
                )
            start_line, end_line = parsed
            if not isinstance(start_line, int) or not isinstance(end_line, int):
                return (
                    f"Invalid `view_range` parameter: {view_range}. "
                    "It should be a list of two integers."
                )
        except Exception:
            return (
                f"Invalid `view_range` parameter: {view_range}. "
                "It should be a list of two integers."
            )

        num_lines = len(lines)
        if start_line < 1 or start_line > num_lines:
            return (
                f"Invalid `view_range` parameter: {view_range}. "
                f"Its first element `{start_line}` should be within the range of "
                f"lines of the file: {[1, num_lines]}."
            )

        if end_line == -1:
            end_line = num_lines
        elif end_line > num_lines:
            warning_message = (
                f"We only show up to {num_lines} since there're only {num_lines} "
                "lines in this file."
            )
            end_line = num_lines

        if end_line < start_line:
            return (
                f"Invalid `view_range` parameter: {view_range}. "
                f"Its second element `{end_line}` should be greater than or equal "
                f"to the first element `{start_line}`."
            )

        # Convert to python slice range.
        line_offset = start_line - 1
        lines = lines[line_offset:end_line]
    else:
        line_offset = 0
    
    output_text = _make_output(_content_from_lines(lines), path, line_offset + 1)
    if warning_message:
        output_text = f"NOTE: {warning_message}\\n{output_text}"
    return output_text

def create_file(path, content):
    """Create a new file with content."""
    if os.path.exists(path):
        return (
            f"Invalid `path` parameter: {path}. "
            f"File already exists at: {path}. Cannot overwrite files using command "
            "`create`."
        )
    _record_undo(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"File created successfully at: {path}"

def str_replace(path, old_str, new_str):
    """Replace string in file."""
    if not os.path.exists(path):
        return (
            f"Invalid `path` parameter: {path}. The path {path} does not exist. "
            "Please provide a valid path."
        )
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    if old_str not in content:
        return (
            f"No replacement was performed, old_str `{old_str}` did not appear "
            f"verbatim in {path}."
        )
    
    count = content.count(old_str)
    if count > 1:
        line_numbers = []
        for i, line in enumerate(content.splitlines(), 1):
            if old_str in line:
                line_numbers.append(i)
        return (
            "No replacement was performed. Multiple occurrences of old_str "
            f"`{old_str}` in lines {line_numbers}. Please ensure it is unique."
        )
    
    _record_undo(path)
    replace_idx = content.find(old_str)
    replacement_line = content[:replace_idx].count("\\n") + 1
    new_content = content.replace(old_str, new_str, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    new_lines = new_content.splitlines()
    start_line = max(1, replacement_line - 4)
    end_line = min(
        len(new_lines), replacement_line + 4 + new_str.count("\\n")
    )
    snippet = "\\n".join(new_lines[start_line - 1:end_line])
    success_message = f"The file {path} has been edited. "
    success_message += _make_output(snippet, f"a snippet of {path}", start_line)
    success_message += (
        "Review the changes and make sure they are as expected. Edit the file "
        "again if necessary."
    )
    return success_message

def insert_line(path, line_num, new_str):
    """Insert content at specific line."""
    if not os.path.exists(path):
        return (
            f"Invalid `path` parameter: {path}. The path {path} does not exist. "
            "Please provide a valid path."
        )
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    line_num = max(0, min(len(lines), int(line_num)))
    lines.insert(line_num, new_str + '\\n')
    
    _record_undo(path)
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    new_str_lines = new_str.split("\\n")
    start_line = max(1, line_num - 4 + 1)
    end_line = min(
        len(lines), line_num + 4 + len(new_str_lines)
    )
    snippet = _content_from_lines(lines[start_line - 1:end_line])
    success_message = f"The file {path} has been edited. "
    success_message += _make_output(
        snippet,
        "a snippet of the edited file",
        start_line,
    )
    success_message += (
        "Review the changes and make sure they are as expected (correct "
        "indentation, no duplicate lines, etc). Edit the file again if necessary."
    )
    return success_message

def undo_edit(path):
    """Revert the last edit made to the file at path."""
    state = _load_undo_state()
    if path not in state:
        return f"No edit history found for {path}."
    info = state.pop(path)
    if info.get("existed"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(info.get("content", ""))
    else:
        if os.path.exists(path):
            os.remove(path)
    _save_undo_state(state)
    if info.get("existed"):
        old_text = info.get("content", "")
        output = (
            f"Last edit to {path} undone successfully. "
            + _make_output("\\n".join(old_text.splitlines()), path)
        )
        return output
    return "Last edit undone successfully. File removed."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['view', 'create', 'str_replace', 'insert', 'undo_edit'])
    parser.add_argument('--path', required=True)
    parser.add_argument('--view_range', default=None)
    parser.add_argument('--file_text', default='')
    parser.add_argument('--old_str', default='')
    parser.add_argument('--new_str', default='')
    parser.add_argument('--insert_line', type=int, default=0)
    
    args = parser.parse_args()
    
    if args.command == 'view':
        print(view_file(args.path, args.view_range))
    elif args.command == 'create':
        print(create_file(args.path, args.file_text))
    elif args.command == 'str_replace':
        print(str_replace(args.path, args.old_str, args.new_str))
    elif args.command == 'insert':
        print(insert_line(args.path, args.insert_line, args.new_str))
    elif args.command == 'undo_edit':
        print(undo_edit(args.path))

if __name__ == '__main__':
    main()
'''
        
        # Write script to temp file and copy to container
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(str_replace_editor_script)
            temp_path = f.name
        
        try:
            # self.runtime_client.copy_to_container(temp_path, "/usr/local/bin/str_replace_editor")
            # self.runtime_client.run("chmod +x /usr/local/bin/str_replace_editor")
            self.runtime_client.copy_to_container(temp_path, "/root/str_replace_editor")
            self.runtime_client.run("chmod +x /root/str_replace_editor")
        except Exception as e:
            print(f"⚠️ Failed to install str_replace_editor in runtime: {e}")
        finally:
            os.unlink(temp_path)

    def reset(self) -> tuple[str, dict]:
        """
        Reset the environment to initial state.

        Returns:
            Tuple containing task instruction and additional info.
        """
        if self._env_initialized:
            self.runtime_client.reset()
            self._setup_tools()
        else:
            self._init_env()

        self.total_steps = 0
        self.done = False
        self.last_test_output = None
        self._task_list = []

        # Get task instruction
        task_instruction = self.runtime_client.get_task_instruction()

        return task_instruction, {}

    def step(self, action: str | dict) -> tuple[str, float, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Action to execute. Can be:
                - str: Raw action string (for XML/text format)
                - dict: Structured action with 'name' and 'arguments'

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if not self._env_initialized:
            self._init_env()

        self.total_steps += 1

        # Parse action
        tool_name, tool_args = self._parse_action(action)

        if not tool_name:
            return "No valid action found.", 0.0, False, {}

        # Check for finish/submit action
        if tool_name.lower() in ["finish", "submit"]:
            obs = self._execute_finish(tool_args)
            reward = 0.0  # Final reward computed separately
            return obs, reward, self.done, {"action": tool_name}

        # Execute action based on tool type
        obs = self._execute_tool(tool_name, tool_args)

        # Reward is computed at the end
        reward = 0.0

        return obs, reward, self.done, {"action": tool_name, "arguments": tool_args}

    def _parse_action(self, action: str | dict) -> tuple[str, dict]:
        """
        Parse action into tool name and arguments.

        Supports multiple formats:
        - Dict with 'name' and 'arguments'
        - OpenAI function call format
        - XML format: <function=name><parameter=key>value</parameter></function>
        """
        if isinstance(action, dict):
            # Direct dict format
            if "name" in action:
                return action["name"], action.get("arguments", {})
            # OpenAI function call format
            if "function" in action:
                fn = action["function"]
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                return fn.get("name", ""), args
            return "", {}

        if isinstance(action, str):
            return self._parse_xml_action(action)

        return "", {}

    def _parse_xml_action(self, action_str: str) -> tuple[str, dict]:
        """
        Parse XML format action string.

        Format: <function=name><parameter=key>value</parameter></function>
        """
        def _parse_xml_param_value_json(raw_value: str) -> Any:
            """Best-effort parse for XML parameter values.

            Keep backward compatibility by returning the original string when
            parsing is uncertain or fails.
            """
            value = raw_value.strip()
            if not value:
                return value

            # Parse likely JSON payloads (arrays/objects), e.g. task_list/view_range.
            if value[0] in "[{":
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value

            # Parse canonical JSON literals.
            if value.lower() in {"true", "false", "null"}:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value

            return value

        # Match function name
        fn_match = re.search(r"<function=(\w+)>", action_str)
        if not fn_match:
            return "", {}

        tool_name = fn_match.group(1)

        # Extract parameters
        args = {}
        param_pattern = r"<parameter=(\w+)>(.*?)</parameter>"
        for match in re.finditer(param_pattern, action_str, re.DOTALL):
            param_name = match.group(1)

            if param_name in PARAM_NAMES_THAT_SHOULD_BE_JSON:
                param_value = _parse_xml_param_value_json(match.group(2))
            else:
                param_value = match.group(2).strip()

            args[param_name] = param_value

        return tool_name, args

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool and return the observation."""
        
        # Map OpenHands tool names to executors
        tool_mapping = {
            "execute_bash": self._execute_bash,
            "str_replace_editor": self._execute_str_replace_editor,
            "think": self._execute_think,
            "task_tracker": self._execute_task_tracker,
            "finish": self._execute_finish,
            "submit": self._execute_finish,
        }

        executor = tool_mapping.get(tool_name.lower())
        if executor:
            return executor(tool_args)
        else:
            return f"Unknown tool: {tool_name}"

    def _execute_bash(self, args: dict) -> str:
        """Execute bash command."""
        command = args.get("command", "")
        timeout = args.get("timeout", self.step_timeout)
        is_input = False if args.get("is_input", 'false').lower() == 'false' else True

        if not command:
            return "ERROR: No previous running command to retrieve logs from."

        # Handle special control sequences
        if is_input and command in ["C-c", "C-d", "C-z"]:
            command = f"echo '{command}' # Control sequence not directly supported"

        result = self.runtime_client.run(command, timeout=int(timeout))
        
        output = result.output
        output += f"\n[Command finished with exit code {result.exit_code}]"
        return self._clip_response(output, MAX_BASH_RESPONSE_LEN_CHAR)

    def _execute_str_replace_editor(self, args: dict) -> str:
        """Execute str_replace_editor command."""
        command = args.get("command", "view")
        path = args.get("path", "")
        
        if not path:
            return "Error: path is required."

        # Build the command based on operation type
        if command == "view":
            view_range = args.get("view_range")
            if view_range:
                cmd = f"/root/str_replace_editor {command} --path '{path}' --view_range '{view_range}'"
            else:
                cmd = f"/root/str_replace_editor {command} --path '{path}'"
        elif command == "create":
            file_text = args.get("file_text", "")
            # Escape special characters for shell
            escaped_text = file_text.replace("'", "'\"'\"'")
            cmd = f"/root/str_replace_editor {command} --path '{path}' --file_text '{escaped_text}'"
        elif command == "str_replace":
            old_str = args.get("old_str", "").replace("'", "'\"'\"'")
            new_str = args.get("new_str", "").replace("'", "'\"'\"'")
            cmd = f"/root/str_replace_editor {command} --path '{path}' --old_str '{old_str}' --new_str '{new_str}'"
        elif command == "insert":
            insert_line = args.get("insert_line", 0)
            new_str = args.get("new_str", "").replace("'", "'\"'\"'")
            cmd = f"/root/str_replace_editor {command} --path '{path}' --insert_line {insert_line} --new_str '{new_str}'"
        elif command == "undo_edit":
            cmd = f"/root/str_replace_editor {command} --path '{path}'"
        else:
            return f"Unknown command: {command}"

        result = self.runtime_client.run(cmd, timeout=self.step_timeout)
        return self._clip_response(result.output, MAX_EDITOR_RESPONSE_LEN_CHAR)

    def _execute_think(self, args: dict) -> str:
        """Execute think tool (no-op, just logs thought)."""
        thought = args.get("thought", "")
        return "Your thought has been logged."

    def _execute_task_tracker(self, args: dict) -> str:
        """Execute task tracker tool."""
        command = args.get("command", "view")
        task_list = args.get("task_list", [])

        if command == "view":
            if not self._task_list:
                return 'No task list found. Use the "plan" command to create one.'
            return self._format_task_list(self._task_list)
        elif command == "plan":
            if not task_list:
                return "No tasks provided."
            normalized_tasks: list[dict[str, str]] = []
            for i, task in enumerate(task_list, 1):
                if not isinstance(task, dict):
                    return (
                        f"Invalid task item at index {i}. "
                        "Each task should be an object."
                    )
                missing = [k for k in ("id", "title", "status") if k not in task]
                if missing:
                    return (
                        f"Invalid task item at index {i}. Missing required fields: "
                        f"{missing}"
                    )
                status = task.get("status")
                if status not in {"todo", "in_progress", "done"}:
                    return (
                        f"Invalid task status at index {i}: {status}. "
                        "Allowed values: ['todo', 'in_progress', 'done']"
                    )
                normalized_tasks.append(
                    {
                        "id": str(task.get("id", "")),
                        "title": str(task.get("title", "")),
                        "status": str(status),
                        "notes": str(task.get("notes", "")),
                    }
                )
            self._task_list = normalized_tasks
            output = f"Task list has been updated with {len(self._task_list)} item(s).\n"
            output += self._format_task_list(self._task_list)
            return output
        else:
            return (
                f"Unknown command: {command}. Supported commands are "
                '"view" and "plan".'
            )

    def _execute_finish(self, args: dict) -> str:
        """Execute finish tool."""
        if "message" not in args:
            return 'Missing required argument "message" in tool call finish'
        message = args.get("message", "")
        self.done = True
        return str(message)

    @staticmethod
    def _format_task_list(task_list: list[dict[str, Any]]) -> str:
        """Format task list for display."""
        if not task_list:
            return "No tasks in the list."

        output = "# Task List\n\n"
        for i, task in enumerate(task_list, 1):
            status = task.get("status", "todo")
            status_icon = {"todo": "⏳", "in_progress": "🔄", "done": "✅"}.get(
                status, "⏳"
            )
            title = task.get("title", "Untitled")
            notes = task.get("notes", "")
            output += f"{i}. {status_icon} {title}\n"
            if notes:
                output += f"   {notes}\n"
            output += "\n"
        return output.strip()

    @staticmethod
    def _clip_response(output: str, max_len: int = MAX_EDITOR_RESPONSE_LEN_CHAR) -> str:
        """Clip overly long tool outputs."""
        if output is None:
            return ""
        if len(output) <= max_len:
            return output
        return f"{output[:max_len]}\n{CLIPPED_NOTICE}"

    def compute_final_reward(self, timeout: int | None = None) -> float:
        """Compute the final reward for the task."""
        if timeout is None:
            timeout = self.reward_timeout
        reward = self.runtime_client.compute_reward(timeout=timeout)
        self.last_test_output = self.runtime_client.get_last_test_output()
        return reward

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self._env_initialized:
            self.runtime_client.close()
            self._env_initialized = False

    def get_patch(self) -> str:
        """Get the current git diff patch."""
        if not self._env_initialized:
            return ""
        return self.runtime_client.get_patch()

    @staticmethod
    def from_dict(extra_info: dict | str) -> "OHEnv":
        """
        Create an environment instance from dictionary configuration.

        Args:
            extra_info: Dictionary containing configuration parameters.

        Returns:
            Initialized OHEnv instance
        """
        import inspect

        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        sig = inspect.signature(OHEnv.__init__)
        init_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in extra_info:
                init_params[param_name] = extra_info[param_name]
        
        init_params["entry"] = extra_info
        return OHEnv(**init_params)

    @staticmethod
    def is_multithread_safe() -> bool:
        """Return whether the environment is multithread safe."""
        return True

    def get_tool_schemas(self) -> list[dict]:
        """
        Get OpenHands-style tool schemas for LLM function calling.
        
        Returns:
            List of tool schemas in OpenAI function calling format.
        """
        from rllm.environments.openhands.tools import get_all_tool_schemas
        return get_all_tool_schemas()
