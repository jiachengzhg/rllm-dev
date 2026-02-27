"""
Execute Bash Tool - OpenHands compatible.

This tool executes bash commands in a persistent shell session.
"""

MAX_RESPONSE_LEN_CHAR = 16000
CLIPPED_NOTICE = (
    "<response clipped><NOTE>Due to the max output limit, only part of the full "
    "response has been shown to you.</NOTE>"
)

# OpenHands-style tool schema (OpenAI function calling format)
EXECUTE_BASH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": """Execute a bash command in the terminal within a persistent shell session.

### Command Execution
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
* Soft timeout: Commands have a soft timeout of 10 seconds, once that's reached, you have the option to continue or interrupt the command.
* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in shell scripts or commands.

### Long-running Commands
* For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.
* For commands that may run for a long time, set the "timeout" parameter to an appropriate value.
* If a bash command returns exit code `-1`, the process hit the soft timeout and is not yet finished.

### Best Practices
* Directory verification: Before creating new directories or files, verify the parent directory exists.
* Directory management: Try to maintain working directory by using absolute paths.

### Output Handling
* Output truncation: If the output exceeds a maximum length, it will be truncated.""",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process."
                },
                "is_input": {
                    "type": "string",
                    "description": "If True, the command is an input to the running process. If False, the command is a bash command to be executed. Default is False.",
                    "enum": ["true", "false"]
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional. Sets a hard timeout in seconds for the command execution."
                }
            },
            "required": ["command"]
        }
    }
}

