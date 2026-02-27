"""
OpenHands-style tool definitions for Code Agent training.

This module provides tool schemas compatible with OpenHands agents,
while using R2E-Gym backend for actual execution.
"""

from rllm.environments.openhands.tools.execute_bash import (
    EXECUTE_BASH_SCHEMA,
)
from rllm.environments.openhands.tools.str_replace_editor import (
    STR_REPLACE_EDITOR_SCHEMA,
)
from rllm.environments.openhands.tools.think import (
    THINK_SCHEMA,
)
from rllm.environments.openhands.tools.finish import (
    FINISH_SCHEMA,
)
from rllm.environments.openhands.tools.task_tracker import (
    TASK_TRACKER_SCHEMA,
)


# All available tool schemas
ALL_TOOL_SCHEMAS = [
    EXECUTE_BASH_SCHEMA,
    STR_REPLACE_EDITOR_SCHEMA,
    THINK_SCHEMA,
    FINISH_SCHEMA,
    TASK_TRACKER_SCHEMA,
]


def get_all_tool_schemas() -> list[dict]:
    """Get all OpenHands tool schemas."""
    return ALL_TOOL_SCHEMAS.copy()


def get_tool_schema(tool_name: str) -> dict | None:
    """Get a specific tool schema by name."""
    schema_map = {
        "execute_bash": EXECUTE_BASH_SCHEMA,
        "str_replace_editor": STR_REPLACE_EDITOR_SCHEMA,
        "think": THINK_SCHEMA,
        "finish": FINISH_SCHEMA,
        "task_tracker": TASK_TRACKER_SCHEMA,
    }
    return schema_map.get(tool_name)


__all__ = [
    "EXECUTE_BASH_SCHEMA",
    "STR_REPLACE_EDITOR_SCHEMA",
    "THINK_SCHEMA",
    "FINISH_SCHEMA",
    "TASK_TRACKER_SCHEMA",
    "ALL_TOOL_SCHEMAS",
    "get_all_tool_schemas",
    "get_tool_schema",
]

