"""
String Replace Editor Tool - OpenHands compatible.

Custom editing tool for viewing, creating and editing files.
"""

MAX_RESPONSE_LEN_CHAR = 16000
CLIPPED_NOTICE = (
    "<response clipped><NOTE>Due to the max output limit, only part of the full "
    "response has been shown to you.</NOTE>"
)

# OpenHands-style tool schema
STR_REPLACE_EDITOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a text file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Before using this tool:
1. Use the view tool to understand the file's contents and context
2. Verify the directory path is correct when creating new files

When making edits:
- Ensure the edit results in idiomatic, correct code
- Do not leave the code in a broken state
- Always use absolute file paths (starting with /)

CRITICAL REQUIREMENTS:
1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation.
2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file.
3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`.""",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`."
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`."
                },
                "file_text": {
                    "type": "string",
                    "description": "Required parameter of `create` command, with the content of the file to be created."
                },
                "old_str": {
                    "type": "string",
                    "description": "Required parameter of `str_replace` command containing the string in `path` to replace."
                },
                "new_str": {
                    "type": "string",
                    "description": "Optional parameter of `str_replace` command containing the new string. Required parameter of `insert` command containing the string to insert."
                },
                "insert_line": {
                    "type": "integer",
                    "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`."
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional parameter of `view` command when `path` points to a file. If provided, shows the indicated line number range, e.g. [11, 12] shows lines 11 and 12."
                }
            },
            "required": ["command", "path"]
        }
    }
}