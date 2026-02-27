"""
Task Tracker Tool - OpenHands compatible.

This tool provides structured task management capabilities.
"""

# OpenHands-style tool schema
TASK_TRACKER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "task_tracker",
        "description": """This tool provides structured task management capabilities for development workflows.
It enables systematic tracking of work items, progress monitoring, and efficient
organization of complex development activities.

## Application Guidelines

Utilize this tool in the following situations:

1. Multi-phase development work - When projects involve multiple sequential or parallel activities
2. Complex implementation tasks - Work requiring systematic planning and coordination
3. Explicit user request for task organization
4. Multiple concurrent requirements - When users present several work items that need coordination
5. Project initiation - Capture and organize user requirements at project start
6. Work commencement - Update task status to in_progress before beginning implementation
7. Task completion - Update status to done and identify any additional work

## Situations Where Tool Usage Is Unnecessary

Avoid using this tool when:
1. Single atomic tasks that require no decomposition
2. Trivial operations where tracking adds no organizational value
3. Simple activities completable in minimal steps
4. Pure information exchange or discussion

## Status Management

1. **Status Values**: Track work using these states:
   - todo: Not yet initiated
   - in_progress: Currently active (maintain single focus)
   - done: Successfully completed

2. **Workflow Practices**:
   - Update status dynamically as work progresses
   - Mark completion immediately upon task finish
   - Limit active work to ONE task at any given time
   - Complete current activities before initiating new ones""",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "plan"],
                    "description": "The command to execute. `view` shows the current task list. `plan` creates or updates the task list."
                },
                "task_list": {
                    "type": "array",
                    "description": "The full task list. Required parameter of `plan` command.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique task identifier"
                            },
                            "title": {
                                "type": "string",
                                "description": "Brief task description"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["todo", "in_progress", "done"],
                                "description": "Current task status"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Optional additional context or details"
                            }
                        },
                        "required": ["title", "status", "id"]
                    }
                }
            },
            "required": ["command"]
        }
    }
}