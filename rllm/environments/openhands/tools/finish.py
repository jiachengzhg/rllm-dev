"""
Finish Tool - OpenHands compatible.

This tool signals completion of the current task.
"""

# OpenHands-style tool schema
FINISH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": """Signals the completion of the current task or conversation.

Use this tool when:
- You have successfully completed the user's requested task
- You cannot proceed further due to technical limitations or missing information

The message should include:
- A clear summary of actions taken and their results
- Any next steps for the user
- Explanation if you're unable to complete the task
- Any follow-up questions if more information is needed""",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Final message to send to the user"
                }
            },
            "required": ["message"]
        }
    }
}