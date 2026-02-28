from enum import Enum


class TerminationReason(Enum):
    MAX_PROMPT_LENGTH_EXCEEDED = "max_prompt_length_exceeded"
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
    ENV_DONE = "env_done"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    ERROR = "error"


class TerminationEvent(Exception):
    def __init__(self, reason: TerminationReason = TerminationReason.UNKNOWN):
        super().__init__(f"Terminated: {reason}")
        self.reason = reason
