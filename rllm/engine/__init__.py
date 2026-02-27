"""Engine module for rLLM.

This module contains the core execution infrastructure for agent trajectory rollout.
"""

from .agent_execution_engine import AgentExecutionEngine, AsyncAgentExecutionEngine

# Avoid importing rollout submodules eagerly to prevent circular imports with workflows
# Import base class only (no side effects) and lazy-load specific engines via __getattr__
from .rollout.rollout_engine import ModelOutput, RolloutEngine

__all__ = [
    "AgentExecutionEngine",
    "AsyncAgentExecutionEngine",
    "RolloutEngine",
    "ModelOutput",
    "VerlEngine",
]


def __getattr__(name):
    if name == "VerlEngine":
        try:
            from .rollout.verl_engine import VerlEngine as _VerlEngine

            return _VerlEngine
        except Exception:
            raise AttributeError(name) from None
    raise AttributeError(name)
