from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

__all__ = ["BaseAgent", "Action", "Step", "Trajectory"]


def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


# Define all agent imports
AGENT_IMPORTS = [
    ("rllm.agents.oh_agent", "OHAgent"),
]

for module_path, class_name in AGENT_IMPORTS:
    imported_class = safe_import(module_path, class_name)
    if imported_class is not None:
        globals()[class_name] = imported_class
        __all__.append(class_name)
