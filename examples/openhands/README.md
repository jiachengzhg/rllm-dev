# OpenHands Agent Training

This example demonstrates training an OpenHands-compatible code agent using the rllm framework.

## Overview

The OHAgent (OpenHands Agent) is designed to solve GitHub issues by:
- Exploring repository structure
- Understanding code context
- Making targeted code changes
- Verifying fixes

## Tools

The agent has access to 5 OpenHands-style tools:

| Tool | Description |
|------|-------------|
| `execute_bash` | Execute bash commands in a persistent shell |
| `str_replace_editor` | View, create, and edit files |
| `think` | Log thoughts for complex reasoning |
| `task_tracker` | Track multi-step tasks |
| `finish` | Signal task completion |

## Quick Start

### 1. Installation

```bash
# Install rllm
cd rllm
pip install -e ./verl
pip install -e ./verl[vllm]
pip install -e .

# Install R2E-Gym for SWE environments
git clone https://github.com/agentica-project/R2E-Gym.git
cd R2E-Gym
pip install -e .
```

### 2. Training

```bash
# Using the training script
bash examples/openhands/train_ohagent.sh

# Or using Python directly
python examples/openhands/train_ohagent.py
```

### 3. Custom Configuration

You can customize training by modifying the configuration:

```bash
python examples/openhands/train_ohagent.py \
    actor_rollout_ref.model.path=YOUR_MODEL \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    env.backend=docker  # or kubernetes
```

## Architecture

### OHAgent (`rllm/agents/oh_agent.py`)

The agent class that:
- Parses model responses to extract tool calls
- Formats observations for the model
- Maintains conversation history
- Supports both XML and OpenAI function calling formats

### OHEnv (`rllm/environments/openhands/oh_env.py`)

The environment class that:
- Wraps R2E-Gym for docker/kubernetes execution
- Provides OpenHands-style tool interfaces
- Computes rewards using test execution
- Supports extensible runtime clients

### Runtime Client (`rllm/environments/openhands/runtime_client.py`)

Abstraction for docker runtime interaction:
- `LocalDockerClient`: Direct local docker
- `RemoteDockerClient`: HTTP-based remote docker (for server separation)

## Extending

### Custom Runtime Client

To use a remote docker server, implement your own client:

```python
from rllm.environments.openhands import RuntimeClient, OHEnv

class MyRemoteClient(RuntimeClient):
    def connect(self, ds, **kwargs):
        # Connect to remote server
        pass
    
    def run(self, command, timeout=90):
        # Execute command via HTTP/gRPC
        pass
    
    # ... implement other methods

# Use in training
env = OHEnv(entry=task, runtime_client=MyRemoteClient(server_url="http://..."))
```

### Custom Tools

To add or modify tools:

```python
from rllm.environments.openhands.tools import OPENHANDS_TOOLS

# Add a new tool
MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "...",
        "parameters": {...}
    }
}

# Register in OHAgent
agent = OHAgent(tools=[...existing_tools..., MY_TOOL_SCHEMA])
```

## Files

```
examples/openhands/
├── README.md                 # This file
├── train_ohagent.py         # Training script
└── train_ohagent.sh         # Shell training script

rllm/agents/
└── oh_agent.py              # OHAgent implementation

rllm/environments/openhands/
├── __init__.py
├── oh_env.py                # OHEnv implementation
└── runtime_client.py        # Docker runtime clients

rllm/tools/openhands_tools/
├── __init__.py
├── execute_bash.py          # Bash execution tool
├── str_replace_editor.py    # File editor tool
├── think.py                 # Think tool
├── finish.py                # Finish tool
└── task_tracker.py          # Task tracker tool
```

## Citation

If you use this code, please cite:

```bibtex
@misc{openhands2025,
  title={OpenHands Agent Training with rLLM},
  author={Your Name},
  year={2025}
}
```

