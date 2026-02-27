"""
Training script for OpenHands-compatible Agent (OHAgent).

This script trains an OHAgent using R2E-Gym dataset on SWE-Bench tasks.
The agent uses OpenHands-style tool calling with:
- execute_bash
- str_replace_editor
- think
- finish
- task_tracker

Usage:
    python train_ohagent.py

Configuration:
    See train_ohagent.sh for full configuration options.
"""

import hydra
from omegaconf import DictConfig

from rllm.agents.oh_agent import OHAgent
from rllm.data import DatasetRegistry
from rllm.environments.openhands.oh_env import OHEnv
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config: DictConfig):
    """
    Main training function.
    
    This function:
    1. Loads the R2E-Gym training dataset
    2. Loads the SWE-Bench-Verified validation dataset
    3. Initializes the OHAgent and OHEnv
    4. Starts PPO training using rllm's AgentTrainer
    
    Args:
        config: Hydra configuration object with training parameters.
    """
    # Load datasets
    # Training on R2E-Gym Subset
    train_dataset = DatasetRegistry.load_dataset("R2E_Gym_Subset", "train")
    
    # Validation on SWE-Bench Verified
    val_dataset = DatasetRegistry.load_dataset("SWE_Bench_Verified", "test")
    
    # Configure agent arguments
    agent_args = {
        "use_fn_calling": config.get("agent", {}).get("use_fn_calling", False),
        "format_model_response": config.get("agent", {}).get("format_model_response", False),
    }
    
    # Configure environment arguments  
    env_args = {
        "backend": config.get("env", {}).get("backend", "kubernetes"),
        "step_timeout": config.get("env", {}).get("step_timeout", 90),
        "reward_timeout": config.get("env", {}).get("reward_timeout", 300),
        "verbose": config.get("env", {}).get("verbose", False),
    }
    
    # Initialize trainer
    trainer = AgentTrainer(
        agent_class=OHAgent,
        env_class=OHEnv,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

