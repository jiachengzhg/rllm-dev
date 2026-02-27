#!/bin/bash
# Training script for OpenHands-compatible Agent (OHAgent)
#
# This script trains an OHAgent using R2E-Gym dataset on SWE-Bench tasks.
# The agent uses OpenHands-style tool calling with:
# - execute_bash
# - str_replace_editor  
# - think
# - finish
# - task_tracker
#
# Usage:
#   bash train_ohagent.sh
#
# Requirements:
#   - Docker or Kubernetes backend configured
#   - GPU cluster with sufficient resources

set -x

# Environment variables for vLLM
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_VISIBLE_DEVICES=0,2,6,7

# Find the rllm package directory
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# Run training
python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${RLLM_DIR}/rllm/data/datasets/R2E_Gym_Subset_Debug/train_verl.parquet\
    data.val_files=${RLLM_DIR}/rllm/data/datasets/SWE_Bench_Verified_Debug/test_verl.parquet \
    data.train_batch_size=2 \
    data.val_batch_size=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=2 \
    actor_rollout_ref.model.path=checkpoints/huggingface/Qwen3-0.6B \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='openhands-agent' \
    trainer.experiment_name='ohagent-debug_test' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=2 \
    trainer.default_hdfs_dir=null \
    rllm.env.name=openhands \
    +rllm.env.env_args.backend=docker \
    +rllm.env.env_args.step_timeout=90 \
    +rllm.env.env_args.reward_timeout=300 \
    +rllm.env.env_args.use_remote=False \
    +rllm.env.env_args.remote_server_url=http://14.103.173.234:1999 \
    rllm.agent.name=ohagent \
    rllm.agent.max_steps=50 \
    +rllm.agent.agent_args.use_fn_calling=False \
    +rllm.agent.agent_args.format_model_response=False \
    +rllm.agent.agent_args.save_completion_text=True \
    rllm.agent.overlong_filter=False \
    rllm.agent.trajectory_timeout=5400 \
    trainer.total_epochs=4  2>&1 | tee ohagent-debug_test.log

