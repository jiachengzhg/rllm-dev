#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Simulate OpenHands rollout without model deployment.

This script mirrors the environment-side flow used in training:
1) Build OHEnv from one sample's `extra_info` + env overrides
2) reset()
3) step(execute_bash) to create /testbed/hello_world.txt
4) step(finish)
5) compute_final_reward()
6) save last_test_output
"""

from __future__ import annotations

import argparse
import importlib
import json
import shlex
from pathlib import Path
from typing import Any

from rllm.environments.openhands.oh_env import OHEnv


def _log(msg: str) -> None:
    print(f"[simulate-rollout] {msg}", flush=True)


def _clip(text: str | None, max_len: int = 1500) -> str:
    if text is None:
        return "<None>"
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... <clipped> ..."


def _load_sample_extra_info(parquet_path: str, index: int) -> dict[str, Any]:
    _log(f"Loading parquet: {parquet_path}")
    pl_spec = importlib.util.find_spec("polars")
    pd_spec = importlib.util.find_spec("pandas")
    if pl_spec is not None:
        import polars as pl

        df = pl.read_parquet(parquet_path)
        n_rows = df.height
        columns = list(df.columns)
        if n_rows == 0:
            raise ValueError(f"Dataset is empty: {parquet_path}")
        if index < 0 or index >= n_rows:
            raise IndexError(f"Index {index} out of range [0, {n_rows - 1}] for {parquet_path}")
        if "extra_info" not in columns:
            raise KeyError(f"`extra_info` column not found in {parquet_path}")
        extra_info = df.row(index, named=True)["extra_info"]
    elif pd_spec is not None:
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        n_rows = len(df)
        columns = list(df.columns)
        if n_rows == 0:
            raise ValueError(f"Dataset is empty: {parquet_path}")
        if index < 0 or index >= n_rows:
            raise IndexError(f"Index {index} out of range [0, {n_rows - 1}] for {parquet_path}")
        if "extra_info" not in columns:
            raise KeyError(f"`extra_info` column not found in {parquet_path}")
        row = df.iloc[index]
        extra_info = row["extra_info"]
    else:
        raise ImportError("Neither `polars` nor `pandas` is available to read parquet files.")

    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    if not isinstance(extra_info, dict):
        raise TypeError(f"`extra_info` at index {index} is not dict/JSON string: {type(extra_info)}")

    _log(f"Loaded sample index={index}; columns={columns}")
    return extra_info


def _write_text(path: Path, text: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "" if text is None else text
    path.write_text(content, encoding="utf-8")
    _log(f"Saved output: {path}")


def _run_single_split(
    split_name: str,
    parquet_path: str,
    sample_index: int,
    output_file: Path,
    env_overrides: dict[str, Any],
    hello_content: str,
) -> None:
    _log(f"========== START {split_name.upper()} ==========")
    extra_info = _load_sample_extra_info(parquet_path, sample_index)

    # Mimic trainer behavior: merge base env args with sample extra_info, then OHEnv.from_dict.
    merged_info = dict(extra_info)
    merged_info.update(env_overrides)
    env = OHEnv.from_dict(merged_info)

    try:
        _log("Calling env.reset() ...")
        observation, info = env.reset()
        _log(f"reset() done, info={info}")
        _log(f"Initial observation (clipped):\n{_clip(observation)}")

        # Use printf instead of heredoc to avoid EOF marker leakage.
        create_cmd = f"printf %s {shlex.quote(hello_content + chr(10))} > /testbed/hello_world.txt"
        create_action = {
            "name": "execute_bash",
            "arguments": {
                "command": create_cmd,
                "timeout": int(env_overrides.get("step_timeout", 90)),
                "is_input": "false",
            },
        }
        _log("Step 1/2: execute_bash -> create /testbed/hello_world.txt")
        obs, reward, done, step_info = env.step(create_action)
        _log(f"step1: reward={reward}, done={done}, info={step_info}")
        _log(f"step1 observation (clipped):\n{_clip(obs)}")

        verify_action = {
            "name": "execute_bash",
            "arguments": {"command": "ls -l /testbed/hello_world.txt && cat /testbed/hello_world.txt", "timeout": int(env_overrides.get("step_timeout", 90)), "is_input": "false"},
        }
        _log("Step 1.5/2: execute_bash -> verify hello_world.txt")
        obs, reward, done, step_info = env.step(verify_action)
        _log(f"verify step: reward={reward}, done={done}, info={step_info}")
        _log(f"verify observation (clipped):\n{_clip(obs)}")

        finish_action = {
            "name": "finish",
            "arguments": {"message": f"simulate rollout finished for {split_name}"},
        }
        _log("Step 2/2: finish")
        obs, reward, done, step_info = env.step(finish_action)
        _log(f"finish step: reward={reward}, done={done}, info={step_info}")
        _log(f"finish observation (clipped):\n{_clip(obs)}")

        _log("Calling env.compute_final_reward() ...")
        final_reward = env.compute_final_reward(timeout=int(env_overrides.get("reward_timeout", 300)))
        _log(f"Final reward: {final_reward}")

        last_test_output = env.last_test_output
        _log(f"last_test_output (clipped):\n{_clip(last_test_output)}")
        _write_text(output_file, last_test_output)
    finally:
        _log("Closing environment ...")
        env.close()

    _log(f"=========== END {split_name.upper()} ===========")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate OHEnv rollout for one train and one test sample.")
    parser.add_argument("--train-files", required=True, help="Train parquet path, e.g. train_verl.parquet")
    parser.add_argument("--test-files", required=True, help="Test/val parquet path, e.g. test_verl.parquet")
    parser.add_argument("--train-index", type=int, default=0, help="Sample index in train parquet")
    parser.add_argument("--test-index", type=int, default=0, help="Sample index in test parquet")
    parser.add_argument("--output-dir", default=".", help="Directory to save output text files")
    parser.add_argument("--train-output-name", default="test_output_train.txt", help="Train output filename")
    parser.add_argument("--test-output-name", default="test_output_test.txt", help="Test output filename")
    parser.add_argument("--backend", default="docker", choices=["docker", "kubernetes"], help="Runtime backend")
    parser.add_argument("--step-timeout", type=int, default=90, help="Step timeout passed to OHEnv")
    parser.add_argument("--reward-timeout", type=int, default=300, help="Reward timeout passed to OHEnv")
    parser.add_argument("--use-remote", action="store_true", help="Whether to use remote docker")
    parser.add_argument("--remote-server-url", default=None, help="Remote docker server URL")
    parser.add_argument("--remote-api-key", default=None, help="Remote docker server API key")
    parser.add_argument("--use-huashan", action="store_true", help="Whether to use huashan MCP runtime")
    parser.add_argument("--huashan-server-url", default=None, help="Huashan MCP server URL")
    parser.add_argument("--hello-content", default="hello world", help="Content written into /testbed/hello_world.txt")
    parser.add_argument("--use-gt-patch", action="store_true", help="Whether to use gt patch")
    args = parser.parse_args()

    env_overrides = {
        "backend": args.backend,
        "step_timeout": args.step_timeout,
        "reward_timeout": args.reward_timeout,
        "use_remote": args.use_remote,
        "remote_server_url": args.remote_server_url,
        "remote_api_key": args.remote_api_key,
        "use_huashan": args.use_huashan,
        "huashan_server_url": args.huashan_server_url,
        "use_gt_patch": args.use_gt_patch,
    }

    output_dir = Path(args.output_dir)
    train_output = output_dir / args.train_output_name
    test_output = output_dir / args.test_output_name

    _run_single_split(
        split_name="train",
        parquet_path=args.train_files,
        sample_index=args.train_index,
        output_file=train_output,
        env_overrides=env_overrides,
        hello_content=args.hello_content,
    )
    _run_single_split(
        split_name="test",
        parquet_path=args.test_files,
        sample_index=args.test_index,
        output_file=test_output,
        env_overrides=env_overrides,
        hello_content=args.hello_content,
    )
    _log("All rollout simulations completed.")


if __name__ == "__main__":
    main()
