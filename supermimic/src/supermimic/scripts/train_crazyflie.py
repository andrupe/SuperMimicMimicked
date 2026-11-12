"""Training script for Crazyflie 2.1 quadcopter on SuperMimic with Brownian velocity control."""

from __future__ import annotations

import argparse
import time

import torch
from mjlab.tasks.registry import load_env_cfg
from supermimic.runners.crazyflie_runner import CrazyflieRunner, CrazyflieRunnerCfg
from supermimic.scripts.train_utils import create_env, setup_device
import supermimic.tasks


def parse_args():
  parser = argparse.ArgumentParser(description="Train Crazyflie 2.1 on SuperMimic with SSQP expert corpus.")
  parser.add_argument("--task-id", type=str, default="SuperMimic-Crazyflie-Corpus", help="Task ID.")
  parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments.")
  parser.add_argument("--iterations", type=int, default=2500, help="Training iterations.")
  parser.add_argument("--collect-steps", type=int, default=8, help="Real sim steps per iteration.")
  parser.add_argument("--wm-batch-size", type=int, default=64, help="Batch size for World Model updates.")
  parser.add_argument("--wm-horizon", type=int, default=4, help="Rollout horizon for World Model supervised training.")
  parser.add_argument("--policy-batch-size", type=int, default=32, help="Batch size for Policy rollouts.")
  parser.add_argument("--horizon", type=int, default=32, help="BPTT policy unroll horizon.")
  parser.add_argument("--replay-ratio", type=int, default=2, help="WM update repeats per iteration.")
  parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda').")
  parser.add_argument("--save-interval", type=int, default=100, help="Checkpoint save interval.")
  parser.add_argument("--seed-steps", type=int, default=50, help="Number of steps per env to pre-seed buffer from corpus.")
  parser.add_argument("--corpus-path", type=str, default="corpus/crazyflie_corpus_50trajs.npz", help="Path to corpus npz.")
  return parser.parse_args()


def main():
  args = parse_args()
  device = setup_device(args.device)

  print(f"[INFO] Initializing Crazyflie SuperMimic training on task: {args.task_id} (device: {device})")
  print(f"[INFO] Parallel environments: {args.num_envs}, Max iterations: {args.iterations}")

  # 1. Load env configuration
  env_cfg = load_env_cfg(args.task_id, play=False)
  env = create_env(env_cfg, num_envs=args.num_envs, device=device)

  # 2. Configure runner
  runner_cfg = CrazyflieRunnerCfg(
    max_iterations=args.iterations,
    num_collect_steps=args.collect_steps,
    replay_ratio=args.replay_ratio,
  )
  runner_cfg.wm.batch_size = args.wm_batch_size
  runner_cfg.wm.horizon = args.wm_horizon
  runner_cfg.policy.batch_size = args.policy_batch_size
  runner_cfg.policy.horizon = args.horizon

  runner = CrazyflieRunner(
    env=env,
    cfg=runner_cfg,
    task_id=args.task_id,
    device=device,
  )

  # Pre-seed replay buffer from corpus
  if args.seed_steps > 0:
    runner.seed_replay_buffer_from_corpus(args.corpus_path, num_steps=args.seed_steps)

  # 3. Main training loop
  print("[INFO] Starting SuperMimic Quadcopter training loop...")
  start_time = time.time()

  for it in range(1, args.iterations + 1):
    step_start = time.time()
    metrics = runner.train_step(it)
    step_duration = time.time() - step_start

    # Console logging
    reward = metrics.get("rollout/mean_step_reward", 0.0)
    wm_loss = metrics.get("wm/Loss/state", 0.0)
    policy_reward = metrics.get("policy/Policy/reward", 0.0)

    log_str = (
      f"Iter {it:04d}/{args.iterations:04d} | "
      f"Sim R: {reward:+.3f} | "
      f"WM Loss: {wm_loss:.4f} | "
      f"BPTT R: {policy_reward:+.3f} | "
      f"Time: {step_duration:.2f}s"
    )
    print(log_str)

    if it % args.save_interval == 0 or it == args.iterations:
      runner.save_checkpoint(it)

  total_time = time.time() - start_time
  print(f"[INFO] Completed {args.iterations} iterations in {total_time:.1f}s ({total_time / args.iterations:.2f}s/iter)!")


if __name__ == "__main__":
  main()

