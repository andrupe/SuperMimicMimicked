"""Training script for Crazyflie 2.1 quadcopter on SuperMimic with Brownian velocity control."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from mjlab.tasks.registry import load_env_cfg
from supermimic.runners.crazyflie_runner import CrazyflieRunner, CrazyflieRunnerCfg
from supermimic.scripts.train_utils import create_env, setup_device
import supermimic.tasks


def parse_args():
  parser = argparse.ArgumentParser(description="Train Crazyflie 2.1 on SuperMimic with SSQP expert corpus.")
  parser.add_argument("--task-id", type=str, default="SuperMimic-Crazyflie-Joint", help="Task ID.")
  parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel environments.")
  parser.add_argument("--iterations", type=int, default=15000, help="Training iterations.")
  parser.add_argument("--collect-steps", type=int, default=8, help="Real sim steps per iteration.")
  parser.add_argument("--wm-batch-size", type=int, default=64, help="Batch size for World Model updates.")
  parser.add_argument("--wm-horizon", type=int, default=8, help="Rollout horizon for World Model supervised training.")
  parser.add_argument("--policy-batch-size", type=int, default=64, help="Batch size for Policy rollouts.")
  parser.add_argument("--policy-num-batches", type=int, default=1, help="Number of policy micro-batches per iteration.")
  parser.add_argument("--grad-accum", action="store_true", help="Accumulate gradients across micro-batches in WM and Policy learners.")
  parser.add_argument("--policy-lr", type=float, default=None, help="Policy learning rate (defaults to runner cfg).")
  parser.add_argument("--wm-lr", type=float, default=None, help="World model learning rate (defaults to runner cfg).")
  parser.add_argument("--horizon", type=int, default=48, help="BPTT policy unroll horizon.")
  parser.add_argument("--history-length", type=int, default=1, help="Actor observation history length.")
  parser.add_argument("--roll-pitch-spin-weight", type=float, default=-0.005, help="WM roll/pitch angular rate regularization weight.")
  parser.add_argument("--tilt-margin-weight", type=float, default=0.0, help="Soft penalty weight above 70 deg tilt in WM.")
  parser.add_argument("--replay-ratio", type=int, default=4, help="WM update repeats per iteration.")
  parser.add_argument("--dynamics-hidden-dims", type=int, nargs="+", default=[512, 512, 512], help="World model MLP hidden dims.")
  parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=[64, 64], help="Actor MLP hidden dims (sized for Crazyflie 2.1 onboard STM32F405).")
  parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[256, 256, 128], help="Critic MLP hidden dims (training-only).")
  parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda').")
  parser.add_argument("--save-interval", type=int, default=500, help="Checkpoint save interval.")
  parser.add_argument("--seed-steps", type=int, default=80, help="Number of steps per env to pre-seed buffer from corpus.")
  parser.add_argument("--corpus-path", type=str, default="corpus/crazyflie_corpus_all90trajs.npz", help="Path to corpus npz.")
  parser.add_argument("--corpus-ratio", type=float, default=None, help="Fraction of envs tracking corpus trajectories (1.0 disables Brownian motion).")
  parser.add_argument("--checkpoint-dir", type=str, default="logs/crazyflie_supermimic", help="Directory for logs and checkpoints.")
  parser.add_argument("--resume-ckpt", type=str, default=None, help="Optional checkpoint path to initialize actor/dynamics/sim_context for finetuning.")
  parser.add_argument("--use-mlflow", action="store_true", help="Enable MLflow experiment tracking.")
  parser.add_argument("--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI.")
  parser.add_argument("--experiment-name", type=str, default="SuperMimic-Crazyflie-MegaCorpus", help="MLflow experiment name.")
  parser.add_argument("--run-name", type=str, default=None, help="MLflow run name.")
  parser.add_argument("--mlflow-log-freq", type=int, default=25, help="Frequency (in iterations) to log metrics to MLflow.")
  return parser.parse_args()


def main():
  args = parse_args()
  device = setup_device(args.device)

  print(f"[INFO] Initializing Crazyflie SuperMimic training on task: {args.task_id} (device: {device})")
  print(f"[INFO] Parallel environments: {args.num_envs}, Max iterations: {args.iterations}")

  # 1. Load env configuration
  env_cfg = load_env_cfg(args.task_id, play=False)
  if "velocity" in env_cfg.commands and hasattr(env_cfg.commands["velocity"], "corpus_path"):
    env_cfg.commands["velocity"].corpus_path = args.corpus_path
  if args.corpus_ratio is not None and "velocity" in env_cfg.commands and hasattr(env_cfg.commands["velocity"], "corpus_ratio"):
    env_cfg.commands["velocity"].corpus_ratio = args.corpus_ratio
  env = create_env(env_cfg, num_envs=args.num_envs, device=device)

  # 2. Configure runner
  runner_cfg = CrazyflieRunnerCfg(
    max_iterations=args.iterations,
    num_collect_steps=args.collect_steps,
    replay_ratio=args.replay_ratio,
    dynamics_hidden_dims=tuple(args.dynamics_hidden_dims),
    actor_hidden_dims=tuple(args.actor_hidden_dims),
    critic_hidden_dims=tuple(args.critic_hidden_dims),
    actor_history_length=args.history_length,
    roll_pitch_spin_weight=args.roll_pitch_spin_weight,
    tilt_margin_weight=args.tilt_margin_weight,
    checkpoint_dir=args.checkpoint_dir,
  )
  runner_cfg.wm.batch_size = args.wm_batch_size
  runner_cfg.wm.horizon = args.wm_horizon
  runner_cfg.wm.accumulate_grad = args.grad_accum
  if args.wm_lr is not None:
    runner_cfg.wm.lr = args.wm_lr
  runner_cfg.policy.batch_size = args.policy_batch_size
  runner_cfg.policy.num_batches = args.policy_num_batches
  runner_cfg.policy.accumulate_grad = args.grad_accum
  if args.policy_lr is not None:
    runner_cfg.policy.lr = args.policy_lr
  runner_cfg.policy.horizon = args.horizon

  runner = CrazyflieRunner(
    env=env,
    cfg=runner_cfg,
    task_id=args.task_id,
    device=device,
  )
  if args.resume_ckpt is not None:
    runner.load_checkpoint(args.resume_ckpt)

  # Pre-seed replay buffer from corpus
  if args.seed_steps > 0:
    runner.seed_replay_buffer_from_corpus(args.corpus_path, num_steps=args.seed_steps)

  # Setup MLflow tracking
  mlflow_run = None
  if args.use_mlflow:
    import mlflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    run_name = args.run_name or f"{args.task_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    mlflow_run = mlflow.start_run(run_name=run_name)
    params = {
      "task_id": args.task_id,
      "num_envs": args.num_envs,
      "iterations": args.iterations,
      "collect_steps": args.collect_steps,
      "wm_batch_size": args.wm_batch_size,
      "wm_horizon": args.wm_horizon,
      "policy_batch_size": args.policy_batch_size,
      "policy_num_batches": args.policy_num_batches,
      "grad_accum": args.grad_accum,
      "policy_lr": runner_cfg.policy.lr,
      "wm_lr": runner_cfg.wm.lr,
      "horizon": args.horizon,
      "replay_ratio": args.replay_ratio,
      "dynamics_hidden_dims": str(tuple(args.dynamics_hidden_dims)),
      "actor_hidden_dims": str(tuple(args.actor_hidden_dims)),
      "seed_steps": args.seed_steps,
      "corpus_path": args.corpus_path,
      "corpus_ratio": args.corpus_ratio if args.corpus_ratio is not None else getattr(env_cfg.commands.get("velocity"), "corpus_ratio", None),
      "checkpoint_dir": args.checkpoint_dir,
      "device": str(device),
    }
    mlflow.log_params(params)
    print(f"[MLflow] Started run '{run_name}' (ID: {mlflow_run.info.run_id}) in experiment '{args.experiment_name}'")

  # 3. Main training loop
  print("[INFO] Starting SuperMimic Quadcopter training loop...")
  start_time = time.time()
  csv_path = Path(runner_cfg.checkpoint_dir) / "train_log.csv"
  csv_path.parent.mkdir(parents=True, exist_ok=True)
  csv_file = open(csv_path, "w", buffering=1)
  csv_file.write("iteration,sim_reward,wm_loss,bptt_reward,step_duration_s,elapsed_s\n")

  for it in range(1, args.iterations + 1):
    step_start = time.time()
    metrics = runner.train_step(it)
    step_duration = time.time() - step_start

    # Console logging
    reward = metrics.get("rollout/mean_step_reward", 0.0)
    wm_loss = metrics.get("wm/Loss/state", 0.0)
    policy_reward = metrics.get("policy/Policy/reward", 0.0)
    elapsed = time.time() - start_time

    csv_file.write(f"{it},{reward:.5f},{wm_loss:.5f},{policy_reward:.5f},{step_duration:.4f},{elapsed:.2f}\n")

    log_str = (
      f"Iter {it:04d}/{args.iterations:04d} | "
      f"Sim R: {reward:+.3f} | "
      f"WM Loss: {wm_loss:.4f} | "
      f"BPTT R: {policy_reward:+.3f} | "
      f"Time: {step_duration:.2f}s"
    )
    print(log_str)

    # MLflow logging
    if args.use_mlflow and (it % args.mlflow_log_freq == 0 or it == args.iterations or it == 1):
      import mlflow
      log_data = {
        "train/sim_reward": float(reward),
        "train/wm_loss": float(wm_loss),
        "train/bptt_reward": float(policy_reward),
        "train/step_duration_s": float(step_duration),
        "train/elapsed_s": float(elapsed),
        "train/iter_per_sec": float(1.0 / max(step_duration, 1e-4)),
      }
      for k, v in metrics.items():
        if k not in ("rollout/mean_step_reward", "wm/Loss/state", "policy/Policy/reward"):
          try:
            log_data[f"raw/{k}"] = float(v)
          except (ValueError, TypeError):
            pass
      mlflow.log_metrics(log_data, step=it)

    if it % args.save_interval == 0 or it == args.iterations:
      runner.save_checkpoint(it)
      if args.use_mlflow:
        import mlflow
        ckpt_file = runner.checkpoint_path / f"checkpoint_{it:06d}.pt"
        if ckpt_file.exists():
          mlflow.log_artifact(str(ckpt_file), artifact_path="checkpoints")

  csv_file.close()

  if args.use_mlflow:
    import mlflow
    if csv_path.exists():
      mlflow.log_artifact(str(csv_path), artifact_path="logs")
    mlflow.end_run()
    print(f"[MLflow] Run successfully completed and logged to {args.mlflow_tracking_uri}")

  total_time = time.time() - start_time
  print(f"[INFO] Completed {args.iterations} iterations in {total_time:.1f}s ({total_time / args.iterations:.2f}s/iter)!")


if __name__ == "__main__":
  main()

