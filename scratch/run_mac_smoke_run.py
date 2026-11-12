"""Driver script for SuperMimic smoke run on Mac CPU with a strict time cap."""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
import torch

# Ensure writable temporary dirs
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DATA_DIR"] = "/tmp/wandb"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs("/tmp/wandb", exist_ok=True)
os.makedirs("/tmp/matplotlib", exist_ok=True)

# Add supermimic and mjlab_research to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "supermimic" / "src"))
sys.path.insert(0, str(ROOT / "mjlab_research" / "src"))

import supermimic.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg, Ratio
from supermimic.scripts.train_utils import create_env


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mac CPU smoke run for SuperMimic")
    parser.add_argument(
        "--runtime-minutes",
        type=float,
        default=50.0,
        help="Maximum training runtime in minutes (default: 50 min, <= 60 min)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20000,
        help="Maximum iterations to train (default: 20000)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Logging interval in iterations (default: 25)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=250,
        help="Checkpoint save interval in iterations (default: 250)",
    )
    return parser.parse_args()


def save_checkpoint(runner, log_dir: Path):
    try:
        world_model_path = str(log_dir / "world_model.pt")
        actor_path = str(log_dir / "actor.pt")
        torch.save(runner._wm.bundle(), world_model_path)
        torch.save(runner.policy_learner.actor.state_dict(), actor_path)
        critic = getattr(runner.policy_learner, "critic", None)
        if critic is not None:
            critic_path = str(log_dir / "critic.pt")
            torch.save(critic.state_dict(), critic_path)
        print(f"[INFO] Checkpoint saved successfully: {actor_path}", flush=True)
    except Exception as e:
        print(f"[WARN] Checkpoint save error: {e}", flush=True)


def main():
    args = parse_args()
    max_runtime_seconds = min(args.runtime_minutes * 60.0, 3600.0)  # Strict <= 1 hour cap

    print("=" * 70, flush=True)
    print("  SuperMimic Smoke Run on Mac (Apple Silicon CPU)", flush=True)
    print(f"  Time budget: {max_runtime_seconds / 60:.1f} minutes max", flush=True)
    print(f"  Max iterations: {args.max_iterations}", flush=True)
    print(f"  Log interval: {args.log_interval} steps", flush=True)
    print("=" * 70, flush=True)

    task_id = "SuperMimic-GenCoord-G1"
    device = "cpu"
    num_envs = 4
    motion_file = str(ROOT / "scratch" / "mock_motion" / "g1_idle_motion.npz")

    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.commands["motion"].motion_file = motion_file
    env_cfg.commands["motion"].sampling_mode = "external"

    runner_cfg = SuperMimicRunnerCfg()
    runner_cfg.max_iterations = args.max_iterations
    runner_cfg.log_interval = args.log_interval
    runner_cfg.save_interval = args.save_interval
    runner_cfg.replay_ratio = 16

    # Lightweight dims optimized for CPU smoke run
    runner_cfg.wm.batch_size = 32
    runner_cfg.wm.num_batches = 1
    runner_cfg.wm.horizon = 4
    runner_cfg.policy.batch_size = 32
    runner_cfg.policy.horizon = 4
    runner_cfg.dynamics_hidden_dims = (128, 128)
    runner_cfg.actor_hidden_dims = (128, 64)
    runner_cfg.critic_hidden_dims = (64, 32)
    runner_cfg.wandb_project = "supermimic_mac_smoke"

    print(f"[INFO] Creating environment with {num_envs} envs on {device}...", flush=True)
    env = create_env(env_cfg, num_envs, device)

    print("[INFO] Initializing SuperMimicRunner...", flush=True)
    runner = SuperMimicRunner(
        env=env,
        cfg=runner_cfg,
        task_id=task_id,
        registry_name=None,
        device=device,
    )

    log_dir = ROOT / "logs" / "mac_smoke_run"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_file = log_dir / "progress.csv"

    # Write CSV header if not exists
    if not csv_file.exists():
        with open(csv_file, "w") as f:
            f.write("timestamp,step,elapsed_sec,wm_loss,policy_reward,steps_per_sec\n")

    start_time = time.time()
    last_save_time = start_time
    total_steps_executed = 0

    # Set up graceful alarm
    def alarm_handler(signum, frame):
        print(f"\n[INFO] Time limit reached ({max_runtime_seconds}s). Saving final checkpoint...", flush=True)
        save_checkpoint(runner, log_dir)
        sys.exit(0)

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(int(max_runtime_seconds))

    print(f"[INFO] Starting training loop (max runtime: {max_runtime_seconds:.0f}s)...", flush=True)
    runner.env.reset()
    runner._seed_sim_history()

    should_train = Ratio(
        runner.env.num_envs
        / runner.cfg.wm.num_batches
        / runner.cfg.wm.batch_size
        / runner.cfg.wm.horizon
        * runner.cfg.replay_ratio
    )

    num_collected_steps = 0
    step_start_time = time.time()
    wm_learn_time = 0.0
    policy_learn_time = 0.0
    loss_dict = {}
    history = []

    try:
        for step in range(runner_cfg.max_iterations):
            elapsed = time.time() - start_time
            if elapsed >= max_runtime_seconds:
                print(f"\n[INFO] Time limit reached ({elapsed:.1f}s >= {max_runtime_seconds:.0f}s). Finishing run...", flush=True)
                break

            total_steps_executed = step + 1
            num_collected_steps += runner.step()

            for _ in range(should_train(step)):
                if runner.replay_buffer.num_transitions < runner.cfg.wm.horizon:
                    continue

                data = (
                    runner.replay_buffer.sample(runner.cfg.wm.horizon, runner.cfg.wm.batch_size)
                    for _ in range(runner.cfg.wm.num_batches)
                )

                t0 = time.time()
                wm_logs = runner.world_model_learner.update(data)
                wm_learn_time += time.time() - t0

                t0 = time.time()
                policy_logs = runner.policy_learner.update()
                policy_learn_time += time.time() - t0

                loss_dict.update(wm_logs)
                loss_dict.update(policy_logs)

            if (step + 1) % runner_cfg.log_interval == 0:
                elapsed_step_time = time.time() - step_start_time
                total_train_time = wm_learn_time + policy_learn_time
                collect_time = elapsed_step_time - total_train_time
                sps = num_collected_steps / elapsed_step_time if elapsed_step_time > 0 else 0.0

                reward = loss_dict.get("Policy/reward", torch.tensor(0.0))
                wm_loss = loss_dict.get("Loss/state", torch.tensor(0.0))
                rew_val = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                wm_val = wm_loss.item() if isinstance(wm_loss, torch.Tensor) else float(wm_loss)

                entry = {
                    "timestamp": time.time(),
                    "step": step + 1,
                    "elapsed_sec": round(elapsed, 1),
                    "wm_loss": round(wm_val, 5),
                    "policy_reward": round(rew_val, 5),
                    "steps_per_sec": round(sps, 1),
                }
                history.append(entry)

                with open(csv_file, "a") as f:
                    f.write(f"{entry['timestamp']},{entry['step']},{entry['elapsed_sec']},{entry['wm_loss']},{entry['policy_reward']},{entry['steps_per_sec']}\n")

                print(
                    f"[{time.strftime('%H:%M:%S')}] Step {step+1:5d} | "
                    f"Elapsed: {elapsed/60:4.1f}m / {max_runtime_seconds/60:.0f}m | "
                    f"WM Loss: {wm_val:.4f} | "
                    f"Policy Rew: {rew_val:+.4f} | "
                    f"SPS: {sps:4.1f} | "
                    f"Collect: {collect_time:.2f}s | "
                    f"Train: {total_train_time:.2f}s",
                    flush=True,
                )

                step_start_time = time.time()
                wm_learn_time = 0.0
                policy_learn_time = 0.0
                num_collected_steps = 0

            if (step + 1) % runner_cfg.save_interval == 0 or (time.time() - last_save_time > 300):
                save_checkpoint(runner, log_dir)
                last_save_time = time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Run interrupted by user.", flush=True)
    finally:
        save_checkpoint(runner, log_dir)
        total_elapsed = time.time() - start_time
        summary = {
            "task_id": task_id,
            "device": device,
            "total_iterations": total_steps_executed,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "total_elapsed_minutes": round(total_elapsed / 60, 2),
            "final_wm_loss": history[-1]["wm_loss"] if history else None,
            "final_policy_reward": history[-1]["policy_reward"] if history else None,
            "checkpoint_dir": str(log_dir),
            "checkpoints": [
                str(log_dir / "world_model.pt"),
                str(log_dir / "actor.pt"),
                str(log_dir / "critic.pt"),
            ],
        }
        with open(log_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("=" * 70, flush=True)
        print("  Smoke run completed!", flush=True)
        print(f"  Total iterations: {total_steps_executed}", flush=True)
        print(f"  Total time: {total_elapsed/60:.2f} minutes", flush=True)
        print(f"  Checkpoints saved to: {log_dir}", flush=True)
        if history:
            print(f"  Initial WM Loss: {history[0]['wm_loss']:.4f} -> Final: {history[-1]['wm_loss']:.4f}", flush=True)
            print(f"  Initial Rew: {history[0]['policy_reward']:+.4f} -> Final: {history[-1]['policy_reward']:+.4f}", flush=True)
        print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
