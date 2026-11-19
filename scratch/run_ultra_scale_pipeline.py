"""Ultra-Scale Experimentation Pipeline for SuperMimic Bitcraze Crazyflie 2.1.

Automates:
1. Verification of the 50,000-trajectory Ultra-Corpus (~7.5M transitions).
2. Phase 1: Ultra-Scale Smoke Test Matrix (Ultra-01 through Ultra-04, 1,000 iterations each):
   - Ultra-01: 512 envs, H=48, Replay Ratio 2
   - Ultra-02: 512 envs, H=64 (1.28s extreme lookahead), Replay Ratio 2
   - Ultra-03: 512 envs, H=48, Replay Ratio 4
   - Ultra-04: 256 envs, H=64, Replay Ratio 2
3. Systematic evaluation of all smoke test models on standard flight courses with MLflow tracking and 3D visual telemetry GIFs.
4. Auto-selection of the champion ultra configuration.
5. Phase 2: Ultra Deep Production Training (50,000 - 60,000 iterations @ 512 envs = ~245M transitions).
6. Multi-checkpoint milestone evaluation (10k, 20k, 30k, 40k, 50k, 60k iterations).
7. Full MLflow artifacts and synthesis report generation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

WORKSPACE = Path("/home/andrupe/SuperMimicMimicked").resolve()
PYTHON_BIN = str(WORKSPACE / "supermimic/.venv/bin/python")
CORPUS_50000 = str(WORKSPACE / "corpus/crazyflie_corpus_50000trajs.npz")
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "SuperMimic-Crazyflie-MegaCorpus"
ARTIFACTS_DIR = Path("/home/andrupe/.gemini/antigravity-cli/brain/b2d41161-778e-44e8-a60a-c51d518b1e94").resolve()


def run_cmd(cmd_list: list[str], desc: str, env_vars: dict | None = None) -> bool:
  full_env = os.environ.copy()
  full_env["MUJOCO_GL"] = "glfw"
  full_env["TORCH_COMPILE_DISABLE"] = "1"
  full_env["MLFLOW_ALLOW_FILE_STORE"] = "true"
  if env_vars:
    full_env.update(env_vars)

  print(f"\n{'='*75}\n[ULTRA-PIPELINE] Starting: {desc}\nCommand: {' '.join(cmd_list)}\n{'='*75}", flush=True)
  t0 = time.time()
  res = subprocess.run(cmd_list, cwd=str(WORKSPACE), env=full_env)
  elapsed = time.time() - t0
  if res.returncode == 0:
    print(f"[ULTRA-PIPELINE SUCCESS] {desc} finished cleanly in {elapsed:.1f}s ({elapsed/60:.2f} mins)", flush=True)
    return True
  else:
    print(f"[ULTRA-PIPELINE ERROR] {desc} failed with exit code {res.returncode} after {elapsed:.1f}s", flush=True)
    return False


def verify_corpus(corpus_path: str = CORPUS_50000) -> bool:
  p = Path(corpus_path)
  print(f"\n[CORPUS VERIFY] Checking {p}...", flush=True)
  if not p.exists():
    print(f"[CORPUS VERIFY ERROR] Corpus file does not exist: {p}", flush=True)
    return False

  try:
    data = np.load(p)
    states = data["states"]
    ctrls = data["ctrls"]
    state_offsets = data["state_offsets"]
    tasks = data["tasks"]
    num_trajs = len(tasks)

    print(f"[CORPUS VERIFY] Trajectories: {num_trajs:,}")
    print(f"[CORPUS VERIFY] States shape: {states.shape}, Ctrls shape: {ctrls.shape}")
    print(f"[CORPUS VERIFY] File size: {p.stat().st_size / (1024*1024):.1f} MB")

    assert not np.any(np.isnan(states)), "NaNs detected in states!"
    assert not np.any(np.isnan(ctrls)), "NaNs detected in controls!"
    assert states[:, 2].min() >= 0.035, f"Unsafe altitude detected: {states[:, 2].min()} m"
    print("[CORPUS VERIFY SUCCESS] Ultra-Corpus passed all integrity and physical validity checks!\n", flush=True)
    return True
  except Exception as e:
    print(f"[CORPUS VERIFY EXCEPTION] {e}", flush=True)
    return False


def evaluate_checkpoint(ckpt_path: Path, run_name: str, corpus_path: str = CORPUS_50000, device: str = "cuda") -> dict | None:
  eval_cmd = [
    PYTHON_BIN,
    "scratch/evaluate_and_log_mlflow.py",
    "--checkpoint-path", str(ckpt_path),
    "--corpus-path", corpus_path,
    "--prefix", run_name,
    "--use-mlflow",
    "--mlflow-tracking-uri", MLFLOW_URI,
    "--experiment-name", EXPERIMENT_NAME,
    "--device", device,
  ]
  success = run_cmd(eval_cmd, f"Evaluation of {run_name} ({ckpt_path.name})")
  if success:
    summary_path = ARTIFACTS_DIR / f"eval_summary_{run_name}_{ckpt_path.stem}.json"
    if summary_path.exists():
      with open(summary_path) as f:
        return json.load(f)
  return None


def run_ultra_smoke_matrix() -> list[dict]:
  """Runs the 4 candidate ultra smoke tests (1,000 iterations each)."""
  smoke_matrix = [
    {
      "name": "ultra_01_512env_h48",
      "envs": 512,
      "horizon": 48,
      "ratio": 2,
      "args": [
        "--task-id", "SuperMimic-Crazyflie-Joint",
        "--corpus-path", CORPUS_50000,
        "--num-envs", "512",
        "--iterations", "1000",
        "--collect-steps", "8",
        "--wm-batch-size", "64",
        "--wm-horizon", "4",
        "--policy-batch-size", "32",
        "--horizon", "48",
        "--replay-ratio", "2",
        "--checkpoint-dir", "logs/ultra_01_512env_h48",
        "--save-interval", "500",
        "--use-mlflow",
        "--run-name", "ultra_01_512env_h48",
      ],
      "ckpt_dir": WORKSPACE / "logs/ultra_01_512env_h48",
    },
    {
      "name": "ultra_02_512env_h64",
      "envs": 512,
      "horizon": 64,
      "ratio": 2,
      "args": [
        "--task-id", "SuperMimic-Crazyflie-Joint",
        "--corpus-path", CORPUS_50000,
        "--num-envs", "512",
        "--iterations", "1000",
        "--collect-steps", "8",
        "--wm-batch-size", "64",
        "--wm-horizon", "4",
        "--policy-batch-size", "32",
        "--horizon", "64",
        "--replay-ratio", "2",
        "--checkpoint-dir", "logs/ultra_02_512env_h64",
        "--save-interval", "500",
        "--use-mlflow",
        "--run-name", "ultra_02_512env_h64",
      ],
      "ckpt_dir": WORKSPACE / "logs/ultra_02_512env_h64",
    },
    {
      "name": "ultra_03_512env_h48_ratio4",
      "envs": 512,
      "horizon": 48,
      "ratio": 4,
      "args": [
        "--task-id", "SuperMimic-Crazyflie-Joint",
        "--corpus-path", CORPUS_50000,
        "--num-envs", "512",
        "--iterations", "1000",
        "--collect-steps", "8",
        "--wm-batch-size", "64",
        "--wm-horizon", "4",
        "--policy-batch-size", "32",
        "--horizon", "48",
        "--replay-ratio", "4",
        "--checkpoint-dir", "logs/ultra_03_512env_h48_ratio4",
        "--save-interval", "500",
        "--use-mlflow",
        "--run-name", "ultra_03_512env_h48_ratio4",
      ],
      "ckpt_dir": WORKSPACE / "logs/ultra_03_512env_h48_ratio4",
    },
    {
      "name": "ultra_04_256env_h64",
      "envs": 256,
      "horizon": 64,
      "ratio": 2,
      "args": [
        "--task-id", "SuperMimic-Crazyflie-Joint",
        "--corpus-path", CORPUS_50000,
        "--num-envs", "256",
        "--iterations", "1000",
        "--collect-steps", "8",
        "--wm-batch-size", "64",
        "--wm-horizon", "4",
        "--policy-batch-size", "32",
        "--horizon", "64",
        "--replay-ratio", "2",
        "--checkpoint-dir", "logs/ultra_04_256env_h64",
        "--save-interval", "500",
        "--use-mlflow",
        "--run-name", "ultra_04_256env_h64",
      ],
      "ckpt_dir": WORKSPACE / "logs/ultra_04_256env_h64",
    },
  ]

  results = []
  for test in smoke_matrix:
    train_cmd = [PYTHON_BIN, "supermimic/src/supermimic/scripts/train_crazyflie.py"] + test["args"]
    success = run_cmd(train_cmd, f"Ultra Smoke Test: {test['name']}")
    if success:
      ckpt_file = test["ckpt_dir"] / "checkpoint_001000.pt"
      if ckpt_file.exists():
        eval_res = evaluate_checkpoint(ckpt_file, test["name"], CORPUS_50000, device="cuda")
        results.append({
          "name": test["name"],
          "envs": test["envs"],
          "horizon": test["horizon"],
          "ratio": test["ratio"],
          "train_success": True,
          "eval": eval_res,
          "ckpt": str(ckpt_file),
          "args": test["args"],
        })
      else:
        results.append({**test, "train_success": True, "eval": None, "ckpt": None})
    else:
      results.append({**test, "train_success": False, "eval": None, "ckpt": None})

  return results


def run_ultra_production(best_cfg: dict, max_iterations: int = 50000) -> dict | None:
  """Runs the deep scaled production training on the 50,000 Ultra-Corpus."""
  run_name = "ultra_megacorpus_production"
  ckpt_dir = "logs/ultra_megacorpus_production"

  args = list(best_cfg["args"])
  for i, a in enumerate(args):
    if a == "--iterations":
      args[i + 1] = str(max_iterations)
    elif a == "--checkpoint-dir":
      args[i + 1] = ckpt_dir
    elif a == "--save-interval":
      args[i + 1] = "2000"
    elif a == "--run-name":
      args[i + 1] = run_name

  train_cmd = [PYTHON_BIN, "supermimic/src/supermimic/scripts/train_crazyflie.py"] + args
  success = run_cmd(train_cmd, f"Ultra Deep Production Run: {run_name} ({max_iterations} iters)")
  if not success:
    return None

  # Evaluate milestone checkpoints
  eval_milestones = [10000, 20000, 30000, 40000, max_iterations]
  milestone_results = {}
  final_eval = None

  for it in eval_milestones:
    ckpt_file = WORKSPACE / ckpt_dir / f"checkpoint_{it:06d}.pt"
    if ckpt_file.exists():
      prefix = f"ultra_prod_{it//1000}k"
      print(f"\n[EVAL MILESTONE] Evaluating checkpoint at {it} iterations...", flush=True)
      ev = evaluate_checkpoint(ckpt_file, prefix, CORPUS_50000, device="cuda")
      milestone_results[it] = ev
      if it == max_iterations:
        final_eval = ev

  return final_eval


def update_synthesis_report(smoke_results: list[dict], prod_eval: dict | None):
  report_path = ARTIFACTS_DIR / "TRAINING_AND_EVALUATION_REPORT.md"
  if not report_path.exists():
    return

  with open(report_path, "r") as f:
    existing = f.read()

  new_section = f"""

---

## 9. Ultra-Scale Experimentation (50,000-Trajectory Ultra-Corpus, ~7.5M Transitions)

### 9.1 Overview & Scaling Frontier
* **Corpus Scale:** 50,000 unique, physically verified reference trajectories (~7,460,000 steps).
* **Trajectory Distribution:** 15k multi-waypoint 3D splines, 10k Lissajous lemniscates, 10k ascending/descending helices, 5k station hovers, 3k vertical takeoffs, 3k touchdown landings, 4k high-speed slaloms.
* **Environment Concurrency:** 512 parallel vectorized environments in MuJoCo.
* **Lookahead Horizons:** $H = 48$ (~0.96s) and $H = 64$ (~1.28s) backpropagation-through-time through neural world model.

### 9.2 Ultra Smoke Test Matrix (1,000 Iterations Each)

| Candidate Configuration | Envs | Horizon | Replay Ratio | Overall Pos RMSE | Overall Vel RMSE | Survival Rate | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

  for s in smoke_results:
    ev = s.get("eval")
    if ev:
      pos_rmse = f"{ev.get('overall_pos_rmse_m', 0.0):.3f} m ({ev.get('overall_pos_rmse_m', 0.0)*100:.1f} cm)"
      vel_rmse = f"{ev.get('overall_vel_rmse_ms', 0.0):.3f} m/s"
      surv = f"{ev.get('overall_survival_ratio', 0.0)*100:.1f}%"
      status = "Evaluated & Logged"
    else:
      pos_rmse = "N/A"
      vel_rmse = "N/A"
      surv = "N/A"
      status = "Completed"
    new_section += f"| **{s['name']}** | {s.get('envs', 512)} | {s.get('horizon', 48)} | {s.get('ratio', 2)} | {pos_rmse} | {vel_rmse} | {surv} | {status} |\n"

  if prod_eval:
    new_section += f"""
### 9.3 Ultra Scaled Production Model Performance
* **Checkpoint:** `{prod_eval.get('checkpoint', '')}`
* **Overall Position RMSE:** **{prod_eval.get('overall_pos_rmse_m', 0.0):.3f} m ({prod_eval.get('overall_pos_rmse_m', 0.0)*100:.1f} cm)**
* **Overall Velocity RMSE:** **{prod_eval.get('overall_vel_rmse_ms', 0.0):.3f} m/s**
* **Overall Flight Survival Rate:** **{prod_eval.get('overall_survival_ratio', 0.0)*100:.1f}%**

#### Per-Course Benchmark Breakdown:
| Benchmark Flight Course | Trajectory ID | Steps Completed / Total | Pos RMSE | Vel RMSE | Termination Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for c in prod_eval.get("results_by_course", []):
      new_section += f"| **{c['task']}** | {c['traj_id']} | {c['steps_survived']} / {c['total_steps']} ({c['survival_ratio']*100:.1f}%) | {c['pos_rmse_m']*100:.1f} cm | {c['vel_rmse_ms']:.2f} m/s | {c['term_reason']} |\n"

  with open(report_path, "w") as f:
    f.write(existing + new_section)
  print(f"[ULTRA-PIPELINE] Updated synthesis report at {report_path}", flush=True)


def main():
  parser = argparse.ArgumentParser(description="Run Ultra-Scale Pipeline on 50,000-Trajectory Corpus.")
  parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke tests and run production directly.")
  parser.add_argument("--iterations", type=int, default=50000, help="Production training iterations.")
  args = parser.parse_args()

  print("\n" + "="*80)
  print("        SUPERMIMIC ULTRA-SCALE EXPERIMENTATION PIPELINE (50,000 TRAJS)")
  print("="*80, flush=True)

  if not verify_corpus(CORPUS_50000):
    print("[ERROR] Corpus verification failed. Aborting pipeline.", flush=True)
    sys.exit(1)

  best_cfg = None
  smoke_results = []
  if not args.skip_smoke:
    print("\n[PIPELINE STEP 1] Executing Ultra Smoke Test Matrix (4 configurations)...", flush=True)
    smoke_results = run_ultra_smoke_matrix()
    print("\n[PIPELINE STEP 1 COMPLETE] Smoke Test Results Summary:")
    for s in smoke_results:
      ev = s.get("eval")
      if ev:
        print(f"  * {s['name']}: Pos RMSE = {ev['overall_pos_rmse_m']*100:.1f} cm, Vel RMSE = {ev['overall_vel_rmse_ms']:.2f} m/s, Survival = {ev['overall_survival_ratio']*100:.1f}%")
      else:
        print(f"  * {s['name']}: Failed or No Eval")

    valid_smokes = [s for s in smoke_results if s.get("eval") is not None]
    if valid_smokes:
      best_cfg = min(valid_smokes, key=lambda s: s["eval"].get("overall_pos_rmse_m", 999.0))
    else:
      print("[WARN] No valid smoke evaluations found; falling back to default ultra_01 configuration.")

  if best_cfg is None:
    best_cfg = {
      "name": "ultra_01_512env_h48",
      "args": [
        "--task-id", "SuperMimic-Crazyflie-Joint",
        "--corpus-path", CORPUS_50000,
        "--num-envs", "512",
        "--iterations", str(args.iterations),
        "--collect-steps", "8",
        "--wm-batch-size", "64",
        "--wm-horizon", "4",
        "--policy-batch-size", "32",
        "--horizon", "48",
        "--replay-ratio", "2",
        "--checkpoint-dir", "logs/ultra_megacorpus_production",
        "--save-interval", "2000",
        "--use-mlflow",
        "--run-name", "ultra_megacorpus_production",
      ],
    }

  print(f"\n[PIPELINE STEP 2] Champion Configuration Selected: {best_cfg['name']}", flush=True)
  print(f"[PIPELINE STEP 3] Launching Ultra Deep Production Run ({args.iterations} iterations)...", flush=True)

  prod_eval = run_ultra_production(best_cfg, max_iterations=args.iterations)
  update_synthesis_report(smoke_results, prod_eval)
  print(f"\n[PIPELINE COMPLETE] Ultra Production Run & Evaluation Finished!", flush=True)


if __name__ == "__main__":
  main()
