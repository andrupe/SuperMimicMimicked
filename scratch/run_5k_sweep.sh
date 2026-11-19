#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=glfw
export TORCH_COMPILE_DISABLE=1
PY=./supermimic/.venv/bin/python

run_exp() {
  local name="$1"
  local hist="$2"
  local spin="$3"
  local tilt="$4"
  echo "=== [$(date -u +%H:%M:%SZ)] Starting ${name} (hist=${hist}, spin=${spin}, tilt=${tilt}) ==="
  $PY supermimic/src/supermimic/scripts/train_crazyflie.py \
    --task-id SuperMimic-Crazyflie-Joint \
    --num-envs 2048 \
    --iterations 5000 \
    --collect-steps 8 \
    --replay-ratio 12 \
    --wm-batch-size 512 \
    --wm-horizon 8 \
    --wm-lr 4e-4 \
    --policy-batch-size 512 \
    --policy-num-batches 1 \
    --policy-lr 3e-4 \
    --horizon 48 \
    --history-length "${hist}" \
    --roll-pitch-spin-weight "${spin}" \
    --tilt-margin-weight "${tilt}" \
    --dynamics-hidden-dims 1024 1024 1024 \
    --actor-hidden-dims 128 128 64 \
    --critic-hidden-dims 256 256 128 \
    --corpus-ratio 1.0 \
    --seed-steps 80 \
    --save-interval 5000 \
    --checkpoint-dir "logs/crazyflie_${name}" \
    --use-mlflow \
    --run-name "crazyflie_${name}" \
    --mlflow-log-freq 25

  echo "=== [$(date -u +%H:%M:%SZ)] Evaluating ${name} ==="
  $PY scratch/eval_checkpoint.py \
    --ckpt "logs/crazyflie_${name}/checkpoint_005000.pt" \
    --name "${name}" \
    --out-json "logs/eval_${name}.json" \
    --gif-dir "gif/${name}" \
    --trajs-per-family 10 \
    --dr-gifs-only
}

run_exp "exp13_5k" 1 -0.012 0.0
run_exp "exp14_5k" 1 -0.008 2.0
run_exp "exp15_5k" 4 -0.012 1.5
run_exp "exp16_5k" 8 -0.008 2.0

echo "=== [$(date -u +%H:%M:%SZ)] All 4 x 5k sweep experiments & evals complete! ==="
