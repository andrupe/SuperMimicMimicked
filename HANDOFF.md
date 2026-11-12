# SuperMimic Crazyflie Handoff

**Branch:** `feature/official-supermimic` (`https://github.com/andrupe/SuperMimicMimicked.git`)  
**Status:** `supermimic/` and `mjlab_research/` are vendored directly in this repo (no submodules required).

---

## 1. Current State & Latest Results

### What Was Fixed
- Calibrated `SiteEffortActionCfg` in `supermimic/src/supermimic/tasks/crazyflie_env_cfgs.py` (`scale=0.0732675`, `offset=0.0809325`, `clip={"rotor_.*": (0.0, 0.1542)}`) so `action = 0.0` maps to exact hover thrust (`0.08093 N` per rotor) instead of `0 N` free-fall.
- Added `SuperMimic-Crazyflie-Corpus` (`CorpusTrajectoryCommand`, RSI, corpus tracking rewards) and exported `corpus/crazyflie_corpus_50trajs.npz`.
- Added replay-buffer seeding from the expert corpus in `supermimic/src/supermimic/runners/crazyflie_runner.py` and differentiable altitude safety reward (`r_alt`) in `supermimic/src/supermimic/envs/crazyflie_world_model_env.py`.

### Latest 2500-Iteration Run (`cpu`, 64 envs, `OMP_NUM_THREADS=8`)
- **Duration:** `685.9s` (`0.27s/iter`)
- **Final metrics (`Iter 2500/2500`):** `Sim R: +0.040 | WM Loss: 1.3927 | BPTT R: +1.365`
- **Rollout evaluation (`logs/crazyflie_supermimic/checkpoint_002500.pt`):**
  - **Figure-8 (`traj_id=5`):** `49/202` steps (terminated on `tilt`), `z=[1.179, 1.506] m`, `mean_pos_err=0.533 m`, `mean_vel_err=1.750 m/s`, `mean_act=[-0.058, 0.121, -0.096, 0.252]`
  - **Helix (`traj_id=3`):** `54/225` steps (terminated on `tilt`), `z=[0.868, 1.455] m`, `mean_pos_err=0.225 m`, `mean_vel_err=0.747 m/s`, `mean_act=[-0.045, 0.152, 0.027, 0.210]`

The quadcopter no longer falls to the ground, **but still develops a persistent roll/pitch bias and hits the `tilt` termination around ~1s (step ~50)**.

---

## 2. Setup & Run Commands (e.g., on Ubuntu Server)

### Clone & Environment Setup
```bash
git clone -b feature/official-supermimic https://github.com/andrupe/SuperMimicMimicked.git
cd SuperMimicMimicked

python3 -m venv .venv
source .venv/bin/activate
pip install -e ./mjlab_research -e ./supermimic
```

### Run Training
Use `--device cuda` if running on a machine with an NVIDIA GPU (or `--device cpu` with `OMP_NUM_THREADS` set):
```bash
export OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/matplotlib
python3 supermimic/src/supermimic/scripts/train_crazyflie.py \
  --task-id SuperMimic-Crazyflie-Corpus \
  --num-envs 64 \
  --wm-batch-size 256 \
  --policy-batch-size 128 \
  --iterations 2500 \
  --collect-steps 8 \
  --wm-horizon 4 \
  --horizon 32 \
  --seed-steps 50 \
  --save-interval 100 \
  --device cpu
```

### Evaluate & Render GIFs
Requires a working offscreen MuJoCo GL context (on headless Linux, set `MUJOCO_GL=egl` or `osmesa`):
```bash
export MPLCONFIGDIR=/tmp/matplotlib
python3 scratch/render_corpus_trajectories.py
```

---

## 3. What To Do Next (If More Work Is Needed)

To fix the remaining ~step-50 `tilt` termination and get full-horizon Figure-8 / Helix tracking:

1. **Inspect `Sim R` (`+0.040`) vs `BPTT R` (`+1.365`) mismatch:**
   - The policy scores high in world-model BPTT rollouts (`+1.365`) while real MuJoCo rewards stay low (`+0.040`), indicating world-model compounding error or reward mismatch between `crazyflie_world_model_env.py` and `crazyflie_env_cfgs.py`.
   - Check multi-step world-model rollout accuracy on `projected_gravity` and `body_ang_vel` over 32 steps.
2. **Check `base_com` domain randomization (`crazyflie_env_cfgs.py`):**
   - `EventManager` currently randomizes `base_com` at startup. Because `QuadcopterPolicy` has no integral/history state beyond the current 17-dim observation, a randomized CoM offset forces an asymmetric hover trim that can drift into `tilt` unless CoM randomization is disabled during initial validation or matched with an integrator/history.
3. **Check `track_ori` / `upright` / attitude stabilization:**
   - Verify `quadcopter_corpus_ori_tracking_reward` in `supermimic/src/supermimic/tasks/crazyflie_mdp.py` and the differentiable orientation/upright terms in `supermimic/src/supermimic/envs/crazyflie_world_model_env.py` are aligned and weighted strongly enough to prevent slow roll/pitch divergence.
4. **Tune rollout horizon & training schedule:**
   - Once `Sim R` climbs steadily alongside `BPTT R`, increase `--iterations` (or `--num-envs` on GPU) and re-run `scratch/render_corpus_trajectories.py` to verify full 202-step (Figure-8) and 225-step (Helix) survival.
