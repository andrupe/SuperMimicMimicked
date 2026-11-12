# Workspace environment (uv, CUDA 13)

Primary project: **`crazyflie_nn`** — quadrotor simulation and control with brax/JAX.
Official **`supermimic`** (built on `mjlab` / MuJoCo-Warp) is integrated in `supermimic/`.

> **Rollback Note**: The legacy `supermimic_vibed` (Isaac Lab 3) is preserved on `main` and tagged as `rollback-to-supermimic-vibed`. To revert, simply run `git checkout rollback-to-supermimic-vibed`.

Target machine: **RTX 2050 (4 GB, sm_86)**, driver 595.91, CUDA 13 capable, 8 cores, 13 GB RAM.
No system CUDA toolkit / `nvcc` is needed — the CUDA runtime comes from the wheels.

## Files

- `pyproject.toml` — dependencies, indexes, overrides, and the optional `isaac` group
- `uv.lock` — pinned graph (default install ≈ 60 packages)
- `.python-version` — `3.12` (brax needs ≥3.11; Isaac Lab needs `<3.13,>=3.12`)

## Setup

```bash
uv sync                  # default: brax/JAX/CUDA + dev tools   (~2 min, 3.9 GB venv)
uv sync --no-dev         # runtime only
```

`uv` lives in `~/.local/bin` — `source ~/.local/bin/env` once per shell.
All commands below work as `uv run <cmd>` without activating the venv.

## Official SuperMimic Stack (`supermimic/` & `mjlab_research/`)

Built on **`mjlab`** (MuJoCo-Warp native, standalone Isaac Lab API, Python 3.13) — no 21 GB Isaac Sim dependency!

```bash
cd supermimic
uv sync
uv run pytest -q                 # verified: 37/37 passing
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --help
```

### Hardware Support
- **Mac (Apple Silicon)**: CPU execution is verified and working for testing, evaluation, and visualization. Mac GPU (`mps`) is **not supported** by NVIDIA Warp (`ValueError: Invalid device identifier: mps`).
- **NVIDIA GPU (Linux)**: Full GPU acceleration for world-model training and BPTT policy learning.

## Verified stack (default group)

| Package | Version | Notes |
|---|---|---|
| `jax` / `jaxlib` / `jax-cuda13-plugin` | 0.12.0.1 | `jax.devices()` → `[CudaDevice(id=0)]` |
| `brax` | 0.14.2 | classic API (`brax.envs.State`, `brax.training.agents.ppo`) still present |
| `flax`, `optax`, `orbax-checkpoint` | current | pulled by brax |
| `mujoco`, `mujoco-mjx` | 3.12.0 | MJX is imported as `from mujoco import mjx` (not `mujoco_mjx`) |
| `numpy` | 2.3.x | |
| `ssqp` | 0.1.2 | module `ssqpy`; C++ SQP/MPC solver, prebuilt abi3 wheel |

`ssqp` ships prebuilt Linux x86_64 wheels that require **AVX2 + FMA** (Haswell / Excavator or
newer). This machine reports both. Verify with:

```bash
uv run python ssqp_smoke_test.py    # double-integrator quickstart + assertions
```

Smoke-tested on the RTX 2050:

```bash
uv run python -c "import jax; print(jax.__version__, jax.devices(), jax.default_backend())"

cd crazyflie_nn && uv run python -c "
import jax
from environment.env import DroneBrax
env = DroneBrax(); s = jax.jit(env.reset)(jax.random.PRNGKey(0))
s = jax.jit(env.step)(s, jax.numpy.zeros(env.action_size))
print('obs', s.obs.shape, 'device', s.obs.devices())"
```

## Using it

```bash
cd crazyflie_nn && uv run python trainPPO.py          # imports are relative to that folder
cd crazyflie_nn && uv run python tools/extract_nn.py --path logs/params/... --out exported_model
uv run pytest crazyflie_nn -q
```

## Notes and gotchas

1. **`Failed to import warp: No module named 'warp'`** — printed by brax when it looks for the optional
   `mjx-warp` backend. Harmless: brax falls back to standard MJX. Install `warp-lang` only if you
   actually want the warp backend.
2. **`torch` / `isaaclab` are absent by default** — that is intentional for the quadrotor work.
   They arrive with `uv sync --group isaac`.
3. **`ulimit -v`** — never run under a virtual-memory cap: Warp/Isaac reserve hundreds of GB of
   *virtual* address space (CUDA VA reservations) even though RSS stays small.
4. **4 GB VRAM** — brax/MJX batch sizes are the thing to watch; scale `num_envs` (or the MJX batch
   dimension) rather than assuming the paper-scale settings fit.
5. **Overrides required by the `isaac` group** (kept so the lock stays resolvable):
   - `urllib3>=2.8.0`: `isaacsim-kernel 6.2.0.0` pins `==2.7.0`, `isaaclab 3.0.0` requires `>=2.8.0`
   - `index-strategy = "unsafe-best-match"` + `https://pypi.nvidia.com`: Isaac Sim pins
     `opencv-python-headless-noffmpeg==4.14.0.94rc3`, which exists only on NVIDIA's index
   - `isaaclab[importers]` is **not** included: it forces `usd-exchange==2.3.0` while `isaaclab`
     requires `3.0.1`. For `supermimic_vibed/tools/mjcf_to_usd.py`, use a throwaway env:
     `uvx --from isaacsim-asset-isolated==6.2.0.1 python supermimic_vibed/tools/mjcf_to_usd.py`
