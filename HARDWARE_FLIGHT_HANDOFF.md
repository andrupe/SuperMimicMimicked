# Hardware Flight Test & Sim-to-Real Handoff (2026-11-19)

This document summarizes everything discovered during Bitcraze Crazyflie 2.1 hardware bring-up, firmware flashing, Lighthouse calibration, flight telemetry analysis, and the exact sim-to-real fixes needed before retraining the next policy.

---

## 1. Hardware & Firmware State

- **Drone Radio URI:** `radio://0/60/2M/E7E7E7E706` (`channel=60`, `speed=2M`, `address=0xE7E7E7E706`, verified via I2C EEPROM & `cflib`).
- **Decks Installed:**
  - Lighthouse Deck (`CONFIG_DECK_LIGHTHOUSE=y`, `MAX_N_BS=4`, `MAX_RATE=50`)
  - Flow Deck (`CONFIG_DECK_FLOW=y`, `CONFIG_DECK_ZRANGER=y`, `CONFIG_DECK_ZRANGER2=y`)
- **Upgraded Thrusters:**
  - Switched to higher-thrust motors/props; firmware built with `CONFIG_CRAZYFLIE_THRUST_UPGRADE_KIT=y` (and `CONFIG_CRAZYFLIE_21_PLUS` disabled) and `CONFIG_ENABLE_THRUST_BAT_COMPENSATED=y`.
- **OOT Controller Slot:**
  - In `crazyflie-firmware` `2026.08+` (`src/modules/src/controller/controller.c`), slot `5` is `Lee` and slot **`6`** is `OutOfTree` (`my_controller.c`).
  - `crazyflie_nn/firmware/my_controller.c` is currently configured for **50 Hz** (`RATE_DO_EXECUTE(RATE_50_HZ, tick)`, `SM_HIST_STRIDE=1`).

---

## 2. Tooling & Repo Changes Made

- **`CRAZYFLIE_DEPLOY.md`**:
  - Added `--shallow-submodules` and `GIT_LFS_SKIP_SMUDGE=1` to avoid `CMSIS_5` HTTP/2 timeouts and LFS stalls.
  - Documented macOS `gmake` + Homebrew `coreutils`/`gnu-sed` `PATH` requirement for `merge_config.sh`, `build/.config` verification, slot `6` (`OutOfTree`), and `.venv_cf` activation.
- **`scratch/fly_trajectory.py`**:
  - Switched OOT controller slot from `5` → `6`.
  - Fixed `send_full_state_setpoint` yaw-rate units: `crtp_commander_generic.c` (`fullStateDecoder`) expects **`millirad/s`** (so `cflib` expects **`rad/s`**, not `deg/s`, despite its docstring). Passing `deg/s` caused `int16_t` overflow (`struct.error`) and 57.3× scale error.
  - Added `commander.enHighLevel = 1` and `supervisor.send_arming_request(True)` before `takeoff`.
  - Added `--pid-only`, `--hover-here`, and automatic 50 Hz CSV telemetry logging to `logs/flights/`.
- **`scratch/check_lighthouse.py`**:
  - Static (no-motor) Lighthouse & Kalman jitter check (`lighthouse.status`, `bsAvailable/Active/Calib/Receive`, `stateEstimate.{x,y,z}` std).
- **`scratch/calibrate_lighthouse.py`**:
  - Voice-guided (`say`) hands-free Lighthouse geometry calibration (`Origin` → `+X` 1 m → `XY` plane → 12 s walk-around), calling `augment_with_ippe()` and `LighthouseConfigWriter(cf, nr_of_base_stations=4)`.

---

## 3. Flight Test Results & Telemetry Diagnosis

### Baseline: Lighthouse + Flow + `THRUST_UPGRADE_KIT` under Stock PID (`1`)
- After recalibrating Lighthouse (`logs/flights/20261119_172206_hover_only.csv`), static estimator jitter dropped from ~30 cm to:
  - `est_x std = 0.0006 m`, `est_y std = 0.0005 m`, `est_z std = 0.0012 m`, `lh_status = 2`.
- Stock PID (`1`) takeoff and hover (`logs/flights/20261119_173018_figure8.csv` and `20261119_173212_hover_only.csv`, `t = 0–5.3 s`) is **rock solid**:
  - `roll ∈ [-0.7°, +1.1°]`, `pitch ∈ [-0.6°, +0.7°]`, smooth climb to `0.8–1.0 m`.

### Failure Mode: Switching to SuperMimic NN (`stabilizer.controller = 6`)
In `logs/flights/20261119_173212_hover_only.csv` (`--hover-only --hover-here`, `t = 5.3 s` onward):
1. **First ~2 s (`t = 5.3–7.3 s`):** Position stayed within ~3 cm of `(-0.16, -0.20, 0.80)`, **but `yaw` immediately spun linearly at ~300 deg/s** (`0° → +180° → -180° → +180°`, ~3 full rotations in 3.5 s).
2. **Growing Attitude Oscillation:** `roll` and `pitch` started as `±5°` ripples and grew exponentially to `±75°` (`roll ∈ [-71.7°, +50.5°]`, `pitch ∈ [-76.6°, +70.2°]`).
3. **Severe Dynamic Battery Sag:** `vbat` started at `3.88 V` at rest, dropped to `3.21 V` on takeoff, and sagged to **`2.99–3.00 V`** dynamically under NN maneuvers.

---

## 4. Root Causes (Sim-to-Real Mismatches) & Retraining Plan

Cross-checking the logs against `crazyflie.xml`, `power_distribution_quadrotor.c`, `crazyflie_env_cfgs.py`, `crazyflie_mdp.py`, and `my_controller.c`:

1. **Yaw Mixing / Rotor Spin Sign Bug (`crazyflie.xml` vs. Real Crazyflie 2.1):**
   - `mjlab_research/src/mjlab/asset_zoo/robots/bitcraze_crazyflie/xmls/crazyflie.xml` defines yaw gears left-vs-right (`FL: -`, `FR: +`, `RL: -`, `RR: +`) instead of diagonal CW/CCW.
   - Real Crazyflie (`power_distribution_quadrotor.c`: `motorForces[0..3]`) uses diagonal CW/CCW (`M4/FL: +`, `M1/FR: -`, `M3/RL: -`, `M2/RR: +`).
   - **Action:** Fix `crazyflie.xml` actuator yaw gears to diagonal CW/CCW (`FL: +0.005956`, `FR: -0.005956`, `RL: -0.005956`, `RR: +0.005956`) and update `my_controller.c` `tz = SM_KM_OVER_KF * (f0 - f1 - f2 + f3)`.

2. **DR Too Narrow for Upgraded Thrusters + Decks (`crazyflie_env_cfgs.py`, `crazyflie_mdp.py`):**
   - Currently `50%` of envs have zero DR, `body_mass` is only `(0.96, 1.04)` around `33 g` (real CF2.1 + Lighthouse + Flow + battery is ~36–42 g), and `thrust_scale_range` is only `(0.96, 1.02)` (whereas `THRUST_UPGRADE_KIT` has much higher thrust/torque gain, exciting the growing `±5° → ±75°` oscillation).
   - **Action:** Apply DR to 100% of envs (or curriculum) with widened ranges:
     - `body_mass` scale: `(0.90, 1.35)`
     - `thrust_scale_range`: `(0.75, 1.50)`
     - `inertia` scale: `(0.80, 1.40)`, `CoM` offset: `±3 mm`
     - `rotor_gain_range`: `(0.94, 1.06)`, `tau_up/down`: `(0.015, 0.045) s`

3. **Dynamic Thrust-Dependent Battery Sag:**
   - Sim only models a tiny linear `sag_rate` (`0–0.0025 / s`), whereas real `vbat` sags `~17%` on takeoff and swings `3.00–3.78 V` with instantaneous thrust.
   - **Action:** Model instantaneous thrust-dependent sag in `CrazyflieMotorAction` (`scale *= clip(1 - k_sag * sum(thrust), 0.70, 1.0)`) + randomized initial SoC.

4. **Control/Estimator Latency & Action-Rate Regularization:**
   - Sim has 0 ms latency; hardware has Kalman + sensor + motor-step delay and showed growing high-frequency roll/pitch oscillations.
   - **Action:** Add `1–2` step (`5–20 ms`) obs/action delay randomization and stronger action-rate / body-rate regularization.

5. **Remove Absolute World `(x, y)` from Policy Observation:**
   - `my_controller.c` (`frame[0..2]`) currently feeds raw `state->position.{x, y, z}` even though `frame[16..18]` already provides body-frame `pos_err_b`.
   - **Action:** Drop world `(x, y)` from the observation vector (keep `z` or `pos_err_b` only) so hover/trajectories are translation-invariant across the arena.
