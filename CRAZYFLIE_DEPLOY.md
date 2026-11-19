# Crazyflie 2.1 Deployment Guide (`exp25_50k`, 100 Hz `stride=2`)

This guide walks through building, flashing, and flying the exported **`exp25_50k`** policy (`92 → 128 → 128 → 64 → 4`, `146.8 KiB` FP32 Flash, `~0.6–1.2 ms` per forward pass) on a Bitcraze Crazyflie 2.1 (STM32F405).

## Included Files
- **`crazyflie_nn/firmware/supermimic_policy.h`** — Exported `exp25_50k` weights + exact `erff` GELU forward pass (`obs_dim=92`, `hidden=(128, 128, 64)`).
- **`crazyflie_nn/firmware/my_controller.c`** — Out-Of-Tree (OOT) firmware controller running at **`100 Hz`** (`RATE_100_HZ`) with a 7-slot `stride=2` (`80 ms`) history ring buffer, exact `crazyflie.xml` `(FL, FR, RL, RR)` torque mixing, quaternion normalization, and NaN/thrust safety clamps.
- **`scratch/fly_trajectory.py`** — Host PC `cflib` trajectory streamer (takes off under stock PID `1`, hands off to NN `5` in hover, streams `50 Hz` setpoints, holds hover at the end, and switches back to PID `1` to land; `Ctrl+C` triggers immediate PID fallback + landing).
- **`corpus/sample_flight_trajs.npz`** — Lightweight 170-trajectory clean-yaw sample corpus covering all 17 flight families.

---

## Step 1 — Install Prerequisites

### On macOS (Apple Silicon or Intel)
```bash
# 1. Install ARM cross-compiler + GNU make
brew install --cask gcc-arm-embedded
brew install make git

# 2. Install Python flight client (inside your preferred venv)
python3 -m venv .venv_cf && source .venv_cf/bin/activate
pip install numpy cflib cfclient
```

### On Ubuntu / Debian Linux
```bash
sudo apt-get update && sudo apt-get install -y gcc-arm-none-eabi make git
pip install numpy cflib cfclient
```

---

## Step 2 — Set Up the Crazyflie Out-Of-Tree (OOT) App

Clone `crazyflie-firmware` next to `SuperMimicMimicked` and copy the controller files into a new OOT app:

```bash
# Adjust REPO_DIR if you cloned SuperMimicMimicked somewhere else
export REPO_DIR="$PWD"

git clone --recursive https://github.com/bitcraze/crazyflie-firmware.git ../crazyflie-firmware
cd ../crazyflie-firmware
cp -r examples/app_hello_world examples/supermimic_rl
cd examples/supermimic_rl

# Copy exported exp25_50k weights and 100 Hz OOT controller
rm -f src/hello_world.c
cp "$REPO_DIR/crazyflie_nn/firmware/my_controller.c" src/
cp "$REPO_DIR/crazyflie_nn/firmware/supermimic_policy.h" src/

# Tell Kbuild to compile my_controller.c
echo "obj-y += my_controller.o" > src/Kbuild

# Enable OOT controller in app-config
cat << 'EOF' > app-config
CONFIG_APP_ENABLE=y
CONFIG_APP_PRIORITY=1
CONFIG_APP_STACKSIZE=512
CONFIG_CONTROLLER_OOT=y
EOF
```

---

## Step 3 — Build and Flash to the Crazyflie 2.1

1. Power off the Crazyflie, then **hold the power button for ~3 seconds** until both blue LEDs blink (bootloader mode).
2. Plug in your **Crazyradio PA** and run:
   ```bash
   # On macOS (use gmake if system make is old, or standard make)
   make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
   make cload
   ```
3. Wait for `Restarting`. The Crazyflie boots safely into stock **PID (`stabilizer.controller = 1`)**, with your NN controller loaded in slot **`5` (`stabilizer.controller = 5`)**.

---

## Step 4 — Pre-Flight Dry Run & Positioning Check

1. Ensure your positioning system (Lighthouse, OptiTrack/Vicon, or Flow deck) is active and the Crazyflie is placed near `(0, 0, 0)` with clear flight volume.
2. Find your Crazyflie radio URI in `cfclient` (default: `radio://0/80/2M/E7E7E7E7E7`).
3. Always check trajectory bounds with `--dry-run` before flying:
   ```bash
   cd "$REPO_DIR"
   python3 scratch/fly_trajectory.py --dry-run --family figure8 --scale 0.5
   ```

---

## Step 5 — Progressive Flight Tests

Press **`Ctrl+C`** at any time during flight to immediately switch `stabilizer.controller` back to **`1` (PID)** and land.

### Test 1 — 5-Second Hover Handoff (Run this first!)
Takes off to `0.8 m` under PID (`1`), switches to the `100 Hz` NN (`5`) for `5 s`, then switches back to PID (`1`) and lands:
```bash
python3 scratch/fly_trajectory.py \
  --hover-only --hover-z 0.8 --duration 5.0 \
  --uri radio://0/80/2M/E7E7E7E7E7
```

### Test 2 — Gentle Half-Scale Maneuvers (`--scale 0.5`)
```bash
python3 scratch/fly_trajectory.py \
  --family point_to_point --scale 0.5 --hover-z 0.8 \
  --uri radio://0/80/2M/E7E7E7E7E7

python3 scratch/fly_trajectory.py \
  --family figure8 --scale 0.5 --hover-z 1.0 \
  --uri radio://0/80/2M/E7E7E7E7E7
```

### Test 3 — Full-Scale Trajectories (`--scale 1.0`)
Available families in `corpus/sample_flight_trajs.npz`:
`hover`, `point_to_point`, `figure8`, `circular_orbit`, `helix`, `slalom`, `polygon_patrol`, `conical_spiral`, `rose_curve`, `crabbed_strafe`, `step_dash_brake`, `vertical_sprint`, `split_s_dive`, `recovery_stabilize`, `perch_approach`, `takeoff`, `landing`.

```bash
python3 scratch/fly_trajectory.py \
  --family figure8 --scale 1.0 --hover-z 1.2 \
  --uri radio://0/80/2M/E7E7E7E7E7
```
