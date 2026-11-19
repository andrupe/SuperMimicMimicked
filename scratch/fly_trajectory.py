"""Option A host-PC trajectory streamer for Crazyflie 2.1 (cflib over Crazyradio).

Streams target setpoints (pos, vel, yaw_rate) at 50 Hz while the onboard OOT
controller (my_controller.c) runs exp25_50k at 100 Hz (stride=2).
Includes --hover-only, --relative origin shifting, --scale, and Ctrl+C PID fallback.
"""

from __future__ import annotations

import argparse
import time
import numpy as np


def load_corpus_trajectory(corpus_path: str, family: str | None, traj_id: int | None):
  data = np.load(corpus_path, allow_pickle=True)
  states = data["states"]
  s_off = data["state_offsets"]
  c_off = data["ctrl_offsets"]
  tasks = data["tasks"]

  if traj_id is None:
    if family is None:
      family = "figure8"
    matches = [i for i, t in enumerate(tasks) if str(t) == family]
    if not matches:
      raise ValueError(f"No trajectories found for family '{family}'. Available: {sorted(set(map(str, tasks)))}")
    traj_id = int(matches[0])

  s = states[s_off[traj_id] : s_off[traj_id + 1], :13]
  H = int(c_off[traj_id + 1] - c_off[traj_id])
  pos = s[:H, 0:3].copy()
  quat = s[:H, 3:7].copy()  # w, x, y, z
  vb = s[:H, 7:10].copy()
  wb = s[:H, 10:13].copy()

  # Rotate body linear velocity to world frame: R(q) * vb
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  vw = np.empty_like(vb)
  vw[:, 0] = (1 - 2 * (y * y + z * z)) * vb[:, 0] + 2 * (x * y - w * z) * vb[:, 1] + 2 * (x * z + w * y) * vb[:, 2]
  vw[:, 1] = 2 * (x * y + w * z) * vb[:, 0] + (1 - 2 * (x * x + z * z)) * vb[:, 1] + 2 * (y * z - w * x) * vb[:, 2]
  vw[:, 2] = 2 * (x * z - w * y) * vb[:, 0] + 2 * (y * z + w * x) * vb[:, 1] + (1 - 2 * (x * x + y * y)) * vb[:, 2]

  yaw_rate_deg = np.degrees(wb[:, 2])
  return traj_id, str(tasks[traj_id]), pos, vw, quat, yaw_rate_deg


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E7")
  p.add_argument("--corpus", type=str, default="corpus/sample_flight_trajs.npz")
  p.add_argument("--family", type=str, default="figure8", help="Trajectory family to fly (if --traj-id unset).")
  p.add_argument("--traj-id", type=int, default=None, help="Explicit trajectory index in corpus.")
  p.add_argument("--hover-only", action="store_true", help="Safe first test: hover at (0, 0, --hover-z) under NN.")
  p.add_argument("--hover-z", type=float, default=0.8, help="Hover height in meters for takeoff / --relative start.")
  p.add_argument("--duration", type=float, default=5.0, help="Duration in seconds for --hover-only.")
  p.add_argument("--relative", action="store_true", default=True, help="Shift trajectory so pos[0] = (0, 0, --hover-z).")
  p.add_argument("--no-relative", dest="relative", action="store_false")
  p.add_argument("--scale", type=float, default=1.0, help="Scale trajectory amplitude & velocity (e.g. 0.5 for gentle test).")
  p.add_argument("--rate", type=float, default=50.0, help="Setpoint stream rate in Hz (onboard NN runs at 100 Hz).")
  p.add_argument("--hover-hold-s", type=float, default=2.0, help="Seconds to hold final hover pose before landing.")
  p.add_argument("--dry-run", action="store_true", help="Print trajectory bounds without connecting to Crazyradio.")
  args = p.parse_args()

  dt = 1.0 / args.rate
  if args.hover_only:
    n = int(args.duration * args.rate)
    pos = np.tile(np.array([[0.0, 0.0, args.hover_z]], dtype=np.float32), (n, 1))
    vw = np.zeros((n, 3), dtype=np.float32)
    quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))
    yaw_rate_deg = np.zeros(n, dtype=np.float32)
    tid, fam = -1, "hover_only"
  else:
    tid, fam, pos, vw, quat, yaw_rate_deg = load_corpus_trajectory(args.corpus, args.family, args.traj_id)
    if args.scale != 1.0:
      pos = pos[0:1] + (pos - pos[0:1]) * args.scale
      vw = vw * args.scale
      yaw_rate_deg = yaw_rate_deg * args.scale
    if args.relative:
      anchor = np.array([0.0, 0.0, args.hover_z], dtype=np.float32)
      pos = pos - pos[0:1] + anchor

  print(
    f"[TRAJ] id={tid} family={fam} steps={len(pos)} dur={len(pos)*dt:.2f}s "
    f"start={pos[0].round(3)} end={pos[-1].round(3)} "
    f"bounds_min={pos.min(axis=0).round(2)} bounds_max={pos.max(axis=0).round(2)}"
  )
  print(f"[STATS] Peak |vel|={np.linalg.norm(vw, axis=-1).max():.2f} m/s, Peak |yaw_rate|={np.abs(yaw_rate_deg).max():.1f} deg/s")
  if args.dry_run:
    return

  import cflib.crtp
  from cflib.crazyflie import Crazyflie
  from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

  cflib.crtp.init_drivers()
  with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
    cf = scf.cf
    try:
      # 1. Take off & settle at pos[0] under stock PID (stabilizer.controller=1)
      cf.param.set_value("stabilizer.controller", "1")
      cf.high_level_commander.takeoff(float(max(0.4, pos[0, 2])), 1.5)
      time.sleep(2.0)
      cf.high_level_commander.go_to(float(pos[0, 0]), float(pos[0, 1]), float(pos[0, 2]), 0.0, 1.5)
      time.sleep(2.0)

      # 2. Switch to OOT SuperMimic NN controller (runs onboard at 100 Hz, stride=2)
      print("[FLIGHT] Switching stabilizer.controller -> 5 (SuperMimic NN 100 Hz)...")
      cf.param.set_value("stabilizer.controller", "5")
      for t in range(len(pos)):
        t0 = time.monotonic()
        cf.commander.send_full_state_setpoint(
          pos[t].tolist(), vw[t].tolist(), [0.0, 0.0, 0.0],
          [float(quat[t, 1]), float(quat[t, 2]), float(quat[t, 3]), float(quat[t, 0])],
          0.0, 0.0, float(yaw_rate_deg[t]),
        )
        rem = dt - (time.monotonic() - t0)
        if rem > 0:
          time.sleep(rem)

      # 3. Hold final pose cleanly before switching back to PID
      for _ in range(int(args.hover_hold_s * args.rate)):
        t0 = time.monotonic()
        cf.commander.send_full_state_setpoint(
          pos[-1].tolist(), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0], 0.0, 0.0, 0.0,
        )
        rem = dt - (time.monotonic() - t0)
        if rem > 0:
          time.sleep(rem)
    except KeyboardInterrupt:
      print("\n[ABORT] Ctrl+C caught — switching back to stock PID (1) and landing!")
    finally:
      cf.param.set_value("stabilizer.controller", "1")
      cf.high_level_commander.land(0.03, 1.5)
      time.sleep(2.0)
      cf.commander.send_stop_setpoint()


if __name__ == "__main__":
  main()
