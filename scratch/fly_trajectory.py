"""Option A host-PC trajectory streamer for Crazyflie 2.1 (cflib over Crazyradio).

Streams target setpoints (pos, vel, yaw_rate) at 50 Hz while the onboard OOT
controller (my_controller.c) runs exp25_50k at 100 Hz (stride=2).
Includes --hover-only, --relative origin shifting, --scale, and Ctrl+C PID fallback.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
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

  yaw_rate_rad = wb[:, 2].copy()
  return traj_id, str(tasks[traj_id]), pos, vw, quat, yaw_rate_rad


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
  p.add_argument("--pid-only", action="store_true", help="Stay on stock PID (1) the entire flight (sanity check Lighthouse + hardware).")
  p.add_argument("--hover-here", action="store_true", help="Anchor at the drone's current (x, y) before takeoff instead of (0, 0).")
  p.add_argument("--log-dir", type=str, default="logs/flights", help="Directory to save CSV flight telemetry logs.")
  p.add_argument("--no-log", action="store_true", help="Disable CSV telemetry logging.")
  p.add_argument("--dry-run", action="store_true", help="Print trajectory bounds without connecting to Crazyradio.")
  args = p.parse_args()

  dt = 1.0 / args.rate
  if args.hover_only:
    n = int(args.duration * args.rate)
    pos = np.tile(np.array([[0.0, 0.0, args.hover_z]], dtype=np.float32), (n, 1))
    vw = np.zeros((n, 3), dtype=np.float32)
    quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))
    yaw_rate_rad = np.zeros(n, dtype=np.float32)
    tid, fam = -1, "hover_only"
  else:
    tid, fam, pos, vw, quat, yaw_rate_rad = load_corpus_trajectory(args.corpus, args.family, args.traj_id)
    if args.scale != 1.0:
      pos = pos[0:1] + (pos - pos[0:1]) * args.scale
      vw = vw * args.scale
      yaw_rate_rad = yaw_rate_rad * args.scale
    if args.relative:
      anchor = np.array([0.0, 0.0, args.hover_z], dtype=np.float32)
      pos = pos - pos[0:1] + anchor

  print(
    f"[TRAJ] id={tid} family={fam} steps={len(pos)} dur={len(pos)*dt:.2f}s "
    f"start={pos[0].round(3)} end={pos[-1].round(3)} "
    f"bounds_min={pos.min(axis=0).round(2)} bounds_max={pos.max(axis=0).round(2)}"
  )
  print(f"[STATS] Peak |vel|={np.linalg.norm(vw, axis=-1).max():.2f} m/s, Peak |yaw_rate|={np.degrees(np.abs(yaw_rate_rad).max()):.1f} deg/s")
  if args.dry_run:
    return

  import cflib.crtp
  from cflib.crazyflie import Crazyflie
  from cflib.crazyflie.log import LogConfig
  from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

  cflib.crtp.init_drivers()
  with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
    cf = scf.cf
    log_file = None
    log_configs = []
    flight_state = {"phase": "init", "tx": pos[0, 0], "ty": pos[0, 1], "tz": pos[0, 2], "tvx": 0.0, "tvy": 0.0, "tvz": 0.0, "tyr": 0.0}
    latest_pos = {"x": 0.0, "y": 0.0, "z": 0.0, "ready": False}
    latest_att = {"stabilizer.roll": float("nan"), "stabilizer.pitch": float("nan"), "stabilizer.yaw": float("nan"), "pm.vbat": float("nan"), "lighthouse.status": -1}

    if not args.no_log:
      os.makedirs(args.log_dir, exist_ok=True)
      stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      log_path = os.path.join(args.log_dir, f"{stamp}_{fam}.csv")
      log_file = open(log_path, "w", newline="")
      writer = csv.writer(log_file)
      writer.writerow([
        "t_s", "phase", "lh_status", "vbat",
        "est_x", "est_y", "est_z", "est_vx", "est_vy", "est_vz",
        "roll", "pitch", "yaw",
        "target_x", "target_y", "target_z", "target_vx", "target_vy", "target_vz", "target_yaw_rate",
      ])
      t_start = time.monotonic()

      def on_pos_vel(_ts, data, _lc):
        latest_pos.update(x=data["stateEstimate.x"], y=data["stateEstimate.y"], z=data["stateEstimate.z"], ready=True)
        writer.writerow([
          f"{time.monotonic() - t_start:.4f}", flight_state["phase"],
          latest_att["lighthouse.status"], f"{latest_att['pm.vbat']:.3f}",
          f"{data['stateEstimate.x']:.4f}", f"{data['stateEstimate.y']:.4f}", f"{data['stateEstimate.z']:.4f}",
          f"{data['stateEstimate.vx']:.4f}", f"{data['stateEstimate.vy']:.4f}", f"{data['stateEstimate.vz']:.4f}",
          f"{latest_att['stabilizer.roll']:.2f}", f"{latest_att['stabilizer.pitch']:.2f}", f"{latest_att['stabilizer.yaw']:.2f}",
          f"{flight_state['tx']:.4f}", f"{flight_state['ty']:.4f}", f"{flight_state['tz']:.4f}",
          f"{flight_state['tvx']:.4f}", f"{flight_state['tvy']:.4f}", f"{flight_state['tvz']:.4f}", f"{flight_state['tyr']:.2f}",
        ])

      lg_pv = LogConfig(name="sm_pos_vel", period_in_ms=20)
      for v in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z", "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"):
        lg_pv.add_variable(v, "float")
      lg_pv.data_received_cb.add_callback(on_pos_vel)

      lg_att = LogConfig(name="sm_att_sys", period_in_ms=20)
      for v in ("stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw", "pm.vbat"):
        lg_att.add_variable(v, "float")
      lg_att.add_variable("lighthouse.status", "uint8_t")
      lg_att.data_received_cb.add_callback(lambda _ts, data, _lc: latest_att.update(data))

      for lg in (lg_pv, lg_att):
        cf.log.add_config(lg)
        lg.start()
        log_configs.append(lg)
      print(f"[LOG] Recording 50 Hz telemetry -> {log_path}")

    if args.hover_here:
      t_wait = time.monotonic()
      while not latest_pos["ready"] and (time.monotonic() - t_wait) < 1.0:
        time.sleep(0.02)
      x0, y0 = latest_pos["x"], latest_pos["y"]
      print(f"[ANCHOR] Shifting trajectory XY from ({pos[0,0]:.2f}, {pos[0,1]:.2f}) -> current ({x0:.2f}, {y0:.2f})")
      pos[:, 0] += x0 - pos[0, 0]
      pos[:, 1] += y0 - pos[0, 1]

    try:
      # 1. Take off & settle at pos[0] under stock PID (stabilizer.controller=1)
      flight_state["phase"] = "takeoff_pid"
      cf.param.set_value("commander.enHighLevel", "1")
      cf.param.set_value("stabilizer.controller", "1")
      if hasattr(cf, "supervisor"):
        cf.supervisor.send_arming_request(True)
      else:
        cf.platform.send_arming_request(True)
      cf.high_level_commander.takeoff(float(max(0.4, pos[0, 2])), 1.5)
      time.sleep(2.0)
      cf.high_level_commander.go_to(float(pos[0, 0]), float(pos[0, 1]), float(pos[0, 2]), 0.0, 1.5)
      time.sleep(2.0)

      # 2. Switch to OOT SuperMimic NN controller (slot 6 in 2026.08+ firmware, runs onboard at 50 Hz, stride=1)
      if args.pid_only:
        print("[FLIGHT] --pid-only: staying on stabilizer.controller = 1 (stock PID)...")
        flight_state["phase"] = "pid_hover"
      else:
        print("[FLIGHT] Switching stabilizer.controller -> 6 (SuperMimic NN 50 Hz)...")
        flight_state["phase"] = "nn"
        cf.param.set_value("stabilizer.controller", "6")
      for t in range(len(pos)):
        t0 = time.monotonic()
        flight_state.update(tx=float(pos[t, 0]), ty=float(pos[t, 1]), tz=float(pos[t, 2]), tvx=float(vw[t, 0]), tvy=float(vw[t, 1]), tvz=float(vw[t, 2]), tyr=float(np.degrees(yaw_rate_rad[t])))
        cf.commander.send_full_state_setpoint(
          pos[t].tolist(), vw[t].tolist(), [0.0, 0.0, 0.0],
          [float(quat[t, 1]), float(quat[t, 2]), float(quat[t, 3]), float(quat[t, 0])],
          0.0, 0.0, float(yaw_rate_rad[t]),
        )
        rem = dt - (time.monotonic() - t0)
        if rem > 0:
          time.sleep(rem)

      # 3. Hold final pose cleanly before switching back to PID
      flight_state.update(phase="hold_nn", tx=float(pos[-1, 0]), ty=float(pos[-1, 1]), tz=float(pos[-1, 2]), tvx=0.0, tvy=0.0, tvz=0.0, tyr=0.0)
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
      flight_state["phase"] = "land_pid"
      cf.param.set_value("stabilizer.controller", "1")
      cf.high_level_commander.land(0.03, 1.5)
      time.sleep(2.0)
      cf.commander.send_stop_setpoint()
      for lg in log_configs:
        try:
          lg.stop()
        except Exception:
          pass
      if log_file is not None:
        log_file.close()
        print(f"[LOG] Saved flight telemetry to {log_path}")


if __name__ == "__main__":
  main()
