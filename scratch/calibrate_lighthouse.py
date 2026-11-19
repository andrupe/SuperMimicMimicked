"""Hands-free, voice-guided Lighthouse geometry calibration over Crazyradio.

Uses macOS `say` to prompt you through Origin -> +X (1 m) -> XY plane ->
12 s walk-around sweep, solves the multi-base-station geometry with cflib,
and writes + persists it directly to the Crazyflie.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import threading
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.mem.lighthouse_memory import LighthouseBsGeometry
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.localization import (
  LhDeck4SensorPositions,
  LighthouseConfigWriter,
  LighthouseMatchedSweepAngleReader,
  LighthouseSweepAngleAverageReader,
)
from cflib.localization.lighthouse_cf_pose_sample import LhCfPoseSample
from cflib.localization.lighthouse_geo_estimation_manager import (
  LhGeoEstimationManager,
  LhGeoInputContainerData,
)


def speak(text: str, use_voice: bool = True) -> None:
  print(f"[CAL] {text}", flush=True)
  if use_voice and shutil.which("say"):
    subprocess.run(["say", text], check=False)


def countdown(seconds: int, use_voice: bool = True) -> None:
  for s in range(seconds, 0, -1):
    speak(str(s), use_voice=use_voice)
    time.sleep(0.35)


def collect_avg_sample(cf: Crazyflie, nr_samples: int = 50, timeout_s: float = 15.0) -> LhCfPoseSample:
  done = threading.Event()
  result: dict = {}

  def on_ready(angles_dict):
    result["angles"] = {bs_id: vecs for bs_id, (_cnt, vecs) in angles_dict.items()}
    done.set()

  reader = LighthouseSweepAngleAverageReader(cf, on_ready)
  reader.nr_of_samples_required = nr_samples
  reader.start_angle_collection()
  if not done.wait(timeout_s):
    raise TimeoutError("Timed out collecting averaged Lighthouse sweeps (check base station visibility).")
  return LhCfPoseSample(angles_calibrated=result["angles"])


def collect_space_samples(cf: Crazyflie, duration_s: float = 12.0, max_samples: int = 250) -> list[LhCfPoseSample]:
  samples: list[LhCfPoseSample] = []
  done = threading.Event()

  def on_data(sample: LhCfPoseSample):
    samples.append(sample)

  reader = LighthouseMatchedSweepAngleReader(
    cf, on_data, timeout_cb=done.set, sample_count=max_samples, min_bs=2
  )
  reader.start(timeout=duration_s)
  try:
    done.wait(duration_s + 1.0)
  finally:
    reader.stop()
  return samples


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--uri", type=str, default="radio://0/60/2M/E7E7E7E706")
  p.add_argument("--wait-s", type=int, default=4, help="Countdown seconds before each static sample.")
  p.add_argument("--walk-s", type=float, default=12.0, help="Seconds to walk around the flight volume.")
  p.add_argument("--no-voice", action="store_true", help="Disable macOS `say` voice prompts.")
  args = p.parse_args()
  voice = not args.no_voice

  cflib.crtp.init_drivers()
  with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="/tmp/cf_cache")) as scf:
    cf = scf.cf
    container = LhGeoInputContainerData(sensor_positions=LhDeck4SensorPositions.positions)

    speak("Place drone at the origin, facing plus X.", voice)
    countdown(args.wait_s, voice)
    speak("Recording origin.", voice)
    container.origin = collect_avg_sample(cf)

    speak("Place drone 1 meter along plus X.", voice)
    countdown(args.wait_s, voice)
    speak("Recording plus X.", voice)
    container.x_axis = [collect_avg_sample(cf)]

    speak("Place drone on the X Y floor plane, in plus Y.", voice)
    countdown(args.wait_s, voice)
    speak("Recording X Y plane.", voice)
    container.xy_plane = [collect_avg_sample(cf)]

    speak(f"Now walk slowly around the flight volume for {int(args.walk_s)} seconds.", voice)
    container.xyz_space = collect_space_samples(cf, duration_s=args.walk_s)
    speak(f"Collected {len(container.xyz_space)} samples. Solving geometry.", voice)

    container.origin.augment_with_ippe(container.sensor_positions)
    for s in container.x_axis + container.xy_plane + container.xyz_space:
      s.augment_with_ippe(container.sensor_positions)
    container.xyz_space = [s for s in container.xyz_space if len(s.ippe_solutions) >= 2]

    solution = LhGeoEstimationManager.estimate_geometry(container)
    if not solution.progress_is_ok:
      details = [
        solution.progress_info,
        solution.general_failure_info,
        solution.origin_sample_info,
        solution.x_axis_samples_info,
        solution.xy_plane_samples_info,
      ]
      msg = "; ".join(d for d in details if d)
      speak(f"Calibration failed: {msg}", voice)
      return

    if not solution.has_converged:
      speak("Warning: solver did not fully converge.", voice)

    geos = {}
    for bs_id, pose in solution.bs_poses.items():
      geo = LighthouseBsGeometry()
      geo.origin = pose.translation.tolist()
      geo.rotation_matrix = pose.rot_matrix.tolist()
      geo.valid = True
      geos[bs_id] = geo
      print(f"  BS {bs_id}: origin={[round(float(x), 3) for x in geo.origin]}")

    if solution.verification_stats:
      print("  Stats:", solution.verification_stats)

    done = threading.Event()
    write_result = {"ok": False}

    def on_stored(ok: bool) -> None:
      write_result["ok"] = ok
      done.set()

    writer = LighthouseConfigWriter(cf, nr_of_base_stations=4)
    writer.write_and_store_config(on_stored, geos=geos)
    if not done.wait(10.0):
      speak("Warning: timed out writing geometry to Crazyflie.", voice)
      return
    if not write_result["ok"]:
      speak("Warning: Crazyflie reported a write failure while storing geometry.", voice)
      return

    speak("Calibration complete and saved to Crazyflie!", voice)


if __name__ == "__main__":
  main()
