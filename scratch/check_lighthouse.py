"""Static Lighthouse & Kalman estimator health check (no motors spun)."""

from __future__ import annotations

import argparse
import time
import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--uri", type=str, default="radio://0/60/2M/E7E7E7E706")
  p.add_argument("--seconds", type=float, default=3.0)
  args = p.parse_args()

  cflib.crtp.init_drivers()
  with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="/tmp/cf_cache")) as scf:
    cf = scf.cf
    lh_logs = sorted(cf.log.toc.toc.get("lighthouse", {}).keys())
    print("deck.bcLighthouse4 =", cf.param.get_value("deck.bcLighthouse4"))
    print("stabilizer.estimator =", cf.param.get_value("stabilizer.estimator"))

    lg = LogConfig(name="lh_diag", period_in_ms=20)
    lg.add_variable("stateEstimate.x", "float")
    lg.add_variable("stateEstimate.y", "float")
    lg.add_variable("stateEstimate.z", "float")
    lg.add_variable("lighthouse.status", "uint8_t")
    for name in ("bsAvailable", "bsActive", "bsCalib", "bsReceive"):
      if name in lh_logs:
        lg.add_variable(f"lighthouse.{name}", cf.log.toc.toc["lighthouse"][name].ctype)

    samples = []
    lg.data_received_cb.add_callback(lambda _ts, data, _lc: samples.append(data))
    cf.log.add_config(lg)
    lg.start()
    time.sleep(args.seconds)
    lg.stop()

    if not samples:
      print("No samples received!")
      return

    print(f"\nCollected {len(samples)} samples over {args.seconds:.1f} s (keep drone still):")
    for k in samples[0]:
      vals = np.array([s[k] for s in samples], dtype=float)
      print(f"  {k:20s}: min={vals.min():8.4f}  max={vals.max():8.4f}  std={vals.std():7.4f}  last={vals[-1]:.4f}")

    z_std = float(np.std([s["stateEstimate.z"] for s in samples]))
    lh_last = int(samples[-1]["lighthouse.status"])
    print("\nVerdict:")
    if lh_last != 2:
      print(f"  [BAD] lighthouse.status={lh_last} (expected 2 = locked). Check base stations / geometry.")
    elif z_std > 0.02:
      print(f"  [BAD] z std={z_std:.3f} m while still (expected < 0.005 m). Re-estimate Lighthouse geometry or check reflections/visibility.")
    else:
      print(f"  [GOOD] lighthouse.status=2 and z std={z_std:.4f} m (< 0.02 m).")


if __name__ == "__main__":
  main()
