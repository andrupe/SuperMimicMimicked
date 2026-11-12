"""Concatenate per-clip motion ``.npz`` files into a packed directory of ``.npy``.

Opening tens of thousands of tiny ``.npz`` clips at startup is dominated by
per-file overhead and scattered I/O. This tool reads them once and writes one
``<field>.npy`` per field plus ``motion_ends.npy`` and ``joint_names.json``.
``MotionLoader`` loads that directory with a handful of sequential reads when
``--motion-file`` points at it.

Example:
  uv run python -m mjlab.scripts.pack_motions \
    --input '~/bones_full/npz/*.npz' --output ~/bones_full/packed
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FIELDS = (
  "joint_pos",
  "joint_vel",
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="glob of per-clip .npz files")
  parser.add_argument("--output", required=True, help="output packed directory")
  parser.add_argument(
    "--workers", type=int, default=min(64, os.cpu_count() or 8), help="reader threads"
  )
  args = parser.parse_args()

  files = sorted(glob.glob(os.path.expanduser(args.input)))
  if not files:
    raise SystemExit(f"No files matched: {args.input}")
  out = os.path.expanduser(args.output)
  os.makedirs(out, exist_ok=True)
  print(f"[pack] {len(files)} clips -> {out} ({args.workers} workers)", flush=True)

  def _read(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {field: data[field] for field in (*FIELDS, "joint_names")}

  with ThreadPoolExecutor(max_workers=args.workers) as pool:
    per_clip = list(pool.map(_read, files))
  print("[pack] read done; concatenating + saving", flush=True)

  joint_names = per_clip[0]["joint_names"].tolist()
  if any(clip["joint_names"].tolist() != joint_names for clip in per_clip):
    raise SystemExit("[pack] the clips have different joint_names")
  with open(os.path.join(out, "joint_names.json"), "w") as fh:
    json.dump(joint_names, fh)
  lengths = np.array([clip["joint_pos"].shape[0] for clip in per_clip], dtype=np.int64)
  np.save(os.path.join(out, "motion_ends.npy"), np.cumsum(lengths))
  names = [os.path.splitext(os.path.basename(f))[0] for f in files]
  with open(os.path.join(out, "motion_names.json"), "w") as fh:
    json.dump(names, fh)
  for field in FIELDS:
    merged = np.concatenate([clip.pop(field) for clip in per_clip], axis=0)
    np.save(os.path.join(out, f"{field}.npy"), merged)
    print(
      f"[pack]   {field}: {tuple(merged.shape)} {merged.dtype} "
      f"({merged.nbytes / 1e9:.1f} GB)",
      flush=True,
    )
    del merged
  print(f"[pack] done: {out} ({int(lengths.sum())} frames)", flush=True)


if __name__ == "__main__":
  main()
