"""Merge 50 SSQP expert trajectories and 40 multi-maneuver trajectories into a single unified 90-trajectory corpus.

Output:
  corpus/crazyflie_corpus_all90trajs.npz
"""

from pathlib import Path
import h5py
import numpy as np

from quadrotor.params import MAX_THRUST
from quadrotor.quaternion import state22_to_17


def merge_corpora(
    npz_50_path: str = "corpus/crazyflie_corpus_50trajs.npz",
    h5_40_path: str = "corpus/crazyflie_multi_maneuvers.h5",
    output_npz_path: str = "corpus/crazyflie_corpus_all90trajs.npz",
):
    print(f"[INFO] Ingesting 50 trajectories from {npz_50_path}...")
    npz_50 = np.load(npz_50_path)
    num_50 = int(npz_50["num_trajs"][0])

    trajectories = []
    # 1. Ingest 50 SSQP trajectories (dt=0.02s)
    for i in range(num_50):
        trajectories.append({
            "state": npz_50[f"state_{i}"].astype(np.float32),
            "ctrl": npz_50[f"ctrl_{i}"].astype(np.float32),
            "dt": float(npz_50[f"dt_{i}"][0]),
            "task": str(npz_50[f"task_{i}"]),
            "source": "ssqp_50",
        })
    print(f"[INFO] Successfully loaded {len(trajectories)} trajectories from 50trajs.npz")

    # 2. Ingest 40 multi-maneuvers (dt=0.005s downsampled by 4 to dt=0.02s)
    print(f"[INFO] Ingesting 40 trajectories from {h5_40_path}...")
    with h5py.File(h5_40_path, "r") as h5_40:
        names_40 = [n.decode() if hasattr(n, "decode") else str(n) for n in h5_40["names"]]
        tasks_40 = [t.decode() if hasattr(t, "decode") else str(t) for t in h5_40["task_types"]]

        for i in range(len(names_40)):
            t = h5_40["trajectories"][f"traj_{i:04d}"]
            s22 = t["states"][:]
            c22 = t["controls"][:]

            # Map 22-state vector to 17-state vector
            s17 = np.array([state22_to_17(x) for x in s22], dtype=np.float32)
            c17 = (c22 / MAX_THRUST).astype(np.float32)

            # Downsample stride 4: 0.005s -> 0.02s
            s17_down = s17[::4].copy()
            # Mean control over 4 substeps
            c_down = c17.reshape(-1, 4, 4).mean(axis=1).astype(np.float32)

            # Synchronize rotor speeds in state to controls
            s17_down[:-1, 13:17] = c_down
            s17_down[-1, 13:17] = c_down[-1]

            # Re-normalize quaternions
            q_norm = np.linalg.norm(s17_down[:, 3:7], axis=-1, keepdims=True)
            s17_down[:, 3:7] /= np.maximum(q_norm, 1e-8)

            trajectories.append({
                "state": s17_down,
                "ctrl": c_down,
                "dt": 0.02,
                "task": tasks_40[i],
                "source": f"multi_maneuver_{names_40[i]}",
            })

    total_trajs = len(trajectories)
    print(f"[INFO] Total merged trajectories: {total_trajs}")

    # 3. Save into unified npz format
    save_dict = {"num_trajs": np.array([total_trajs], dtype=np.int32)}
    task_counts = {}
    for idx, t in enumerate(trajectories):
        save_dict[f"state_{idx}"] = t["state"]
        save_dict[f"ctrl_{idx}"] = t["ctrl"]
        save_dict[f"dt_{idx}"] = np.array([t["dt"]], dtype=np.float32)
        save_dict[f"task_{idx}"] = np.array(t["task"])
        task_counts[t["task"]] = task_counts.get(t["task"], 0) + 1

    out_file = Path(output_npz_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, **save_dict)
    print(f"[INFO] Successfully saved {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    print(f"[INFO] Task breakdown: {task_counts}")


if __name__ == "__main__":
    merge_corpora()
