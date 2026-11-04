import numpy as np
import argparse
import pickle
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Neural Network to CSV")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--out", type=str, default="exported_model")
    return parser.parse_args()


def save_csv(name, data, out_dir):
    arr = np.array(data)
    path = os.path.join(out_dir, f"{name}.csv")

    if arr.ndim == 1:
        np.savetxt(path, arr.reshape(1, -1), delimiter=",", fmt="%.18f")
    else:
        np.savetxt(path, arr, delimiter=",", fmt="%.7f")

    print(f"Saved CSV: {name}.csv {arr.shape}")


def main():
    args = parse_args()

    csv_dir = os.path.join(args.out, "csv")

    os.makedirs(csv_dir, exist_ok=True)

    try:
        with open(args.path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print("Error in loading file")
        print(e)
        exit(1)

    normalizer = data[0]
    norm_dict = normalizer.__dict__ if hasattr(normalizer, "__dict__") else normalizer

    save_csv("OBS_MEAN", norm_dict["mean"], csv_dir)
    save_csv("OBS_STD",  norm_dict["std"], csv_dir)

    layers = ["hidden_0", "hidden_1", "hidden_2"]
    policy_params = data[1]["params"]

    for layer in layers:
        if layer in policy_params:
            save_csv(f"{layer}_weights",policy_params[layer]["kernel"].T ,csv_dir)
            save_csv(f"{layer}_bias",policy_params[layer]["bias"],csv_dir)

if __name__ == "__main__":
    main()
