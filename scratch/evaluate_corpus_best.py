import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import jax
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table

from quadrotor.supermimic import (
    SuperMimicPolicy, SuperMimicDataset, load_checkpoint,
    _jit_closed_loop, OBS_DIM, ACT_DIM
)

console = Console()
corpus_path = "corpus/crazyflie_corpus_50trajs.h5"
ckpt_path = "logs/supermimic_10h/checkpoints/checkpoint_best.npz"

ds = SuperMimicDataset.from_hdf5(corpus_path)
model = SuperMimicPolicy()
rng = jax.random.PRNGKey(42)
dummy_params = model.init(rng, jnp.zeros((1, OBS_DIM)))["params"]
import optax
tx = optax.adam(1e-3)
dummy_opt = tx.init(dummy_params)

loaded = load_checkpoint(ckpt_path, dummy_params, dummy_opt)
params = loaded["params"]
obs_mean = jnp.asarray(loaded["obs_mean"])
obs_std = jnp.asarray(loaded["obs_std"])

console.print(f"[bold cyan]Loaded Best Checkpoint:[/] Epoch {loaded['epoch']}, Step {loaded['step']}")
console.print(f"Recorded Best Val RMSE: {loaded['val_loss']:.2f} cm\n")

# Evaluate across all trajectories categorized by task_type
task_results = {}
rng_key = jax.random.PRNGKey(100)

for traj in ds.trajectories:
    task = traj["task_type"]
    if task not in task_results:
        task_results[task] = {"nom_rmse": [], "noise_rmse": [], "dr_rmse": [], "max_err": []}
        
    xs_ref = jnp.asarray(traj["xs_22"])
    rng_key, k1, k2, k3 = jax.random.split(rng_key, 4)
    
    # 1. Nominal
    errs_nom, fin_nom = _jit_closed_loop(params, xs_ref, obs_mean, obs_std, k1, add_noise=False, dr=False)
    e_np = np.asarray(errs_nom)
    task_results[task]["nom_rmse"].append(np.sqrt(np.mean(np.square(e_np))) * 100)
    task_results[task]["max_err"].append(np.max(e_np) * 100)
    
    # 2. Sensor noise (5mm pos, 0.005 gyro)
    errs_noise, fin_noise = _jit_closed_loop(params, xs_ref, obs_mean, obs_std, k2, add_noise=True, dr=False)
    task_results[task]["noise_rmse"].append(np.sqrt(np.mean(np.square(np.asarray(errs_noise)))) * 100)
    
    # 3. Domain randomized (mass, inertia, lag)
    errs_dr, fin_dr = _jit_closed_loop(params, xs_ref, obs_mean, obs_std, k3, add_noise=False, dr=True)
    task_results[task]["dr_rmse"].append(np.sqrt(np.mean(np.square(np.asarray(errs_dr)))) * 100)

table = Table(title="SuperMimic Best Model Sim2Real Evaluation by Task Family", header_style="bold magenta")
table.add_column("Task Family", style="cyan", no_wrap=True)
table.add_column("Trajs", justify="center")
table.add_column("Nominal RMSE (cm)", justify="right", style="green")
table.add_column("Sensor Noise RMSE (cm)", justify="right", style="green")
table.add_column("Domain Rand RMSE (cm)", justify="right", style="yellow")
table.add_column("Mean Peak Drift (cm)", justify="right", style="yellow")

for task, d in sorted(task_results.items()):
    table.add_row(
        task,
        str(len(d["nom_rmse"])),
        f"{np.mean(d['nom_rmse']):.2f}",
        f"{np.mean(d['noise_rmse']):.2f}",
        f"{np.mean(d['dr_rmse']):.2f}",
        f"{np.mean(d['max_err']):.2f}",
    )

all_nom = [e for d in task_results.values() for e in d["nom_rmse"]]
all_noise = [e for d in task_results.values() for e in d["noise_rmse"]]
all_dr = [e for d in task_results.values() for e in d["dr_rmse"]]
all_max = [e for d in task_results.values() for e in d["max_err"]]

table.add_section()
table.add_row(
    "[bold]OVERALL CORPUS[/bold]",
    str(len(all_nom)),
    f"[bold]{np.mean(all_nom):.2f}[/bold]",
    f"[bold]{np.mean(all_noise):.2f}[/bold]",
    f"[bold]{np.mean(all_dr):.2f}[/bold]",
    f"[bold]{np.mean(all_max):.2f}[/bold]",
)

console.print(table)
