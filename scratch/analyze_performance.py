import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import jax
import jax.numpy as jnp
from quadrotor.supermimic import (
    SuperMimicPolicy, SuperMimicDataset, load_checkpoint,
    compute_obs, normalize_obs, norm_action_to_rpm
)
from crazyflie_nn.dynamics.dynamics import rk4
from quadrotor.params import HOVER_RPM, MAX_THRUST

corpus_path = "corpus/crazyflie_corpus_50trajs.h5"
ckpt_path = "logs/supermimic_10h/checkpoints/checkpoint_best.npz"

ds = SuperMimicDataset.from_hdf5(corpus_path)
model = SuperMimicPolicy()
rng = jax.random.PRNGKey(42)
dummy_params = model.init(rng, jnp.zeros((1, 22)))["params"]
import optax
dummy_opt = optax.adam(1e-3).init(dummy_params)

loaded = load_checkpoint(ckpt_path, dummy_params, dummy_opt)
params = loaded["params"]
obs_mean = jnp.asarray(loaded["obs_mean"])
obs_std = jnp.asarray(loaded["obs_std"])

print("Analyzing controller behavior in detail...")

# Let's inspect Hover trajectories vs Figure-8 trajectories
for target_task in ["hover", "point_to_point", "figure8"]:
    matching = [t for t in ds.trajectories if t["task_type"] == target_task]
    if not matching:
        continue
    traj = matching[0]
    xs_ref = traj["xs_22"]
    us_ref_rpm = traj["us_rpm"]
    H = len(xs_ref) - 1
    
    s = xs_ref[0].copy()
    a_prev = jnp.zeros(4)
    
    positions = []
    actions = []
    rpms = []
    errors = []
    
    for t in range(H):
        p_target = xs_ref[t + 1, 0:3]
        o = compute_obs(s, p_target, a_prev)
        o_norm = normalize_obs(o, obs_mean, obs_std)
        a = np.array(model.apply({"params": params}, o_norm))
        u_rpm = np.array(norm_action_to_rpm(a))
        
        def sub(it, st):
            return rk4(st, jnp.asarray(u_rpm), jax.random.PRNGKey(0), 0.005)
        s = np.array(jax.lax.fori_loop(0, 4, sub, jnp.asarray(s)))
        a_prev = a
        
        pos_err = np.linalg.norm(s[0:3] - xs_ref[t + 1, 0:3])
        positions.append(s[0:3])
        actions.append(a)
        rpms.append(u_rpm)
        errors.append(pos_err)
        
    positions = np.array(positions)
    rpms = np.array(rpms)
    errors = np.array(errors)
    
    print(f"\n================ Task: {target_task.upper()} (H = {H} steps, {H*0.02:.1f} s) ================")
    print(f"Mean Pos Error       : {np.mean(errors)*100:.2f} cm")
    print(f"Final Pos Error      : {errors[-1]*100:.2f} cm")
    print(f"Max Pos Error        : {np.max(errors)*100:.2f} cm")
    
    # Check Z steady-state error
    z_actual = positions[:, 2]
    z_ref = xs_ref[1:, 2]
    z_err = z_actual - z_ref
    print(f"Mean Z (Alt) Error   : {np.mean(z_err)*100:.2f} cm (min {np.min(z_err)*100:.2f}, max {np.max(z_err)*100:.2f})")
    
    # Check control saturation
    min_rpm = np.min(rpms)
    max_rpm = np.max(rpms)
    mean_rpm = np.mean(rpms)
    print(f"RPM Range            : min {min_rpm:.1f}, max {max_rpm:.1f}, mean {mean_rpm:.1f} (Hover RPM = {HOVER_RPM:.1f})")
    
    # Check attitude tilt
    tilts = []
    for pos_t in positions:
        # R is in state 3:12
        pass
    print(f"Trajectory Start Ref : {xs_ref[0, 0:3]}")
    print(f"Trajectory End Ref   : {xs_ref[-1, 0:3]}")
    print(f"Actual End Pos       : {positions[-1]}")
