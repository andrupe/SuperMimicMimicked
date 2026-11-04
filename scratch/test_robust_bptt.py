import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from quadrotor.supermimic import (
    SuperMimicDataset, SuperMimicPolicy, compute_obs, normalize_obs,
    norm_action_to_rpm, train_behavior_cloning
)
from crazyflie_nn.dynamics.dynamics import rk4

print("Training SuperMimic with perturbed BPTT rollouts...")

ds = SuperMimicDataset.from_hdf5("corpus/crazyflie_corpus_20trajs.h5")
params, _ = train_behavior_cloning(ds, epochs=100, batch_size=256, lr=1e-3)

model = SuperMimicPolicy()
tx = optax.adam(learning_rate=5e-4)
opt_state = tx.init(params)

obs_mean_jax = jnp.asarray(ds.obs_mean)
obs_std_jax = jnp.asarray(ds.obs_std)

# Build windows
windows = []
horizon = 20
for traj in ds.trajectories:
    xs = traj["xs_22"]
    H = len(xs) - 1
    for st in range(0, H - horizon, 5):
        windows.append(xs[st:st + horizon + 1])
windows_jax = jnp.asarray(windows, dtype=jnp.float32)
M = len(windows_jax)
print(f"BPTT windows: {M}, horizon: {horizon}")

def single_traj_loss(p, ref_traj, rng_key):
    k1, k2, k3 = jax.random.split(rng_key, 3)
    # Perturb initial state to teach active recovery
    p_noise = jax.random.normal(k1, shape=(3,)) * 0.03 # 3cm initial pos perturbation
    v_noise = jax.random.normal(k2, shape=(3,)) * 0.05 # 5cm/s initial vel perturbation
    
    s0 = ref_traj[0].at[0:3].add(p_noise).at[12:15].add(v_noise)
    a_init = jnp.zeros(4, dtype=jnp.float32)
    
    def step_fn(carry, t):
        s_curr, a_prev, key = carry
        p_target = ref_traj[t + 1, 0:3]
        o = compute_obs(s_curr, p_target, a_prev)
        
        # Add small sensor noise during training for robustness
        key, kn = jax.random.split(key)
        o = o.at[0:3].add(jax.random.normal(kn, shape=(3,)) * 0.005)
        
        o_norm = normalize_obs(o, obs_mean_jax, obs_std_jax)
        a = model.apply({"params": p}, o_norm)
        u_rpm = norm_action_to_rpm(a)
        
        def sub(i, s):
            return rk4(s, u_rpm, jax.random.PRNGKey(0), 0.005)
        s_next = jax.lax.fori_loop(0, 4, sub, s_curr)
        
        # Stage tracking penalty
        pos_err = jnp.sum(jnp.square(s_curr[0:3] - ref_traj[t, 0:3]))
        vel_err = jnp.sum(jnp.square(s_curr[12:15] - ref_traj[t, 12:15]))
        act_reg = jnp.sum(jnp.square(a - a_prev))
        
        loss = pos_err + 0.1 * vel_err + 0.01 * act_reg
        return (s_next, a, key), loss

    timesteps = jnp.arange(horizon)
    _, losses = jax.lax.scan(step_fn, (s0, a_init, k3), timesteps)
    return jnp.mean(losses)

@jax.jit
def bptt_step(p, opt_s, batch_refs, rng_keys):
    def loss_fn(p_curr):
        losses = jax.vmap(lambda ref, k: single_traj_loss(p_curr, ref, k))(batch_refs, rng_keys)
        return jnp.mean(losses)
    l, g = jax.value_and_grad(loss_fn)(p)
    u, opt_s = tx.update(g, opt_s, p)
    p = optax.apply_updates(p, u)
    return p, opt_s, l

rng = jax.random.PRNGKey(0)
B = 32
num_b = M // B

t0 = time.perf_counter()
for ep in range(50):
    rng, pk = jax.random.split(rng)
    perm = jax.random.permutation(pk, M)
    ep_l = 0.0
    for b in range(num_b):
        rng, k_b = jax.random.split(rng)
        batch_keys = jax.random.split(k_b, B)
        idx = perm[b*B:(b+1)*B]
        params, opt_state, l_val = bptt_step(params, opt_state, windows_jax[idx], batch_keys)
        ep_l += float(l_val)
    if ep % 10 == 0:
        print(f"Epoch {ep:2d} | BPTT Loss: {ep_l / num_b:.6f}")

print(f"BPTT finished in {time.perf_counter() - t0:.2f} s")

# Now test closed-loop on Trajectory 1 (helix) and Trajectory 0!
for i in [0, 1, 2]:
    traj = ds.trajectories[i]
    xs_ref = traj["xs_22"]
    H = len(xs_ref) - 1
    s = xs_ref[0].copy()
    a_prev = jnp.zeros(4)
    errs = []
    for t in range(H):
        p_target = xs_ref[t + 1, 0:3]
        o = compute_obs(s, p_target, a_prev)
        o_norm = normalize_obs(o, ds.obs_mean, ds.obs_std)
        a = model.apply({"params": params}, o_norm)
        u_rpm = norm_action_to_rpm(a)
        
        def sub(it, st):
            return rk4(st, u_rpm, jax.random.PRNGKey(0), 0.005)
        s = jax.lax.fori_loop(0, 4, sub, s)
        a_prev = a
        errs.append(float(jnp.linalg.norm(s[0:3] - xs_ref[t + 1, 0:3])))
    rmse_cm = np.sqrt(np.mean(np.square(errs))) * 100
    max_cm = np.max(errs) * 100
    print(f"Traj {i} ({traj['task_type']:15s}): RMSE = {rmse_cm:.2f} cm | Max Drift = {max_cm:.2f} cm")
