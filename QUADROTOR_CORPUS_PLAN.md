# Quadrotor expert-trajectory corpus with SSQP — setup plan

Status: **approach validated end-to-end** with a prototype (`quadrotor_ssqp_proto.py`).
Everything in §3 and §4 is empirically verified on this machine, not inferred from docs.

---

## 1. What SSQP does and does not give you

| | |
|---|---|
| Presets | `Go2, A1, Panda, G1, Talos, DoublePendulum` — **no quadrotor** |
| Actuators | joint torques/velocities via Pinocchio; **no rotor/ESC model** |
| Manifold states | **supported** (`state_size` > `state_diff_size` + `right_plus`/`right_minus`) — this is what makes quaternion attitude work |
| QP backends in 0.1.2 | `ADMM`, `HPIPM` (PIQP is on `main`, not in the wheel) |
| Trajectory I/O | `solver.write(path)`, `ssqpy.solvers.loadTrajectory(path)` |

Consequence: the quadrotor must be supplied as **custom `StageDynamics` / `StageCost` /
`StageConstraints`**. No URDF/MJCF route is attractive here — the Bitcraze MJCF has no actuated
joints, and SSQP's XML path produces torque-controlled joints, not rotor thrusts.

## 2. Model: match `crazyflie_nn/dynamics/dynamics.py` exactly

Their simulator is the ground truth the corpus should be consistent with:

```
state  : p(3), R_wb(3x3), v_b(3), w_b(3), omega(4)          (22)
control: omega_cmd(4)  [RPM], first-order motor lag  tau_m = 0.1 s
params : M = 0.033 kg, L = 0.046 m, KF = 2.25e-8, KM = 1.34e-10,
         I (3x3, off-diagonals ~0.8e-6), Kaero = diag(9.18e-7, 9.18e-7, 10.3e-7),
         omega_max = 2618 RPM, DT = 0.005 (RK4)
```

### Recommended OCP state/control

```
x (17):  p(3), q(4, quat), v_b(3), w_b(3), omega(4)
dx(16):  dp(3), dtheta(3), dv(3), dw(3), domega(4)
u (4):   rotor speeds (or omega^2 — see below), bounded [0, 2618]
```

- **Keep the motor lag** (`omega_dot = (u - omega)/tau_m`): without it the controls are
  non-physical steps that the JAX sim cannot reproduce.
- **Rotor mixing** (from their `dynamics.py`, an exact linear map on `omega_i^2`):
  ```
  T      = KF * sum(omega_i^2)
  tau_phi   = KF*L/sqrt(2) * (-w1^2 - w2^2 + w3^2 + w4^2)
  tau_theta = KF*L/sqrt(2) * (-w1^2 + w2^2 + w3^2 - w4^2)
  tau_psi   = KM           * (-w1^2 + w2^2 - w3^2 + w4^2)
  ```
  Using **u = omega^2** makes the box constraints linear and the dynamics smooth; the drag term
  `sum|omega_i|` becomes `sum sqrt(u_i)`, which is fine but adds a sqrt kink near 0.
- **Do not run the OCP at 200 Hz.** `H=600` stages × Python callbacks is unaffordable
  (see §4). Use `dt = 0.02–0.05` for the OCP and put **N RK4 substeps of 5 ms inside
  `evaluate()`**, so `f_k` is one OCP step but still integrates with the sim's own RK4.
- **Rescale internally.** The Crazyflie is tiny (M=0.033, I~1.7e-5): the QP is badly
  conditioned at SI scale. Either normalize (unit mass, unit inertia, `omega/omega_max`) inside
  `evaluate`/`gradient` and rescale on output, or keep the prototype's 1 kg / 1e-2 scale and add a
  documented similarity transform. The prototype deliberately used "normalized quadrotor" numbers.

## 3. SSQP wiring — verified recipe (with the six gotchas)

```python
model = ssqpy.model.Model(H, dt)          # no urdf/mjcf: fully custom system
model.finalize(
    dynamics=dynamics,                    # list, len H+1, ONE OBJECT PER STAGE
    costs=all_costs,                      # list, len H+1, terminal included
    terminal_cost=terminal,               # also required
    constraints=constraint_list,          # FLAT list, len H+1
    init_states=init_states,              # (H+1, nx) full state vectors
    init_controls=init_controls,          # (H, nu)
)
solver = ssqpy.solvers.ssqp(model, params)
state, x, u = solver.solve()
```

Gotchas that cost real debugging time:

1. **`hessprod_xx/xu/uu` are mandatory** on `StageDynamics` and `StageConstraints` in this
   wheel — the nanobind trampoline raises `pure virtual function` even though the docs say
   "optional, defaults to zero". Return zeros (Gauss-Newton).
2. **`costs` must have exactly `H+1` entries** *and* `terminal_cost` must be passed.
   `H` entries → `Costs size mismatch`; `H+1` without `terminal_cost` →
   `Terminal stage costs are undefined`.
3. **`constraints` is a flat list of `H+1` objects**, not a list of lists (nested → `std::bad_cast`
   or `Constraints size mismatch`).
4. **The terminal stage passes an empty control vector** — every cost/constraint must guard
   `u.size == 0`. This was the first runtime crash after `finalize` succeeded.
5. **`gradient` must be tangent-space**: `dX[i,j] = (f(x ⊕ ε e_j) ⊖ f(x)) / ε` with
   `n_dx × n_dx`, and `dU` is `n_dx × n_u`.
6. **`init_states` must be full `nx`-dimensional states**, not just positions. A partial guess
   surfaces as a bare `std::bad_cast` from `finalize`, which is unhelpful.
7. **`init_states` only seeds the guess — `x0` is fixed by `dynamics[0].init_state()`.**
   If you pass `init_states` but not a stage-0 dynamics object with the right `init_state()`,
   the OCP starts from the origin and the solution is garbage (symptom: initial tracking error
   equal to the distance from the origin, thrust pinned at a bound). Subclass the stage-0
   dynamics and override `init_state()`/`init_control()`.

### Verified model behaviour

```
state      : SolverState.Success, 7 SQP iterations
residual   : max ||x_{k+1} ⊖ f(x_k,u_k)|| = 7.4e-12 over an independent rollout
accuracy   : 3.3 cm terminal position error on a 2.7 m, 3 s transfer
```

The `7e-12` residual is the important one: it proves the quaternion `right_plus`/`right_minus`
pair and the tangent-space Jacobian are consistent with `evaluate`.

## 4. Performance budget (measured, H=60 / 3 s / 17-state-class problem)

| Item | Time |
|---|---|
| `evaluate` (one stage) | 63 µs |
| `gradient` (tangent-space finite differences) | **1 976 µs** (31× evaluate) |
| cost `evaluate`+`gradient` | 34 µs |
| one SQP iteration (61 stages) | ~123 ms |
| full solve, HPIPM, 7 iterations | **1 150 ms** |
| full solve, ADMM | **14 452 ms**, failed to converge in 100 iterations |

**Use HPIPM.** ADMM is 12× slower *and* did not reach tolerance.

Analytic Jacobians are the single biggest lever: replacing the 16 finite-difference `evaluate`
calls per stage with hand-derived tangent Jacobians should land at ~100–150 µs/stage, i.e.
**~80–120 ms per trajectory (~10×)**. The derivatives are mechanical:
`∂v'/∂dθ = -dt·R·skew(e_z T/m)`, `∂(attitude diff)/∂dw = dt/2·I`, etc.

### Corpus projections (8 cores, one process per core)

| Trajectories | FD gradients (1.15 s) | Analytic (~0.1 s) |
|---|---|---|
| 1 000 | 2.4 min | 13 s |
| 10 000 | 24 min | 2.1 min |
| 100 000 | 4.0 h | 21 min |
| 1 000 000 | 40 h | 3.5 h |

Storage (states 61×17 + controls 60×4 float32 ≈ 5.1 kB/traj, plus metadata): **~0.6 GB per
100 k**.

## 5. Pipeline to build

1. **Parameter module** — single source of truth for M, I, KF, KM, L, Kaero, tau_m, omega_max,
   imported by both the OCP and the JAX sim (avoid drift between corpus and training).
2. **`QuadrotorDynamics(StageDynamics)`** — 17/16 state, 4 RPM controls, N RK4 substeps,
   analytic tangent Jacobians.
3. **Costs** — quadratic tracking (position / attitude / body velocity / rate / control effort) +
   terminal pose; `TrackingCost` in the prototype is the template, instantiated per stage from a
   reference array.
4. **Constraints** — rotor box `[0, omega_max]`; add `|v|`, altitude floor, tilt limit
   (`cos θ > 0.7`), and obstacle half-spaces for a richer corpus.
5. **Task sampler** — start pose, goal pose, yaw profile, duration `T ∈ [1.5, 4] s`,
   optional via-points; sample inside a feasible envelope (start with reachable
   `‖Δp‖ ≤ v_max·T`, `Δz` moderate), and **reject/retry** when `SolverState != Success`.
6. **Initial guess** — quintic position profile + hover attitude. Verified to converge in 7
   iterations on a 2.7 m transfer; it is a good default for all tasks.
7. **Parallel driver** — `multiprocessing.Pool(8)`; threads do **not** help because every
   callback crosses into Python (GIL). Model rebuild is ~0.8 ms, so build per task.
8. **Validation gate per trajectory** — `SolverState.Success`, residual `< 1e-8`, controls inside
   bounds, no NaN, max accel below a threshold, and a rollout in the **JAX sim** (domain-randomized
   off) to confirm the corpus is reproducible by the trainer.
9. **Storage** — HDF5 (chunked, one dataset per field) or Zarr:
   `x (N, H+1, 17)`, `u (N, H, 4)`, `dt`, `solver_state`, `cost`, `iters`, `task_params(JSON)`.
   Deterministic seeds per task; keep the rejected-task log.

## 6. Risks / decisions to make early

- **Conditioning** at Crazyflie scale — decide normalize-vs-similarity-transform before writing
  the dynamics class, it touches everything (cost weights, bounds, Jacobians).
- **Control parameterization** — `omega` vs `omega^2`: bounds/linear mixing favour `omega^2`,
  transfer to the RPM-trained policy favours `omega`. Pick one and document the conversion.
- **Drag nonlinearity** — `sum|omega| · Kaero · v_b` couples control magnitude and velocity;
  it makes the model realistic but adds state-dependent Jacobian terms. Start without it, add
  after the corpus pipeline is proven.
- **ADMM fallback** — if HPIPM turns out to struggle on a task family, ADMM needs ~50× more
  iterations; budget accordingly rather than assuming solver swaps are free.
- **Domain randomization** — the JAX sim randomizes KF/KM/M/I *inside* `dynamics()`. Expert
  trajectories must be generated for the **nominal** parameters; randomization stays a
  training-side phenomenon (otherwise "expert" is undefined).

## 7. Immediate next steps

1. Move the prototype's model code into `quadrotor/` with the real Crazyflie parameters.
2. Add analytic Jacobians; re-verify the `1e-12`-class residual and the ~10× speedup.
3. Add RK4 substeps inside `evaluate` and confirm the OCP trajectory replays in the JAX sim.
4. Write `generate_corpus.py` (sampler → solve → validate → HDF5) with a `--dry-run` that
   solves 10 tasks and prints the accept rate.
5. Only then scale to 10⁵ trajectories.
