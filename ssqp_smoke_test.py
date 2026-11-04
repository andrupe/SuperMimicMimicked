"""Smoke test for the `ssqp` package (module `ssqpy`).

Mirrors the official double-integrator quickstart
(https://lar.upatras.gr/ssqp/getting-started/quickstart/) and adds assertions:
a point mass starting at p=0 with v=0 must be driven to p=1 within the horizon.

Run with:  uv run python ssqp_smoke_test.py
"""

import numpy as np

import ssqpy
import ssqpy.model.costs as costs

H, DT = 10, 0.1
TARGET = 1.0


class DoubleIntegrator(ssqpy.model.dynamics.StageDynamics):
    """Semi-implicit Euler double integrator, acceleration as control."""

    def output_size(self):
        return 2

    def state_size(self):
        return 2

    def control_size(self):
        return 1

    def evaluate(self, x, u):
        p, v = x[:]
        vn = v + u[0] * DT
        pn = p + vn * DT
        return np.array([pn, vn])

    def gradient(self, x, u):
        dX = np.zeros((2, 2))
        dX[0, 0] = 1.0
        dX[0, 1] = DT
        dX[1, 1] = 1.0

        dU = np.zeros((2, 1))
        dU[0, 0] = DT**2
        dU[1, 0] = DT
        return dX, dU

    def hessprod_xx(self, x, u, lambda_):
        return np.zeros((2, 2))

    def hessprod_xu(self, x, u, lambda_):
        return np.zeros((2, 1))

    def hessprod_uu(self, x, u, lambda_):
        return np.zeros((1, 1))


class Cost(costs.StageCost):
    """Penalise distance to the target position."""

    def evaluate(self, x, u):
        return 0.5 * (x[0] - TARGET) ** 2

    def gradient(self, x, u):
        dx = np.zeros(2)
        dx[0] = x[0] - TARGET
        return dx, np.zeros(1)

    def hessian_xx(self, x, u):
        hxx = np.zeros((2, 2))
        hxx[0, 0] = 1.0
        return hxx

    def hessian_xu(self, x, u):
        return np.zeros((2, 1))

    def hessian_uu(self, x, u):
        return np.zeros((1, 1))


def main():
    print("ssqpy version:", getattr(ssqpy, "__version__", "unknown"))

    stage_cost = Cost()
    model = ssqpy.model.Model(H, DT)
    model.finalize(
        # one dynamics instance per stage: reusing a single object triggers
        # "multiple uses of the same StageDynamics object" and is unsafe if
        # stages are expected to hold different state
        dynamics=[DoubleIntegrator() for _ in range(H + 1)],
        stage_cost=stage_cost,
        terminal_cost=stage_cost,
    )

    solver = ssqpy.solvers.ssqp(
        model, qp_solver_type=ssqpy.solvers.QPSolverType.ADMM
    )
    state, x, u = solver.solve()

    x0 = np.asarray(x.stage(0))
    xN = np.asarray(x.stage(H))
    u0 = np.asarray(u.stage(0))

    print("solver state :", state)
    print("x0           :", x0)
    print("xN           :", xN)
    print("u0           :", u0)

    # --- assertions: the solve actually did something sensible ---
    assert x0.shape == (2,), x0.shape
    assert u0.shape == (1,), u0.shape
    assert abs(x0[0]) < 1e-6, f"initial position should be 0, got {x0[0]}"
    assert abs(xN[0] - TARGET) < 1e-2, f"final position {xN[0]} not within 1e-2 of {TARGET}"
    assert abs(xN[1]) < 0.2, f"final velocity {xN[1]} should be near zero"
    cost_N = stage_cost.evaluate(xN, np.zeros(1))
    assert cost_N < 5e-5, f"terminal cost {cost_N} too high"
    assert np.all(np.isfinite(x.toarray())), "non-finite state trajectory"
    assert np.all(np.isfinite(u.toarray())), "non-finite control trajectory"

    print(f"\nOK — reached p={xN[0]:.6f} (target {TARGET}), terminal cost {cost_N:.3e}")


if __name__ == "__main__":
    main()
