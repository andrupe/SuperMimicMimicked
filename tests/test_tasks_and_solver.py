"""Test trajectory solver and sampler across all task families."""

import numpy as np
import pytest
from quadrotor.sampler import sample_task, TaskType
from quadrotor.solver import solve_task


def test_solve_point_to_point():
    rng = np.random.default_rng(42)
    task = sample_task(TaskType.POINT_TO_POINT, rng=rng, T=2.0, dt=0.04)
    res = solve_task(task, max_iter=80, tolerance=1e-5, use_analytic=True)

    assert res["success"] is True, f"Validation failed: {res['validation']}"
    assert res["solver_state"] == "Success"
    assert res["validation"]["max_track_err"] < 0.20
    assert res["validation"]["final_track_err"] < 0.10
    assert res["validation"]["max_residual"] < 1e-6


def test_solve_figure8():
    rng = np.random.default_rng(101)
    task = sample_task(TaskType.FIGURE8, rng=rng, T=3.5, dt=0.05)
    res = solve_task(task, max_iter=80, tolerance=1e-5, use_analytic=True)

    assert res["success"] is True, f"Validation failed: {res['validation']}"
    assert res["solver_state"] == "Success"
    assert res["validation"]["max_residual"] < 1e-6
    assert res["validation"]["max_tilt_deg"] < 45.0


def test_solve_helix():
    rng = np.random.default_rng(202)
    task = sample_task(TaskType.HELIX, rng=rng, T=3.5, dt=0.05)
    res = solve_task(task, max_iter=80, tolerance=1e-5, use_analytic=True)

    assert res["success"] is True, f"Validation failed: {res['validation']}"
    assert res["solver_state"] == "Success"
    assert res["validation"]["max_residual"] < 1e-6


def test_solve_hover():
    rng = np.random.default_rng(303)
    task = sample_task(TaskType.HOVER, rng=rng, T=2.0, dt=0.04)
    res = solve_task(task, max_iter=80, tolerance=1e-5, use_analytic=True)

    assert res["success"] is True, f"Validation failed: {res['validation']}"
    assert res["solver_state"] == "Success"
    assert res["validation"]["final_track_err"] < 0.10
