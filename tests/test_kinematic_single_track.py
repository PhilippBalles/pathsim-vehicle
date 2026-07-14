"""Tests for the KinematicSingleTrack block."""

import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Scope

from pathsim_vehicle.kinematicSingleTrack import KinematicSingleTrack


def test_state_is_pose_only():
    """The dynamic state is the pose [psi, X, Y] (length 3)."""
    car = KinematicSingleTrack()
    assert len(car.initial_value) == 3
    assert car.L == car.l_f + car.l_r


def test_algebraic_relations():
    """v_y, r follow the exact no-slip relations r = v_x*tan(delta)/L, v_y = l_r*r."""
    car = KinematicSingleTrack(l_f=1.2, l_r=1.4)
    delta, v_x = 0.10, 12.0
    v_y, r, psi, X, Y = car._func_alg(np.zeros(3), np.array([delta, v_x]), 0.0)

    r_exp = v_x * np.tan(delta) / car.L
    assert np.isclose(r, r_exp)
    assert np.isclose(v_y, car.l_r * r_exp)
    # pose is passed straight through from the state
    assert (psi, X, Y) == (0.0, 0.0, 0.0)


def test_smooth_through_standstill_and_reverse():
    """No 1/v_x term: standstill gives zero motion, reverse stays finite."""
    car = KinematicSingleTrack()
    standstill = car._func_dyn(np.zeros(3), np.array([0.3, 0.0]), 0.0)
    assert np.allclose(standstill, 0.0)

    reverse = car._func_dyn(np.zeros(3), np.array([0.3, -8.0]), 0.0)
    assert np.all(np.isfinite(reverse))


def test_analytic_jacobian_matches_central_difference():
    """_jac_dyn equals a central difference of _func_dyn, including at v_x = 0 and reverse."""
    car = KinematicSingleTrack()
    rng = np.random.default_rng(0)
    h = 1e-6
    max_err = 0.0
    for _ in range(500):
        x = np.array([rng.uniform(-np.pi, np.pi),
                      rng.uniform(-50, 50), rng.uniform(-50, 50)])
        u = np.array([rng.uniform(-1.3, 1.3), rng.uniform(-30, 30)])
        Ja = car._jac_dyn(x, u, 0.0)
        Jn = np.zeros((3, 3))
        for j in range(3):
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            Jn[:, j] = (car._func_dyn(xp, u, 0.0) - car._func_dyn(xm, u, 0.0)) / (2 * h)
        max_err = max(max_err, np.abs(Ja - Jn).max())
    assert max_err < 1e-6


def test_constant_steer_traces_predicted_circle():
    """In a PathSim graph, constant (delta, v_x) traces a closed circle of the
    predicted kinematic radius and the algebraic outputs settle to r, v_y."""
    delta, v_x = 0.05, 10.0
    c_delta, c_vx = Constant(delta), Constant(v_x)
    car = KinematicSingleTrack(l_f=1.2, l_r=1.4)
    sc = Scope(labels=["v_y", "r", "psi", "X", "Y"])

    conns = ([Connection(c_delta, car[0]), Connection(c_vx, car[1])]
             + [Connection(car[i], sc[i]) for i in range(5)])
    sim = Simulation([c_delta, c_vx, car, sc], conns, dt=0.005, log=False)

    r = v_x * np.tan(delta) / car.L
    sim.run(2 * np.pi / abs(r))            # exactly one revolution

    t, data = sc.read()
    v_y_out, r_out, X, Y = data[0], data[1], data[3], data[4]

    # algebraic outputs settle to the exact relations (mid-run, transient gone)
    mid = len(t) // 2
    assert np.isclose(r_out[mid], r)
    assert np.isclose(v_y_out[mid], car.l_r * r)

    # the path is a circle of the predicted CG radius R = |v| / |r|
    R_pred = np.hypot(v_x, car.l_r * r) / abs(r)
    assert np.isclose((X.max() - X.min()) / 2, R_pred, rtol=0.02)
    assert np.isclose((Y.max() - Y.min()) / 2, R_pred, rtol=0.02)

    # one full revolution returns to the start
    assert abs(X[-1] - X[0]) < 0.2 and abs(Y[-1] - Y[0]) < 0.2
