"""Tests for the SingleTrack block."""

import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Scope

from pathsim_vehicle.singleTrack import SingleTrack


def test_state_is_six_dimensional():
    """The state is the full [v_x, v_y, r, psi, X, Y] vector (length 6)."""
    car = SingleTrack()
    assert len(car.initial_value) == 6


def test_rhs_matches_equations_of_motion():
    """The rhs is the Newton-Euler EOM plus exact pose kinematics."""
    car = SingleTrack(m=1500.0, I_z=3000.0, l_f=1.2, l_r=1.4)
    x = np.array([10.0, 0.5, 0.2, 0.3, 4.0, -2.0])
    u = np.array([500.0, 800.0, -200.0, 600.0])
    v_x, v_y, r, psi = x[0], x[1], x[2], x[3]
    F_x_f, F_y_f, F_x_r, F_y_r = u

    dv_x, dv_y, dr, dpsi, dX, dY = car._func_dyn(x, u, 0.0)

    assert np.isclose(dv_x, (F_x_f + F_x_r) / car.m + v_y * r)
    assert np.isclose(dv_y, (F_y_f + F_y_r) / car.m - v_x * r)
    assert np.isclose(dr, (car.l_f * F_y_f - car.l_r * F_y_r) / car.I_z)
    assert np.isclose(dpsi, r)
    assert np.isclose(dX, v_x * np.cos(psi) - v_y * np.sin(psi))
    assert np.isclose(dY, v_x * np.sin(psi) + v_y * np.cos(psi))


def test_smooth_through_standstill_and_reverse():
    """No 1/v_x anywhere: the rhs stays finite at v_x = 0 and in reverse."""
    car = SingleTrack()
    u = np.array([300.0, 400.0, -100.0, 200.0])

    standstill = car._func_dyn(np.zeros(6), u, 0.0)
    assert np.all(np.isfinite(standstill))

    reverse = car._func_dyn(np.array([-8.0, 0.2, 0.1, 0.5, 0.0, 0.0]), u, 0.0)
    assert np.all(np.isfinite(reverse))


def test_analytic_jacobian_matches_central_difference():
    """_jac_dyn equals a central difference of _func_dyn, including at v_x ~ 0 and reverse."""
    car = SingleTrack()
    rng = np.random.default_rng(0)
    h = 1e-6
    max_err = 0.0
    for _ in range(500):
        x = np.array([rng.uniform(-30, 30), rng.uniform(-5, 5),
                      rng.uniform(-1.5, 1.5), rng.uniform(-np.pi, np.pi),
                      rng.uniform(-50, 50), rng.uniform(-50, 50)])
        u = rng.uniform(-5000, 5000, size=4)
        Ja = car._jac_dyn(x, u, 0.0)
        Jn = np.zeros((6, 6))
        for j in range(6):
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            Jn[:, j] = (car._func_dyn(xp, u, 0.0) - car._func_dyn(xm, u, 0.0)) / (2 * h)
        max_err = max(max_err, np.abs(Ja - Jn).max())
    assert max_err < 1e-6


def test_steady_force_balance_traces_circle():
    """In a PathSim graph, axle forces in steady-state balance (zero net yaw
    moment, lateral force = m*v_x*r) hold v_x, r constant and trace a closed
    circle of radius v_x / r."""
    m, I_z, l_f, l_r = 1500.0, 3000.0, 1.2, 1.4
    L = l_f + l_r
    v_x0, r0 = 15.0, 0.3

    # steady state with v_y = 0: F_y_f + F_y_r = m*v_x0*r0, l_f*F_y_f = l_r*F_y_r
    F_y_f = m * v_x0 * r0 * l_r / L
    F_y_r = m * v_x0 * r0 * l_f / L

    car = SingleTrack(m=m, I_z=I_z, l_f=l_f, l_r=l_r,
                      initial_value=[v_x0, 0.0, r0, 0.0, 0.0, 0.0])
    consts = [Constant(0.0), Constant(F_y_f), Constant(0.0), Constant(F_y_r)]
    sc = Scope(labels=["v_x", "v_y", "r", "psi", "X", "Y"])

    conns = ([Connection(consts[i], car[i]) for i in range(4)]
             + [Connection(car[i], sc[i]) for i in range(6)])
    sim = Simulation(consts + [car, sc], conns, dt=0.005, log=False)

    sim.run(2 * np.pi / r0)                # exactly one revolution

    t, data = sc.read()
    v_x, v_y, r, X, Y = data[0], data[1], data[2], data[4], data[5]

    # the balance is an equilibrium: velocities stay at their initial values
    mid = len(t) // 2
    assert np.isclose(v_x[mid], v_x0)
    assert np.isclose(v_y[mid], 0.0, atol=1e-8)
    assert np.isclose(r[mid], r0)

    # the path is a circle of radius R = v_x0 / r0
    R_pred = v_x0 / r0
    assert np.isclose((X.max() - X.min()) / 2, R_pred, rtol=0.02)
    assert np.isclose((Y.max() - Y.min()) / 2, R_pred, rtol=0.02)

    # one full revolution returns to the start
    assert abs(X[-1] - X[0]) < 0.2 and abs(Y[-1] - Y[0]) < 0.2
