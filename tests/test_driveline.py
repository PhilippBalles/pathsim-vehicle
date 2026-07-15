"""Tests for the Driveline block, including the closed loop with Wheel and
SingleTrack (torque-driven car)."""

import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Scope

from pathsim_vehicle.driveline import Driveline
from pathsim_vehicle.wheel import Wheel
from pathsim_vehicle.singleTrack import SingleTrack


def test_rhs_is_the_torque_balance():
    """The rhs is I_w*domega/dt = T_d - T_b - M_y, independent of omega."""
    dl = Driveline(I_w=1.2)
    T_d, T_b, M_y = 300.0, 50.0, 120.0
    u = np.array([T_d, T_b, M_y])

    domega = dl._func_dyn(np.array([40.0]), u, 0.0)
    assert np.isclose(domega[0], (T_d - T_b - M_y) / dl.I_w)

    # no state feedback: same derivative at any omega
    assert np.allclose(dl._func_dyn(np.array([-7.0]), u, 0.0), domega)


def test_analytic_jacobian_is_zero():
    """The rhs does not depend on the state -> 1x1 zero Jacobian."""
    dl = Driveline()
    J = dl._jac_dyn(np.array([40.0]), np.array([300.0, 50.0, 120.0]), 0.0)
    assert J.shape == (1, 1)
    assert np.allclose(J, 0.0)


def test_initial_spin_speed():
    """omega_0 lands in the (one-element) initial state vector."""
    assert np.allclose(Driveline(omega_0=66.7).initial_value, [66.7])
    assert np.allclose(Driveline().initial_value, [0.0])


def _torque_driven_car_sim(T_d, T_b, v_x0, duration, omega_scale=1.0):
    """Full torque-driven car: SingleTrack + one Wheel and one Driveline per
    axle (LinearTire), straight driving. ``T_d``/``T_b`` are applied to both
    drivelines; ``omega_scale`` scales the rolling-start spin speed."""
    car = SingleTrack(m=1500.0, I_z=3000.0, l_f=1.2, l_r=1.4,
                      initial_value=[v_x0, 0.0, 0.0, 0.0, 0.0, 0.0])
    wf, wr = Wheel(l=1.2), Wheel(l=-1.4)
    omega_0 = omega_scale * v_x0 / wf.R_w
    df, dr = Driveline(omega_0=omega_0), Driveline(omega_0=omega_0)
    c_zero, c_Td, c_Tb = Constant(0.0), Constant(T_d), Constant(T_b)
    sc = Scope(labels=["v_x", "omega_f", "omega_r"])

    conns = [
        Connection(c_zero, wf["delta"], wr["delta"]),
        # driveline spin speeds to the wheels
        Connection(df["omega"], wf["omega"], sc[1]),
        Connection(dr["omega"], wr["omega"], sc[2]),
        # chassis velocity feedback and static axle loads to both wheels
        Connection(car["v_x"], wf["v_x"], wr["v_x"], sc[0]),
        Connection(car["v_y"], wf["v_y"], wr["v_y"]),
        Connection(car["r"], wf["r"], wr["r"]),
        Connection(car["F_z_f"], wf["F_z"]),
        Connection(car["F_z_r"], wr["F_z"]),
        # body-frame axle forces into the chassis
        Connection(wf["F_x"], car["F_x_f"]),
        Connection(wf["F_y"], car["F_y_f"]),
        Connection(wr["F_x"], car["F_x_r"]),
        Connection(wr["F_y"], car["F_y_r"]),
        # load torques back into the drivelines, torques in
        Connection(wf["M_y"], df["M_y"]),
        Connection(wr["M_y"], dr["M_y"]),
        Connection(c_Td, df["T_d"], dr["T_d"]),
        Connection(c_Tb, df["T_b"], dr["T_b"]),
        ]

    sim = Simulation([c_zero, c_Td, c_Tb, wf, wr, df, dr, car, sc], conns,
                     dt=0.001, log=False)
    sim.run(duration)
    return sc.read(), wf.R_w


def test_closed_loop_free_rolling_settles():
    """Zero torque: the (stable) slip dynamics find pure rolling on their own,
    even from a 20% overspin start."""
    (t, data), R_w = _torque_driven_car_sim(T_d=0.0, T_b=0.0, v_x0=20.0,
                                            duration=2.0, omega_scale=1.2)
    v_x, omega_f, omega_r = data[0], data[1], data[2]
    assert np.isclose(R_w * omega_f[-1], v_x[-1], atol=1e-6)
    assert np.isclose(R_w * omega_r[-1], v_x[-1], atol=1e-6)


def test_closed_loop_accelerates_at_effective_mass_rate():
    """Constant drive torque on both axles: quasi-steady slip acceleration
    dv_x/dt = 2*T_d/R_w / (m + 2*I_w/R_w^2) (verified in derive_driveline.py;
    the quasi-steady approximation is good to ~1e-4)."""
    T_d = 300.0
    (t, data), R_w = _torque_driven_car_sim(T_d=T_d, T_b=0.0, v_x0=20.0,
                                            duration=4.0)
    v_x = data[0]
    m, I_w = 1500.0, 1.2
    a_pred = (2 * T_d / R_w) / (m + 2 * I_w / R_w**2)

    mid = len(t) // 2
    a_meas = (v_x[-1] - v_x[mid]) / (t[-1] - t[mid])
    assert np.isclose(a_meas, a_pred, rtol=1e-3)


def test_closed_loop_brake_torque_decelerates():
    """The same magnitude on the T_b ports decelerates at the same rate."""
    T_b = 300.0
    (t, data), R_w = _torque_driven_car_sim(T_d=0.0, T_b=T_b, v_x0=20.0,
                                            duration=4.0)
    v_x = data[0]
    m, I_w = 1500.0, 1.2
    a_pred = -(2 * T_b / R_w) / (m + 2 * I_w / R_w**2)

    mid = len(t) // 2
    a_meas = (v_x[-1] - v_x[mid]) / (t[-1] - t[mid])
    assert v_x[-1] < v_x[0]
    assert np.isclose(a_meas, a_pred, rtol=1e-3)
