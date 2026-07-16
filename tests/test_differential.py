"""Tests for the Differential block, including the closed loop as a center
differential on the single-track car."""

import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Scope

from pathsim_vehicle.differential import Differential
from pathsim_vehicle.wheel import Wheel
from pathsim_vehicle.singleTrack import SingleTrack


def test_algebraic_map_matches_equations():
    """T_l = T_r = i*T_in/2 (equal-torque split) and
    omega_in = i*(omega_l + omega_r)/2 (Willis + final drive)."""
    diff = Differential(i=3.5)
    T_in, omega_l, omega_r = 120.0, 60.0, 48.0
    T_l, T_r, omega_in = diff._func_alg(np.array([T_in, omega_l, omega_r]))

    assert np.isclose(T_l, 0.5 * diff.i * T_in)
    assert np.isclose(T_r, T_l)
    assert np.isclose(omega_in, 0.5 * diff.i * (omega_l + omega_r))


def test_map_is_lossless():
    """Power balance T_in*omega_in = T_l*omega_l + T_r*omega_r at random
    operating points -- the ideal differential neither stores nor
    dissipates energy."""
    diff = Differential(i=3.7)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = rng.uniform(-200, 200, size=3)
        T_l, T_r, omega_in = diff._func_alg(u)
        assert np.isclose(u[0] * omega_in, T_l * u[1] + T_r * u[2])


def test_torque_and_speed_paths_are_decoupled():
    """The torque outputs depend only on T_in; the speed output only on the
    side speeds (unconnected side speeds corrupt omega_in, never the split)."""
    diff = Differential()
    y_wired = diff._func_alg(np.array([100.0, 60.0, 48.0]))
    y_bare = diff._func_alg(np.array([100.0, 0.0, 0.0]))
    assert np.allclose(y_wired[:2], y_bare[:2])
    assert y_bare[2] == 0.0


def _center_diff_car_sim(T_in, v_x0, duration, i=3.5):
    """Single-track car driven through a center differential: one torque
    source feeds both axles' T_d ports, wheel speeds are reported back."""
    car = SingleTrack(m=1500.0, I_z=3000.0, l_f=1.2, l_r=1.4,
                      initial_value=[v_x0, 0.0, 0.0, 0.0, 0.0, 0.0])
    omega_0 = v_x0 / 0.30
    wf = Wheel(l=1.2, omega_0=omega_0)
    wr = Wheel(l=-1.4, omega_0=omega_0)
    diff = Differential(i=i)
    c_zero, c_Tin = Constant(0.0), Constant(T_in)
    sc = Scope(labels=["v_x", "omega_in"])

    conns = [
        Connection(c_zero, wf["delta"], wr["delta"]),
        # torque source -> differential -> wheels ("l" = front, "r" = rear)
        Connection(c_Tin, diff["T_in"]),
        Connection(diff["T_l"], wf["T_d"]),
        Connection(diff["T_r"], wr["T_d"]),
        # wheel speeds back to the differential
        Connection(wf["omega"], diff["omega_l"]),
        Connection(wr["omega"], diff["omega_r"]),
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
        # input-shaft speed for inspection (an engine map would consume this)
        Connection(diff["omega_in"], sc[1]),
        ]

    sim = Simulation([c_zero, c_Tin, diff, wf, wr, car, sc], conns,
                     dt=0.005, log=False)
    sim.run(duration)
    return sc.read(), wf.R_w


def test_closed_loop_center_diff_accelerates_at_effective_mass_rate():
    """Constant input torque through the center diff: dv_x/dt =
    i*T_in*R_w / (m*R_w^2 + 2*I_w) (verified in derive_differential.py)."""
    T_in, i = 100.0, 3.5
    (t, data), R_w = _center_diff_car_sim(T_in=T_in, v_x0=20.0, duration=4.0,
                                          i=i)
    v_x = data[0]
    m, I_w = 1500.0, 1.2
    a_pred = i * T_in * R_w / (m * R_w**2 + 2 * I_w)

    mid = len(t) // 2
    a_meas = (v_x[-1] - v_x[mid]) / (t[-1] - t[mid])
    assert np.isclose(a_meas, a_pred, rtol=1e-3)


def test_closed_loop_input_shaft_speed_reports_rolling_speed():
    """At pure rolling the reported input-shaft speed is i * v_x / R_w."""
    (t, data), R_w = _center_diff_car_sim(T_in=0.0, v_x0=20.0, duration=1.0,
                                          i=3.5)
    v_x, omega_in = data[0], data[1]
    assert np.isclose(omega_in[-1], 3.5 * v_x[-1] / R_w, rtol=1e-6)


def test_open_diff_cannot_steer_the_speed_difference():
    """Two Wheels behind the diff on a wheel bench (constant chassis speed,
    so each load torque is affine in the own spin speed) with UNEQUAL brake
    torques: the speed-difference trajectory is identical for very different
    input torques -- the input appears only in the sum dynamics (no torque
    vectoring with an open differential)."""

    def rig(T_in):
        omega_0 = 20.0 / 0.30
        wl, wr = Wheel(omega_0=omega_0), Wheel(omega_0=omega_0)
        diff = Differential(i=3.5)
        c_Tin, c_Tbl, c_Tbr = Constant(T_in), Constant(50.0), Constant(10.0)
        c_vx, c_Fz = Constant(20.0), Constant(4000.0)
        sc = Scope(labels=["omega_l", "omega_r"])
        conns = [
            Connection(c_Tin, diff["T_in"]),
            Connection(diff["T_l"], wl["T_d"]),
            Connection(diff["T_r"], wr["T_d"]),
            Connection(c_Tbl, wl["T_b"]),
            Connection(c_Tbr, wr["T_b"]),
            Connection(c_vx, wl["v_x"], wr["v_x"]),
            Connection(c_Fz, wl["F_z"], wr["F_z"]),
            Connection(wl["omega"], diff["omega_l"], sc[0]),
            Connection(wr["omega"], diff["omega_r"], sc[1]),
            ]
        sim = Simulation([c_Tin, c_Tbl, c_Tbr, c_vx, c_Fz, diff, wl, wr, sc],
                         conns, dt=0.005, log=False)
        sim.run(2.0)
        t, data = sc.read()
        return data[0] - data[1]

    assert np.allclose(rig(0.0), rig(200.0), rtol=0.0, atol=1e-9)
