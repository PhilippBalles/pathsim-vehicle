"""Tests for the Wheel block, including the closed loop with SingleTrack."""

import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Scope

from pathsim_vehicle.wheel import Wheel
from pathsim_vehicle.singleTrack import SingleTrack
from pathsim_vehicle.tiremodels import LinearTire, SimplePacejka


def test_default_tire_is_linear():
    """With no tire argument the Wheel constructs a LinearTire."""
    assert isinstance(Wheel().tire, LinearTire)


def test_pure_rolling_gives_zero_forces():
    """omega = v_cx / R_w and no lateral motion -> kappa = alpha = 0 -> no forces."""
    wheel = Wheel()
    v_x = 20.0
    y = wheel._func_alg(np.array([0.0, v_x / wheel.R_w, v_x, 0.0, 0.0, 4000.0]))
    assert np.allclose(y, 0.0)


def test_standstill_is_well_defined():
    """All inputs zero -> zero slip (atan2(0,0) = 0) -> zero forces, no NaN."""
    wheel = Wheel()
    assert np.allclose(wheel._func_alg(np.zeros(6)), 0.0)


def test_algebraic_chain_matches_equations():
    """_func_alg reproduces the documented kinematics -> tire -> rotation chain."""
    wheel = Wheel(l=1.2, s=0.0, R_w=0.30, v_eps=1.0)
    delta, omega, v_x, v_y, r, F_z = 0.08, 40.0, 15.0, 0.4, 0.25, 4000.0
    F_x, F_y, M_y = wheel._func_alg(np.array([delta, omega, v_x, v_y, r, F_z]))

    v_cx = v_x - wheel.s * r
    v_cy = v_y + wheel.l * r
    kappa = (wheel.R_w * omega - v_cx) / np.sqrt(v_cx**2 + wheel.v_eps**2)
    alpha = delta - np.arctan2(v_cy, v_cx)
    F_x_t, F_y_t = wheel.tire.forces(kappa, alpha, F_z)

    assert np.isclose(F_x, F_x_t * np.cos(delta) - F_y_t * np.sin(delta))
    assert np.isclose(F_y, F_x_t * np.sin(delta) + F_y_t * np.cos(delta))
    assert np.isclose(M_y, wheel.R_w * F_x_t)


def test_negative_load_input_is_clamped():
    """A (nonphysical) negative F_z on the wire is clamped to zero before the
    tire call: a load-scaled tire produces no forces."""
    wheel = Wheel(tire=SimplePacejka())
    y = wheel._func_alg(np.array([0.0, 0.0, 20.0, 0.0, 0.0, -4000.0]))
    assert np.allclose(y, 0.0)


def test_braking_slip_gives_negative_force_and_torque():
    """An underspinning wheel (R_w*omega < v_cx) brakes and loads the driveline."""
    wheel = Wheel()
    v_x = 20.0
    F_x, F_y, M_y = wheel._func_alg(
        np.array([0.0, 0.5 * v_x / wheel.R_w, v_x, 0.0, 0.0, 4000.0]))
    assert F_x < 0.0
    assert np.isclose(F_y, 0.0)
    assert M_y < 0.0


def test_tire_model_is_swappable_and_pacejka_saturates():
    """A locked wheel (omega = 0) at speed: the LinearTire force grows with the
    slip while SimplePacejka saturates at mu_x * F_z."""
    F_z = 4000.0
    u = np.array([0.0, 0.0, 20.0, 0.0, 0.0, F_z])
    linear = Wheel(tire=LinearTire())
    pacejka = Wheel(tire=SimplePacejka())

    F_x_lin = linear._func_alg(u)[0]
    F_x_pac = pacejka._func_alg(u)[0]

    assert abs(F_x_pac) <= pacejka.tire.mu_x * F_z
    assert abs(F_x_lin) > 10 * abs(F_x_pac)


def _closed_loop_sim(delta, omega, v_x0, duration, tire=None):
    """Build and run the Wheel + SingleTrack architecture: one wheel per axle,
    front wheel steered, chassis velocities and static axle loads fed to both
    wheels. ``tire`` is an optional factory called with the wheel's static
    axle load, ``tire(F_z) -> TireModel``."""
    car = SingleTrack(m=1500.0, I_z=3000.0, l_f=1.2, l_r=1.4,
                      initial_value=[v_x0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def make_tire(F_z):
        return None if tire is None else tire(F_z)

    wf = Wheel(l=1.2, tire=make_tire(car.F_z_f))
    wr = Wheel(l=-1.4, tire=make_tire(car.F_z_r))
    c_delta, c_zero, c_omega = Constant(delta), Constant(0.0), Constant(omega)
    sc = Scope(labels=["v_x", "v_y", "r", "psi", "X", "Y"])

    conns = [
        Connection(c_delta, wf["delta"]),
        Connection(c_zero, wr["delta"]),
        Connection(c_omega, wf["omega"], wr["omega"]),
        # chassis velocity feedback to both wheels
        Connection(car["v_x"], wf["v_x"], wr["v_x"]),
        Connection(car["v_y"], wf["v_y"], wr["v_y"]),
        Connection(car["r"], wf["r"], wr["r"]),
        # static axle loads to the wheels
        Connection(car["F_z_f"], wf["F_z"]),
        Connection(car["F_z_r"], wr["F_z"]),
        # body-frame axle forces into the chassis
        Connection(wf["F_x"], car["F_x_f"]),
        Connection(wf["F_y"], car["F_y_f"]),
        Connection(wr["F_x"], car["F_x_r"]),
        Connection(wr["F_y"], car["F_y_r"]),
        ] + [Connection(car[i], sc[i]) for i in range(6)]

    sim = Simulation([c_delta, c_zero, c_omega, wf, wr, car, sc], conns,
                     dt=0.005, log=False)
    sim.run(duration)
    return sc.read()


def test_closed_loop_settles_at_rolling_speed():
    """Straight driving with constant omega: the only longitudinal equilibrium
    is kappa = 0, so v_x converges to R_w * omega."""
    omega, R_w = 40.0, 0.30
    t, data = _closed_loop_sim(delta=0.0, omega=omega, v_x0=8.0, duration=5.0)
    v_x, v_y, r = data[0], data[1], data[2]
    assert np.isclose(v_x[-1], R_w * omega, atol=1e-6)
    assert np.allclose(v_y, 0.0) and np.allclose(r, 0.0)


def test_closed_loop_steady_cornering_matches_linear_bicycle():
    """With LinearTire the steady-state yaw rate matches the classical
    single-track relation r = v_x * delta / (L + K_us * v_x^2) with the
    understeer gradient K_us = m/L * (l_r/C_f - l_f/C_r)."""
    delta = 0.05
    t, data = _closed_loop_sim(delta=delta, omega=10.0 / 0.30, v_x0=10.0,
                               duration=20.0)
    v_x, v_y, r = data[0], data[1], data[2]

    # steady state reached (yaw rate flat over the second half)
    assert np.isclose(r[len(t) // 2], r[-1], rtol=1e-3)

    m, l_f, l_r, C = 1500.0, 1.2, 1.4, 8.0e4
    L = l_f + l_r
    K_us = m / L * (l_r - l_f) / C
    r_pred = v_x[-1] * delta / (L + K_us * v_x[-1]**2)
    assert np.isclose(r[-1], r_pred, rtol=0.02)


def test_closed_loop_pacejka_is_neutral_then_friction_limited():
    """With the static axle loads on the wire and load-proportional peak
    forces, C ~ F_z cancels the lever arms (l_f*F_z_f = l_r*F_z_r): the car
    is NEUTRAL. Below the friction limit SimplePacejka therefore matches a
    LinearTire bridged at the axle loads, both at the neutral yaw rate
    r = v_x*delta/L. Beyond the limit (a_y demand > mu_y*g) the unbounded
    linear tire corners at unphysical lateral acceleration while the
    saturating tire breaks away and cannot sustain the demanded circle."""
    pac = SimplePacejka()
    bridged = lambda F_z: LinearTire(C_kappa=pac.B_x * pac.C_x * pac.mu_x * F_z,
                                     C_alpha=pac.B_y * pac.C_y * pac.mu_y * F_z)
    pacejka = lambda F_z: SimplePacejka()
    kwargs = dict(omega=20.0 / 0.30, v_x0=20.0, duration=15.0)
    L, g = 1.2 + 1.4, 9.81

    for delta in (0.01, 0.03):            # below the limit: rungs agree, neutral
        _, d_lin = _closed_loop_sim(delta=delta, tire=bridged, **kwargs)
        _, d_pac = _closed_loop_sim(delta=delta, tire=pacejka, **kwargs)
        assert np.isclose(d_pac[2][-1], d_lin[2][-1], rtol=0.02)
        assert np.isclose(d_pac[2][-1], d_pac[0][-1] * delta / L, rtol=0.02)

    delta = 0.08                          # a_y demand v_x^2*delta/L > mu_y*g
    _, d_lin = _closed_loop_sim(delta=delta, tire=bridged, **kwargs)
    _, d_pac = _closed_loop_sim(delta=delta, tire=pacejka, **kwargs)
    a_y_lin = d_lin[0][-1] * d_lin[2][-1]
    a_y_pac = d_pac[0][-1] * d_pac[2][-1]
    assert a_y_lin > pac.mu_y * g         # linear tire: unphysical grip
    assert a_y_pac < pac.mu_y * g         # saturating tire: circle not sustained
