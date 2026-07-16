"""Tests for the Wheel block, including the torque-driven closed loop with
SingleTrack."""

import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Scope, Adder, PID

from pathsim_vehicle.wheel import Wheel
from pathsim_vehicle.singleTrack import SingleTrack
from pathsim_vehicle.tiremodels import LinearTire, SimplePacejka


def test_default_tire_is_linear():
    """With no tire argument the Wheel constructs a LinearTire."""
    assert isinstance(Wheel().tire, LinearTire)


def test_initial_spin_speed():
    """omega_0 lands in the (one-element) initial state vector."""
    assert np.allclose(Wheel(omega_0=66.7).initial_value, [66.7])
    assert np.allclose(Wheel().initial_value, [0.0])


def test_rhs_is_the_torque_balance():
    """The rhs is I_w*domega/dt = T_d - T_b - R_w*F_x_t with the load torque
    computed internally: at pure rolling (F_x_t = 0) the net torque alone
    accelerates the wheel; an overspinning state reduces the derivative."""
    wheel = Wheel(I_w=1.2)
    v_x, T_d, T_b = 20.0, 300.0, 50.0
    omega_roll = v_x / wheel.R_w
    u = np.array([0.0, T_d, T_b, v_x, 0.0, 0.0, 4000.0])

    domega = wheel._func_dyn(np.array([omega_roll]), u, 0.0)
    assert np.isclose(domega[0], (T_d - T_b) / wheel.I_w)

    # state feedback: overspin -> F_x_t > 0 -> load torque opposes the spin
    domega_over = wheel._func_dyn(np.array([1.2 * omega_roll]), u, 0.0)
    assert domega_over[0] < domega[0]


def test_pure_rolling_gives_zero_forces():
    """omega = v_cx / R_w and no lateral motion -> kappa = alpha = 0 -> no
    forces; the omega output reports the state."""
    wheel = Wheel()
    v_x = 20.0
    omega = v_x / wheel.R_w
    y = wheel._func_alg(np.array([omega]),
                        np.array([0.0, 0.0, 0.0, v_x, 0.0, 0.0, 4000.0]), 0.0)
    assert np.allclose([y[0], y[1], y[3]], 0.0)
    assert np.isclose(y[2], omega)


def test_standstill_is_well_defined():
    """All inputs and the state zero -> zero slip (atan2(0,0) = 0) -> zero
    forces, no NaN."""
    wheel = Wheel()
    assert np.allclose(wheel._func_alg(np.zeros(1), np.zeros(7), 0.0), 0.0)


def test_algebraic_chain_matches_equations():
    """_func_alg reproduces the documented kinematics -> tire -> rotation chain."""
    wheel = Wheel(l=1.2, s=0.0, R_w=0.30, v_eps=1.0)
    delta, omega, v_x, v_y, r, F_z = 0.08, 40.0, 15.0, 0.4, 0.25, 4000.0
    F_x, F_y, omega_out, M_y = wheel._func_alg(
        np.array([omega]), np.array([delta, 0.0, 0.0, v_x, v_y, r, F_z]), 0.0)

    v_cx = v_x - wheel.s * r
    v_cy = v_y + wheel.l * r
    kappa = (wheel.R_w * omega - v_cx) / np.sqrt(v_cx**2 + wheel.v_eps**2)
    alpha = delta - np.arctan2(v_cy, v_cx)
    F_x_t, F_y_t = wheel.tire.forces(kappa, alpha, F_z)

    assert np.isclose(F_x, F_x_t * np.cos(delta) - F_y_t * np.sin(delta))
    assert np.isclose(F_y, F_x_t * np.sin(delta) + F_y_t * np.cos(delta))
    assert np.isclose(omega_out, omega)
    assert np.isclose(M_y, wheel.R_w * F_x_t)


def test_negative_load_input_is_clamped():
    """A (nonphysical) negative F_z on the wire is clamped to zero before the
    tire call: a load-scaled tire produces no forces."""
    wheel = Wheel(tire=SimplePacejka())
    y = wheel._func_alg(np.zeros(1),
                        np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0, -4000.0]), 0.0)
    assert np.allclose(y, 0.0)


def test_braking_slip_gives_negative_force_and_torque():
    """An underspinning wheel (R_w*omega < v_cx) brakes; the load torque
    observation goes negative (the road spins the wheel up)."""
    wheel = Wheel()
    v_x = 20.0
    F_x, F_y, _, M_y = wheel._func_alg(
        np.array([0.5 * v_x / wheel.R_w]),
        np.array([0.0, 0.0, 0.0, v_x, 0.0, 0.0, 4000.0]), 0.0)
    assert F_x < 0.0
    assert np.isclose(F_y, 0.0)
    assert M_y < 0.0


def test_tire_model_is_swappable_and_pacejka_saturates():
    """A locked wheel (omega = 0) at speed: the LinearTire force grows with the
    slip while SimplePacejka saturates at mu_x * F_z."""
    F_z = 4000.0
    x, u = np.zeros(1), np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0, F_z])
    linear = Wheel(tire=LinearTire())
    pacejka = Wheel(tire=SimplePacejka())

    F_x_lin = linear._func_alg(x, u, 0.0)[0]
    F_x_pac = pacejka._func_alg(x, u, 0.0)[0]

    assert abs(F_x_pac) <= pacejka.tire.mu_x * F_z
    assert abs(F_x_lin) > 10 * abs(F_x_pac)


def _torque_driven_car_sim(T_d, T_b, v_x0, duration, omega_scale=1.0):
    """Straight-driving torque-driven car: SingleTrack + one Wheel per axle
    (LinearTire). ``T_d``/``T_b`` are applied to both wheels; ``omega_scale``
    scales the rolling-start spin speed."""
    car = SingleTrack(m=1500.0, I_z=3000.0, l_f=1.2, l_r=1.4,
                      initial_value=[v_x0, 0.0, 0.0, 0.0, 0.0, 0.0])
    omega_0 = omega_scale * v_x0 / 0.30
    wf = Wheel(l=1.2, omega_0=omega_0)
    wr = Wheel(l=-1.4, omega_0=omega_0)
    c_zero, c_Td, c_Tb = Constant(0.0), Constant(T_d), Constant(T_b)
    sc = Scope(labels=["v_x", "omega_f", "omega_r"])

    conns = [
        Connection(c_zero, wf["delta"], wr["delta"]),
        # torque sources into the wheels
        Connection(c_Td, wf["T_d"], wr["T_d"]),
        Connection(c_Tb, wf["T_b"], wr["T_b"]),
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
        # wheel spin speeds for inspection
        Connection(wf["omega"], sc[1]),
        Connection(wr["omega"], sc[2]),
        ]

    sim = Simulation([c_zero, c_Td, c_Tb, wf, wr, car, sc], conns,
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
    dv_x/dt = 2*T_d/R_w / (m + 2*I_w/R_w^2) (verified in derive_wheel.py;
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


def _cruise_cornering_sim(delta, v_ref, duration, tire=None):
    """Cornering at a PID-held speed: a cruise controller turns the speed
    error into a drive torque on both axles (the physical replacement for
    the retired prescribed-omega source). ``tire`` is an optional factory
    called with the wheel's static axle load, ``tire(F_z) -> TireModel``."""
    car = SingleTrack(m=1500.0, I_z=3000.0, l_f=1.2, l_r=1.4,
                      initial_value=[v_ref, 0.0, 0.0, 0.0, 0.0, 0.0])

    def make_tire(F_z):
        return None if tire is None else tire(F_z)

    omega_0 = v_ref / 0.30
    wf = Wheel(l=1.2, omega_0=omega_0, tire=make_tire(car.F_z_f))
    wr = Wheel(l=-1.4, omega_0=omega_0, tire=make_tire(car.F_z_r))
    c_delta, c_zero, c_ref = Constant(delta), Constant(0.0), Constant(v_ref)
    err = Adder("+-")
    pid = PID(Kp=1000.0, Ki=500.0)
    sc = Scope(labels=["v_x", "v_y", "r", "psi", "X", "Y"])

    conns = [
        Connection(c_delta, wf["delta"]),
        Connection(c_zero, wr["delta"]),
        # cruise control: speed error -> PID -> drive torque on both axles
        Connection(c_ref, err[0]),
        Connection(car["v_x"], err[1], wf["v_x"], wr["v_x"]),
        Connection(err, pid),
        Connection(pid, wf["T_d"], wr["T_d"]),
        # chassis velocity feedback and static axle loads to both wheels
        Connection(car["v_y"], wf["v_y"], wr["v_y"]),
        Connection(car["r"], wf["r"], wr["r"]),
        Connection(car["F_z_f"], wf["F_z"]),
        Connection(car["F_z_r"], wr["F_z"]),
        # body-frame axle forces into the chassis
        Connection(wf["F_x"], car["F_x_f"]),
        Connection(wf["F_y"], car["F_y_f"]),
        Connection(wr["F_x"], car["F_x_r"]),
        Connection(wr["F_y"], car["F_y_r"]),
        ] + [Connection(car[i], sc[i]) for i in range(6)]

    sim = Simulation([c_delta, c_zero, c_ref, err, pid, wf, wr, car, sc],
                     conns, dt=0.001, log=False)
    sim.run(duration)
    return sc.read()


def test_closed_loop_steady_cornering_matches_linear_bicycle():
    """With LinearTire the steady-state yaw rate matches the classical
    single-track relation r = v_x * delta / (L + K_us * v_x^2) with the
    understeer gradient K_us = m/L * (l_r/C_f - l_f/C_r)."""
    delta, v_ref = 0.05, 10.0
    t, data = _cruise_cornering_sim(delta=delta, v_ref=v_ref, duration=20.0)
    v_x, v_y, r = data[0], data[1], data[2]

    # steady state reached (yaw rate flat over the second half)
    assert np.isclose(r[len(t) // 2], r[-1], rtol=1e-3)
    # the cruise controller holds the reference speed exactly (integral action)
    assert np.isclose(v_x[-1], v_ref, rtol=1e-4)

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
    kwargs = dict(v_ref=20.0, duration=15.0)
    L, g = 1.2 + 1.4, 9.81

    for delta in (0.01, 0.03):            # below the limit: rungs agree, neutral
        _, d_lin = _cruise_cornering_sim(delta=delta, tire=bridged, **kwargs)
        _, d_pac = _cruise_cornering_sim(delta=delta, tire=pacejka, **kwargs)
        assert np.isclose(d_pac[2][-1], d_lin[2][-1], rtol=0.02)
        assert np.isclose(d_pac[2][-1], d_pac[0][-1] * delta / L, rtol=0.02)

    delta = 0.08                          # a_y demand v_x^2*delta/L > mu_y*g
    _, d_lin = _cruise_cornering_sim(delta=delta, tire=bridged, **kwargs)
    _, d_pac = _cruise_cornering_sim(delta=delta, tire=pacejka, **kwargs)
    a_y_lin = d_lin[0][-1] * d_lin[2][-1]
    a_y_pac = d_pac[0][-1] * d_pac[2][-1]
    assert a_y_lin > pac.mu_y * g         # linear tire: unphysical grip
    assert a_y_pac < pac.mu_y * g         # saturating tire: circle not sustained
