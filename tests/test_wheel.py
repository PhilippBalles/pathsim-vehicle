"""Tests for the Wheel block, including the torque-driven closed loop with
SingleTrack. Closed-loop rigs run at dt = 0.005: the contact-patch
relaxation caps the fast wheel mode at the speed-independent omega_n
(~158 1/s at the defaults), which is the point of the design."""

import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Scope, Adder, PID

from pathsim_vehicle.wheel import Wheel
from pathsim_vehicle.singleTrack import SingleTrack
from pathsim_vehicle.tiremodels import LinearTire, SimplePacejka


def test_default_tire_is_linear():
    """With no tire argument the Wheel constructs a LinearTire."""
    assert isinstance(Wheel().tire, LinearTire)


def test_initial_state():
    """omega_0 lands in state 0; the patch deflections start at zero
    (exact for pure rolling)."""
    assert np.allclose(Wheel(omega_0=66.7).initial_value, [66.7, 0.0, 0.0])
    assert np.allclose(Wheel().initial_value, [0.0, 0.0, 0.0])


def test_rhs_is_the_torque_balance():
    """I_w*domega/dt = T_d - T_b - R_w*F_x_t with the force from the PATCH
    state: an unloaded patch transmits nothing regardless of the spin
    speed; a loaded patch opposes the spin."""
    wheel = Wheel(I_w=1.2)
    v_x, T_d, T_b = 20.0, 300.0, 50.0
    u = np.array([0.0, T_d, T_b, v_x, 0.0, 0.0, 4000.0])

    # unloaded patch (u_x = 0): net torque alone accelerates the wheel
    domega = wheel._func_dyn(np.array([v_x / wheel.R_w, 0.0, 0.0]), u, 0.0)[0]
    assert np.isclose(domega, (T_d - T_b) / wheel.I_w)

    # loaded patch: F_x_t = C_kappa * u_x / sigma_x opposes the spin
    u_x = 0.003
    F_x_t = wheel.tire.C_kappa * u_x / wheel.sigma_x
    domega_loaded = wheel._func_dyn(
        np.array([v_x / wheel.R_w, u_x, 0.0]), u, 0.0)[0]
    assert np.isclose(domega_loaded,
                      (T_d - T_b - wheel.R_w * F_x_t) / wheel.I_w)


def test_patch_relaxes_toward_the_steady_slips():
    """du = (v_bar/sigma) * (sigma*slip_ss - u): the patch loads toward the
    rigid steady-state slips and is stationary exactly there."""
    wheel = Wheel()
    delta, omega, v_x = 0.05, 40.0, 15.0
    u = np.array([delta, 0.0, 0.0, v_x, 0.0, 0.0, 4000.0])

    v_bar = np.sqrt(v_x**2 + wheel.v_eps**2)
    kappa_ss = (wheel.R_w * omega - v_x) / v_bar
    alpha_ss = delta - np.arctan2(0.0, v_x)

    # empty patch: derivatives point toward the steady deflections
    _, du_x, du_y = wheel._func_dyn(np.array([omega, 0.0, 0.0]), u, 0.0)
    assert np.isclose(du_x, v_bar * kappa_ss)
    assert np.isclose(du_y, v_bar * alpha_ss)

    # steady patch: derivatives vanish
    x_star = np.array([omega, wheel.sigma_x * kappa_ss,
                       wheel.sigma_y * alpha_ss])
    _, du_x, du_y = wheel._func_dyn(x_star, u, 0.0)
    assert np.isclose(du_x, 0.0, atol=1e-12)
    assert np.isclose(du_y, 0.0, atol=1e-12)


def test_low_speed_damping_ramp():
    """Below v_low the transmitted force carries the carcass-damping term
    d(v_bar)*du_x; above v_low the force is exactly the tire law's."""
    wheel = Wheel()
    x = np.array([10.0, 0.0, 0.0])          # spinning wheel, empty patch

    # at standstill: F = tire(0) + d_low_ramped * du_x, tire term is zero
    u_still = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4000.0])
    F_x = wheel._func_alg(x, u_still, 0.0)[0]
    v_bar = wheel.v_eps
    du_x = wheel.R_w * x[0]                  # v_bar*kappa_ss at v_cx = 0
    d = wheel.d_low * 0.5 * (1 + np.cos(np.pi * v_bar / wheel.v_low))
    assert np.isclose(F_x, d * du_x)

    # at speed: damping fully ramped out, empty patch transmits nothing
    u_fast = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 4000.0])
    assert np.isclose(wheel._func_alg(x, u_fast, 0.0)[0], 0.0)


def test_pure_rolling_gives_zero_forces():
    """omega = v_cx / R_w with an empty patch: no slip targets, no patch
    motion, no forces; the omega output reports the state."""
    wheel = Wheel()
    v_x = 20.0
    omega = v_x / wheel.R_w
    y = wheel._func_alg(np.array([omega, 0.0, 0.0]),
                        np.array([0.0, 0.0, 0.0, v_x, 0.0, 0.0, 4000.0]), 0.0)
    assert np.allclose([y[0], y[1], y[3]], 0.0)
    assert np.isclose(y[2], omega)


def test_standstill_is_well_defined():
    """All states and inputs zero -> zero slip targets (atan2(0,0) = 0),
    zero patch motion, zero forces, no NaN."""
    wheel = Wheel()
    assert np.allclose(wheel._func_alg(np.zeros(3), np.zeros(7), 0.0), 0.0)
    assert np.allclose(wheel._func_dyn(np.zeros(3), np.zeros(7), 0.0), 0.0)


def test_algebraic_chain_matches_equations():
    """_func_alg evaluates the tire at the TRANSIENT slips u/sigma and
    rotates by the steer angle (above v_low, so no damping term)."""
    wheel = Wheel(l=1.2, s=0.0, R_w=0.30, v_eps=1.0)
    delta, omega, v_x, v_y, r, F_z = 0.08, 40.0, 15.0, 0.4, 0.25, 4000.0
    u_x, u_y = 0.004, -0.01
    F_x, F_y, omega_out, M_y = wheel._func_alg(
        np.array([omega, u_x, u_y]),
        np.array([delta, 0.0, 0.0, v_x, v_y, r, F_z]), 0.0)

    F_x_t, F_y_t = wheel.tire.forces(u_x / wheel.sigma_x,
                                     u_y / wheel.sigma_y, F_z)

    assert np.isclose(F_x, F_x_t * np.cos(delta) - F_y_t * np.sin(delta))
    assert np.isclose(F_y, F_x_t * np.sin(delta) + F_y_t * np.cos(delta))
    assert np.isclose(omega_out, omega)
    assert np.isclose(M_y, wheel.R_w * F_x_t)


def test_negative_load_input_is_clamped():
    """A (nonphysical) negative F_z on the wire is clamped to zero before
    the tire call: a load-scaled tire produces no forces even with a
    loaded patch."""
    wheel = Wheel(tire=SimplePacejka())
    y = wheel._func_alg(np.array([0.0, 0.05, 0.1]),
                        np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0, -4000.0]),
                        0.0)
    assert np.allclose(y, 0.0)


def test_braking_slip_gives_negative_force_and_torque():
    """An underspinning wheel with the patch at its steady deflection
    brakes; the load torque observation goes negative (the road spins the
    wheel up)."""
    wheel = Wheel()
    v_x = 20.0
    omega = 0.5 * v_x / wheel.R_w
    kappa_ss = (wheel.R_w * omega - v_x) / np.sqrt(v_x**2 + wheel.v_eps**2)
    F_x, F_y, _, M_y = wheel._func_alg(
        np.array([omega, wheel.sigma_x * kappa_ss, 0.0]),
        np.array([0.0, 0.0, 0.0, v_x, 0.0, 0.0, 4000.0]), 0.0)
    assert F_x < 0.0
    assert np.isclose(F_y, 0.0)
    assert M_y < 0.0


def test_tire_model_is_swappable_and_pacejka_saturates():
    """A locked wheel (omega = 0) at speed, patch at steady deflection:
    the LinearTire force grows with the slip while SimplePacejka
    saturates at mu_x * F_z."""
    F_z, v_x = 4000.0, 20.0
    linear = Wheel(tire=LinearTire())
    pacejka = Wheel(tire=SimplePacejka())
    kappa_ss = -v_x / np.sqrt(v_x**2 + linear.v_eps**2)
    x = np.array([0.0, linear.sigma_x * kappa_ss, 0.0])
    u = np.array([0.0, 0.0, 0.0, v_x, 0.0, 0.0, F_z])

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
                     dt=0.005, log=False)
    sim.run(duration)
    return sc.read(), wf.R_w


def test_closed_loop_free_rolling_settles():
    """Zero torque: the (damped) patch-spin dynamics find pure rolling on
    their own, even from a 20% overspin start."""
    (t, data), R_w = _torque_driven_car_sim(T_d=0.0, T_b=0.0, v_x0=20.0,
                                            duration=2.0, omega_scale=1.2)
    v_x, omega_f, omega_r = data[0], data[1], data[2]
    assert np.isclose(R_w * omega_f[-1], v_x[-1], atol=1e-6)
    assert np.isclose(R_w * omega_r[-1], v_x[-1], atol=1e-6)


def test_closed_loop_accelerates_at_effective_mass_rate():
    """Constant drive torque on both axles: quasi-steady acceleration
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


def test_closed_loop_standstill_launch_at_the_recommended_step():
    """The payoff of the relaxation states: a standstill launch (the
    stiffest regime of the rigid-slip wheel, eigenvalue ~ -7600 1/s) runs
    stably and accurately at dt = 0.005 and reaches the effective-mass
    speed."""
    T_d, duration = 300.0, 4.0
    (t, data), R_w = _torque_driven_car_sim(T_d=T_d, T_b=0.0, v_x0=0.0,
                                            duration=duration)
    v_x = data[0]
    m, I_w = 1500.0, 1.2
    a_pred = (2 * T_d / R_w) / (m + 2 * I_w / R_w**2)

    assert np.isfinite(v_x).all()
    # the launch transient is over quickly; the mean acceleration over the
    # full run already matches the quasi-steady prediction to < 1%
    assert np.isclose(v_x[-1] / duration, a_pred, rtol=1e-2)


def _cruise_cornering_sim(delta, v_ref, duration, tire=None):
    """Cornering at a PID-held speed: a cruise controller turns the speed
    error into a drive torque on both axles. ``tire`` is an optional
    factory called with the wheel's static axle load,
    ``tire(F_z) -> TireModel``."""
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
                     conns, dt=0.005, log=False)
    sim.run(duration)
    return sc.read()


def test_closed_loop_steady_cornering_matches_linear_bicycle():
    """With LinearTire the steady-state yaw rate matches the classical
    single-track relation r = v_x * delta / (L + K_us * v_x^2) with the
    understeer gradient K_us = m/L * (l_r/C_f - l_f/C_r). The patch
    states change nothing stationary (exact reduction, derive_wheel.py)."""
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
    linear tire holds the circle at unphysical lateral acceleration while
    the saturating tire breaks away and leaves the neutral relation
    entirely (plow-out or spin, depending on the transients -- with the
    lagged patch force the breakaway here is a spin)."""
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
    assert a_y_lin > pac.mu_y * g         # linear tire: unphysical grip...
    assert np.isclose(d_lin[2][-1], d_lin[0][-1] * delta / L,
                      rtol=0.02)          # ...and the circle is held
    # saturating tire: the demanded circle is NOT sustained -- the yaw rate
    # departs from the neutral relation entirely
    r_neutral = d_pac[0][-1] * delta / L
    assert not np.isclose(d_pac[2][-1], r_neutral, rtol=0.10)
