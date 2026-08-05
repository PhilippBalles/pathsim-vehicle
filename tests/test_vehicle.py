#########################################################################################
##
##                  Tests for the SingleTrackVehicle assembled subsystem
##
#########################################################################################

import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Scope

from pathsim_vehicle.wheel import Wheel
from pathsim_vehicle.singleTrack import SingleTrack
from pathsim_vehicle.vehicle import SingleTrackVehicle


# shared vehicle parameters and maneuver -------------------------------------------------

PARAMS = dict(m=1500.0, I_z=3000.0, l_f=1.2, l_r=1.4)
V_X0 = 5.0
DELTA = 0.04      # front steer [rad]
T_D_R = 400.0     # rear drive torque [N m]
DT = 0.005
DURATION = 3.0


def _hand_wired_sim():
    """Reference: the three blocks wired by hand at the top level, exactly
    the idiom used in test_wheel.py. Only ``delta`` (front) and ``T`` (rear)
    are driven; every other command port is left unconnected (reads 0)."""
    chassis = SingleTrack(initial_value=[V_X0, 0.0, 0.0, 0.0, 0.0, 0.0], **PARAMS)
    front = Wheel(l=+PARAMS["l_f"], s=0.0)
    rear  = Wheel(l=-PARAMS["l_r"], s=0.0)
    c_delta, c_Tdr = Constant(DELTA), Constant(T_D_R)
    sc = Scope(labels=["v_x", "v_y", "r", "psi", "X", "Y"])

    conns = [
        Connection(chassis["v_x"], front["v_x"], rear["v_x"]),
        Connection(chassis["v_y"], front["v_y"], rear["v_y"]),
        Connection(chassis["r"],   front["r"],   rear["r"]),
        Connection(chassis["F_z_f"], front["F_z"]),
        Connection(chassis["F_z_r"], rear["F_z"]),
        Connection(front["F_x"], chassis["F_x_f"]),
        Connection(front["F_y"], chassis["F_y_f"]),
        Connection(rear["F_x"],  chassis["F_x_r"]),
        Connection(rear["F_y"],  chassis["F_y_r"]),
        Connection(c_delta, front["delta"]),
        Connection(c_Tdr,   rear["T"]),
        ] + [Connection(chassis[i], sc[i]) for i in range(6)]

    sim = Simulation([chassis, front, rear, c_delta, c_Tdr, sc], conns,
                     dt=DT, log=False)
    sim.run(DURATION)
    return sc.read()


def _factory_sim():
    """The same car built by the SingleTrackVehicle subclass, driven through
    its Interface with the identical commands."""
    car = SingleTrackVehicle(initial_value=[V_X0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             **PARAMS)
    c_delta, c_Tdr = Constant(DELTA), Constant(T_D_R)
    sc = Scope(labels=["v_x", "v_y", "r", "psi", "X", "Y"])

    names = ["v_x", "v_y", "r", "psi", "X", "Y"]
    conns = [
        Connection(c_delta, car["delta"]),
        Connection(c_Tdr,   car["T_r"]),
        ] + [Connection(car[nm], sc[i]) for i, nm in enumerate(names)]

    sim = Simulation([car, c_delta, c_Tdr, sc], conns, dt=DT, log=False)
    sim.run(DURATION)
    return sc.read()


# the gate: bit-for-bit equivalence ------------------------------------------------------

def test_factory_matches_hand_wired():
    """The assembled subsystem must reproduce the hand-wired car exactly.
    A macroscopic divergence here means a swapped port, a wrong mounting
    offset, or the register_port_map inversion got flipped; the tolerance is
    tight enough that only such structural bugs can trip it, loose enough to
    tolerate benign floating-point reordering across the subsystem boundary."""
    t_ref, d_ref = _hand_wired_sim()
    t_sub, d_sub = _factory_sim()

    assert np.allclose(t_ref, t_sub)
    assert len(d_ref) == len(d_sub) == 6
    for ref, sub in zip(d_ref, d_sub):
        assert np.allclose(ref, sub, atol=1e-9, rtol=1e-9)


# smoke / physical sanity ----------------------------------------------------------------

def test_standstill_launch_is_loop_free_at_the_recommended_step():
    """A standstill launch on the rear axle runs stably at dt=0.005 (the
    relaxation-enabled step) and accelerates the car forward — proof the
    assembled graph carries no stiff algebraic loop across the boundary."""
    car = SingleTrackVehicle(initial_value=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             **PARAMS)
    c_Tdr = Constant(T_D_R)
    sc = Scope(labels=["v_x"])
    conns = [Connection(c_Tdr, car["T_r"]), Connection(car["v_x"], sc[0])]
    sim = Simulation([car, c_Tdr, sc], conns, dt=DT, log=False)
    sim.run(4.0)

    t, data = sc.read()
    v_x = data[0] if data.ndim > 1 else data
    assert np.isfinite(v_x).all()
    assert v_x[-1] > 1.0            # the car actually launched
    assert v_x[-1] > v_x[0]         # forward


def test_front_drive_launches_via_the_front_axle():
    """FWD case: only ``T_f`` is connected (``T_r`` left open, so the rear
    axle freewheels). The car launches, and record=True proves the drive
    force appears on the FRONT wheel and not the rear — covering the
    ``interface["T_f"] -> front["T"]`` path that every other test leaves
    unexercised, and catching a T_f/T_r swap a bare launch assertion misses."""
    car = SingleTrackVehicle(initial_value=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             record=True, **PARAMS)
    c_Tdf = Constant(T_D_R)
    sc = Scope(labels=["v_x"])
    conns = [Connection(c_Tdf, car["T_f"]), Connection(car["v_x"], sc[0])]
    sim = Simulation([car, c_Tdf, sc], conns, dt=DT, log=False)
    sim.run(4.0)

    t, data = sc.read()
    v_x = data[0] if data.ndim > 1 else data
    assert np.isfinite(v_x).all()
    assert v_x[-1] > 1.0            # the car actually launched
    assert v_x[-1] > v_x[0]         # forward

    _, d_front = car.scope_front.read()    # [F_x, F_y, omega, M_y]
    _, d_rear  = car.scope_rear.read()
    half = d_front.shape[1] // 2
    # the front sustains a large forward (tractive) force; the undriven rear
    # carries only the small drag that spins its wheel up from omega_0=0 to
    # rolling, so the front dominates by two orders of magnitude. A T_f/T_r
    # swap would move the big force to the rear and flip both assertions.
    assert np.mean(d_front[0, half:]) > 100.0
    assert (np.mean(np.abs(d_front[0, half:]))
            > 10.0 * np.mean(np.abs(d_rear[0, half:])))


def test_steer_command_develops_yaw_of_the_right_sign():
    """A positive (left) steer while the car rolls forward develops a
    positive yaw rate — the assembled car corners."""
    car = SingleTrackVehicle(initial_value=[15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             **PARAMS)
    c_delta, c_Tdr = Constant(0.03), Constant(200.0)
    sc = Scope(labels=["r"])
    conns = [Connection(c_delta, car["delta"]),
             Connection(c_Tdr, car["T_r"]),
             Connection(car["r"], sc[0])]
    sim = Simulation([car, c_delta, c_Tdr, sc], conns, dt=DT, log=False)
    sim.run(2.0)

    t, data = sc.read()
    r = data[0] if data.ndim > 1 else data
    assert np.isfinite(r).all()
    assert np.mean(r[len(r) // 2:]) > 0.0   # settled into a left turn


# observability --------------------------------------------------------------------------

def test_record_exposes_internal_wheel_forces():
    """record=True drops diagnostic scopes inside the subsystem; after a run
    the rear (driven) wheel shows a non-zero F_x, and a steer produces a
    non-zero front F_y — signals that never reach the Interface."""
    car = SingleTrackVehicle(initial_value=[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             record=True, **PARAMS)
    c_delta, c_Tdr = Constant(0.04), Constant(400.0)
    conns = [Connection(c_delta, car["delta"]),
             Connection(c_Tdr, car["T_r"])]
    sim = Simulation([car, c_delta, c_Tdr], conns, dt=DT, log=False)
    sim.run(1.5)

    t_f, data_f = car.scope_front.read()   # [F_x, F_y, omega, M_y]
    t_r, data_r = car.scope_rear.read()
    assert data_f.shape[0] == 4 and data_r.shape[0] == 4
    assert np.any(np.abs(data_r[0]) > 1.0)   # rear F_x: axle is driven
    assert np.any(np.abs(data_f[1]) > 1.0)   # front F_y: steer builds lateral force


# interface contract ---------------------------------------------------------------------

def test_interface_contract_and_handles():
    """The car exposes the declared command inputs and motion outputs, and
    keeps the three constituent blocks reachable as handles."""
    car = SingleTrackVehicle(**PARAMS)

    for name in ["delta", "delta_r", "T_f", "T_r"]:
        assert name in car.inputs
    for name in ["v_x", "v_y", "r", "psi", "X", "Y"]:
        assert name in car.outputs

    assert isinstance(car.front, Wheel)
    assert isinstance(car.rear, Wheel)
    assert isinstance(car.chassis, SingleTrack)
    # the geometry lives only on the wheels' mounting offsets
    assert car.front.l == +PARAMS["l_f"]
    assert car.rear.l == -PARAMS["l_r"]
