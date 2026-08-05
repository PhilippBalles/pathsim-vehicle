#########################################################################################
##
##                  Single-Track Vehicle (assembled subsystem)
##
#########################################################################################



# IMPORTS ===============================================================================

from pathsim import Subsystem, Interface, Connection
from pathsim.blocks import Scope
from pathsim.solvers import SSPRK22

from .wheel import Wheel
from .singleTrack import SingleTrack


# BLOCK Definitions ================================================================================

class SingleTrackVehicle(Subsystem):
    """Front/rear single-track (bicycle) car, assembled from existing blocks.

    This is an *assembly*, not new physics: it instantiates one
    :class:`SingleTrack` chassis and two :class:`Wheel` blocks (front and
    rear axle), wires them, and exposes the car's physical connector through
    the subsystem :class:`~pathsim.subsystem.Interface`. Neither ``Wheel``
    nor ``SingleTrack`` is modified — every equation still lives, and is
    still gated, in those blocks. Because it subclasses
    :class:`~pathsim.subsystem.Subsystem`, the vehicle *is* an ordinary
    PathSim block: it nests inside larger systems, composes into the solver
    (``set_solver``/``step``/``solve`` and algebraic-loop handling all
    recurse), and checkpoints under its own type name.

    The Interface carries only what a real car exposes at its boundary —
    **commands in, motion out**. The three constituent blocks stay reachable
    as the attributes ``front``, ``rear`` and ``chassis`` (so any diagnostic
    signal, e.g. ``vehicle.front["F_y"]``, can be wired to a top-level
    ``Scope``); with ``record=True`` the factory additionally drops
    diagnostic scopes *inside* the subsystem, returned as ``scope_front`` and
    ``scope_rear``.

    The only place the front/rear geometry enters is each wheel's mounting
    offset — ``l = +l_f`` (front) / ``l = -l_r`` (rear), ``s = 0`` on the
    single-track centre line. Centralising those offsets here is the point of
    the block: it is the most error-prone part of the hand wiring.

    No algebraic loop crosses the boundary. The wheels have feedthrough
    (steer/velocity -> force) but ``SingleTrack``'s output is state-only, so
    the chassis->wheel edge is algebraic while wheel->chassis closes through
    the chassis integrator. The interface->interface path is therefore not
    algebraic, and the vehicle reports as a loop-free block to the top level.


    Interface Input Ports (commands; every one optional, unconnected -> 0)
    ---------------------------------------------------------------------
    delta : float
        front steer angle [rad]
    delta_r : float
        rear steer angle [rad] (0 for a conventional front-steer car; a
        four-wheel-steering controller simply connects this port)
    T_f : float
        front-axle net spin-axis torque [N m] (signed drive minus brake;
        sum several sources with an ``Adder`` upstream)
    T_r : float
        rear-axle net spin-axis torque [N m]

    Interface Output Ports (motion; the chassis state)
    --------------------------------------------------
    v_x, v_y : float
        body-frame longitudinal / lateral velocity [m/s]
    r : float
        yaw rate [rad/s]
    psi : float
        yaw angle [rad]
    X, Y : float
        global position [m]


    Parameters
    ----------
    m : float
        vehicle mass [kg]
    I_z : float
        yaw moment of inertia [kg m^2]
    l_f : float
        CG-to-front-axle distance [m]
    l_r : float
        CG-to-rear-axle distance [m]
    g : float
        gravitational acceleration [m/s^2]
    tire_f, tire_r : TireModel | None
        tire force laws for the front / rear wheel. ``None`` lets each
        ``Wheel`` install its own default (``LinearTire``); the laws are
        stateless so distinct instances are only cosmetic.
    initial_value : array_like | None
        chassis initial state ``[v_x, v_y, r, psi, X, Y]``; wheels start at
        ``omega_0 = 0``.
    record : bool
        if ``True``, wire a diagnostic ``Scope`` to each wheel's
        ``F_x, F_y, omega, M_y`` inside the subsystem, exposed as
        ``scope_front`` / ``scope_rear`` (read with ``.read()`` after a run).
    """

    def __init__(self, m=1500.0, I_z=3000.0, l_f=1.2, l_r=1.4, g=9.81,
                 tire_f=None, tire_r=None, initial_value=None, record=False):

        # --- the physical parts (real blocks; both kept unchanged) ---
        self.chassis = SingleTrack(m=m, I_z=I_z, l_f=l_f, l_r=l_r, g=g,
                                   initial_value=initial_value)
        # mounting offsets are the ONLY place the front/rear geometry enters
        self.front = Wheel(l=+l_f, s=0.0, tire=tire_f)
        self.rear  = Wheel(l=-l_r, s=0.0, tire=tire_r)

        # --- boundary: the Interface is the car's physical connector ---
        interface = Interface()
        # Subsystem inversion: interface OUTPUTS feed the internal blocks, so
        # the car's command INPUTS live in port_map_out; the car's motion
        # OUTPUTS (fed from the chassis) live in port_map_in. This mapping
        # must be registered before any Connection references the interface.
        interface.register_port_map(
            port_map_in={"v_x": 0, "v_y": 1, "r": 2, "psi": 3, "X": 4, "Y": 5},
            port_map_out={"delta": 0, "delta_r": 1, "T_f": 2, "T_r": 3},
            )

        blocks = [interface, self.chassis, self.front, self.rear]

        conns = [
            # chassis motion -> both axles' velocity inputs
            Connection(self.chassis["v_x"], self.front["v_x"], self.rear["v_x"]),
            Connection(self.chassis["v_y"], self.front["v_y"], self.rear["v_y"]),
            Connection(self.chassis["r"],   self.front["r"],   self.rear["r"]),
            # static axle loads -> each wheel's vertical load
            Connection(self.chassis["F_z_f"], self.front["F_z"]),
            Connection(self.chassis["F_z_r"], self.rear["F_z"]),
            # body-frame axle forces -> chassis inputs
            Connection(self.front["F_x"], self.chassis["F_x_f"]),
            Connection(self.front["F_y"], self.chassis["F_y_f"]),
            Connection(self.rear["F_x"],  self.chassis["F_x_r"]),
            Connection(self.rear["F_y"],  self.chassis["F_y_r"]),
            # boundary: commands in (unconnected top-level ports read 0)
            Connection(interface["delta"],   self.front["delta"]),
            Connection(interface["delta_r"], self.rear["delta"]),
            Connection(interface["T_f"],     self.front["T"]),
            Connection(interface["T_r"],     self.rear["T"]),
            # boundary: motion out
            Connection(self.chassis["v_x"],  interface["v_x"]),
            Connection(self.chassis["v_y"],  interface["v_y"]),
            Connection(self.chassis["r"],    interface["r"]),
            Connection(self.chassis["psi"],  interface["psi"]),
            Connection(self.chassis["X"],    interface["X"]),
            Connection(self.chassis["Y"],    interface["Y"]),
            ]

        # TEMPORARY SHIM (pending an upstream core fix — see the finding doc
        # "dynamicalsystem_subsystem_issue.md"). Subsystem assembles its
        # internal graph at construction, which probes each block's feedthrough
        # via len(block); DynamicalSystem.__len__ reads engine.state, but the
        # parent Simulation only assigns engines later (via Subsystem.set_solver).
        # Give the internal blocks a provisional engine so graph assembly can
        # run; the parent overwrites it on set_solver, so the choice of SSPRK22
        # here is immaterial. Remove this once __len__ falls back to
        # initial_value when engine is None.
        for _blk in (self.chassis, self.front, self.rear):
            _blk.set_solver(SSPRK22, None)

        # --- optional diagnostic scopes, recorded INSIDE the subsystem ---
        self.scope_front = None
        self.scope_rear  = None
        if record:
            self.scope_front = Scope(labels=["F_x", "F_y", "omega", "M_y"])
            self.scope_rear  = Scope(labels=["F_x", "F_y", "omega", "M_y"])
            blocks += [self.scope_front, self.scope_rear]
            conns += [
                Connection(self.front["F_x"],   self.scope_front[0]),
                Connection(self.front["F_y"],   self.scope_front[1]),
                Connection(self.front["omega"], self.scope_front[2]),
                Connection(self.front["M_y"],   self.scope_front[3]),
                Connection(self.rear["F_x"],    self.scope_rear[0]),
                Connection(self.rear["F_y"],    self.scope_rear[1]),
                Connection(self.rear["omega"],  self.scope_rear[2]),
                Connection(self.rear["M_y"],    self.scope_rear[3]),
                ]

        super().__init__(blocks, conns)
