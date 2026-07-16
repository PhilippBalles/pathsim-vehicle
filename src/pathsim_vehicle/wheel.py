#########################################################################################
##
##                  Wheel Block
##
#########################################################################################



# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem

from .tiremodels import LinearTire


# BLOCK Definitions ================================================================================

class Wheel(DynamicalSystem):
    """Per-corner spinning rigid body with a contact patch.

    The block is the physical wheel — rim, tire, hub, brake disc — as one
    block. It owns the three things the force-input :class:`SingleTrack`
    chassis deliberately does not: the steering rotation, the tire force
    law, and the spin inertia. Its single state is the wheel spin speed
    :math:`\\omega`, integrated from the Newton-Euler torque balance about
    the spin axis (:math:`y^t`, ISO 8855; positive :math:`\\omega` rolls
    the vehicle forward):

    .. math::

        I_w\\,\\dot\\omega = T_d - T_b - M_y,
        \\qquad
        M_y = R_w\\,F_x^{t},

    with the spin inertia :math:`I_w` (rim, tire, hub, brake disc, and any
    rigidly coupled rotating parts), the drive torque :math:`T_d` (from a
    motor, engine, or differential), the brake torque :math:`T_b`, and the
    road-reaction (load) torque :math:`M_y` computed internally from the
    tire chain. Both torque inputs are *signed* and enter the balance
    exactly as written — the block applies no sign logic of its own. In
    particular a constant positive :math:`T_b` at standstill will spin the
    wheel *backwards*: physically a brake opposes motion (Coulomb friction
    with stiction), and that direction-opposing logic belongs to a future
    brake block upstream of the ``T_b`` port, not to this block. Unlike
    the chassis blocks, the right-hand side *does* depend on the own
    state: :math:`M_y` grows with the slip ratio, which grows with
    :math:`\\omega` — for a linear tire in straight driving
    :math:`\\partial\\dot\\omega/\\partial\\omega =
    -R_w^2 C_\\kappa/(I_w \\bar v) < 0`, a stable, milliseconds-fast spin
    mode (verified in ``derive_wheel.py``), so an explicit solver needs
    :math:`\\mathrm{d}t \\lesssim 10^{-3}\\,\\mathrm{s}` at the defaults.

    Around that state sits the algebraic force chain, evaluated on every
    update both for the outputs and for the :math:`M_y` inside the
    balance. The contact point sits at body-frame position :math:`(l, s)`
    (:math:`l = +l_f` front, :math:`-l_r` rear; :math:`s = 0` on the
    single-track centre line), and its velocity follows from rigid-body
    kinematics with :math:`\\boldsymbol{\\omega} = (0, 0, r)`:

    .. math::

        v_{cx} = v_x - s\\,r,
        \\qquad
        v_{cy} = v_y + l\\,r.

    The longitudinal slip ratio compares the circumferential speed
    :math:`R_w \\omega` (from the block's own state) with :math:`v_{cx}`;
    the slip angle is the angle between the wheel heading and the contact
    velocity:

    .. math::

        \\kappa = \\frac{R_w\\,\\omega - v_{cx}}
                        {\\sqrt{v_{cx}^{2} + v_{\\varepsilon}^{2}}},
        \\qquad
        \\alpha = \\delta - \\operatorname{atan2}(v_{cy},\\ v_{cx}).

    The smooth norm :math:`\\sqrt{v_{cx}^2 + v_\\varepsilon^2}` removes
    the only :math:`1/v_{cx}` singularity (:math:`v_\\varepsilon` a small
    low-speed guard, not a physical parameter); ``atan2`` makes
    :math:`\\alpha` well defined in every quadrant, including reverse
    (:math:`v_{cx} < 0`), and :math:`0` at standstill. The slip pair
    drives the tire model, a stateless object returning the wheel-frame
    forces

    .. math::

        (F_x^{t},\\ F_y^{t}) = \\texttt{tire.forces}(\\kappa,\\ \\alpha,\\ F_z),

    which are then rotated into the body frame by the steer angle:

    .. math::

        \\begin{aligned}
        F_x^{b} &= F_x^{t}\\cos\\delta - F_y^{t}\\sin\\delta, \\\\
        F_y^{b} &= F_x^{t}\\sin\\delta + F_y^{t}\\cos\\delta.
        \\end{aligned}

    The two body-frame forces feed :class:`SingleTrack`'s force inputs
    (one ``Wheel`` per axle, or one per corner in a dual-track model).
    Both the chassis and the wheel are integrators, so with pure sources
    on the torque ports the graph is free of algebraic loops. One caveat
    when a *coupling block* such as the :class:`Differential` both feeds
    ``T_d`` and reads ``omega`` back: PathSim's feedthrough detection is
    per-*block*, not per-port, so this cycle is conservatively flagged as
    an algebraic loop even though ``omega`` is a pure state output with no
    instantaneous dependence on ``T_d`` — the fixed-point stage PathSim
    then runs converges immediately. The tire force law lives in a
    stateless :class:`TireModel` that the ``Wheel`` *calls* as an ordinary
    method — it is not a block and not part of the connection graph. The
    vertical load :math:`F_z` is an *input port*, fed by the chassis:
    :class:`SingleTrack` outputs its static axle loads, and a
    higher-fidelity chassis (dual track with load transfer) outputs
    dynamic loads through the same wire — every rung of the chassis ladder
    presents identical plumbing to the ``Wheel``. Camber and the
    self-aligning moment :math:`M_z` are out of scope in this version.
    Both frames follow ISO 8855 (x forward, y left, z up); the tire frame
    is the body frame rotated by :math:`\\delta` about :math:`+z`, with
    the lateral axis :math:`y^{t}` the wheel spin axis.

    Wiring caveats: the chassis feedback :math:`(v_x, v_y, r)` and
    :math:`F_z` are mandatory — an unconnected input reads 0.0 in PathSim,
    so a forgotten :math:`F_z` wire means zero vertical load, zero tire
    forces, and a car that does not move, without any error. The torque
    ports are *meaningfully* optional: ``T_d`` unconnected is
    freewheeling, ``T_b`` unconnected is no brake. For a rolling start at
    chassis speed :math:`v_{x,0}` choose ``omega_0`` :math:`=
    v_{x,0}/R_w`; the default 0 at nonzero chassis speed means a
    locked-wheel transient (:math:`\\kappa \\approx -1`) until the stable
    slip dynamics catch up. Rolling resistance and bearing drag are out of
    scope; they can enter later as an upstream resistance-torque block on
    the ``T_b`` path. The load input is clamped to :math:`F_z \\ge 0`
    before the tire call. No analytic Jacobian is supplied: the state
    Jacobian goes through the tire model, so the ``DynamicOperator``
    differentiates numerically (an analytic chain-rule Jacobian is a
    documented, additive future upgrade). The block stays agnostic to
    vehicle mass and geometry: the load arrives over the wire, and the
    signed lever arm :math:`l` (:math:`+l_f`/:math:`-l_r`) is passed at
    construction.


    Input Ports
    -----------
    delta : float
        steer angle of this wheel [rad]
    T_d : float
        drive torque (signed, from motor/engine/differential) [N m]
    T_b : float
        brake torque (signed, subtracted as-is) [N m]
    v_x : float
        chassis longitudinal velocity [m/s]
    v_y : float
        chassis lateral velocity [m/s]
    r : float
        chassis yaw rate [rad/s]
    F_z : float
        vertical load (from chassis) [N]

    Output Ports
    ------------
    F_x : float
        body-frame longitudinal force [N]
    F_y : float
        body-frame lateral force [N]
    omega : float
        wheel spin speed (= the state) [rad/s]
    M_y : float
        spin-axis load torque R_w * F_x_t (observation) [N m]


    Parameters
    ----------
    l : float
        Signed longitudinal contact-point position [m];
        +l_f front, -l_r rear.
    s : float
        Lateral contact-point offset [m]; 0 for single-track.
    R_w : float
        Effective rolling radius [m].
    I_w : float
        Spin inertia about the wheel axis [kg m^2].
    v_eps : float
        Low-speed guard for the slip-ratio denominator [m/s].
    omega_0 : float
        Initial wheel spin speed [rad/s].
    tire : TireModel, optional
        Tire force model instance; defaults to ``LinearTire()``.


    """

    # port labels for semantic access
    input_port_labels  = {"delta": 0, "T_d": 1, "T_b": 2, "v_x": 3,
                          "v_y": 4, "r": 5, "F_z": 6}
    output_port_labels = {"F_x": 0, "F_y": 1, "omega": 2, "M_y": 3}

    def __init__(self, l=1.2, s=0.0, R_w=0.30, I_w=1.2, v_eps=1.0,
                 omega_0=0.0, tire=None):

        # wheel parameters
        self.l = l
        self.s = s
        self.R_w = R_w
        self.I_w = I_w
        self.v_eps = v_eps

        # tire force model (stateless, called in-process)
        self.tire = tire if tire is not None else LinearTire()

        super().__init__(
            func_dyn=self._func_dyn,
            func_alg=self._func_alg,
            initial_value=np.asarray([omega_0], dtype=float),
            )

        # Pre-size the input register to the seven declared ports.
        # DynamicalSystem probes ``func_alg`` for feedthrough in ``__len__``
        # (during graph assembly, before any connection has grown the
        # register), and both equations unpack all seven inputs.
        self.inputs.resize(len(self.input_port_labels))


    def _tire_chain(self, omega, u):
        """Shared slip/tire chain of both system equations: contact-point
        kinematics -> slip pair -> tire call.

        Parameters
        ----------
        omega : float
            Wheel spin speed (the state).
        u : array[float]
            Input vector ``[delta, T_d, T_b, v_x, v_y, r, F_z]``.

        Returns
        -------
        delta, T_d, T_b, F_x_t, F_y_t : float
            Steer angle, the two torque inputs, and the tire-frame forces.
        """
        delta, T_d, T_b, v_x, v_y, r, F_z = u

        # contact-point velocity (body frame); s = 0 for single track
        v_cx = v_x - self.s * r
        v_cy = v_y + self.l * r

        # slip quantities (sqrt guard on kappa; atan2 valid in all quadrants)
        kappa = (self.R_w * omega - v_cx) / np.sqrt(v_cx**2 + self.v_eps**2)
        alpha = delta - np.arctan2(v_cy, v_cx)

        # tire force law in the wheel frame; load input clamped non-negative
        F_x_t, F_y_t = self.tire.forces(kappa, alpha, max(F_z, 0.0))

        return delta, T_d, T_b, F_x_t, F_y_t


    def _func_dyn(self, x, u, t):
        """Torque balance about the spin axis.

        Parameters
        ----------
        x : array[float]
            State vector ``[omega]``.
        u : array[float]
            Input vector ``[delta, T_d, T_b, v_x, v_y, r, F_z]``.
        t : float
            Time.

        Returns
        -------
        dxdt : array[float]
            State derivative ``[domega/dt]``.
        """
        _, T_d, T_b, F_x_t, _ = self._tire_chain(x[0], u)

        # spin-axis load torque closes the balance internally
        return np.array([(T_d - T_b - self.R_w * F_x_t) / self.I_w])


    def _func_alg(self, x, u, t):
        """Output equation: body-frame forces, the spin speed, and the
        spin-axis load torque (observation).

        Parameters
        ----------
        x : array[float]
            State vector ``[omega]``.
        u : array[float]
            Input vector ``[delta, T_d, T_b, v_x, v_y, r, F_z]``.
        t : float
            Time.

        Returns
        -------
        y : array[float]
            Output vector ``[F_x, F_y, omega, M_y]``.
        """
        omega = x[0]
        delta, _, _, F_x_t, F_y_t = self._tire_chain(omega, u)

        # rotate wheel-frame forces into the body frame by the steer angle
        F_x_b = F_x_t * np.cos(delta) - F_y_t * np.sin(delta)
        F_y_b = F_x_t * np.sin(delta) + F_y_t * np.cos(delta)

        return np.array([F_x_b, F_y_b, omega, self.R_w * F_x_t])
