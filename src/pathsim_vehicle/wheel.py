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
    """Per-corner spinning rigid body with a compliant contact patch.

    The block is the physical wheel — rim, hub, brake disc, tire carcass,
    contact patch — as one block. It owns the three things the
    force-input :class:`SingleTrack` chassis deliberately does not: the
    steering rotation, the tire force law, and *all wheel-local
    dynamics*. Its three states are the wheel spin speed :math:`\\omega`,
    integrated from the Newton-Euler torque balance about the spin axis,
    and the two carcass deflections :math:`u_x, u_y`, whose first-order
    relaxation dynamics make the tire force build up over the relaxation
    lengths :math:`\\sigma_x, \\sigma_y` instead of instantaneously.

    The contact point sits at body-frame position :math:`(l, s)`
    (:math:`l = +l_f` front, :math:`-l_r` rear; :math:`s = 0` on the
    single-track centre line); its velocity follows from rigid-body
    kinematics, :math:`v_{cx} = v_x - s\\,r`, :math:`v_{cy} = v_y +
    l\\,r`. The *steady-state* slips are the rigid targets

    .. math::

        \\kappa_{ss} = \\frac{R_w\\,\\omega - v_{cx}}{\\bar v},
        \\quad
        \\bar v = \\sqrt{v_{cx}^2 + v_\\varepsilon^2},
        \\qquad
        \\alpha_{ss} = \\delta - \\operatorname{atan2}(v_{cy}, v_{cx}),

    (the smooth norm :math:`\\bar v` removes the only :math:`1/v_{cx}`
    singularity; ``atan2`` is valid in every quadrant, including reverse,
    and 0 at standstill). The patch states relax toward them with
    distance-based first-order dynamics (the standard single-contact-point
    transient, Pacejka §7.2),

    .. math::

        \\dot u_x = \\frac{\\bar v}{\\sigma_x}
                    (\\sigma_x \\kappa_{ss} - u_x),
        \\qquad
        \\dot u_y = \\frac{\\bar v}{\\sigma_y}
                    (\\sigma_y \\alpha_{ss} - u_y),

    and the tire law is evaluated at the *transient* slips
    :math:`\\kappa' = u_x/\\sigma_x`, :math:`\\alpha' = u_y/\\sigma_y`:

    .. math::

        (F_x^{t0},\\ F_y^{t}) =
        \\texttt{tire.forces}(\\kappa',\\ \\alpha',\\ F_z).

    At steady state the transient slips reduce *exactly* to the rigid
    ones, so every stationary result of the rigid-slip wheel survives
    unchanged (verified in ``derive_wheel.py``). Because the pure
    relaxation model leaves the patch–spin pair almost undamped at crawl
    speeds, the block adds the carcass *material* damping the first-order
    model omits, as a longitudinal damper ramped out by the crossover
    speed ``v_low`` (Pacejka's low-speed correction):

    .. math::

        F_x^{t} = F_x^{t0} + d(\\bar v)\\,\\dot u_x,
        \\qquad
        d(\\bar v) = d_{low} \\tfrac{1}{2}
        \\big(1 + \\cos(\\pi \\bar v / v_{low})\\big)
        \\ \\text{for } \\bar v < v_{low}, \\text{ else } 0.

    The spin state closes the balance internally
    (:math:`y^t` axis, ISO 8855; positive :math:`\\omega` rolls forward):

    .. math::

        I_w\\,\\dot\\omega = T - M_y,
        \\qquad
        M_y = R_w\\,F_x^{t},

    with the net spin-axis torque :math:`T` — the signed sum of every
    drive and brake source acting on the wheel (motor, engine,
    differential, brake). It enters exactly as written; the block applies
    no sign logic of its own. A constant positive :math:`T` at standstill
    spins the wheel *forwards*, a negative one *backwards*: physically a
    brake opposes motion (Coulomb friction with stiction), and that
    direction-opposing logic belongs to a future brake block upstream of
    this port. Because the port carries one net torque, drive and brake
    (or any several sources) are summed with an ``Adder`` before it — the
    same visible injection pattern the chassis force ports use. The
    tire-frame forces are rotated into the body
    frame by the steer angle, :math:`F_x^b = F_x^t\\cos\\delta -
    F_y^t\\sin\\delta`, :math:`F_y^b = F_x^t\\sin\\delta +
    F_y^t\\cos\\delta`.

    The point of the relaxation states is numerical as much as physical:
    the rigid-slip wheel has a real eigenvalue :math:`-R_w^2
    C_\\kappa/(I_w \\bar v)` that blows up like :math:`1/\\bar v` at low
    speed (about −7600 1/s at standstill), while the relaxed patch–spin
    pair caps the fast mode at the *speed-independent* natural frequency
    :math:`\\omega_n = R_w\\sqrt{C_\\kappa/(I_w \\sigma_x)}` (about
    158 1/s at the defaults). An explicit solver therefore runs at
    :math:`\\mathrm{d}t \\approx 5\\times10^{-3}` s from standstill to
    top speed. One verified warning: when an explicit step is too large
    for the fast mode, the guard nonlinearity *saturates* the instability
    — the simulation returns plausible-looking, silently wrong
    trajectories rather than NaNs. Stay at or below the recommended step.

    The tire force law lives in a stateless :class:`TireModel` that the
    ``Wheel`` *calls* as an ordinary method — it is not a block, not part
    of the connection graph, and owns no dynamics, so every law behind
    the contract ``forces(kappa, alpha, F_z)`` (linear, Dugoff,
    simplified or full Magic Formula) plugs in unchanged; combined-slip
    laws simply read both transient slips. The relaxation lengths are
    carcass properties and live on the ``Wheel`` (the block that owns the
    carcass states), keeping the tire contract forces-only. The vertical
    load :math:`F_z` is an *input port*, fed by the chassis:
    :class:`SingleTrack` outputs its static axle loads, and a
    higher-fidelity chassis outputs dynamic loads through the same wire —
    every rung of the chassis ladder presents identical plumbing to the
    ``Wheel``. Camber and the self-aligning moment :math:`M_z` are out of
    scope in this version. Both frames follow ISO 8855 (x forward, y
    left, z up); the tire frame is the body frame rotated by
    :math:`\\delta` about :math:`+z`.

    Wiring caveats: the chassis feedback :math:`(v_x, v_y, r)` and
    :math:`F_z` are mandatory — an unconnected input reads 0.0 in
    PathSim, so a forgotten :math:`F_z` wire means zero vertical load,
    zero tire forces, and a car that does not move, without any error.
    The torque port is *meaningfully* optional: ``T`` unconnected reads
    0.0 — no drive and no brake, i.e. a coasting wheel. For a rolling start at
    chassis speed :math:`v_{x,0}` choose ``omega_0`` :math:`=
    v_{x,0}/R_w`; the patch deflections start at zero, which is exact for
    pure rolling. Using the guarded :math:`\\bar v` in the patch decay
    keeps the formulation single-cased at all speeds; the price is that a
    parked, wound-up patch relaxes on the timescale
    :math:`\\sigma_x/v_\\varepsilon` (~0.3 s) instead of holding forever
    — true standstill stiction belongs to a future brake block. Rolling
    resistance and bearing drag are out of scope; they can enter later as
    an upstream resistance-torque source summed into ``T``. The load
    input is clamped to :math:`F_z \\ge 0` before the tire call. No
    analytic Jacobian is supplied: the state Jacobian goes through the
    tire model, so the ``DynamicOperator`` differentiates numerically (an
    analytic chain-rule Jacobian is a documented, additive future
    upgrade). The block stays agnostic to vehicle mass and geometry: the
    load arrives over the wire, and the signed lever arm :math:`l` is
    passed at construction. When a coupling block such as the
    :class:`Differential` both feeds ``T`` and reads ``omega`` back,
    PathSim's per-*block* feedthrough detection conservatively flags the
    cycle as an algebraic loop even though ``omega`` is a pure state
    output; the fixed-point stage it then runs converges immediately.


    Input Ports
    -----------
    delta : float
        steer angle of this wheel [rad]
    T : float
        net spin-axis torque (signed; drive minus brake, summed
        externally with an ``Adder`` when there is more than one
        source) [N m]
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
        wheel spin speed (state 0) [rad/s]
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
    sigma_x : float
        Longitudinal relaxation length [m].
    sigma_y : float
        Lateral relaxation length [m].
    v_eps : float
        Low-speed guard for the slip denominator [m/s].
    d_low : float
        Low-speed carcass damping [N s/m]; the default gives the
        standstill patch-spin mode a damping ratio of about 0.7 at the
        default tire stiffness (derived in ``derive_wheel.py``).
    v_low : float
        Crossover speed above which the low-speed damping is fully
        ramped out [m/s].
    omega_0 : float
        Initial wheel spin speed [rad/s].
    tire : TireModel, optional
        Tire force model instance; defaults to ``LinearTire()``.


    """

    # port labels for semantic access
    input_port_labels  = {"delta": 0, "T": 1, "v_x": 2,
                          "v_y": 3, "r": 4, "F_z": 5}
    output_port_labels = {"F_x": 0, "F_y": 1, "omega": 2, "M_y": 3}

    def __init__(self, l=1.2, s=0.0, R_w=0.30, I_w=1.2, sigma_x=0.30,
                 sigma_y=0.50, v_eps=1.0, d_low=2900.0, v_low=2.5,
                 omega_0=0.0, tire=None):

        # wheel parameters
        self.l = l
        self.s = s
        self.R_w = R_w
        self.I_w = I_w
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.v_eps = v_eps
        self.d_low = d_low
        self.v_low = v_low

        # tire force model (stateless, called in-process)
        self.tire = tire if tire is not None else LinearTire()

        super().__init__(
            func_dyn=self._func_dyn,
            func_alg=self._func_alg,
            initial_value=np.asarray([omega_0, 0.0, 0.0], dtype=float),
            )

        # Pre-size the input register to the six declared ports.
        # DynamicalSystem probes ``func_alg`` for feedthrough in ``__len__``
        # (during graph assembly, before any connection has grown the
        # register), and both equations unpack all six inputs.
        self.inputs.resize(len(self.input_port_labels))


    def _tire_chain(self, x, u):
        """Shared kinematics/patch/tire chain of both system equations:
        contact-point kinematics -> steady slip targets -> patch
        relaxation -> tire call at the transient slips -> low-speed
        damping.

        Parameters
        ----------
        x : array[float]
            State vector ``[omega, u_x, u_y]``.
        u : array[float]
            Input vector ``[delta, T, v_x, v_y, r, F_z]``.

        Returns
        -------
        delta, T, F_x_t, F_y_t, du_x, du_y : float
            Steer angle, the net spin-axis torque, the transmitted
            tire-frame forces, and the patch derivatives.
        """
        omega, u_x, u_y = x
        delta, T, v_x, v_y, r, F_z = u

        # contact-point velocity (body frame); s = 0 for single track
        v_cx = v_x - self.s * r
        v_cy = v_y + self.l * r
        v_bar = np.sqrt(v_cx**2 + self.v_eps**2)

        # steady-state slip targets (sqrt guard; atan2 valid in all quadrants)
        kappa_ss = (self.R_w * omega - v_cx) / v_bar
        alpha_ss = delta - np.arctan2(v_cy, v_cx)

        # patch relaxation toward the targets (first-order, distance-based)
        du_x = (v_bar / self.sigma_x) * (self.sigma_x * kappa_ss - u_x)
        du_y = (v_bar / self.sigma_y) * (self.sigma_y * alpha_ss - u_y)

        # tire force law at the TRANSIENT slips; load clamped non-negative
        F_x_t0, F_y_t = self.tire.forces(u_x / self.sigma_x,
                                         u_y / self.sigma_y,
                                         max(F_z, 0.0))

        # low-speed carcass damping, ramped out by v_low
        if v_bar < self.v_low:
            d = self.d_low * 0.5 * (1 + np.cos(np.pi * v_bar / self.v_low))
        else:
            d = 0.0
        F_x_t = F_x_t0 + d * du_x

        return delta, T, F_x_t, F_y_t, du_x, du_y


    def _func_dyn(self, x, u, t):
        """Patch relaxation and torque balance about the spin axis.

        Parameters
        ----------
        x : array[float]
            State vector ``[omega, u_x, u_y]``.
        u : array[float]
            Input vector ``[delta, T, v_x, v_y, r, F_z]``.
        t : float
            Time.

        Returns
        -------
        dxdt : array[float]
            State derivative ``[domega/dt, du_x/dt, du_y/dt]``.
        """
        _, T, F_x_t, _, du_x, du_y = self._tire_chain(x, u)

        # spin-axis load torque closes the balance internally
        domega = (T - self.R_w * F_x_t) / self.I_w

        return np.array([domega, du_x, du_y])


    def _func_alg(self, x, u, t):
        """Output equation: body-frame forces, the spin speed, and the
        spin-axis load torque (observation).

        Parameters
        ----------
        x : array[float]
            State vector ``[omega, u_x, u_y]``.
        u : array[float]
            Input vector ``[delta, T, v_x, v_y, r, F_z]``.
        t : float
            Time.

        Returns
        -------
        y : array[float]
            Output vector ``[F_x, F_y, omega, M_y]``.
        """
        delta, _, F_x_t, F_y_t, _, _ = self._tire_chain(x, u)

        # rotate wheel-frame forces into the body frame by the steer angle
        F_x_b = F_x_t * np.cos(delta) - F_y_t * np.sin(delta)
        F_y_b = F_x_t * np.sin(delta) + F_y_t * np.cos(delta)

        return np.array([F_x_b, F_y_b, x[0], self.R_w * F_x_t])
