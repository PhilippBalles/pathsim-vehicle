#########################################################################################
##
##                  Wheel Block
##
#########################################################################################



# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks._block import Block
from pathsim.optim.operator import Operator

from .tiremodels import LinearTire


# BLOCK Definitions ================================================================================

class Wheel(Block):
    """Per-corner bridge between the chassis and the contact patch.

    The block is *purely algebraic* (stateless, feedthrough): given the
    steer angle :math:`\\delta`, the wheel spin speed :math:`\\omega`, and
    the chassis velocity feedback :math:`(v_x, v_y, r)`, it forms the
    contact-point velocities, the longitudinal slip ratio :math:`\\kappa`
    and the slip angle :math:`\\alpha`, evaluates a swappable tire-force
    model, rotates the resulting tire-frame forces into the body frame,
    and outputs the two body-frame force components together with the
    spin-axis reaction torque :math:`M_y = R_w F_x^{t}`. It owns exactly
    the two things the force-input :class:`SingleTrack` chassis
    deliberately does not: the steering rotation and the tire force law.

    The two body-frame forces feed :class:`SingleTrack`'s force inputs
    (one ``Wheel`` per axle, or one per corner in a dual-track model);
    because :class:`SingleTrack` is an integrator, the feedback
    ``SingleTrack -> Wheel -> SingleTrack`` closes through the chassis
    state and forms *no algebraic loop*. The wheel spin speed
    :math:`\\omega` is an *input*, not a state: a future driveline owns
    the spin inertia and integrates
    :math:`I_w \\dot\\omega = T_d - T_b - M_y`, so the ``Wheel`` only
    returns the load torque :math:`M_y` that closes that balance. The tire
    force law lives in a stateless :class:`TireModel` that the ``Wheel``
    *calls* as an ordinary method — it is not a block and not part of the
    connection graph. The vertical load :math:`F_z` is a static parameter
    for now (a future suspension/load-transfer block will feed it through
    a port); camber and the self-aligning moment :math:`M_z` are out of
    scope in this version. Both frames follow ISO 8855 (x forward, y left,
    z up); the tire frame is the body frame rotated by :math:`\\delta`
    about :math:`+z`, with the lateral axis :math:`y^{t}` the wheel spin
    axis.

    The block evaluates the following chain on every update; there is no
    integration, so the equations below *are* the block. The contact point
    sits at body-frame position :math:`(l, s)` (:math:`l = +l_f` front,
    :math:`-l_r` rear; :math:`s = 0` on the single-track centre line), and
    its velocity follows from rigid-body kinematics with
    :math:`\\boldsymbol{\\omega} = (0, 0, r)`:

    .. math::

        v_{cx} = v_x - s\\,r,
        \\qquad
        v_{cy} = v_y + l\\,r.

    The longitudinal slip ratio compares the circumferential speed
    :math:`R_w \\omega` with :math:`v_{cx}`; the slip angle is the angle
    between the wheel heading and the contact velocity:

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

    which are then rotated into the body frame by the steer angle and
    reduced to the spin-axis reaction torque:

    .. math::

        \\begin{aligned}
        F_x^{b} &= F_x^{t}\\cos\\delta - F_y^{t}\\sin\\delta, \\\\
        F_y^{b} &= F_x^{t}\\sin\\delta + F_y^{t}\\cos\\delta, \\\\
        M_y     &= R_w\\,F_x^{t}.
        \\end{aligned}

    All five input ports are mandatory and must be connected. The vertical
    load :math:`F_z` is clamped to :math:`F_z \\ge 0` before the tire
    call; promoting it to an input port when a suspension/load-transfer
    block exists is a localized, additive change that does not touch the
    tire models. No analytic Jacobian is supplied: the ``Operator``
    differentiates the 3x5 feedthrough numerically, matching the lean
    :class:`Function` reference (an analytic chain-rule Jacobian is a
    documented, additive future upgrade). Parameters may be passed
    directly or supplied by a ``Vehicle`` factory — the natural home for
    the *derived* quantities, the signed lever arm :math:`l`
    (:math:`+l_f`/:math:`-l_r`) and the static axle load
    :math:`F_z = m g\\,l_r / (l_f + l_r)` (front) — which keeps the
    ``Wheel`` agnostic to vehicle mass and geometry.


    Input Ports
    -----------
    delta : float
        steer angle of this wheel [rad]
    omega : float
        wheel spin speed (from driveline) [rad/s]
    v_x : float
        chassis longitudinal velocity [m/s]
    v_y : float
        chassis lateral velocity [m/s]
    r : float
        chassis yaw rate [rad/s]

    Output Ports
    ------------
    F_x : float
        body-frame longitudinal force [N]
    F_y : float
        body-frame lateral force [N]
    M_y : float
        spin-axis reaction torque R_w * F_x_t [N m]


    Parameters
    ----------
    l : float
        Signed longitudinal contact-point position [m];
        +l_f front, -l_r rear.
    s : float
        Lateral contact-point offset [m]; 0 for single-track.
    R_w : float
        Effective rolling radius [m].
    F_z : float
        Static vertical load [N] (placeholder; set by the vehicle).
    v_eps : float
        Low-speed guard for the slip-ratio denominator [m/s].
    tire : TireModel, optional
        Tire force model instance; defaults to ``LinearTire()``.


    """

    # port labels for semantic access
    input_port_labels  = {"delta": 0, "omega": 1, "v_x": 2, "v_y": 3, "r": 4}
    output_port_labels = {"F_x": 0, "F_y": 1, "M_y": 2}

    def __init__(self, l=1.2, s=0.0, R_w=0.30, F_z=4000.0, v_eps=1.0, tire=None):
        super().__init__()

        # wheel parameters
        self.l = l
        self.s = s
        self.R_w = R_w
        self.F_z = F_z
        self.v_eps = v_eps

        # tire force model (stateless, called in-process)
        self.tire = tire if tire is not None else LinearTire()

        # algebraic operator wrapping the input -> output map
        self.op_alg = Operator(func=self._func_alg)


    def update(self, t):
        """Evaluate the wheel as part of the algebraic component of the
        global system DAE.

        Parameters
        ----------
        t : float
            evaluation time
        """

        #apply operator to get output
        y = self.op_alg(self.inputs.to_array())
        self.outputs.update_from_array(y)


    def _func_alg(self, u):
        """Algebraic map from chassis/driveline inputs to body-frame
        forces and spin-axis reaction torque.

        Parameters
        ----------
        u : array[float]
            Input vector ``[delta, omega, v_x, v_y, r]``.

        Returns
        -------
        y : array[float]
            Output vector ``[F_x, F_y, M_y]``.
        """
        delta, omega, v_x, v_y, r = u

        # contact-point velocity (body frame); s = 0 for single track
        v_cx = v_x - self.s * r
        v_cy = v_y + self.l * r

        # slip quantities (sqrt guard on kappa; atan2 valid in all quadrants)
        kappa = (self.R_w * omega - v_cx) / np.sqrt(v_cx**2 + self.v_eps**2)
        alpha = delta - np.arctan2(v_cy, v_cx)

        # tire force law in the wheel frame (method call on the held model)
        F_x_t, F_y_t = self.tire.forces(kappa, alpha, max(self.F_z, 0.0))

        # rotate wheel-frame forces into the body frame by the steer angle
        F_x_b = F_x_t * np.cos(delta) - F_y_t * np.sin(delta)
        F_y_b = F_x_t * np.sin(delta) + F_y_t * np.cos(delta)

        # spin-axis reaction (load) torque for the driveline
        M_y = self.R_w * F_x_t

        return np.array([F_x_b, F_y_b, M_y])
