#########################################################################################
##
##                  Single-track Block
##
#########################################################################################



# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem


# BLOCK Definitions ================================================================================

class SingleTrack(DynamicalSystem):
    """Nonlinear, force-driven single-track (bicycle) vehicle model.

    The four body-frame axle force resultants — longitudinal and lateral,
    front and rear — are supplied as *inputs*; the block integrates the
    full planar rigid-body equations of motion, appends the exact pose
    kinematics, and outputs the six-state vector
    :math:`[v_x, v_y, r, \\psi, X, Y]` together with the static axle loads
    :math:`F_{z,f}, F_{z,r}`. Unlike :class:`LinearSingleTrack`,
    the longitudinal velocity :math:`v_x` is a *state* rather than an
    input, and there is no internal tire model, no steering input, and no
    small-angle assumption: the steering angle and the tire force law are
    owned by an upstream ``Wheel`` block that delivers the already-rotated
    body-frame forces to this block's ports. Because the forces are
    external, the model contains *no division by* :math:`v_x` — it is
    smooth everywhere and remains valid through standstill
    (:math:`v_x = 0`) and in reverse (:math:`v_x < 0`).

    The two wheels of each axle are lumped onto the vehicle centre line
    (ISO 8855: x forward, y left, z up), so a longitudinal force has no
    moment arm and the only yaw moment is :math:`l_f F_{y,f} - l_r F_{y,r}`;
    tire self-aligning moments are not represented.

    The axle loads are published as *output ports* even though they are
    static here (pure functions of mass and geometry): the chassis is the
    block that owns :math:`m, g, l_f, l_r`, and routing :math:`F_z` over a
    wire gives every rung of the chassis ladder the same plumbing — a
    higher-fidelity chassis (dual track with load transfer) outputs dynamic
    loads through identical ports, and the downstream ``Wheel`` never
    changes. On a level road, the planar moment balance of the gravity
    force about the rear and front contact points gives
    (:math:`L = l_f + l_r`)

    .. math::

        F_{z,f} = \\frac{m\\,g\\,l_r}{L},
        \\qquad
        F_{z,r} = \\frac{m\\,g\\,l_f}{L},
        \\qquad
        F_{z,f} + F_{z,r} = m\\,g.

    This block integrates the planar rigid-body equations directly; there
    is no reduction or linearization, so the equations below *are* the
    model. The CG velocity :math:`(v_x, v_y)` is expressed in the body
    frame, which rotates with the yaw rate :math:`r` about :math:`+z`; the
    resulting transport (Coriolis/centripetal) terms appear as
    :math:`+m\\,v_y\\,r` and :math:`-m\\,v_x\\,r` in the equations of
    motion. Newton's law and the yaw balance about the CG give

    .. math::

        \\begin{aligned}
        m\\,\\dot v_x &= F_{x,f} + F_{x,r} + m\\,v_y\\,r, \\\\
        m\\,\\dot v_y &= F_{y,f} + F_{y,r} - m\\,v_x\\,r, \\\\
        I_z\\,\\dot r &= l_f\\,F_{y,f} - l_r\\,F_{y,r},
        \\end{aligned}

    where :math:`F_{x,f}, F_{y,f}, F_{x,r}, F_{y,r}` are the body-frame
    axle forces taken directly from the input ports. They are produced
    upstream from tire-frame forces :math:`(\\cdot)^t` and the steer
    angles :math:`\\delta` (front), :math:`\\delta_r` (rear) by the planar
    rotation

    .. math::

        \\begin{aligned}
        F_{x,f} &= F_{x,f}^{t}\\cos\\delta   - F_{y,f}^{t}\\sin\\delta, \\\\
        F_{y,f} &= F_{x,f}^{t}\\sin\\delta   + F_{y,f}^{t}\\cos\\delta, \\\\
        F_{x,r} &= F_{x,r}^{t}\\cos\\delta_r - F_{y,r}^{t}\\sin\\delta_r, \\\\
        F_{y,r} &= F_{x,r}^{t}\\sin\\delta_r + F_{y,r}^{t}\\cos\\delta_r,
        \\end{aligned}

    which, together with the tire law that produces
    :math:`F_{x,f}^{t}, \\dots`, lives in the ``Wheel`` block.
    ``SingleTrack`` consumes the left-hand sides directly and never sees
    :math:`\\delta`. The pose kinematics are appended in their exact
    nonlinear form:

    .. math::

        \\dot\\psi = r,
        \\qquad
        \\dot X = v_x\\cos\\psi - v_y\\sin\\psi,
        \\qquad
        \\dot Y = v_x\\sin\\psi + v_y\\cos\\psi.

    The output appends the static axle loads to the state,
    :math:`\\mathbf{y} = [v_x, v_y, r, \\psi, X, Y, F_{z,f}, F_{z,r}]^\\top`,
    and in particular does *not* depend on the input: the block has no
    feedthrough, so feedback loops through this block still close through
    the integrator and form no algebraic loop.
    All four input ports are mandatory and must be connected. There are no
    cornering-stiffness or :math:`v_{x,\\mathrm{eps}}` parameters: this
    block has neither a tire model nor a :math:`1/v_x` singularity, so its
    right-hand side and Jacobian are smooth at :math:`v_x = 0` and for
    :math:`v_x < 0`.


    Input Ports
    -----------
    F_x_f : float
        front-axle longitudinal force, body frame [N]
    F_y_f : float
        front-axle lateral force, body frame [N]
    F_x_r : float
        rear-axle longitudinal force, body frame [N]
    F_y_r : float
        rear-axle lateral force, body frame [N]

    Output Ports
    ------------
    v_x : float
        longitudinal velocity [m/s]
    v_y : float
        lateral velocity [m/s]
    r : float
        yaw rate [rad/s]
    psi : float
        yaw angle [rad]
    X : float
        vehicle position along the global X-axis [m]
    Y : float
        vehicle position along the global Y-axis [m]
    F_z_f : float
        static front-axle vertical load [N]
    F_z_r : float
        static rear-axle vertical load [N]


    Parameters
    ----------
    m : float
        Vehicle mass [kg].
    I_z : float
        Yaw moment of inertia [kg m^2].
    l_f : float
        Distance from CG to front axle [m].
    l_r : float
        Distance from CG to rear axle [m].
    g : float
        Gravitational acceleration [m/s^2].
    initial_value : array_like, optional
        Initial state vector ``[v_x, v_y, r, psi, X, Y]``


    """

    # port labels for semantic access
    input_port_labels  = {"F_x_f": 0, "F_y_f": 1, "F_x_r": 2, "F_y_r": 3}
    output_port_labels = {"v_x": 0, "v_y": 1, "r": 2, "psi": 3, "X": 4, "Y": 5,
                          "F_z_f": 6, "F_z_r": 7}

    def __init__(self, m=1500.0, I_z=3000.0, l_f=1.2, l_r=1.4, g=9.81,
                 initial_value=None):

        # vehicle parameters
        self.m = m
        self.I_z = I_z
        self.l_f = l_f
        self.l_r = l_r
        self.g = g

        # static axle loads (moment balance about the contact points)
        self.F_z_f = m * g * l_r / (l_f + l_r)
        self.F_z_r = m * g * l_f / (l_f + l_r)

        if initial_value is None:
            initial_value = np.zeros(6)

        super().__init__(
            func_dyn=self._func_dyn,
            func_alg=self._func_alg,
            initial_value=np.asarray(initial_value, dtype=float),
            jac_dyn=self._jac_dyn,
            )


    def _func_dyn(self, x, u, t):
        """Right-hand side of the nonlinear single-track ODEs.

        Parameters
        ----------
        x : array[float]
            State vector ``[v_x, v_y, r, psi, X, Y]``.
        u : array[float]
            Input vector ``[F_x_f, F_y_f, F_x_r, F_y_r]``.
        t : float
            Time.

        Returns
        -------
        dxdt : array[float]
            State derivative vector.
        """
        v_x, v_y, r, psi, X, Y = x
        F_x_f, F_y_f, F_x_r, F_y_r = u

        # Newton-Euler EOM (body frame); transport terms +v_y*r, -v_x*r
        dv_x = (F_x_f + F_x_r) / self.m + v_y * r
        dv_y = (F_y_f + F_y_r) / self.m - v_x * r
        dr   = (self.l_f * F_y_f - self.l_r * F_y_r) / self.I_z

        # exact pose kinematics
        dpsi = r
        dX   = v_x * np.cos(psi) - v_y * np.sin(psi)
        dY   = v_x * np.sin(psi) + v_y * np.cos(psi)

        return np.array([dv_x, dv_y, dr, dpsi, dX, dY])


    def _func_alg(self, x, u, t):
        """Output equation: full state plus the precomputed static axle
        loads. Independent of ``u`` — the base class's passthrough
        detection therefore reports no feedthrough, preserving the
        loop-free closure through the integrator.

        Parameters
        ----------
        x : array[float]
            State vector ``[v_x, v_y, r, psi, X, Y]``.
        u : array[float]
            Input vector (unused).
        t : float
            Time.

        Returns
        -------
        y : array[float]
            Output vector ``[v_x, v_y, r, psi, X, Y, F_z_f, F_z_r]``.
        """
        return np.concatenate([x, [self.F_z_f, self.F_z_r]])


    def _jac_dyn(self, x, u, t):
        """Analytic state Jacobian df/dx of the single-track equations.

        The forces are inputs (dF/dx = 0): the only state dependence is the
        transport coupling (v_y*r, -v_x*r) and the pose kinematics, so the
        dr row is identically zero, the X, Y columns never feed back, and J
        carries no parameters.
        """
        v_x, v_y, r, psi = x[0], x[1], x[2], x[3]

        J = np.zeros((6, 6))
        J[0, 1] =  r
        J[0, 2] =  v_y
        J[1, 0] = -r
        J[1, 2] = -v_x
        J[3, 2] =  1
        J[4, 0] =  np.cos(psi)
        J[4, 1] = -np.sin(psi)
        J[4, 3] = -v_x * np.sin(psi) - v_y * np.cos(psi)
        J[5, 0] =  np.sin(psi)
        J[5, 1] =  np.cos(psi)
        J[5, 3] =  v_x * np.cos(psi) - v_y * np.sin(psi)
        return J
