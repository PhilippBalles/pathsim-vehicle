#########################################################################################
##
##                  Kinematic single-track Block
##
#########################################################################################



# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem


# BLOCK Definitions ================================================================================

class KinematicSingleTrack(DynamicalSystem):
    """Kinematic (zero tire-slip) single-track (bicycle) vehicle model with
    external longitudinal velocity input.

    The vehicle turns by geometry alone: each wheel rolls along its own
    heading without lateral slip, so the lateral velocity :math:`v_y` and
    yaw rate :math:`r` follow algebraically from the steering angle and
    forward speed, leaving only the pose :math:`[\\psi, X, Y]` to integrate.
    The block outputs the state :math:`[v_y, r, \\psi, X, Y]` — the same
    signature as :class:`LinearSingleTrack`, so it is a drop-in replacement.

    The model is valid at low lateral acceleration, where tire slip is
    negligible (roughly :math:`a_y \\lesssim 4\\,\\mathrm{m/s^2}` on dry
    asphalt, i.e. low speed or gentle manoeuvres). Unlike
    :class:`LinearSingleTrack` it makes *no* small-angle approximation and
    carries no mass, inertia, or tire parameters — only the geometry
    :math:`l_f, l_r`. There is no :math:`1/v_x` term anywhere, so the model
    is globally smooth and remains well defined through standstill
    (:math:`v_x = 0 \\Rightarrow v_y = r = 0`) and reverse; it is singular
    only at the non-physical steering angle :math:`|\\delta| \\to \\pi/2`.

    In the body frame (ISO 8855: x forward, y left, z up) the front wheel is
    steered by :math:`\\delta` and the rear wheel is fixed
    (:math:`\\delta_r = 0`). The zero-slip assumption — each wheel's
    contact-point velocity is aligned with its rolling direction — gives the
    two constraints

    .. math::

        v_y - l_r\\,r = 0,
        \\qquad
        -v_x\\sin\\delta + (v_y + l_f\\,r)\\cos\\delta = 0,

    the first from the unsteered rear wheel, the second from the front wheel
    rotated by :math:`\\delta`. Solving them for :math:`v_y` and :math:`r`
    yields the exact algebraic kinematic relations

    .. math::

        r = \\frac{v_x\\tan\\delta}{L},
        \\qquad
        v_y = l_r\\,r = \\frac{l_r\\,v_x\\tan\\delta}{L},
        \\qquad
        L = l_f + l_r.

    The pose kinematics are the exact nonlinear form

    .. math::

        \\dot\\psi = r,
        \\qquad
        \\dot X = v_x\\cos\\psi - v_y\\sin\\psi,
        \\qquad
        \\dot Y = v_x\\sin\\psi + v_y\\cos\\psi.

    Because :math:`v_y` and :math:`r` are algebraic in the inputs, the
    dynamic state reduces to the pose :math:`\\mathbf{x} = [\\psi, X, Y]`;
    :math:`v_y` and :math:`r` are emitted by the output equation
    ``_func_alg``, which therefore has direct feedthrough — hence the block
    subclasses :class:`DynamicalSystem` rather than :class:`ODE`.


    Input Ports
    -----------
    delta : float
        front-axle steering angle [rad]
    v_x : float
        longitudinal velocity [m/s]

    Output Ports
    ------------
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


    Parameters
    ----------
    l_f : float
        Distance from CG to front axle [m].
    l_r : float
        Distance from CG to rear axle [m].
    initial_value : array_like, optional
        Initial pose state vector ``[psi, X, Y]`` (length 3).


    """

    # port labels for semantic access
    input_port_labels  = {"delta": 0, "v_x": 1}
    output_port_labels = {"v_y": 0, "r": 1, "psi": 2, "X": 3, "Y": 4}

    def __init__(self, l_f=1.2, l_r=1.4, initial_value=None):

        # vehicle parameters
        self.l_f = l_f
        self.l_r = l_r
        self.L = l_f + l_r          # wheelbase (derived once)

        if initial_value is None:
            initial_value = np.zeros(3)

        super().__init__(
            func_dyn=self._func_dyn,
            func_alg=self._func_alg,
            initial_value=np.asarray(initial_value, dtype=float),
            jac_dyn=self._jac_dyn,
            )

        # Pre-size the input register to the two declared ports. DynamicalSystem
        # probes ``func_alg`` for feedthrough in ``__len__`` (during graph
        # assembly, before any connection has grown the register), and that
        # output equation indexes ``u[1]`` -> both ports must exist up front.
        self.inputs.resize(len(self.input_port_labels))


    def _func_dyn(self, x, u, t):
        """Right-hand side of the kinematic single-track ODEs (pose only).

        Parameters
        ----------
        x : array[float]
            State vector ``[psi, X, Y]``.
        u : array[float]
            Input vector ``[delta, v_x]``.
        t : float
            Time.

        Returns
        -------
        dxdt : array[float]
            State derivative vector ``[dpsi, dX, dY]``.
        """
        psi, X, Y = x
        delta, v_x = u[0], u[1]

        # algebraic kinematic relations (exact, no-slip constraint)
        r   = v_x * np.tan(delta) / self.L
        v_y = self.l_r * r

        # exact pose kinematics
        dpsi = r
        dX   = v_x * np.cos(psi) - v_y * np.sin(psi)
        dY   = v_x * np.sin(psi) + v_y * np.cos(psi)

        return np.array([dpsi, dX, dY])


    def _func_alg(self, x, u, t):
        """Output equation y = [v_y, r, psi, X, Y].

        ``v_y`` and ``r`` are algebraic in the inputs (direct feedthrough);
        the pose ``psi, X, Y`` is passed through from the state.

        Parameters
        ----------
        x : array[float]
            State vector ``[psi, X, Y]``.
        u : array[float]
            Input vector ``[delta, v_x]``.
        t : float
            Time.

        Returns
        -------
        y : array[float]
            Output vector ``[v_y, r, psi, X, Y]``.
        """
        psi, X, Y = x
        delta, v_x = u[0], u[1]

        r   = v_x * np.tan(delta) / self.L
        v_y = self.l_r * r

        return np.array([v_y, r, psi, X, Y])


    def _jac_dyn(self, x, u, t):
        """Analytic state Jacobian df_dyn/dx of the pose kinematics.

        ``_func_dyn`` depends on the state only through ``psi`` (in dX, dY),
        so only the first column is non-zero.
        """
        psi = x[0]
        delta, v_x = u[0], u[1]
        r   = v_x * np.tan(delta) / self.L
        v_y = self.l_r * r

        J = np.zeros((3, 3))
        J[1, 0] = -v_x * np.sin(psi) - v_y * np.cos(psi)
        J[2, 0] =  v_x * np.cos(psi) - v_y * np.sin(psi)
        return J
