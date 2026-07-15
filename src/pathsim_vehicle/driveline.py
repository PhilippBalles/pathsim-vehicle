#########################################################################################
##
##                  Driveline Block
##
#########################################################################################



# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.ode import ODE


# BLOCK Definitions ================================================================================

class Driveline(ODE):
    """Per-wheel spin dynamics: a single integrator for the wheel spin
    speed :math:`\\omega` driven by the torque balance about the spin axis.

    The block closes the gap the :class:`Wheel` block deliberately left
    open — there, :math:`\\omega` is an *input* and the block returns the
    spin-axis load torque :math:`M_y = R_w F_x^t`; without a ``Driveline``
    the spin speed has to be prescribed by a source block. With one
    ``Driveline`` per ``Wheel``, the wheel speed is instead *earned* from
    drive and brake torques working against the road reaction. The engine
    or motor itself is *not* part of this block: it is an upstream torque
    source feeding the ``T_d`` port — a :class:`Constant` today, a motor
    block with a torque map later. This is the base rung of the driveline
    ladder: a per-wheel electric machine (hub-motor EV) is exactly N
    independent instances of this block; shared drivelines compose in
    front of it (the :class:`Differential` coupling block, and later an
    ICE powertrain chain). On the current single-track chassis, where
    wheels are lumped per axle, "per wheel" reads as "per axle".

    Newton-Euler for the wheel about its spin axis (:math:`y^t`,
    ISO 8855; positive :math:`\\omega` rolls the vehicle forward):

    .. math::

        I_w\\,\\dot\\omega = T_d - T_b - M_y,

    with the spin inertia :math:`I_w` (wheel, tire, and any rigidly
    coupled rotating parts — brake disc, hub-motor rotor, referred
    driveline inertia), the drive torque :math:`T_d`, the brake torque
    :math:`T_b`, and the road-reaction (load) torque :math:`M_y = R_w
    F_x^t` delivered by the ``Wheel`` block. A driving tire loads the
    driveline (:math:`M_y > 0` opposes the spin), a braking tire feeds
    energy back. All three torques are *signed* inputs and the block
    applies no sign logic of its own: in particular a constant positive
    :math:`T_b` at standstill will spin the wheel *backwards* —
    physically a brake opposes motion (Coulomb friction with stiction),
    and that direction-opposing logic belongs to a future brake block
    upstream of the ``T_b`` port, not to this rung. Keeping ``T_d`` and
    ``T_b`` as separate ports (rather than one net torque) lets the drive
    and brake sources be wired, inspected, and replaced independently.

    The right-hand side does not depend on the state :math:`\\omega` at
    all — all coupling comes back through the connection graph
    (``omega -> Wheel -> M_y``) — so the analytic state Jacobian is the
    :math:`1\\times1` zero matrix. The intended per-wheel loop
    ``Driveline -> Wheel -> Driveline`` closes through this integrator,
    alongside the chassis loop ``SingleTrack -> Wheel -> SingleTrack``,
    so the full car graph stays free of algebraic loops. At steady state
    the balance gives the classic drive-force relation
    :math:`F_x^t = (T_d - T_b)/R_w`; with zero torque the (stable,
    millisecond-fast) slip dynamics find pure rolling
    :math:`\\omega \\to v_{cx}/R_w` on their own; and under constant net
    torque the vehicle accelerates with the effective mass
    :math:`m + I_w/R_w^2` per driven wheel — all verified symbolically
    and numerically in ``derive_driveline.py``.

    Wiring caveats, in the spirit of the ``F_z`` warning on the
    :class:`Wheel`: an unconnected input reads 0.0 in PathSim.
    Unconnected ``T_d`` or ``T_b`` is *meaningful* (freewheeling, no
    brake), but a forgotten ``M_y`` wire decouples the wheel from the
    road — it spins up under drive torque without ever loading the tire,
    without any error. Wire ``Wheel.M_y -> Driveline.M_y`` always. For a
    rolling start at chassis speed :math:`v_{x,0}` choose
    ``omega_0 = v_x0 / R_w``; the default ``omega_0 = 0`` at nonzero
    chassis speed means a locked-wheel transient until the slip dynamics
    catch up. Rolling resistance, bearing drag, and aerodynamic wheel
    losses are out of scope at this rung; they can enter later as an
    upstream resistance-torque block on the ``T_b`` path.


    Input Ports
    -----------
    T_d : float
        drive torque (signed, from motor/engine) [N m]
    T_b : float
        brake torque (signed, subtracted as-is) [N m]
    M_y : float
        spin-axis load torque (from Wheel) [N m]

    Output Ports
    ------------
    omega : float
        wheel spin speed [rad/s]


    Parameters
    ----------
    I_w : float
        Spin inertia of the wheel and rigidly coupled rotating parts
        [kg m^2], > 0.
    omega_0 : float
        Initial wheel spin speed [rad/s].


    """

    # port labels for semantic access
    input_port_labels  = {"T_d": 0, "T_b": 1, "M_y": 2}
    output_port_labels = {"omega": 0}

    def __init__(self, I_w=1.2, omega_0=0.0):

        # driveline parameters
        self.I_w = I_w

        super().__init__(
            func=self._func_dyn,
            initial_value=np.asarray([omega_0], dtype=float),
            jac=self._jac_dyn,
            )


    def _func_dyn(self, x, u, t):
        """Right-hand side: torque balance about the spin axis.

        Parameters
        ----------
        x : array[float]
            State vector ``[omega]`` (unused — no state feedback).
        u : array[float]
            Input vector ``[T_d, T_b, M_y]``.
        t : float
            evaluation time

        Returns
        -------
        dx : array[float]
            State derivative ``[domega]``.
        """
        T_d, T_b, M_y = u
        return np.array([(T_d - T_b - M_y) / self.I_w])


    def _jac_dyn(self, x, u, t):
        """Analytic state Jacobian: the rhs is independent of omega.

        Parameters
        ----------
        x : array[float]
            State vector ``[omega]``.
        u : array[float]
            Input vector ``[T_d, T_b, M_y]``.
        t : float
            evaluation time

        Returns
        -------
        J : array[float]
            1x1 zero matrix.
        """
        return np.zeros((1, 1))
