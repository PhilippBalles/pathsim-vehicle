#########################################################################################
##
##                  Differential Block
##
#########################################################################################



# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks._block import Block
from pathsim.optim.operator import Operator


# BLOCK Definitions ================================================================================

class Differential(Block):
    """Ideal open differential, including the final-drive ratio, coupling
    one torque source to two :class:`Wheel` blocks.

    The block is the *coupling rung* of the driveline ladder and is
    *purely algebraic* (stateless, massless, lossless): the spin inertias
    live in the two ``Wheel`` blocks it feeds, and any carrier or
    pinion inertia is referred into their ``I_w`` (or neglected). The two
    output sides are labelled ``l``/``r`` (left/right on a dual-track
    axle); on the current single-track chassis, where wheels are lumped
    per *axle*, the same block reads as a *center* differential with
    ``l`` -> front and ``r`` -> rear.

    A bevel-gear differential is a planetary gear with stationary ratio
    :math:`i_0 = -1` between the side gears; the Willis equation gives
    the carrier speed as the mean side speed, and the final drive
    connects the input shaft to the carrier with ratio :math:`i`:

    .. math::

        \\omega_{in} = i\\,\\frac{\\omega_l + \\omega_r}{2},
        \\qquad
        T_c = i\\,T_{in}.

    The torque split follows from the ideal-block assumptions alone:
    with massless internals and no losses the power identity
    :math:`T_c\\,\\omega_c = T_l\\,\\omega_l + T_r\\,\\omega_r` must hold
    for *all* side speeds (they are unconstrained), so coefficient
    matching gives

    .. math::

        T_l = T_r = \\frac{T_c}{2} = \\frac{i\\,T_{in}}{2}.

    An open differential is an *equal-torque* device: speed differences
    cost it nothing and torque differences are impossible. Behind two
    identical ``Wheel`` blocks, the input torque appears only in the
    speed-*sum* dynamics; the speed-*difference* dynamics are driven
    purely by the load imbalance — on split friction the unloaded side
    spins up and no choice of :math:`T_{in}` can steer the difference.
    This is why torque vectoring needs per-wheel torque sources, not this
    block. Both derivations and the two-wheel effective-mass result
    :math:`\\dot v_x = i\\,T_{in}\\,R_w/(m R_w^2 + 2 I_w)` are verified
    symbolically and numerically in ``derive_differential.py``. Sign
    conventions follow the ``Wheel``: all speeds positive rolling
    forward, positive :math:`T_{in}` drives the vehicle forward.

    The torque path and the speed path are decoupled: the torque outputs
    depend only on ``T_in``, the speed output only on the side speeds.
    Every loop through this block
    (``Differential -> Wheel -> Differential``) closes through the
    ``Wheel`` integrators — ``omega`` is a pure state output with no
    instantaneous dependence on ``T``. PathSim's per-*block*
    feedthrough detection cannot see that and conservatively flags the
    cycle as an algebraic loop; the fixed-point stage it then runs
    converges immediately, so the topology costs only the loop-solver
    bookkeeping. Wiring caveats: unconnected ``omega_l``/``omega_r``
    (they read 0.0) corrupt *only* the reported ``omega_in`` — harmless
    while ``T_in`` comes from a :class:`Constant`, a silent failure the
    moment an engine block computes its torque from ``omega_in``; wire
    both side speeds whenever anything consumes ``omega_in``. Unconnected
    ``T_in`` is meaningful (coasting). Brake torques act on the wheel,
    not on the input shaft: sum them into each ``Wheel``'s single ``T``
    port with an ``Adder`` *behind* the differential, one per side (the
    differential's drive-torque output plus the signed brake torque). Fidelity grows additively: a
    limited-slip differential adds an antisymmetric coupling torque
    :math:`\\pm T_{LSD}(\\omega_l - \\omega_r)` (future block or
    extension); a *locked* differential is the stiff limit enforcing
    equal speeds — as a hard constraint it merges the two spin inertias
    and is out of scope at this rung.


    Input Ports
    -----------
    T_in : float
        input-shaft torque (signed, from source) [N m]
    omega_l : float
        left (front) side speed, from Wheel [rad/s]
    omega_r : float
        right (rear) side speed, from Wheel [rad/s]

    Output Ports
    ------------
    T_l : float
        left (front) drive torque i*T_in/2 [N m]
    T_r : float
        right (rear) drive torque i*T_in/2 [N m]
    omega_in : float
        input-shaft speed i*(omega_l + omega_r)/2 [rad/s]


    Parameters
    ----------
    i : float
        Final-drive ratio (input turns i times faster than the
        carrier) [-], > 0.


    """

    # port labels for semantic access
    input_port_labels  = {"T_in": 0, "omega_l": 1, "omega_r": 2}
    output_port_labels = {"T_l": 0, "T_r": 1, "omega_in": 2}

    def __init__(self, i=3.5):
        super().__init__()

        # final-drive ratio
        self.i = i

        # algebraic operator wrapping the input -> output map
        self.op_alg = Operator(func=self._func_alg)

        # Pre-size the input register to the three declared ports. As part
        # of the Wheel <-> Differential cycle this block is evaluated by
        # PathSim's algebraic-loop stage before any connection has grown
        # the register, and ``_func_alg`` unpacks all three inputs.
        self.inputs.resize(len(self.input_port_labels))


    def update(self, t):
        """Evaluate the differential as part of the algebraic component
        of the global system DAE.

        Parameters
        ----------
        t : float
            evaluation time
        """

        #apply operator to get output
        y = self.op_alg(self.inputs.to_array())
        self.outputs.update_from_array(y)


    def _func_alg(self, u):
        """Algebraic map from input torque and side speeds to side
        torques and input-shaft speed.

        Parameters
        ----------
        u : array[float]
            Input vector ``[T_in, omega_l, omega_r]``.

        Returns
        -------
        y : array[float]
            Output vector ``[T_l, T_r, omega_in]``.
        """
        T_in, omega_l, omega_r = u

        # equal-torque split with final-drive ratio
        T_side = 0.5 * self.i * T_in

        # Willis equation (carrier = mean side speed) with final-drive ratio
        omega_in = 0.5 * self.i * (omega_l + omega_r)

        return np.array([T_side, T_side, omega_in])
