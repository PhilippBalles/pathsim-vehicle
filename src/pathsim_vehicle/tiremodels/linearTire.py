#########################################################################################
##
##                  Linear tire model (ladder rung 1)
##
#########################################################################################



# IMPORTS ===============================================================================

from .tireModel import TireModel


# TIRE MODEL Definitions ================================================================

class LinearTire(TireModel):
    """Stateless, load-independent linear slip/cornering law (ladder rung 1).

    The classical linear-bicycle tire model — longitudinal force
    proportional to slip ratio, lateral force proportional to slip angle:

    .. math::

        F_x^{t} = C_\\kappa\\,\\kappa,
        \\qquad
        F_y^{t} = C_\\alpha\\,\\alpha.

    The law is strictly linear and load-independent:

    - **No load sensitivity**: :math:`F_z` does not enter the force law; a
      more heavily loaded tire is not stiffer in this model. :math:`F_z` is
      accepted only so every :class:`TireModel` shares one signature.
    - **Constant Jacobian**: the force Jacobian is the constant diagonal
      matrix :math:`\\operatorname{diag}(C_\\kappa, C_\\alpha)`; the two
      axes do not couple and the slopes never change.
    - **Odd, non-saturating**: each force is odd in its slip and grows
      without bound: there is no friction limit, so the tire never
      saturates.

    The model is therefore valid only in the small-slip, linear range —
    roughly :math:`|\\alpha|` below a few degrees and lateral acceleration
    within the linear tire regime (:math:`a_y \\lesssim 4\\,\\mathrm{m/s^2}`
    on dry asphalt for a typical passenger car), the same envelope as
    :class:`LinearSingleTrack`. Beyond it the saturating
    :class:`SimplePacejka` rung is required.

    ``LinearTire`` is the linearization of the Magic Formula about zero
    slip: near zero slip a :class:`SimplePacejka` tire *is* a ``LinearTire``
    with :math:`C_\\kappa = B_x C_x \\mu_x F_z` and
    :math:`C_\\alpha = B_y C_y \\mu_y F_z` at the operating load — at once
    the formal sense in which this model is the tangent of the Magic
    Formula and the practical recipe for choosing the stiffnesses to match
    a given Pacejka set near centre. The default ``C_alpha`` matches
    :class:`LinearSingleTrack`'s axle cornering stiffness, so a
    single-track built from ``Wheel`` + ``LinearTire`` reproduces its
    linear handling.


    Parameters
    ----------
    C_kappa : float
        Slip (longitudinal) stiffness [N], > 0.
    C_alpha : float
        Cornering (lateral) stiffness [N/rad], > 0.


    """

    def __init__(self, C_kappa=1.0e5, C_alpha=8.0e4):

        # tire stiffnesses
        self.C_kappa = C_kappa
        self.C_alpha = C_alpha


    def forces(self, kappa, alpha, F_z):
        """Linear tire-frame forces; ``F_z`` is accepted but unused."""
        return self.C_kappa * kappa, self.C_alpha * alpha
