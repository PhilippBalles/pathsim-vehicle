#########################################################################################
##
##                  Simplified Pacejka tire model (ladder rung 2)
##
#########################################################################################



# IMPORTS ===============================================================================

import numpy as np

from .tireModel import TireModel


# HELPERS ===============================================================================

def magic_sine(s, B, C, D, E):
    """Magic-Formula sine kernel shared by all Pacejka-family rungs.

    .. math::

        \\mathrm{MF}(s; B, C, D, E) =
        D\\,\\sin\\!\\Big[C\\arctan\\!\\big(Bs - E(Bs - \\arctan Bs)\\big)\\Big]

    The initial slope is the product :math:`BCD` (the curvature factor
    :math:`E` does not affect it), the curve is odd in ``s``, and since
    :math:`|\\sin| \\le 1` the force is bounded by the peak :math:`D`.

    Parameters
    ----------
    s : float
        Slip quantity (slip ratio or slip angle).
    B : float
        Stiffness factor.
    C : float
        Shape factor.
    D : float
        Peak force [N].
    E : float
        Curvature factor.

    Returns
    -------
    F : float
        Force at slip ``s`` [N].
    """
    bs = B * s
    return D * np.sin(C * np.arctan(bs - E * (bs - np.arctan(bs))))


# TIRE MODEL Definitions ================================================================

class SimplePacejka(TireModel):
    """Simplified Pacejka "Magic Formula", independently per axis (rung 2).

    The middle rung of the tire-model ladder: it keeps the one feature
    :class:`LinearTire` lacks — a *friction limit*, so the force saturates
    at a load-scaled peak :math:`D = \\mu F_z` instead of growing without
    bound — while staying far simpler than ``MagicFormula61``: it is *pure
    slip* (the longitudinal force depends only on :math:`\\kappa`, the
    lateral only on :math:`\\alpha`; no combined-slip coupling), with no
    horizontal or vertical shifts, no camber, no inflation pressure, and no
    self-aligning moment.

    Both forces are the Magic-Formula sine kernel evaluated independently
    on each axis, with the peak scaled by the vertical load:

    .. math::

        F_x^{t} = \\mathrm{MF}(\\kappa;\\,B_x, C_x, D_x, E_x),
        \\quad
        F_y^{t} = \\mathrm{MF}(\\alpha;\\,B_y, C_y, D_y, E_y),
        \\qquad
        D_x = \\mu_x F_z,\\;\\; D_y = \\mu_y F_z.

    The four parameters per axis have the standard Pacejka roles:

    - **B — stiffness factor**: sets the initial slope through the product
      :math:`BCD`.
    - **C — shape factor**: sets the high-slip asymptote and, for
      :math:`C \\ge 1`, guarantees the curve reaches its peak.
    - **D = mu*F_z — peak force**: the friction limit; :math:`\\mu` is the
      peak friction coefficient and the peak force scales linearly with
      load — the simplest possible load sensitivity.
    - **E — curvature factor** (:math:`\\le 1`): shapes the region around
      the peak without changing the initial slope.

    The kernel expands as :math:`\\mathrm{MF}(s) = (BCD)\\,s + O(s^3)`, so
    near zero slip a ``SimplePacejka`` tire *is* a :class:`LinearTire` with
    :math:`C_\\kappa = B_x C_x \\mu_x F_z` and
    :math:`C_\\alpha = B_y C_y \\mu_y F_z` at the operating load. The curve
    is odd and, unlike :class:`LinearTire`, bounded and saturating.

    Combined slip (the friction-circle coupling), the nonlinear load
    dependence, camber, inflation pressure, and the moments
    :math:`M_z, M_x, M_y` are all absent by design; they are
    ``MagicFormula61``'s additions.

    The defaults are representative dry-asphalt passenger-car values,
    chosen so the initial slopes at the ``Wheel``'s nominal load
    :math:`F_z = 4000\\,\\mathrm{N}` are
    :math:`C_\\kappa \\approx 1.0 \\times 10^5\\,\\mathrm{N}` and
    :math:`C_\\alpha \\approx 8.0 \\times 10^4\\,\\mathrm{N/rad}` —
    matching :class:`LinearTire`'s defaults, so the two rungs agree near
    centre and diverge only as the tire loads up.


    Parameters
    ----------
    B_x : float
        Longitudinal stiffness factor.
    C_x : float
        Longitudinal shape factor, > 1.
    mu_x : float
        Longitudinal peak friction coefficient.
    E_x : float
        Longitudinal curvature factor, <= 1.
    B_y : float
        Lateral stiffness factor.
    C_y : float
        Lateral shape factor, > 1.
    mu_y : float
        Lateral peak friction coefficient.
    E_y : float
        Lateral curvature factor, <= 1.


    """

    def __init__(self, B_x=18.0, C_x=1.40, mu_x=1.0, E_x=0.97,
                 B_y=15.0, C_y=1.30, mu_y=1.0, E_y=0.97):

        # longitudinal Pacejka factors
        self.B_x = B_x
        self.C_x = C_x
        self.mu_x = mu_x
        self.E_x = E_x

        # lateral Pacejka factors
        self.B_y = B_y
        self.C_y = C_y
        self.mu_y = mu_y
        self.E_y = E_y


    def forces(self, kappa, alpha, F_z):
        """Pure-slip Magic-Formula forces with load-scaled peaks."""
        F_x_t = magic_sine(kappa, self.B_x, self.C_x, self.mu_x * F_z, self.E_x)
        F_y_t = magic_sine(alpha, self.B_y, self.C_y, self.mu_y * F_z, self.E_y)
        return F_x_t, F_y_t
