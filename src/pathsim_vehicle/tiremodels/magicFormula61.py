#########################################################################################
##
##                  Pacejka Magic Formula 6.1.2 tire model (ladder rung 3)
##
#########################################################################################



# IMPORTS ===============================================================================

import os

import numpy as np

from .tireModel import TireModel
from .simplePacejka import magic_sine


# DEFAULT PARAMETER SET =================================================================

# Canonical Pacejka-book default coefficient set (MF6.1, FNOMIN = 4000 N passenger
# car), shipped so the model is usable without a fitted .tir. Matches the Wheel's
# nominal load, so it sits directly under SimplePacejka's defaults.
DEFAULT_TIR = os.path.join(os.path.dirname(__file__), "data", "PacejkaBook_Defaults.tir")


# HELPERS ===============================================================================

def read_tir(path):
    """Parse a Tyre Property File (``.tir``) into a ``{KEY: value}`` dictionary.

    The ``.tir`` is the MF-Tyre/MF-Swift text format: ``[SECTION]`` headers over
    ``KEY = VALUE`` lines, with ``$`` starting an inline comment and ``!`` a
    full-line one. Section headers are organisational only, so the result is a
    single flat dictionary keeping *every* group (model, dimensions, vertical,
    scaling factors, and all five force/moment coefficient groups). Numeric
    values become ``float``; the few string values (e.g. ``TYRESIDE``) are kept
    as stripped strings. A coefficient absent from the file is simply not a key;
    :meth:`MagicFormula61.forces` reads every coefficient through a
    ``get(key, 0.0)`` accessor, so a missing optional term is inert (0).

    Parameters
    ----------
    path : str
        Path to the ``.tir`` file.

    Returns
    -------
    params : dict
        Flattened ``KEY -> float | str`` coefficient dictionary.

    Raises
    ------
    ValueError
        If ``FITTYP`` is missing or not in ``{61, 62}`` (MF6.1/6.2 share this
        layout; 6.2 only populates a few extra terms).
    """
    params = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("!") or line.startswith("["):
                continue
            line = line.split("$", 1)[0].strip()   # drop inline comment
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'").strip('"').strip()
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val                  # keep string values verbatim

    if "FITTYP" not in params or int(params["FITTYP"]) not in (61, 62):
        raise ValueError(
            "read_tir: expected a Magic-Formula 6.1/6.2 file (FITTYP in {61, 62}), "
            f"got FITTYP={params.get('FITTYP')!r} from {path!r}")
    return params


def magic_cos(z, B, C, E):
    """Magic-Formula *cosine* kernel used for the combined-slip weighting.

    .. math::

        \\mathrm{MF}_{\\cos}(z; B, C, E) =
        \\cos\\!\\Big[C\\arctan\\!\\big(Bz - E(Bz - \\arctan Bz)\\big)\\Big]

    Same argument as :func:`~pathsim_vehicle.tiremodels.simplePacejka.magic_sine`
    but with an outer cosine and no peak factor: it equals 1 at ``z = 0`` and
    falls below 1 away from it, which is exactly the friction-circle weighting
    that degrades one force in the presence of cross-slip.

    Parameters
    ----------
    z : float
        Weighting argument (shifted cross-slip).
    B, C, E : float
        Stiffness, shape, and curvature factors of the weighting.

    Returns
    -------
    G : float
        Cosine weighting (dimensionless).
    """
    bz = B * z
    return np.cos(C * np.arctan(bz - E * (bz - np.arctan(bz))))


# TIRE MODEL Definitions ================================================================

class MagicFormula61(TireModel):
    """Pacejka "Magic Formula" MF6.1.2 forces, pure + combined slip (rung 3).

    The top rung of the tire-model ladder and the first with **combined slip**
    — the :math:`\\kappa\\!\\leftrightarrow\\!\\alpha` friction-circle coupling
    that :class:`SimplePacejka` lacks — and with the full load dependence
    (through :math:`df_z`) of the cornering/slip stiffnesses, the peak, and the
    curvature. The pure-slip forces are the Magic-Formula sine with
    load-dependent coefficients and shifts; combined slip multiplies each by a
    cosine weighting (and adds the :math:`\\kappa`-induced side force
    :math:`S_{Vy\\kappa}`):

    .. math::

        F_x^{t} = G_{x\\alpha}\\,F_{x0},
        \\qquad
        F_y^{t} = G_{y\\kappa}\\,F_{y0} + S_{Vy\\kappa}.

    Because the weightings normalise to 1 at zero cross-slip
    (:math:`G_{x\\alpha}|_{\\alpha^*=0} = G_{y\\kappa}|_{\\kappa=0} = 1`,
    :math:`S_{Vy\\kappa}|_{\\kappa=0} = 0`), the combined forces reduce exactly
    to the pure forces there, and — with the shifts off — the pure force is
    exactly :class:`SimplePacejka`'s ``magic_sine``; its initial slope is the
    (now load-dependent) stiffness, so near zero slip it is a
    :class:`LinearTire`. These reductions are the ladder cross-checks verified in
    ``derive_mf61.py``.

    Scope (v1 = forces-only)
    ------------------------
    Returns the two tire-frame **forces** only, evaluated at camber
    :math:`\\gamma^* = 0`, nominal inflation (:math:`dp_i = 0`), unit turn-slip
    weights (:math:`\\zeta_i = 1`) and forward motion
    (:math:`\\operatorname{sgn}V_{cx} = +1`, so :math:`\\alpha^* = \\tan\\alpha`).
    The parser still reads the *whole* ``.tir``, so the self-aligning moment
    :math:`M_z`, the overturning :math:`M_x`, the rolling-resistance
    :math:`M_y`, and the camber/pressure/velocity-sign inputs are *parsed but
    deferred* to a v2 (an additive upgrade, not a rewrite). Full coefficient
    expansion: the companion ``pacejka.tex`` (eqs 4.E1–4.E30, 4.E50–4.E67);
    top-level structure and this evaluation point: ``mf61.tex``.

    Parameters
    ----------
    coeffs : dict, optional
        A parsed ``.tir`` coefficient dictionary (see :func:`read_tir`). Defaults
        to the shipped Pacejka-book passenger-car set.


    """

    def __init__(self, coeffs=None):
        # whole .tir kept; forces() reads coefficients via get(key, 0.0)
        self.p = read_tir(DEFAULT_TIR) if coeffs is None else dict(coeffs)

    @classmethod
    def from_tir(cls, path):
        """Construct from a ``.tir`` file path (parse, then build)."""
        return cls(read_tir(path))

    def forces(self, kappa, alpha, F_z):
        """MF6.1.2 tire-frame forces (v1: gamma=0, dp_i=0, zeta=1, sgn(V_cx)=+1)."""
        p = self.p
        def g(k):
            return float(p.get(k, 0.0))

        eps = 1.0e-6
        F_z0 = g("LFZO") * g("FNOMIN")
        dfz = (F_z - F_z0) / F_z0
        # alpha* = tan(alpha) * sgn(V_cx), with sgn(V_cx) = +1 (v1 forward motion).
        # Sign reconciliation: standard .tir sets are ISO/TYDEX, where the cornering
        # stiffness is negative (dF_y/dalpha < 0). The Wheel and the rung-1/2 tires
        # use the opposite convention (dF_y/dalpha > 0), so the slip-angle sign is
        # flipped here to present that convention to the Wheel -- exactly OCD's
        # `-slip_angle` feed into MF52 (mf61.tex, "Sign convention").
        a = -np.tan(alpha)

        def _sgn(z):
            return np.where(z >= 0.0, 1.0, -1.0)

        def _clamp_E(E):
            return np.minimum(E, 1.0)           # Magic-Formula curvature (<= 1)

        # --- pure longitudinal F_x0 (eqs 4.E9-4.E18 at gamma=0, dp_i=0) ---
        C_x = g("PCX1") * g("LCX")
        mu_x = (g("PDX1") + g("PDX2") * dfz) * g("LMUX")
        D_x = mu_x * F_z
        K_xk = F_z * (g("PKX1") + g("PKX2") * dfz) * np.exp(g("PKX3") * dfz) * g("LKX")
        B_x = K_xk / (C_x * D_x + eps)
        S_Hx = (g("PHX1") + g("PHX2") * dfz) * g("LHX")
        S_Vx = F_z * (g("PVX1") + g("PVX2") * dfz) * g("LVX") * g("LMUX")
        kappa_x = kappa + S_Hx
        E_x = _clamp_E((g("PEX1") + g("PEX2") * dfz + g("PEX3") * dfz**2)
                       * (1.0 - g("PEX4") * _sgn(kappa_x)) * g("LEX"))
        F_x0 = magic_sine(kappa_x, B_x, C_x, D_x, E_x) + S_Vx

        # --- pure lateral F_y0 (eqs 4.E19-4.E29 at gamma=0, dp_i=0) ---
        C_y = g("PCY1") * g("LCY")
        mu_y = (g("PDY1") + g("PDY2") * dfz) * g("LMUY")
        D_y = mu_y * F_z
        K_ya = g("PKY1") * F_z0 * np.sin(g("PKY4") * np.arctan(F_z / (g("PKY2") * F_z0))) * g("LKY")
        B_y = K_ya / (C_y * D_y + eps)
        S_Hy = (g("PHY1") + g("PHY2") * dfz) * g("LHY")
        S_Vy = F_z * (g("PVY1") + g("PVY2") * dfz) * g("LVY") * g("LMUY")
        alpha_y = a + S_Hy
        E_y = _clamp_E((g("PEY1") + g("PEY2") * dfz)
                       * (1.0 - g("PEY3") * _sgn(alpha_y)) * g("LEY"))
        F_y0 = magic_sine(alpha_y, B_y, C_y, D_y, E_y) + S_Vy

        # --- combined-slip weightings (eqs 4.E50-4.E67 at gamma=0) ---
        B_xa = g("RBX1") * np.cos(np.arctan(g("RBX2") * kappa)) * g("LXAL")
        E_xa = _clamp_E(g("REX1") + g("REX2") * dfz)
        S_Hxa = g("RHX1")
        C_xa = g("RCX1")
        G_xa = (magic_cos(a + S_Hxa, B_xa, C_xa, E_xa)
                / magic_cos(S_Hxa, B_xa, C_xa, E_xa))

        B_yk = g("RBY1") * np.cos(np.arctan(g("RBY2") * (a - g("RBY3")))) * g("LYKA")
        E_yk = _clamp_E(g("REY1") + g("REY2") * dfz)
        S_Hyk = g("RHY1") + g("RHY2") * dfz
        C_yk = g("RCY1")
        G_yk = (magic_cos(kappa + S_Hyk, B_yk, C_yk, E_yk)
                / magic_cos(S_Hyk, B_yk, C_yk, E_yk))

        D_Vyk = mu_y * F_z * (g("RVY1") + g("RVY2") * dfz) * np.cos(np.arctan(g("RVY4") * a))
        S_Vyk = D_Vyk * np.sin(g("RVY5") * np.arctan(g("RVY6") * kappa)) * g("LVYKA")

        return G_xa * F_x0, G_yk * F_y0 + S_Vyk
