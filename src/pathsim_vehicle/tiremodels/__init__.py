"""Swappable tire force laws for the ``Wheel`` block.

The tire-model ladder is ``LinearTire`` -> ``SimplePacejka`` ->
``MagicFormula61`` (rung 3), one module per rung.
Every rung is a stateless :class:`TireModel` object (*not* a block) that the
``Wheel`` calls as ``forces(kappa, alpha, F_z) -> (F_x_t, F_y_t)``: a pure
function of the longitudinal slip ratio, the slip angle, and the (clamped,
non-negative) vertical load, returning the two contact forces in the tire
(wheel) frame. The ``Wheel`` owns everything around this call — it forms the
slips from the contact-point kinematics, clamps ``F_z >= 0``, and rotates
the returned forces into the body frame by the steer angle — so the tire
models are frame-agnostic and carry no state.

In this version the contract is *forces-only*: the self-aligning moment
``M_z`` and the overturning/rolling moments are out of scope (a documented,
additive upgrade — ``MagicFormula61`` parses their coefficients but defers
their evaluation to a v2).
"""

from .tireModel import TireModel
from .linearTire import LinearTire
from .simplePacejka import SimplePacejka, magic_sine
from .magicFormula61 import MagicFormula61, magic_cos, read_tir

__all__ = ["TireModel", "LinearTire", "SimplePacejka", "magic_sine",
           "MagicFormula61", "magic_cos", "read_tir"]
