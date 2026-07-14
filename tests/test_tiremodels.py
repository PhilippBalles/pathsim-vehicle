"""Tests for the tire-model ladder (TireModel, LinearTire, SimplePacejka)."""

import numpy as np
import pytest

from pathsim_vehicle.tiremodels import TireModel, LinearTire, SimplePacejka, magic_sine


def test_base_contract_is_abstract():
    """The base TireModel declares the contract but implements nothing."""
    with pytest.raises(NotImplementedError):
        TireModel().forces(0.0, 0.0, 4000.0)


def test_linear_tire_is_linear_and_decoupled():
    """F_x = C_kappa*kappa and F_y = C_alpha*alpha, each axis independent."""
    tire = LinearTire(C_kappa=1.0e5, C_alpha=8.0e4)
    kappa, alpha = 0.02, -0.05
    F_x, F_y = tire.forces(kappa, alpha, 4000.0)
    assert np.isclose(F_x, tire.C_kappa * kappa)
    assert np.isclose(F_y, tire.C_alpha * alpha)


def test_linear_tire_ignores_load():
    """F_z is accepted for interface uniformity but unused."""
    tire = LinearTire()
    assert tire.forces(0.03, 0.01, 0.0) == tire.forces(0.03, 0.01, 9000.0)


def test_simple_pacejka_is_odd_and_bounded():
    """Both forces are odd in their slip and bounded by the peak mu*F_z."""
    tire = SimplePacejka()
    F_z = 4000.0
    assert tire.forces(0.0, 0.0, F_z) == (0.0, 0.0)
    for s in [0.01, 0.05, 0.2, 0.8]:
        F_x_p, F_y_p = tire.forces(s, s, F_z)
        F_x_m, F_y_m = tire.forces(-s, -s, F_z)
        assert np.isclose(F_x_p, -F_x_m)
        assert np.isclose(F_y_p, -F_y_m)
        assert abs(F_x_p) <= tire.mu_x * F_z
        assert abs(F_y_p) <= tire.mu_y * F_z


def test_simple_pacejka_attains_peak_mu_Fz():
    """For C > 1 the curve actually reaches the load-scaled peak D = mu*F_z."""
    tire = SimplePacejka()
    F_z = 4000.0
    slips = np.linspace(0.0, 4.0, 40001)
    F_x = magic_sine(slips, tire.B_x, tire.C_x, tire.mu_x * F_z, tire.E_x)
    F_y = magic_sine(slips, tire.B_y, tire.C_y, tire.mu_y * F_z, tire.E_y)
    assert np.isclose(F_x.max(), tire.mu_x * F_z, rtol=1e-4)
    assert np.isclose(F_y.max(), tire.mu_y * F_z, rtol=1e-4)


def test_simple_pacejka_peak_scales_linearly_with_load():
    """The load enters only through D = mu*F_z, so forces scale linearly in F_z."""
    tire = SimplePacejka()
    F_x_1, F_y_1 = tire.forces(0.1, 0.08, 3000.0)
    F_x_2, F_y_2 = tire.forces(0.1, 0.08, 6000.0)
    assert np.isclose(F_x_2, 2 * F_x_1)
    assert np.isclose(F_y_2, 2 * F_y_1)


def test_small_slip_reduces_to_linear_tire():
    """Near zero slip SimplePacejka is a LinearTire with the bridge
    stiffnesses C_kappa = B_x*C_x*mu_x*F_z, C_alpha = B_y*C_y*mu_y*F_z."""
    tire = SimplePacejka()
    F_z = 4000.0
    linear = LinearTire(C_kappa=tire.B_x * tire.C_x * tire.mu_x * F_z,
                        C_alpha=tire.B_y * tire.C_y * tire.mu_y * F_z)
    s = 1e-4
    F_x_p, F_y_p = tire.forces(s, s, F_z)
    F_x_l, F_y_l = linear.forces(s, s, F_z)
    assert np.isclose(F_x_p, F_x_l, rtol=1e-4)
    assert np.isclose(F_y_p, F_y_l, rtol=1e-4)


def test_default_slopes_match_linear_tire_defaults():
    """At the nominal load F_z = 4000 N the default initial slopes are
    ~1.0e5 N and ~8.0e4 N/rad, matching LinearTire's defaults."""
    tire = SimplePacejka()
    F_z = 4000.0
    C_kappa = tire.B_x * tire.C_x * tire.mu_x * F_z
    C_alpha = tire.B_y * tire.C_y * tire.mu_y * F_z
    assert np.isclose(C_kappa, 1.0e5, rtol=0.05)
    assert np.isclose(C_alpha, 8.0e4, rtol=0.05)
