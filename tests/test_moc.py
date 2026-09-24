import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine_design import moc, nozzle


def _source_flow(x, r, g):
    """Exact conical source flow from the origin: (theta, nu, mu) at (x, r)."""
    M = moc.mach_from_area(x * x + r * r, g)       # sonic sphere of unit radius
    return math.atan2(r, x), moc.prandtl_meyer(M, g), math.asin(1 / M)


def test_compatibility_relations_match_source_flow():
    # The axisymmetric source terms must reproduce an exact solution.
    g, x, r, h = 1.2, 3.0, 0.8, 1e-6
    th, nu, mu = _source_flow(x, r, g)
    p = (x, r, th, nu, 0.0, mu)
    for direction, invariant, q in [(th - mu, lambda t, n: t + n, moc._q_minus(p)),
                                    (th + mu, lambda t, n: t - n, moc._q_plus(p))]:
        x2, r2 = x + h * math.cos(direction), r + h * math.sin(direction)
        th2, nu2, _ = _source_flow(x2, r2, g)
        slope = (invariant(th2, nu2) - invariant(th, nu)) / (x2 - x)
        assert slope == pytest.approx(q, rel=1e-4)


def test_planar_ideal_nozzle_matches_exact_theory():
    moc.AXISYMMETRIC = False
    try:
        g, Me = 1.4, 2.4
        c = moc.ideal_contour(Me, g, n=40)
        # Planar minimum-length theory: wall angle at the end of the expansion is nu_e / 2.
        assert c.theta_attach_deg == pytest.approx(math.degrees(moc.prandtl_meyer(Me, g)) / 2, abs=0.1)
        assert c.r[-1] == pytest.approx(moc.area_ratio(Me, g), rel=3e-3)
        assert moc.analyze(c.x, c.r, g, n=40).efficiency == pytest.approx(1.0, abs=1e-3)
    finally:
        moc.AXISYMMETRIC = True


@pytest.mark.parametrize("alpha", [10.0, 15.0, 20.0])
def test_conical_nozzle_matches_source_flow_thrust(alpha):
    # In source flow the exit surface is a spherical cap; axial momentum and
    # pressure over it give F = p (1 + g M^2) A_exit at the cap's area ratio.
    g, eps, Rd = 1.2, 10.0, 2.0
    a = math.radians(alpha)
    t = np.linspace(0, a, 60)
    x, r = Rd * np.sin(t), 1 + Rd * (1 - np.cos(t))
    xl = np.linspace(x[-1], x[-1] + (math.sqrt(eps) - r[-1]) / math.tan(a), 200)[1:]
    x, r = np.concatenate([x, xl]), np.concatenate([r, r[-1] + math.tan(a) * (xl - x[-1])])

    def f(M):
        return moc.pressure_ratio(M, g) * (1 + g * M * M)

    theory = f(moc.mach_from_area(eps * 2 / (1 + math.cos(a)), g)) / f(moc.mach_from_area(eps, g))
    assert moc.analyze(x, r, g, n=30).efficiency == pytest.approx(theory, abs=3e-3)


def test_ideal_contour_reanalysed_gives_uniform_axial_exit():
    # Designed by the inverse method, checked by the independent direct march.
    g, eps = 1.188, 4.53
    c = moc.ideal_contour(moc.mach_from_area(eps, g), g, n=40)
    assert c.r[-1] ** 2 == pytest.approx(eps, rel=0.01)
    flow = moc.analyze(c.x, c.r, g, n=40)
    assert flow.efficiency == pytest.approx(1.0, abs=3e-3)
    assert abs(math.degrees(flow.wall[-1][moc.TH])) < 0.1


def test_rao_bell_close_to_moc_truncated_ideal_contour():
    # The chart-based Rao approximation should perform like an MOC-designed
    # truncated ideal contour of the same length.
    g, eps = 1.188, 4.53
    xr, rr, _, _ = nozzle.diverging_contour(eps, 0.8, "rao")
    t = moc.tic_contour(eps, g, 0.8)
    assert t.x[-1] == pytest.approx(xr[-1], rel=0.01)
    assert t.r[-1] == pytest.approx(math.sqrt(eps), rel=1e-6)
    eff_rao = moc.analyze(xr, rr, g).momentum_efficiency
    eff_tic = moc.analyze(t.x, t.r, g).momentum_efficiency
    assert eff_rao == pytest.approx(eff_tic, abs=3e-3)
    assert 0.97 < eff_rao < 0.99
