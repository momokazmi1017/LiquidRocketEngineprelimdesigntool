import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine_design import cooling, injector, nozzle
from engine_design.combustion import G0, performance
from engine_design.engine import EngineSpec, design
from engine_design.propellants import ETHANOL_75, LCH4, LH2, LOX, RP1

PSI = 6894.757
ATM = 101_325.0

# Sutton & Biblarz, Rocket Propulsion Elements, Table 5-5:
# Pc = 1000 psia, optimum expansion to 1 atm, shifting equilibrium.
SUTTON_5_5 = [
    # fuel, O/F, Tc [K], c* [m/s], Isp [s]
    (LH2, 4.02, 2999, 2386, 389),
    (RP1, 2.56, 3676, 1799, 300),
    (LCH4, 3.21, 3526, 1835, 311),
]


@pytest.mark.parametrize("fuel, of, Tc, cstar, isp", SUTTON_5_5, ids=lambda v: getattr(v, "name", None))
def test_performance_matches_sutton(fuel, of, Tc, cstar, isp):
    p = performance(LOX, fuel, of, 1000 * PSI, pe=ATM)
    assert p.isp(ATM) == pytest.approx(isp, rel=0.01)
    assert p.cstar == pytest.approx(cstar, rel=0.02)
    assert p.Tc == pytest.approx(Tc, rel=0.02)


def test_given_eps_roundtrips_exit_pressure():
    p1 = performance(LOX, RP1, 2.4, 50e5, pe=ATM)
    p2 = performance(LOX, RP1, 2.4, 50e5, eps=p1.eps)
    assert p2.pe == pytest.approx(ATM, rel=1e-4)


def test_shifting_beats_frozen():
    kw = dict(pc=30e5, eps=10.0)
    assert performance(LOX, LCH4, 3.0, shifting=True, **kw).isp_vac > \
        performance(LOX, LCH4, 3.0, shifting=False, **kw).isp_vac


def test_mach_area_relation():
    # gamma = 1.4, A/A* = 1.6875 <-> M = 2 (supersonic), M ~ 0.3722 (subsonic)
    assert cooling.mach_from_area(1.6875, 1.4, supersonic=True) == pytest.approx(2.0, rel=1e-4)
    assert cooling.mach_from_area(1.6875, 1.4, supersonic=False) == pytest.approx(0.3722, rel=1e-3)


def test_contour_geometry():
    Rt, eps, cr, Lstar = 0.02, 8.0, 5.0, 1.0
    g = nozzle.chamber_contour(Rt, eps, cr, Lstar)
    assert g.r.min() == pytest.approx(Rt, rel=1e-6)
    assert g.r[-1] == pytest.approx(Rt * np.sqrt(eps), rel=1e-6)
    assert g.r[0] == pytest.approx(Rt * np.sqrt(cr), rel=1e-6)
    assert np.all(np.diff(g.x) >= -1e-12)
    # Integrated volume up to the throat reproduces L* * At
    m = g.x <= 0
    V = np.trapezoid(np.pi * g.r[m] ** 2, g.x[m])
    assert V == pytest.approx(Lstar * np.pi * Rt ** 2, rel=0.01)


def test_orifice_flow_roundtrip():
    o = injector.size_orifices("ox", mdot=1.0, rho=1141.0, dp=5e5, n=10, cd=0.7)
    area = 10 * np.pi * o.d ** 2 / 4
    assert 0.7 * area * np.sqrt(2 * 1141.0 * 5e5) == pytest.approx(1.0, rel=1e-9)


def test_engine_design_closes_on_thrust():
    spec = EngineSpec("t", LOX, ETHANOL_75, thrust=5000.0, pc=25e5, of=1.3,
                      coolant=cooling.ETHANOL_75,
                      channels=cooling.Channels(72, 0.8e-3, 1.0e-3, 0.7e-3))
    d = design(spec)
    # F = mdot * Isp * g0, and F = Cf * Pc * At
    assert d.mdot * d.isp * G0 == pytest.approx(5000.0, rel=1e-9)
    assert d.cf * spec.pc * d.At == pytest.approx(5000.0, rel=1e-9)
    # Energy balance: coolant temperature rise matches integrated wall heat load
    c = d.cool
    assert c.Q_total == pytest.approx(d.mdot_fuel * spec.coolant.cp * (c.T_cool_out - c.T_cool[-1]), rel=1e-6)
    # Peak heat flux sits at the throat
    assert abs(c.x[np.argmax(c.q)]) < 0.1 * d.geom.Rt * 2
