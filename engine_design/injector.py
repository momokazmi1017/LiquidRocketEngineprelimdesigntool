"""Injector orifice sizing for an unlike-impinging doublet pattern.

Orifice flow:      mdot = Cd * A * sqrt(2 * rho * dp)
Resultant angle:   tan(delta) = (mo vo sin(ao) - mf vf sin(af)) / (mo vo cos(ao) + mf vf cos(af))
where ao, af are each jet's angle from the chamber axis. A resultant near 0 deg
keeps the combined spray axial and away from the wall.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class OrificeSet:
    name: str
    n: int
    d: float        # orifice diameter, m
    v: float        # injection velocity, m/s
    dp: float       # pressure drop, Pa
    mdot: float     # total, kg/s


@dataclass
class InjectorDesign:
    ox: OrificeSet
    fuel: OrificeSet
    resultant_angle_deg: float
    momentum_ratio: float    # ox / fuel


def size_orifices(name, mdot, rho, dp, n, cd=0.7) -> OrificeSet:
    area_each = mdot / (n * cd * np.sqrt(2.0 * rho * dp))
    d = np.sqrt(4.0 * area_each / np.pi)
    v = cd * np.sqrt(2.0 * dp / rho)   # jet velocity downstream of the vena contracta, based on geometric area
    return OrificeSet(name, n, d, v, dp, mdot)


def unlike_doublet(mdot_ox, rho_ox, mdot_f, rho_f, pc, n_elements,
                   dp_frac=0.20, cd=0.7, angle_ox=30.0, angle_f=30.0) -> InjectorDesign:
    """Size an unlike doublet with one ox and one fuel orifice per element.

    dp_frac: injector pressure drop as a fraction of chamber pressure (15–25 % is
    typical for combustion stability / feed-coupling margin).
    """
    dp = dp_frac * pc
    ox = size_orifices("oxidizer", mdot_ox, rho_ox, dp, n_elements, cd)
    fu = size_orifices("fuel", mdot_f, rho_f, dp, n_elements, cd)
    ao, af = np.radians(angle_ox), np.radians(angle_f)
    mo, mf = mdot_ox * ox.v, mdot_f * fu.v
    delta = np.degrees(np.arctan2(mo * np.sin(ao) - mf * np.sin(af),
                                  mo * np.cos(ao) + mf * np.cos(af)))
    return InjectorDesign(ox, fu, delta, mo / mf)
