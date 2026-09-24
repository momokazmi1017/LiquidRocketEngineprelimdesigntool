"""Thrust chamber and nozzle contour generation.

Converging section: cylinder -> fillet (radius Rf) -> straight cone (half-angle
theta_c) -> upstream throat arc (1.5 Rt).

Diverging section, either:
- "rao": Rao thrust-optimised parabola (TOP) approximation — downstream
  throat arc (0.382 Rt) to the inflection angle theta_n, then a quadratic
  Bezier to the exit at angle theta_e. theta_n and theta_e come from Rao's
  charts as reproduced in Huzel & Huang (Fig. 4-16), digitised to about
  +/- 1 deg.
- "tic": truncated ideal contour designed with the method of characteristics
  (moc.py), with the same length convention.

The throat sits at x = 0; the injector face is at a negative x.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import moc
from .moc import cone_length

# Approximate Rao angles for a bell whose length is a fraction of a 15 deg cone.
_RAO_EPS = np.array([4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0])
_RAO_TABLE = {
    # length fraction: (theta_n [deg], theta_e [deg]) at each _RAO_EPS
    0.8: (np.array([21.5, 23.0, 26.0, 28.5, 30.0, 31.0, 32.0, 34.0]),
          np.array([14.0, 13.0, 11.0, 9.0, 8.0, 7.5, 7.0, 6.0])),
    0.9: (np.array([20.0, 21.0, 24.0, 26.5, 27.5, 28.5, 29.0, 31.0]),
          np.array([11.5, 10.5, 9.0, 7.5, 7.0, 6.5, 6.0, 5.0])),
}


def rao_angles(eps: float, length_frac: float = 0.8) -> tuple[float, float]:
    """(theta_n, theta_e) in degrees for a Rao TOP bell. Interpolated in log(eps)."""
    if length_frac not in _RAO_TABLE:
        raise ValueError(f"Rao chart angles are tabulated for length fractions {sorted(_RAO_TABLE)}; "
                         f"use bell='tic' for other lengths")
    tn, te = _RAO_TABLE[length_frac]
    x = np.log(np.clip(eps, _RAO_EPS[0], _RAO_EPS[-1]))
    return float(np.interp(x, np.log(_RAO_EPS), tn)), float(np.interp(x, np.log(_RAO_EPS), te))


@dataclass
class ChamberGeometry:
    x: np.ndarray            # axial stations, m (throat at 0)
    r: np.ndarray            # wall radius, m
    Rt: float
    Rc: float                # chamber radius
    Re: float                # exit radius
    L_cyl: float             # cylindrical length
    L_chamber: float         # injector face to throat
    L_nozzle: float          # throat to exit
    V_chamber: float         # volume injector -> throat, m^3
    theta_n: float
    theta_e: float
    R_throat_curv: float     # mean throat radius of curvature (for Bartz)


def _arc(xc, rc, R, a0, a1, n):
    """Points on a circle of radius R centred at (xc, rc) between angles a0 and a1 (rad)."""
    a = np.linspace(a0, a1, n)
    return xc + R * np.cos(a), rc + R * np.sin(a)


def _bezier(p0, p1, p2, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    return pts[:, 0], pts[:, 1]


def _converging(Rt, Rc, theta_c, Rf, Ru, n):
    """Fillet + cone + upstream throat arc, returned from the chamber end to the throat."""
    tc = np.radians(theta_c)
    # Upstream throat arc, centre (0, Rt + Ru); wall slope angle goes theta_c -> 0.
    xa, ra = _arc(0.0, Rt + Ru, Ru, -np.pi / 2 - tc, -np.pi / 2, n)
    x1, r1 = xa[0], ra[0]
    # Fillet tangent to the cylinder; its downstream end is where the cone starts.
    r2 = Rc - Rf * (1 - np.cos(tc))
    if r2 <= r1:
        raise ValueError("contraction ratio too small for the chosen fillet/throat radii")
    x2 = x1 - (r2 - r1) / np.tan(tc)
    xf = x2 - Rf * np.sin(tc)
    xfil, rfil = _arc(xf, Rc - Rf, Rf, np.pi / 2, np.pi / 2 - tc, n)
    xcone = np.linspace(x2, x1, n)
    rcone = r2 - (xcone - x2) * np.tan(tc)
    x = np.concatenate([xfil, xcone[1:], xa[1:]])
    r = np.concatenate([rfil, rcone[1:], ra[1:]])
    return x, r


def diverging_contour(eps: float, length_frac: float = 0.8, bell: str = "rao",
                      gamma: float | None = None, n: int = 60):
    """Bell from the throat to the exit, normalised by Rt: (x, r, theta_n, theta_e).

    bell="rao": Rao thrust-optimised parabola approximation (chart angles).
    bell="tic": truncated ideal contour designed by the method of characteristics
                (needs gamma); theta_n is the attachment angle of the throat arc.
    """
    Rd = 0.382
    if bell == "tic":
        if gamma is None:
            raise ValueError("bell='tic' needs gamma")
        t = _tic_cached(round(eps, 4), round(gamma, 4), round(length_frac, 4))
        return t.x, t.r, t.theta_attach_deg, t.theta_exit_deg
    if bell != "rao":
        raise ValueError(f"unknown bell type {bell!r}")
    tn, te = rao_angles(eps, length_frac)
    xd, rd = _arc(0.0, 1 + Rd, Rd, -np.pi / 2, -np.pi / 2 + np.radians(tn), n)
    N = np.array([xd[-1], rd[-1]])
    Ln = length_frac * cone_length(eps)
    E = np.array([Ln, np.sqrt(eps)])
    m1, m2 = np.tan(np.radians(tn)), np.tan(np.radians(te))
    c1, c2 = N[1] - m1 * N[0], E[1] - m2 * E[0]
    Q = np.array([(c2 - c1) / (m1 - m2), (m1 * c2 - m2 * c1) / (m1 - m2)])
    xb, rb = _bezier(N, Q, E, 3 * n)
    return np.concatenate([xd, xb[1:]]), np.concatenate([rd, rb[1:]]), tn, te


@lru_cache(maxsize=32)
def _tic_cached(eps, gamma, length_frac):
    return moc.tic_contour(eps, gamma, length_frac)


def chamber_contour(Rt: float, eps: float, contraction_ratio: float, L_star: float,
                    theta_c: float = 30.0, length_frac: float = 0.8, n: int = 60,
                    bell: str = "rao", gamma: float | None = None) -> ChamberGeometry:
    """Build the full inner wall contour from the injector face to the nozzle exit."""
    At = np.pi * Rt ** 2
    Rc = Rt * np.sqrt(contraction_ratio)
    Re = Rt * np.sqrt(eps)
    Ru, Rd = 1.5 * Rt, 0.382 * Rt
    Rf = 0.5 * Rc

    # Converging section and its volume
    xcv, rcv = _converging(Rt, Rc, theta_c, Rf, Ru, n)
    V_conv = np.trapezoid(np.pi * rcv ** 2, xcv)

    # Cylinder length from the characteristic length L* = Vc / At
    V_c = L_star * At
    L_cyl = (V_c - V_conv) / (np.pi * Rc ** 2)
    if L_cyl < 0:
        raise ValueError(f"L* = {L_star} m is too short: converging section alone exceeds it")
    x_inj = xcv[0] - L_cyl
    xcyl = np.linspace(x_inj, xcv[0], n)
    rcyl = np.full_like(xcyl, Rc)

    # Diverging bell (normalised shape, scaled by Rt)
    xd, rd, tn, te = diverging_contour(eps, length_frac, bell, gamma, n)
    xd, rd = Rt * xd, Rt * rd

    x = np.concatenate([xcyl, xcv[1:], xd[1:]])
    r = np.concatenate([rcyl, rcv[1:], rd[1:]])

    return ChamberGeometry(
        x=x, r=r, Rt=Rt, Rc=Rc, Re=Re, L_cyl=L_cyl, L_chamber=-x_inj, L_nozzle=float(xd[-1]),
        V_chamber=V_c, theta_n=tn, theta_e=te, R_throat_curv=0.5 * (Ru + Rd),
    )
