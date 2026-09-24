"""Axisymmetric method of characteristics (MOC) for supersonic nozzle flow.

Irrotational, isentropic, constant-gamma flow. All lengths are normalised by
the throat radius Rt, and the throat is at x = 0.

Characteristics and compatibility relations (theta = flow angle, nu =
Prandtl-Meyer angle, mu = Mach angle):
    C+ : dr/dx = tan(theta + mu),   d(theta - nu) = -sin(theta) sin(mu) / (r cos(theta + mu)) dx
    C- : dr/dx = tan(theta - mu),   d(theta + nu) = +sin(theta) sin(mu) / (r cos(theta - mu)) dx
The right-hand sides are the axisymmetric source terms; without them these are
the planar Riemann invariants. Each unit process uses a predictor-corrector
with averaged coefficients.

Two uses:
- analyze(): march a characteristic net through a given wall contour (direct
  problem) and integrate the wall pressure to get thrust. This gives the
  nozzle's real divergence efficiency.
- ideal_contour(): design the wall that turns the flow back to uniform,
  axial flow at a chosen exit Mach number (inverse problem). Truncating it
  gives a practical bell (truncated ideal contour, TIC).

The net starts from uniform, just-supersonic flow across the throat plane.
MOC cannot represent shocks. Where characteristics of the same family cross
(a compression steepening into a weak shock) the crossings are counted and
reported; if the net breaks down entirely the march raises ShockFormed.
Reference: Zucrow & Hoffman, Gas Dynamics Vol. 2, ch. 15-16.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import brentq

# A characteristic-net point: (x, r, theta, nu, M, mu)
X, R, TH, NU, MA, MU = range(6)

# Set False for planar (2-D) flow: the source terms vanish and "r" becomes the
# half-height. Used to check the solver against exact planar results.
AXISYMMETRIC = True


# --- Gas dynamics -----------------------------------------------------------

def prandtl_meyer(M: float, g: float) -> float:
    a = math.sqrt((g + 1) / (g - 1))
    b = math.sqrt(M * M - 1)
    return a * math.atan(b / a) - math.atan(b)


def mach_from_nu(nu: float, g: float, M0: float = 2.0) -> float:
    """Invert the Prandtl-Meyer function (Newton; bisection fallback near M = 1)."""
    if nu <= 1e-12:
        return 1.0 + 1e-9
    M = max(M0, 1.0 + 1e-6)
    for _ in range(50):
        f = prandtl_meyer(M, g) - nu
        dfdM = math.sqrt(M * M - 1) / (M * (1 + 0.5 * (g - 1) * M * M))
        if dfdM <= 0:
            break
        M_new = M - f / dfdM
        if M_new <= 1.0:
            M_new = 0.5 * (M + 1.0)
        if abs(M_new - M) < 1e-12:
            return M_new
        M = M_new
    else:
        return M
    return brentq(lambda m: prandtl_meyer(m, g) - nu, 1.0 + 1e-12, 100.0, xtol=1e-14)


def area_ratio(M: float, g: float) -> float:
    return (1 / M) * ((2 / (g + 1)) * (1 + 0.5 * (g - 1) * M * M)) ** ((g + 1) / (2 * (g - 1)))


def mach_from_area(eps: float, g: float) -> float:
    """Supersonic Mach number for area ratio A/A*."""
    return brentq(lambda M: area_ratio(M, g) - eps, 1.0 + 1e-9, 50.0)


def mass_flux(M, g):
    """rho V / (rho0 a0): mass flux normalised by stagnation density and sound speed."""
    return M * (1 + 0.5 * (g - 1) * M * M) ** (-(g + 1) / (2 * (g - 1)))


def pressure_ratio(M, g):
    """p / p0."""
    return (1 + 0.5 * (g - 1) * M * M) ** (-g / (g - 1))


def _point(x, r, th, nu, g, M_guess=2.0):
    M = mach_from_nu(nu, g, M_guess)
    return (x, r, th, nu, M, math.asin(1.0 / M))


def _ring(r):
    """Flow area per unit length of a surface at radius r (planar: per unit depth)."""
    return 2 * math.pi * r if AXISYMMETRIC else 1.0


def _disk(r):
    """Cross-sectional area out to radius r."""
    return math.pi * r * r if AXISYMMETRIC else r


# --- Source terms and unit processes ----------------------------------------

# Near the axis sin(theta)/r is 0/0 (its limit is the finite d(theta)/dr), and a
# small error in theta at tiny r is amplified without bound. Within this radius
# a point's source term is not used; the other end of the characteristic
# segment supplies it instead.
_AXIS = 2e-2
N_CORRECTOR = 3


def _q_minus(p):
    if not AXISYMMETRIC:
        return 0.0
    if p[R] < _AXIS:
        return None
    return math.sin(p[TH]) * math.sin(p[MU]) / (p[R] * math.cos(p[TH] - p[MU]))


def _q_plus(p):
    if not AXISYMMETRIC:
        return 0.0
    if p[R] < _AXIS:
        return None
    return -math.sin(p[TH]) * math.sin(p[MU]) / (p[R] * math.cos(p[TH] + p[MU]))


def _avg(a, b):
    if a is None:
        return b if b is not None else 0.0
    if b is None:
        return a
    return 0.5 * (a + b)


def interior_point(p1, p2, g):
    """Intersection of the C+ through p1 and the C- through p2."""
    lp = math.tan(p1[TH] + p1[MU])
    lm = math.tan(p2[TH] - p2[MU])
    qp0, qm0 = _q_plus(p1), _q_minus(p2)
    qp, qm = qp0, qm0
    p3 = None
    for _ in range(N_CORRECTOR + 1):
        x3 = (p2[R] - p1[R] + lp * p1[X] - lm * p2[X]) / (lp - lm)
        r3 = p1[R] + lp * (x3 - p1[X])
        a = p1[TH] - p1[NU] + (qp or 0.0) * (x3 - p1[X])   # (theta - nu) at p3
        b = p2[TH] + p2[NU] + (qm or 0.0) * (x3 - p2[X])   # (theta + nu) at p3
        p3 = _point(x3, max(r3, 0.0), 0.5 * (a + b), 0.5 * (b - a), g, p1[MA])
        lp = math.tan(0.5 * (p1[TH] + p1[MU] + p3[TH] + p3[MU]))
        lm = math.tan(0.5 * (p2[TH] - p2[MU] + p3[TH] - p3[MU]))
        qp, qm = _avg(qp0, _q_plus(p3)), _avg(qm0, _q_minus(p3))
    return p3


def axis_point(p2, g):
    """Where the C- through p2 reaches the axis (theta = 0, r = 0)."""
    lm = math.tan(p2[TH] - p2[MU])
    qm = _q_minus(p2) or 0.0
    p3 = None
    for _ in range(N_CORRECTOR + 1):
        x3 = p2[X] - p2[R] / lm
        p3 = _point(x3, 0.0, 0.0, p2[TH] + p2[NU] + qm * (x3 - p2[X]), g, p2[MA])
        lm = math.tan(0.5 * (p2[TH] - p2[MU] - p3[MU]))
    return p3


def wall_point(p1, wall, g):
    """Where the C+ through p1 meets the wall; the flow there follows the wall slope."""
    lp = math.tan(p1[TH] + p1[MU])
    qp0 = _q_plus(p1) or 0.0
    qp = qp0
    p3 = None
    for _ in range(N_CORRECTOR + 1):
        x3 = wall.intersect_line(p1[X], p1[R], lp)
        th3 = math.atan(wall.slope(x3))
        nu3 = th3 - (p1[TH] - p1[NU] + qp * (x3 - p1[X]))
        p3 = _point(x3, wall.r(x3), th3, nu3, g, p1[MA])
        lp = math.tan(0.5 * (p1[TH] + p1[MU] + p3[TH] + p3[MU]))
        qp = _avg(qp0, _q_plus(p3))
    return p3


# --- Wall description -------------------------------------------------------

class Wall:
    """Nozzle wall r(x) from tabulated points (normalised by Rt), extended
    linearly past the last point. The extension is only used to locate the
    wall point that overshoots the end of the wall; that point is replaced by
    one exactly at the end (see _march_to_wall_point)."""

    def __init__(self, x, r):
        x, r = np.asarray(x, float), np.asarray(r, float)
        keep = np.concatenate([[True], np.diff(x) > 1e-12])
        self.x, self.rr = x[keep], r[keep]
        self.dr = np.gradient(self.rr, self.x)
        self.x_end = float(self.x[-1])

    def r(self, x):
        if x <= self.x_end:
            return float(np.interp(x, self.x, self.rr))
        return float(self.rr[-1] + self.dr[-1] * (x - self.x_end))

    def slope(self, x):
        return float(np.interp(min(x, self.x_end), self.x, self.dr))

    def intersect_line(self, x1, r1, m):
        """Where the line r = r1 + m (x - x1) through (x1, r1) meets the wall.

        Normally (x1, r1) is inside the nozzle and the meeting point is
        downstream. Where the wall curves back inward, discretisation can leave
        a point a hair outside the wall; the line then meets the wall slightly
        upstream, and that point is returned instead."""
        def f(x):
            return self.r(x) - (r1 + m * (x - x1))
        f1 = f(x1)
        if f1 == 0:
            return x1
        step = max(1e-6, 0.02 * abs(f1) / max(abs(m), 1e-3))
        direction = 1.0 if f1 > 0 else -1.0
        a, b = x1, x1 + direction * step
        while (f(b) > 0) == (f1 > 0):
            a, b = b, b + 2 * (b - a)
            if abs(b - x1) > 1e3 or b < 0:
                raise RuntimeError("characteristic never reaches the wall")
        return brentq(f, min(a, b), max(a, b), xtol=1e-12)


def throat_arc(R_d=0.382, theta_max_deg=60.0, n=400):
    """Downstream throat arc of radius R_d (normalised by Rt)."""
    t = np.linspace(0.0, math.radians(theta_max_deg), n)
    return Wall(R_d * np.sin(t), 1.0 + R_d * (1 - np.cos(t)))


# --- Characteristic net -----------------------------------------------------

M_START = 1.01


def initial_line(g, n=40, M0=M_START):
    """Uniform, parallel flow at M0 (just supersonic) across the throat plane.

    This ignores the curvature of the real sonic line; the error is confined to
    the region near the throat, while divergence losses are set by the
    downstream wall. (Sauer's transonic solution is not valid for the tight
    0.382 Rt downstream throat radius of Rao-type bells.)
    """
    nu, mu = prandtl_meyer(M0, g), math.asin(1 / M0)
    return [(0.0, float(y), 0.0, nu, M0, mu) for y in np.linspace(0.0, 1.0, n)]


class ShockFormed(RuntimeError):
    """Characteristics of the same family crossed: a compression steepened into a shock."""


MERGE = True


def _merge_crossings(row):
    """Merge characteristics of the same family that have crossed.

    Along a valid row the points move monotonically away from the axis. A
    point out of order means a compression has steepened into a weak shock,
    which MOC cannot represent. The classical remedy is to coalesce the
    crossing characteristics into one: the out-of-order point is dropped (the
    weak shock's small entropy rise is neglected). The axis point and the last
    point (on the wall, or on the final characteristic) are always kept.

    Returns (cleaned row, number of merges).
    """
    for p in row:
        if not (math.isfinite(p[NU]) and math.isfinite(p[X]) and math.isfinite(p[R])):
            raise ShockFormed(f"net broke down near x = {p[X]:.3f} Rt")
    if not MERGE:
        return row, sum(1 for a, b in zip(row[:-1], row[1:-1]) if b[R] <= a[R])
    out = [row[0]]
    merges = 0
    for p in row[1:-1]:
        if p[R] > out[-1][R] and p[R] < row[-1][R]:
            out.append(p)
        else:
            merges += 1
    if len(row) > 1:
        out.append(row[-1])
    return out, merges


def _march_to_wall_point(g, wall, n, reached, x_exact, shrink=True):
    """March the net from the throat until the wall point satisfies reached(w).

    That wall point is replaced by one at x_exact(prev, w) (the Prandtl-Meyer
    angle is interpolated linearly in x between the neighbouring wall points),
    so results vary continuously with where the march stops.

    With shrink=True the net then continues only within that point's domain of
    dependence: each row is one point shorter and its top point lies on the C-
    from the stop point, until that C- reaches the axis.

    Returns (all points, C- from the stop point to the axis (or None),
    wall points from the throat to the stop point, merge count).
    """
    row = initial_line(g, n)
    pts = list(row)
    walls = [row[-1]]
    crossings = 0
    try:
        for _ in range(20000):
            mid = [interior_point(row[k], row[k + 1], g) for k in range(len(row) - 1)]
            mid, merged = _merge_crossings(mid)
            crossings += merged
            nxt = [axis_point(mid[0], g)]
            nxt += [interior_point(mid[k], mid[k + 1], g) for k in range(len(mid) - 1)]
            w = wall_point(mid[-1], wall, g)
            done = reached(w)
            if done:
                prev = row[-1]
                x = x_exact(prev, w)
                f = (x - prev[X]) / (w[X] - prev[X])
                w = _point(x, wall.r(x), math.atan(wall.slope(x)), prev[NU] + f * (w[NU] - prev[NU]), g, w[MA])
            nxt.append(w)
            walls.append(w)
            nxt, merged = _merge_crossings(nxt)
            crossings += merged
            pts += mid + nxt
            row = nxt
            if done:
                break
        else:
            raise RuntimeError("characteristic net did not reach the stopping point")
        if not shrink:
            return np.array(pts), None, walls, crossings

        last = [row[-1]]
        while len(row) > 1:
            mid = [interior_point(row[k], row[k + 1], g) for k in range(len(row) - 1)]
            if len(mid) > 1:
                mid, merged = _merge_crossings(mid)
                crossings += merged
            nxt = [axis_point(mid[0], g)]
            nxt += [interior_point(mid[k], mid[k + 1], g) for k in range(len(mid) - 1)]
            if len(nxt) > 1:
                nxt, merged = _merge_crossings(nxt)
                crossings += merged
            pts += mid + nxt
            last += [mid[-1], nxt[-1]]
            row = nxt
    except (ValueError, ZeroDivisionError) as e:
        raise ShockFormed(f"net broke down near x = {row[0][X]:.3f} Rt ({e})") from e
    return np.array(pts), last, walls, crossings


def _segment_fluxes(a, b, g):
    """(mass flow, axial thrust) across a short segment of a C- characteristic.

    The flow crosses a characteristic at the Mach angle mu; the downstream
    normal of a C- segment has axial component sin(mu - theta). Fluxes are
    normalised by rho0 a0 Rt^2 and p0 Rt^2 respectively.
    """
    ds = math.hypot(b[X] - a[X], b[R] - a[R])
    M, th, mu = 0.5 * (a[MA] + b[MA]), 0.5 * (a[TH] + b[TH]), 0.5 * (a[MU] + b[MU])
    area = _ring(0.5 * (a[R] + b[R])) * ds
    mdot = mass_flux(M, g) * math.sin(mu) * area
    thrust = pressure_ratio(M, g) * (g * M * M * math.cos(th) * math.sin(mu) + math.sin(mu - th)) * area
    return mdot, thrust


def _c_minus_fluxes(pts, g):
    m = f = 0.0
    for a, b in zip(pts, pts[1:]):
        dm, df = _segment_fluxes(a, b, g)
        m, f = m + dm, f + df
    return m, f


def line_mass_flow(pts, g):
    """Mass flow through a constant-x line of points (normalised by rho0 a0 Rt^2)."""
    r = np.array([p[R] for p in pts])
    f = np.array([mass_flux(p[MA], g) * math.cos(p[TH]) * _ring(p[R]) for p in pts])
    return float(np.trapezoid(f, r))


class FieldInterpolator:
    """(theta, M) at arbitrary (x, r) from net points. Linear inside the net;
    nearest point in the thin sliver between a convex wall and the net's hull."""

    def __init__(self, points):
        xy, vals = points[:, [X, R]], points[:, [TH, MA]]
        self._lin = LinearNDInterpolator(xy, vals)
        self._near = NearestNDInterpolator(xy, vals)

    def __call__(self, x, r):
        x, r = np.atleast_1d(x).astype(float), np.atleast_1d(r).astype(float)
        v = self._lin(x, r)
        bad = np.isnan(v[:, 0])
        if bad.any():
            v[bad] = self._near(x[bad], r[bad])
        return v


# --- Direct problem: flow through a given wall ------------------------------

@dataclass
class NozzleFlow:
    gamma: float
    points: np.ndarray        # (N, 6) array of net points
    wall: np.ndarray          # net points on the wall, throat -> exit lip
    x_exit: float
    r_exit: float
    mdot: float               # through the throat (normalised by rho0 a0 Rt^2)
    thrust_vac: float         # vacuum thrust, normalised by p0 Rt^2
    efficiency: float         # vacuum thrust / 1-D ideal vacuum thrust, same mass flow and exit area
    momentum_efficiency: float  # jet-momentum efficiency (see analyze)
    discharge_coeff: float    # mdot / 1-D choked mdot through the geometric throat
    merges: int               # weak-shock merges of crossing characteristics


def analyze(wall_x, wall_r, g, n=50) -> NozzleFlow:
    """MOC analysis of a nozzle whose wall runs from the throat (0, 1) to the exit lip.

    Vacuum thrust from axial momentum conservation: the thrust across the throat
    plus the pressure acting on the diverging wall,
        F = integral over throat of (p + rho u^2) dA  +  integral over wall of p dA_x.
    Only net points on the wall are needed, so the march stops at the exit lip.
    """
    wall = Wall(wall_x, wall_r)
    x_e, r_e = wall.x_end, float(wall_r[-1])
    pts, _, walls, merges = _march_to_wall_point(g, wall, n, reached=lambda w: w[X] >= x_e,
                                                 x_exact=lambda prev, w: x_e, shrink=False)
    initial = initial_line(g, n)
    r0 = np.array([p[R] for p in initial])
    M0 = np.array([p[MA] for p in initial])
    th0 = np.array([p[TH] for p in initial])
    ring0 = np.array([_ring(r) for r in r0])
    thrust_throat = float(np.trapezoid(pressure_ratio(M0, g) * (1 + g * M0 ** 2 * np.cos(th0) ** 2) * ring0, r0))

    w = np.array(walls)
    area = np.array([_disk(r) for r in w[:, R]])
    thrust = thrust_throat + float(np.trapezoid(pressure_ratio(w[:, MA], g), area))

    mdot = line_mass_flow(initial, g)
    cd = mdot / (mass_flux(1.0, g) * _disk(1.0))
    # 1-D reference with the same mass flow and exit area. The effective throat
    # is Cd * At, so the effective area ratio is (exit area) / Cd.
    Me = mach_from_area(_disk(r_e) / _disk(1.0) / cd, g)
    thrust_1d = pressure_ratio(Me, g) * (1 + g * Me ** 2) * _disk(r_e)

    # Divergence loses jet momentum; the exit pressure-area term is nearly
    # unchanged. Attributing the whole thrust deficit to momentum gives the
    # efficiency that applies at optimum expansion (pe = pa), where thrust is
    # all momentum: 1 - deficit / (mdot V) with mdot V = gamma p M^2 A (1-D).
    momentum_1d = pressure_ratio(Me, g) * g * Me ** 2 * _disk(r_e)
    momentum_eff = 1 - (thrust_1d - thrust) / momentum_1d

    return NozzleFlow(
        gamma=g, points=pts, wall=w, x_exit=x_e, r_exit=r_e, mdot=mdot, thrust_vac=thrust,
        efficiency=thrust / thrust_1d, momentum_efficiency=momentum_eff, discharge_coeff=cd,
        merges=merges,
    )


# --- Inverse problem: ideal (uniform-exit) contour --------------------------

@dataclass
class IdealContour:
    gamma: float
    M_exit: float
    x: np.ndarray             # wall, throat -> exit (normalised by Rt)
    r: np.ndarray
    theta_attach_deg: float   # wall angle where the throat arc ends (point T)
    kernel: np.ndarray        # net points upstream of the C- from T
    turning: np.ndarray       # net points in the turning region
    mdot: float


def _kernel(g, arc, theta_T, n):
    """Kernel for a throat arc that ends at wall angle theta_T. Returns (points, TA)."""
    tan_t = math.tan(theta_T)
    pts, TA, _, _ = _march_to_wall_point(
        g, arc, n, reached=lambda w: w[TH] >= theta_T,
        x_exact=lambda prev, w: brentq(lambda x: arc.slope(x) - tan_t, prev[X], w[X]))
    return pts, TA


def ideal_contour(M_exit: float, g: float, R_d: float = 0.382, n: int = 50,
                  n_turn: int = 120) -> IdealContour:
    """Design the wall that gives uniform, axial flow at M_exit.

    1. Kernel: flow through a throat arc of radius R_d that ends at angle
       theta_T (point T). The C- from T reaches the axis at A. Bisection on
       theta_T puts M at A equal to M_exit.
    2. Downstream of the C+ from A (line EX) the flow is uniform. The region
       between TA and EX is a Goursat problem solved by MOC.
    3. The wall is the streamline through T: on each C- in the turning region,
       it sits where the enclosed mass flow equals the mass flow across TA.
    """
    arc = throat_arc(R_d)

    # Bracket theta_T, widening the upper bound gradually: very large arc
    # angles overexpand the flow and can form a shock.
    lo = math.radians(1.0)
    for hi_deg in range(10, 51, 5):
        hi = math.radians(hi_deg)
        if _kernel(g, arc, hi, n)[1][-1][MA] >= M_exit:
            break
        lo = hi
    else:
        raise ValueError(f"M_exit = {M_exit:.2f} needs a throat arc beyond 50 deg")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _kernel(g, arc, mid, n)[1][-1][MA] >= M_exit:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-7:
            break
    kernel, TA = _kernel(g, arc, hi, n)
    M_exit, nu_e, x_A = TA[-1][MA], TA[-1][NU], TA[-1][X]

    # The net thins out toward the axis as the flow expands, so TA's points are
    # unevenly spaced. Resample it evenly in arc length (linear in theta and nu).
    ta = np.array(TA)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ta[:, X]), np.diff(ta[:, R])))])
    s_new = np.linspace(0.0, s[-1], n_turn)
    TA = [_point(float(np.interp(si, s, ta[:, X])), float(np.interp(si, s, ta[:, R])),
                 float(np.interp(si, s, ta[:, TH])), float(np.interp(si, s, ta[:, NU])), g)
          for si in s_new]
    TA[-1] = tuple(ta[-1])      # keep A exactly on the axis
    x_T, r_T, theta_T = TA[0][X], TA[0][R], TA[0][TH]

    # Nozzle mass flow, measured across TA itself so the wall search below is
    # consistent with the numerical net: T must lie exactly on the dividing streamline.
    mdot, _ = _c_minus_fluxes(TA[::-1], g)

    # EX: straight C+ from A carrying uniform flow, out to the exit lip.
    mu_e = math.asin(1 / M_exit)
    flux_e = mass_flux(M_exit, g)
    r_exit = brentq(lambda rr: flux_e * _disk(rr) - mdot, 0.0, 100.0)
    ex_r = np.linspace(0.0, r_exit, int(1.5 * n_turn))
    EX = [(x_A + rr / math.tan(mu_e), float(rr), 0.0, nu_e, M_exit, mu_e) for rr in ex_r]

    # Goursat grid: grid[i][k] lies on the C+ from TA[i] and the C- from EX[k].
    m = len(TA) - 1
    grid = [[None] * len(EX) for _ in range(m + 1)]
    for i in range(m + 1):
        grid[i][0] = TA[i]
    for k in range(len(EX)):
        grid[m][k] = EX[k]
    for k in range(1, len(EX)):
        for i in range(m - 1, -1, -1):
            grid[i][k] = interior_point(grid[i][k - 1], grid[i + 1][k], g)

    # Wall: on each C- (fixed k), accumulate mass flow upward from EX[k].
    wall_pts = [(x_T, r_T)]
    for k in range(1, len(EX) - 1):
        m_acc = flux_e * _disk(EX[k][R])       # uniform flow crossing EX below EX[k]
        for i in range(m, 0, -1):
            a, b = grid[i][k], grid[i - 1][k]
            dm, _ = _segment_fluxes(a, b, g)
            if m_acc + dm >= mdot:
                frac = (mdot - m_acc) / dm
                wall_pts.append((a[X] + frac * (b[X] - a[X]), a[R] + frac * (b[R] - a[R])))
                break
            m_acc += dm
    wall_pts.append((EX[-1][X], EX[-1][R]))
    wall_pts = np.array(wall_pts)
    wall_pts = wall_pts[np.concatenate([[True], np.diff(wall_pts[:, 0]) > 0])]

    # Full wall: throat arc up to T, then the turning contour.
    t = np.linspace(0, theta_T, 80)
    x = np.concatenate([R_d * np.sin(t), wall_pts[1:, 0]])
    r = np.concatenate([1.0 + R_d * (1 - np.cos(t)), wall_pts[1:, 1]])

    turning = np.array([p for row in grid for p in row])
    return IdealContour(g, M_exit, x, r, math.degrees(theta_T), kernel, turning, mdot)


def truncate(contour: IdealContour, eps: float):
    """Cut an ideal contour where its exit area gives area ratio eps. Returns (x, r)."""
    r_cut = math.sqrt(eps) if AXISYMMETRIC else eps
    if r_cut >= contour.r[-1]:
        raise ValueError("ideal contour exit area is smaller than the requested area ratio")
    i = int(np.argmax(contour.r >= r_cut))
    x_cut = float(np.interp(r_cut, contour.r[i - 1:i + 1], contour.x[i - 1:i + 1]))
    return np.concatenate([contour.x[:i], [x_cut]]), np.concatenate([contour.r[:i], [r_cut]])


def cone_length(eps: float) -> float:
    """Length (in Rt) of a 15 deg conical nozzle with area ratio eps, the usual bell reference."""
    return (math.sqrt(eps) - 1) / math.tan(math.radians(15.0))


@dataclass
class TIC:
    x: np.ndarray             # wall from the throat to the exit, normalised by Rt
    r: np.ndarray
    M_design: float           # design Mach number of the parent ideal contour
    theta_attach_deg: float
    theta_exit_deg: float


def tic_contour(eps: float, g: float, length_frac: float = 0.8, R_d: float = 0.382) -> TIC:
    """Truncated ideal contour with area ratio eps and length length_frac x the 15 deg cone.

    A longer (higher design Mach) ideal contour cut at area ratio eps is
    shorter; the design Mach number is found by bisection so the cut contour
    has the requested length.
    """
    target = length_frac * cone_length(eps)
    M_lo = mach_from_area(eps, g)
    full = ideal_contour(M_lo, g, R_d)
    if full.x[-1] <= target:
        # Even the untruncated ideal nozzle is shorter than requested.
        return TIC(full.x, full.r, full.M_exit, full.theta_attach_deg, 0.0)

    def length(M):
        return truncate(ideal_contour(M, g, R_d), eps)[0][-1]

    M_hi = M_lo + 0.5
    while length(M_hi) > target:
        M_lo, M_hi = M_hi, M_hi + 0.5
    for _ in range(25):
        M = 0.5 * (M_lo + M_hi)
        if length(M) > target:
            M_lo = M
        else:
            M_hi = M
        if M_hi - M_lo < 1e-3:
            break
    c = ideal_contour(M_hi, g, R_d)
    x, r = truncate(c, eps)
    theta_e = math.degrees(math.atan((r[-1] - r[-2]) / (x[-1] - x[-2])))
    return TIC(x, r, c.M_exit, c.theta_attach_deg, theta_e)
