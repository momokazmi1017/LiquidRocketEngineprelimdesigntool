"""1-D regenerative cooling analysis.

Gas side: Bartz (1957) correlation with the sigma property-variation factor,
          adiabatic wall temperature with recovery factor r = Pr^(1/3).
Wall:     1-D conduction through the hot-wall thickness.
Coolant:  Dittus-Boelter in rectangular channels with rib-fin efficiency,
          Blasius friction factor for pressure drop.

The coolant (all of the fuel) enters at the nozzle exit and flows toward the
injector (counterflow). At each station the three thermal resistances are
solved in series, then the coolant energy and momentum balances are marched.

Simplifications, stated so the results are read in context: no film cooling,
no radiation, no soot/carbon deposit resistance, no axial wall conduction,
single-phase coolant with temperature-dependent viscosity only, and ideal-gas
isentropic Mach number with the chamber frozen gamma.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .nozzle import ChamberGeometry


@dataclass(frozen=True)
class Coolant:
    name: str
    rho: float      # kg/m^3
    cp: float       # J/kg-K
    k: float        # W/m-K
    mu_ref: float   # Pa-s at T_ref
    T_ref: float    # K
    B_visc: float   # Andrade constant: mu = mu_ref * exp(B (1/T - 1/T_ref))
    T_limit: float  # K, onset of boiling / decomposition concern at operating pressure

    def mu(self, T):
        return self.mu_ref * np.exp(self.B_visc * (1.0 / T - 1.0 / self.T_ref))


# Approximate liquid properties near room temperature.
ETHANOL = Coolant("Ethanol", 789.0, 2440.0, 0.167, 1.07e-3, 298.15, 1700.0, 470.0)
ETHANOL_75 = Coolant("Ethanol-75", 855.0, 3000.0, 0.25, 2.2e-3, 298.15, 2000.0, 480.0)
RP1 = Coolant("RP-1", 810.0, 2000.0, 0.13, 1.6e-3, 298.15, 1500.0, 560.0)


@dataclass(frozen=True)
class Wall:
    name: str
    k: float        # W/m-K
    T_limit: float  # K, allowable hot-wall temperature


CUCRZR = Wall("CuCrZr", 320.0, 800.0)
GRCOP42 = Wall("GRCop-42", 300.0, 900.0)
INCONEL718 = Wall("Inconel 718", 20.0, 1000.0)


@dataclass
class Channels:
    n: int              # number of channels
    height: float       # m
    rib_width: float    # m
    wall_thickness: float  # hot-wall thickness, m


@dataclass
class CoolingResult:
    x: np.ndarray
    r: np.ndarray
    mach: np.ndarray
    h_gas: np.ndarray
    h_cool: np.ndarray        # effective, referenced to gas-side area
    q: np.ndarray             # W/m^2
    T_aw: np.ndarray
    T_wg: np.ndarray          # hot-gas-side wall
    T_wc: np.ndarray          # coolant-side wall
    T_cool: np.ndarray
    p_cool: np.ndarray
    v_cool: np.ndarray
    channel_width: np.ndarray
    Q_total: float            # W
    T_cool_out: float         # K, after the last (injector-end) station
    p_cool_out: float         # Pa

    def summary(self) -> dict:
        i = int(np.argmax(self.T_wg))
        return {
            "max_heat_flux_MW_m2": float(self.q.max() / 1e6),
            "max_hot_wall_T_K": float(self.T_wg[i]),
            "max_hot_wall_x_mm": float(self.x[i] * 1e3),
            "coolant_T_in_K": float(self.T_cool[-1]),
            "coolant_T_out_K": float(self.T_cool_out),
            "coolant_dp_bar": float((self.p_cool[-1] - self.p_cool_out) / 1e5),
            "max_coolant_velocity_m_s": float(self.v_cool.max()),
            "min_channel_width_mm": float(self.channel_width.min() * 1e3),
            "total_heat_load_kW": float(self.Q_total / 1e3),
        }


def mach_from_area(area_ratio: float, gamma: float, supersonic: bool) -> float:
    if area_ratio <= 1.0 + 1e-12:
        return 1.0
    g = gamma

    def f(M):
        return (1 / M) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * M * M)) ** ((g + 1) / (2 * (g - 1))) - area_ratio

    return brentq(f, 1.0 + 1e-9, 50.0) if supersonic else brentq(f, 1e-6, 1.0 - 1e-9)


def bartz_prefactor(Dt, pc, cstar, cp, gamma, T0, molar_mass, R_curv):
    """Throat-referenced Bartz coefficient (W/m^2-K) before the (At/A)^0.9 and sigma factors."""
    # Bartz viscosity correlation, originally mu[lb/in-s] = 46.6e-10 M^0.5 T[R]^0.6
    mu = 46.6e-10 * molar_mass ** 0.5 * (T0 * 1.8) ** 0.6 * 17.858
    Pr = 4 * gamma / (9 * gamma - 5)
    return (0.026 / Dt ** 0.2) * (mu ** 0.2 * cp / Pr ** 0.6) * (pc / cstar) ** 0.8 * (Dt / R_curv) ** 0.1, Pr


def regen_analysis(geom: ChamberGeometry, *, pc, cstar, T0, gamma, cp_gas, molar_mass,
                   mdot_coolant, coolant: Coolant, wall: Wall, channels: Channels,
                   T_cool_in=298.15, p_cool_in=None, n_stations=400) -> CoolingResult:
    # Resample the contour uniformly in arc length.
    s_raw = np.concatenate([[0], np.cumsum(np.hypot(np.diff(geom.x), np.diff(geom.r)))])
    s = np.linspace(0, s_raw[-1], n_stations)
    x = np.interp(s, s_raw, geom.x)
    r = np.interp(s, s_raw, geom.r)
    ds = np.gradient(s)

    Dt = 2 * geom.Rt
    C_bartz, Pr = bartz_prefactor(Dt, pc, cstar, cp_gas, gamma, T0, molar_mass, geom.R_throat_curv)
    rec = Pr ** (1 / 3)
    g = gamma

    ch = channels
    n = len(x)
    out = {k: np.zeros(n) for k in
           ("mach", "h_gas", "h_cool", "q", "T_aw", "T_wg", "T_wc", "T_cool", "p_cool", "v_cool", "width")}
    if p_cool_in is None:
        p_cool_in = 1.5 * pc
    T_c, p_c = T_cool_in, p_cool_in
    Q_total = 0.0

    # March from nozzle exit (last index) to injector (index 0).
    for i in range(n - 1, -1, -1):
        area_ratio = (r[i] / geom.Rt) ** 2
        M = mach_from_area(area_ratio, g, supersonic=x[i] > 0)
        stag = 1 + (g - 1) / 2 * M * M
        T_aw = T0 * (1 + rec * (g - 1) / 2 * M * M) / stag

        # Coolant side
        width = (2 * np.pi * (r[i] + ch.wall_thickness) - ch.n * ch.rib_width) / ch.n
        if width <= 0:
            raise ValueError(f"channels do not fit at x = {x[i]*1e3:.1f} mm; reduce count or rib width")
        A_flow = ch.n * width * ch.height
        Dh = 2 * width * ch.height / (width + ch.height)
        v = mdot_coolant / (coolant.rho * A_flow)
        mu = coolant.mu(T_c)
        Re = coolant.rho * v * Dh / mu
        Pr_c = mu * coolant.cp / coolant.k
        h_c = 0.023 * Re ** 0.8 * Pr_c ** 0.4 * coolant.k / Dh
        m_fin = np.sqrt(2 * h_c / (wall.k * ch.rib_width))
        eta_fin = np.tanh(m_fin * ch.height) / (m_fin * ch.height)
        h_c_eff = h_c * ch.n * (width + 2 * eta_fin * ch.height) / (2 * np.pi * r[i])

        # Gas side (Bartz). Iterate on hot-wall temperature because sigma depends on it.
        T_wg = 0.5 * (T_aw + T_c)
        for _ in range(20):
            sigma = 1.0 / ((0.5 * T_wg / T0 * stag + 0.5) ** 0.68 * stag ** 0.12)
            h_g = C_bartz * (1.0 / area_ratio) ** 0.9 * sigma
            q = (T_aw - T_c) / (1 / h_g + ch.wall_thickness / wall.k + 1 / h_c_eff)
            T_new = T_aw - q / h_g
            if abs(T_new - T_wg) < 0.01:
                T_wg = T_new
                break
            T_wg = T_new
        T_wc = T_wg - q * ch.wall_thickness / wall.k

        for key, val in (("mach", M), ("h_gas", h_g), ("h_cool", h_c_eff), ("q", q), ("T_aw", T_aw),
                         ("T_wg", T_wg), ("T_wc", T_wc), ("T_cool", T_c), ("p_cool", p_c),
                         ("v_cool", v), ("width", width)):
            out[key][i] = val

        # March coolant to the next (upstream) station.
        dQ = q * 2 * np.pi * r[i] * ds[i]
        Q_total += dQ
        T_c += dQ / (mdot_coolant * coolant.cp)
        f = 0.316 * Re ** -0.25
        p_c -= f * ds[i] / Dh * 0.5 * coolant.rho * v * v

    return CoolingResult(
        x=x, r=r, mach=out["mach"], h_gas=out["h_gas"], h_cool=out["h_cool"], q=out["q"],
        T_aw=out["T_aw"], T_wg=out["T_wg"], T_wc=out["T_wc"], T_cool=out["T_cool"],
        p_cool=out["p_cool"], v_cool=out["v_cool"], channel_width=out["width"], Q_total=Q_total,
        T_cool_out=T_c, p_cool_out=p_c,
    )
