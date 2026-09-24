"""Fuel film cooling: a fuel-rich wall layer that entrains core gas.

Model (two-stream mixing):
- A fraction f of the fuel is injected along the wall at the injector face.
  The rest burns in the core at O/F_core = mdot_ox / ((1 - f) mdot_fuel).
- Core gas mixes into the wall layer at a rate proportional to the local core
  mass flux G = mdot_core / (pi r^2) acting over the wall perimeter:
      d(m_entrained)/ds = K_t * G * 2 pi r = 2 K_t mdot_core / r
  K_t is a turbulent mixing coefficient. It is not known a priori, so it is
  treated as a calibration parameter and its sensitivity is reported.
- The wall layer's stagnation temperature is the adiabatic equilibrium
  temperature at the layer's local mixture ratio. As O/F_wall -> 0 (pure film
  fuel near the injector) it approaches the fuel's boiling temperature, which is
  what a liquid film holds the wall at.
- Performance: the core and wall-layer streams expand through the same nozzle
  and are mass-averaged at the throat. The wall layer burns fuel-rich, which is
  the Isp cost of film cooling.
"""
import warnings
from dataclasses import dataclass

import numpy as np

from .combustion import chamber_temperature, performance
from .nozzle import ChamberGeometry

OF_MIN_EQUILIBRIUM = 0.1   # below this the equilibrium solver is not used


@dataclass
class FilmResult:
    film_fraction: float
    K_t: float
    x: np.ndarray               # m, along the contour
    of_wall: np.ndarray         # wall-layer mixture ratio
    T0_wall: np.ndarray         # K, wall-layer stagnation temperature
    of_core: float
    Tc_core: float
    entrained_at_throat: float  # fraction of core flow mixed into the wall layer by the throat
    of_wall_throat: float
    isp_ideal: float            # two-stream mass-averaged ideal Isp at pa, s
    cstar_ideal: float          # two-stream mass-averaged c*, m/s
    isp_core: float
    isp_wall: float

    def T0_at(self, x):
        return np.interp(x, self.x, self.T0_wall)


def wall_layer(geom: ChamberGeometry, ox, fuel, of: float, pc: float, eps: float, pa: float,
               mdot: float, film_fraction: float, K_t: float, T_film0: float) -> FilmResult:
    mdot_ox = mdot * of / (1 + of)
    mdot_f = mdot - mdot_ox
    m_film = film_fraction * mdot_f
    mdot_core = mdot - m_film
    of_core = mdot_ox / (mdot_f - m_film)
    ox_frac_core = of_core / (1 + of_core)

    # Entrained core mass along the wall (arc length), integrated with the trapezoid rule.
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(geom.x), np.diff(geom.r)))])
    rate = 2 * K_t * mdot_core / geom.r
    m_e = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(s))])
    m_e = np.minimum(m_e, mdot_core)

    of_wall = m_e * ox_frac_core / (m_film + m_e * (1 - ox_frac_core))

    # Wall-layer temperature vs O/F: equilibrium table above OF_MIN_EQUILIBRIUM,
    # linear blend down to the film boiling temperature at O/F = 0.
    of_tab = np.linspace(OF_MIN_EQUILIBRIUM, of_core, 25)
    T_tab = np.array([chamber_temperature(ox, fuel, o, pc) for o in of_tab])
    T0_wall = np.interp(of_wall, np.concatenate([[0.0], of_tab]), np.concatenate([[T_film0], T_tab]))

    # Two-stream performance, split at the throat.
    i_t = int(np.argmin(np.abs(geom.x)))
    m_wall_t = m_film + m_e[i_t]
    of_wall_t = max(of_wall[i_t], OF_MIN_EQUILIBRIUM)
    p_core = performance(ox, fuel, of_core, pc, eps=eps)
    with warnings.catch_warnings():
        # A very fuel-rich wall stream can expand slightly below the 300 K lower
        # bound of the NASA fits at the nozzle exit; the extrapolation is small
        # and the stream carries only a few percent of the flow.
        warnings.simplefilter("ignore", UserWarning)
        p_wall = performance(ox, fuel, of_wall_t, pc, eps=eps)
    w_wall = m_wall_t / mdot
    isp = (1 - w_wall) * p_core.isp(pa) + w_wall * p_wall.isp(pa)
    cstar = (1 - w_wall) * p_core.cstar + w_wall * p_wall.cstar

    return FilmResult(
        film_fraction=film_fraction, K_t=K_t, x=geom.x.copy(), of_wall=of_wall, T0_wall=T0_wall,
        of_core=of_core, Tc_core=p_core.Tc, entrained_at_throat=m_e[i_t] / mdot_core,
        of_wall_throat=of_wall[i_t], isp_ideal=isp, cstar_ideal=cstar,
        isp_core=p_core.isp(pa), isp_wall=p_wall.isp(pa),
    )
