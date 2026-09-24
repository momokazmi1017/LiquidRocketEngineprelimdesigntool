"""Chemical-equilibrium rocket performance (CEA-style) using Cantera.

Method
------
1. Chamber: adiabatic, constant-pressure equilibrium (HP) with the reactant
   enthalpy of the liquid propellants at their storage temperature.
2. Nozzle: isentropic expansion from the chamber state. For each static
   pressure p the state is found at constant entropy, either re-equilibrating
   the composition (shifting equilibrium) or holding it fixed (frozen).
   Velocity comes from the energy equation  v = sqrt(2 (h0 - h)).
3. Throat: the pressure that maximises mass flux G = rho * v (choked flow).
   This avoids needing an equilibrium sound speed.
4. Exit: the pressure where G_throat / G = area ratio, or a given exit pressure.
"""
from dataclasses import dataclass

import cantera as ct
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .propellants import Propellant

G0 = 9.80665  # m/s^2

_GAS = None


def _gas() -> ct.Solution:
    """Ideal-gas product mixture: all C/H/O/N species in the NASA database with <= 2 carbons."""
    global _GAS
    if _GAS is None:
        species = [
            s for s in ct.Species.list_from_file("nasa_gas.yaml")
            if set(s.composition) <= {"C", "H", "O", "N"} and s.composition.get("C", 0) <= 2
        ]
        _GAS = ct.Solution(thermo="ideal-gas", species=species)
    return _GAS


@dataclass
class Performance:
    oxidizer: str
    fuel: str
    of: float
    pc: float            # Pa
    # chamber
    Tc: float            # K
    molar_mass_c: float  # kg/kmol
    gamma_c: float       # frozen cp/cv at chamber
    cp_c: float          # frozen cp at chamber, J/kg-K
    # throat
    pt: float
    Tt: float
    cstar: float         # m/s
    # exit
    eps: float
    pe: float
    Te: float
    ve: float            # m/s
    isp_vac: float       # s
    shifting: bool

    def isp(self, pa: float) -> float:
        """Ideal specific impulse (s) at ambient pressure pa (Pa)."""
        # Ae/mdot = eps * At/mdot = eps * c*/pc
        return (self.ve + (self.pe - pa) * self.eps * self.cstar / self.pc) / G0

    def cf(self, pa: float) -> float:
        """Ideal thrust coefficient at ambient pressure pa."""
        return self.isp(pa) * G0 / self.cstar


class _Expansion:
    """Holds the chamber state and evaluates isentropic nozzle states."""

    def __init__(self, ox: Propellant, fuel: Propellant, of: float, pc: float, shifting: bool):
        gas = _gas()
        w_ox, w_f = of / (1 + of), 1 / (1 + of)

        atoms = {}
        for prop, w in ((ox, w_ox), (fuel, w_f)):
            for el, n in prop.elements_per_kg().items():
                atoms[el] = atoms.get(el, 0.0) + w * n
        h0 = w_ox * ox.h_per_kg() + w_f * fuel.h_per_kg()

        # Start from atoms, equilibrate at a guess temperature, then impose the
        # reactant enthalpy and find the adiabatic flame state. Very fuel-rich
        # mixtures burn cool and need a lower starting guess to converge.
        for T_guess in (3000.0, 1000.0):
            try:
                gas.TPX = T_guess, pc, {el: n for el, n in atoms.items() if n > 0}
                gas.equilibrate("TP")
                gas.HP = h0, pc
                gas.equilibrate("HP")
                break
            except ct.CanteraError:
                if T_guess == 1000.0:
                    raise

        self.gas = gas
        self.pc = pc
        self.h0 = h0
        self.s0 = gas.s
        self.Tc = gas.T
        self.Yc = gas.Y.copy()
        self.molar_mass_c = gas.mean_molecular_weight
        self.gamma_c = gas.cp / gas.cv
        self.cp_c = gas.cp
        self.shifting = shifting

    def state(self, p: float):
        """Return (T, rho, v) at static pressure p after isentropic expansion."""
        gas = self.gas
        gas.TPY = self.Tc, self.pc, self.Yc
        gas.SP = self.s0, p
        if self.shifting:
            gas.equilibrate("SP")
        v = np.sqrt(max(2.0 * (self.h0 - gas.h), 0.0))
        return gas.T, gas.density, v

    def mass_flux(self, p: float) -> float:
        _, rho, v = self.state(p)
        return rho * v


def performance(ox: Propellant, fuel: Propellant, of: float, pc: float,
                eps: float | None = None, pe: float | None = None,
                shifting: bool = True) -> Performance:
    """Ideal rocket performance. Give either area ratio `eps` or exit pressure `pe` (Pa)."""
    if (eps is None) == (pe is None):
        raise ValueError("specify exactly one of eps or pe")

    ex = _Expansion(ox, fuel, of, pc, shifting)

    res = minimize_scalar(lambda p: -ex.mass_flux(p), bounds=(0.3 * pc, 0.9 * pc),
                          method="bounded", options={"xatol": 1e-6 * pc})
    pt = res.x
    Tt, rho_t, v_t = ex.state(pt)
    Gt = rho_t * v_t
    cstar = pc / Gt

    if pe is None:
        # Step down from the throat until the area ratio is bracketed, so the
        # search never probes pressures far below the answer (where very
        # fuel-rich mixtures would expand to unphysical temperatures).
        def f(p):
            return Gt / ex.mass_flux(p) - eps

        hi = lo = 0.999 * pt
        while f(lo) < 0:
            hi, lo = lo, 0.3 * lo
            if lo < 1e-6 * pc:
                raise ValueError(f"area ratio {eps} not reached above 1e-6 * pc")
        pe = brentq(f, lo, hi, xtol=1e-9 * pc)
    Te, rho_e, ve = ex.state(pe)
    eps = Gt / (rho_e * ve)

    isp_vac = (ve + pe / (rho_e * ve)) / G0

    return Performance(
        oxidizer=ox.name, fuel=fuel.name, of=of, pc=pc,
        Tc=ex.Tc, molar_mass_c=ex.molar_mass_c, gamma_c=ex.gamma_c, cp_c=ex.cp_c,
        pt=pt, Tt=Tt, cstar=cstar,
        eps=eps, pe=pe, Te=Te, ve=ve, isp_vac=isp_vac, shifting=shifting,
    )


def chamber_temperature(ox: Propellant, fuel: Propellant, of: float, pc: float) -> float:
    """Adiabatic equilibrium flame temperature (K) at mixture ratio `of` and pressure pc."""
    return _Expansion(ox, fuel, of, pc, shifting=True).Tc


def of_sweep(ox, fuel, pc, of_values, pa=101_325.0, eps=None, pe=None, shifting=True):
    """Performance over a range of mixture ratios. Returns a list of Performance."""
    return [performance(ox, fuel, of, pc, eps=eps, pe=pe, shifting=shifting) for of in of_values]


def optimum_of(ox, fuel, pc, of_bounds, pa=101_325.0, eps=None, pe=None, shifting=True) -> float:
    """Mixture ratio that maximises ideal Isp at ambient pressure pa."""
    res = minimize_scalar(
        lambda of: -performance(ox, fuel, of, pc, eps=eps, pe=pe, shifting=shifting).isp(pa),
        bounds=of_bounds, method="bounded", options={"xatol": 1e-3},
    )
    return res.x
