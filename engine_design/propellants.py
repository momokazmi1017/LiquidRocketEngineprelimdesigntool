"""Propellant definitions.

Each propellant is described by its elemental formula, its enthalpy at the
storage condition (J/mol, including heat of formation and, for cryogens, the
sensible/latent enthalpy relative to 298 K gas), molar mass and liquid density.
Enthalpies follow the NASA CEA reactant library (thermo.inp) conventions.

Blends (e.g. 75 % ethanol / 25 % water) are built with `blend()`.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Propellant:
    name: str
    formula: dict           # element -> atoms per molecule, e.g. {"C": 2, "H": 6, "O": 1}
    h_mol: float            # J/mol at storage temperature
    molar_mass: float       # kg/mol
    density: float          # kg/m^3 (liquid, at storage temperature)
    T_storage: float        # K
    # Mass-weighted components for blends; a pure propellant lists itself.
    components: tuple = field(default=(), compare=False)

    def elements_per_kg(self) -> dict:
        """Moles of each element per kg of propellant."""
        if self.components:
            out = {}
            for prop, w in self.components:
                for el, n in prop.elements_per_kg().items():
                    out[el] = out.get(el, 0.0) + w * n
            return out
        return {el: n / self.molar_mass for el, n in self.formula.items()}

    def h_per_kg(self) -> float:
        """Enthalpy per kg of propellant (J/kg)."""
        if self.components:
            return sum(w * p.h_per_kg() for p, w in self.components)
        return self.h_mol / self.molar_mass


# --- Oxidizers --------------------------------------------------------------
LOX = Propellant("LOX", {"O": 2}, -12_979.0, 0.031999, 1141.0, 90.17)

# --- Fuels ------------------------------------------------------------------
LH2 = Propellant("LH2", {"H": 2}, -9_012.0, 0.002016, 70.8, 20.27)
LCH4 = Propellant("LCH4", {"C": 1, "H": 4}, -89_233.0, 0.016043, 422.6, 111.64)
ETHANOL = Propellant("Ethanol", {"C": 2, "H": 6, "O": 1}, -277_510.0, 0.046069, 789.0, 298.15)
WATER = Propellant("Water", {"H": 2, "O": 1}, -285_830.0, 0.018015, 997.0, 298.15)
# CEA's RP-1 surrogate: CH1.9423
RP1 = Propellant("RP-1", {"C": 1, "H": 1.9423}, -24_717.7, 0.013969, 810.0, 298.15)


def blend(name: str, parts: list[tuple[Propellant, float]]) -> Propellant:
    """Mix propellants by mass fraction, e.g. blend("E75", [(ETHANOL, .75), (WATER, .25)])."""
    total = sum(w for _, w in parts)
    parts = tuple((p, w / total) for p, w in parts)
    density = 1.0 / sum(w / p.density for p, w in parts)   # ideal (volume-additive) mixing
    return Propellant(name, {}, 0.0, 0.0, density, 298.15, components=parts)


ETHANOL_75 = blend("Ethanol-75", [(ETHANOL, 0.75), (WATER, 0.25)])
