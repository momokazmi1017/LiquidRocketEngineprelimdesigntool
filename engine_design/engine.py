"""Top-level engine sizing: requirements in, full preliminary design out."""
from dataclasses import dataclass, field

import numpy as np

from . import cooling, injector, nozzle
from .combustion import G0, Performance, optimum_of, performance
from .propellants import Propellant


@dataclass
class EngineSpec:
    name: str
    oxidizer: Propellant
    fuel: Propellant
    thrust: float                 # N, at the design ambient pressure
    pc: float                     # Pa
    pa: float = 101_325.0         # design ambient pressure, Pa
    of: float | None = None       # None -> Isp-optimal
    of_bounds: tuple = (0.8, 3.0)
    pe: float | None = None       # exit pressure; defaults to pa (optimum expansion)
    eps: float | None = None      # or specify area ratio instead
    eta_cstar: float = 0.95       # combustion efficiency
    eta_cf: float = 0.98          # nozzle efficiency (divergence, boundary layer)
    L_star: float = 1.1           # characteristic length, m
    contraction_ratio: float = 6.0
    theta_c: float = 30.0         # converging half-angle, deg
    bell_fraction: float = 0.8
    # Cooling
    coolant: cooling.Coolant | None = None   # None -> no regen analysis
    wall: cooling.Wall = cooling.CUCRZR
    channels: cooling.Channels | None = None
    # Injector
    n_elements: int = 12
    injector_dp_frac: float = 0.20
    injector_cd: float = 0.7


@dataclass
class EngineDesign:
    spec: EngineSpec
    perf: Performance
    of: float
    isp_ideal: float
    isp: float
    cstar: float
    cf: float
    mdot: float
    mdot_ox: float
    mdot_fuel: float
    At: float
    geom: nozzle.ChamberGeometry
    inj: injector.InjectorDesign
    cool: cooling.CoolingResult | None = field(default=None)

    def summary(self) -> dict:
        s, g, p = self.spec, self.geom, self.perf
        d = {
            "name": s.name,
            "propellants": f"{s.oxidizer.name} / {s.fuel.name}",
            "thrust_N": s.thrust,
            "chamber_pressure_bar": s.pc / 1e5,
            "mixture_ratio": self.of,
            "chamber_temperature_K": p.Tc,
            "gamma_frozen": p.gamma_c,
            "cstar_ideal_m_s": p.cstar,
            "cstar_delivered_m_s": self.cstar,
            "thrust_coefficient": self.cf,
            "isp_ideal_s": self.isp_ideal,
            "isp_delivered_s": self.isp,
            "isp_vac_ideal_s": p.isp_vac,
            "mdot_total_kg_s": self.mdot,
            "mdot_ox_kg_s": self.mdot_ox,
            "mdot_fuel_kg_s": self.mdot_fuel,
            "area_ratio": p.eps,
            "exit_pressure_bar": p.pe / 1e5,
            "throat_diameter_mm": 2e3 * g.Rt,
            "chamber_diameter_mm": 2e3 * g.Rc,
            "exit_diameter_mm": 2e3 * g.Re,
            "chamber_length_mm": 1e3 * g.L_chamber,
            "nozzle_length_mm": 1e3 * g.L_nozzle,
            "overall_length_mm": 1e3 * (g.L_chamber + g.L_nozzle),
            "rao_theta_n_deg": g.theta_n,
            "rao_theta_e_deg": g.theta_e,
            "injector_ox_orifice_mm": self.inj.ox.d * 1e3,
            "injector_fuel_orifice_mm": self.inj.fuel.d * 1e3,
            "injector_ox_velocity_m_s": self.inj.ox.v,
            "injector_fuel_velocity_m_s": self.inj.fuel.v,
            "injector_resultant_angle_deg": self.inj.resultant_angle_deg,
        }
        if self.cool is not None:
            d.update({f"cooling_{k}": v for k, v in self.cool.summary().items()})
        return d


def design(spec: EngineSpec) -> EngineDesign:
    pe = spec.pe if (spec.pe is not None or spec.eps is not None) else spec.pa
    exp = {"eps": spec.eps} if spec.eps is not None else {"pe": pe}

    of = spec.of if spec.of is not None else optimum_of(
        spec.oxidizer, spec.fuel, spec.pc, spec.of_bounds, pa=spec.pa, **exp)
    perf = performance(spec.oxidizer, spec.fuel, of, spec.pc, **exp)

    # Delivered performance: Isp = eta_c* * eta_cf * Isp_ideal
    isp_ideal = perf.isp(spec.pa)
    isp = spec.eta_cstar * spec.eta_cf * isp_ideal
    cstar = spec.eta_cstar * perf.cstar
    cf = isp * G0 / cstar

    mdot = spec.thrust / (isp * G0)
    mdot_ox = mdot * of / (1 + of)
    mdot_f = mdot - mdot_ox
    At = mdot * cstar / spec.pc
    Rt = np.sqrt(At / np.pi)

    geom = nozzle.chamber_contour(Rt, perf.eps, spec.contraction_ratio, spec.L_star,
                                  spec.theta_c, spec.bell_fraction)

    inj = injector.unlike_doublet(mdot_ox, spec.oxidizer.density, mdot_f, spec.fuel.density,
                                  spec.pc, spec.n_elements, spec.injector_dp_frac, spec.injector_cd)

    cool = None
    if spec.coolant is not None:
        cool = cooling.regen_analysis(
            geom, pc=spec.pc, cstar=perf.cstar, T0=perf.Tc, gamma=perf.gamma_c,
            cp_gas=perf.cp_c, molar_mass=perf.molar_mass_c, mdot_coolant=mdot_f,
            coolant=spec.coolant, wall=spec.wall, channels=spec.channels,
        )

    return EngineDesign(spec, perf, of, isp_ideal, isp, cstar, cf, mdot, mdot_ox, mdot_f,
                        At, geom, inj, cool)
