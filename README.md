# Liquid Rocket Engine Preliminary Design Tool

A Python toolkit that takes top-level requirements (thrust, chamber pressure,
propellants) and produces a complete preliminary thrust-chamber design:
equilibrium combustion performance, chamber and bell-nozzle geometry,
injector orifice sizing, and a 1-D regenerative cooling analysis.

The combustion solver is validated against published reference data (see
[Validation](#validation)) and every physical relation is covered by tests.

## Example: E5-75, a 5 kN LOX / 75 % ethanol engine

`python examples/design_5kN_lox_ethanol.py` → [`outputs/E5-75/`](outputs/E5-75)

| Requirement | Value |
|---|---|
| Thrust (sea level) | 5.0 kN |
| Chamber pressure | 25 bar |
| Propellants | LOX / 75 % ethanol + 25 % water |
| Mixture ratio | 1.30 (fuel-rich of the 1.38 Isp peak, for cooling margin) |
| Efficiencies | η<sub>c\*</sub> = 0.94, η<sub>Cf</sub> = 0.98 |

| Result | Value |
|---|---|
| Chamber temperature | 3081 K |
| Isp, ideal / delivered (sea level) | 247.3 s / 227.8 s |
| Isp, vacuum (ideal) | 278.2 s |
| Total mass flow | 2.24 kg/s (1.27 ox, 0.97 fuel) |
| Throat / chamber / exit diameter | 42.1 / 103.2 / 89.6 mm |
| Area ratio | 4.53 |
| Overall length | 286 mm |
| Peak heat flux (throat) | 21.4 MW/m² |
| Peak hot-wall temperature | 758 K (CuCrZr limit 800 K) |
| Coolant temperature rise | 298 → 478 K |
| Coolant pressure drop | 3.1 bar |

![Mixture ratio sweep](outputs/E5-75/of_sweep.png)
![Chamber contour](outputs/E5-75/contour.png)
![Regenerative cooling](outputs/E5-75/cooling.png)

### Cooling channel trade study

The first channel layout (40 channels × 2.0 mm deep) gave only 6 m/s coolant
velocity and a 1124 K hot wall, far above the copper-alloy limit. Sweeping channel
count, depth, rib width and wall thickness:

| Channels | Depth (mm) | Rib (mm) | Wall (mm) | Coolant v (m/s) | Hot wall (K) | Δp (bar) |
|---|---|---|---|---|---|---|
| 40 | 2.0 | 1.2 | 1.0 | 6.3 | 1124 | 0.1 |
| 40 | 1.0 | 1.2 | 1.0 | 12.6 | 985 | 0.9 |
| 60 | 1.0 | 1.0 | 0.8 | 14.7 | 858 | 1.3 |
| 60 | 0.8 | 1.0 | 0.8 | 18.4 | 827 | 2.3 |
| **72** | **0.8** | **1.0** | **0.7** | **22.0** | **758** | **3.1** |
| 80 | 0.7 | 0.9 | 0.7 | 25.1 | 725 | 4.6 |

72 channels was selected: it clears the wall limit with 42 K of margin while keeping
the minimum channel width (0.9 mm) manufacturable and the pressure drop modest.

**Open issue:** the coolant leaves at ~478 K, at the estimated boiling onset of the
ethanol/water blend at this pressure. Options for the next revision: a small
fuel film-cooling fraction at the injector, raising coolant pressure, or a lower
mixture ratio.

## Method

| Module | What it does |
|---|---|
| [`combustion.py`](engine_design/combustion.py) | Adiabatic HP equilibrium with Cantera + NASA thermo data, using liquid-propellant enthalpies. Isentropic expansion with shifting or frozen composition. The throat is found by maximising mass flux ρv, which avoids needing an equilibrium sound speed. The exit comes from area ratio or exit pressure. |
| [`nozzle.py`](engine_design/nozzle.py) | Chamber volume from L\*, a converging section (fillet, cone, 1.5 Rt arc), and a Rao thrust-optimised parabolic bell (0.382 Rt arc, then a quadratic Bézier between θn and θe). |
| [`injector.py`](engine_design/injector.py) | Unlike-doublet orifice sizing from ṁ = C<sub>d</sub>A√(2ρΔp), plus the spray resultant angle from the momentum balance. |
| [`cooling.py`](engine_design/cooling.py) | Bartz gas-side coefficient with the σ correction and recovery-factor adiabatic wall temperature. 1-D wall conduction. Dittus-Boelter coolant side with rib fin efficiency. Counterflow march of coolant energy and pressure (Blasius friction). |
| [`engine.py`](engine_design/engine.py) | Ties everything together: optional Isp-optimal O/F, delivered Isp = η<sub>c\*</sub>η<sub>Cf</sub>·Isp<sub>ideal</sub>, ṁ = F / (Isp g₀), A<sub>t</sub> = ṁ c\* / P<sub>c</sub>. |

## Validation

Ideal performance against Sutton & Biblarz, *Rocket Propulsion Elements*, Table 5-5
(P<sub>c</sub> = 1000 psia, optimum expansion to 1 atm, shifting equilibrium):

| Propellants | O/F | T<sub>c</sub> (K) ref / this | c\* (m/s) ref / this | Isp (s) ref / this |
|---|---|---|---|---|
| LOX / LH2 | 4.02 | 2999 / 2956 | 2386 / 2417 | 389 / 389.1 |
| LOX / RP-1 | 2.56 | 3676 / 3666 | 1799 / 1796 | 300 / 299.6 |
| LOX / CH4 | 3.21 | 3526 / 3526 | 1835 / 1855 | 311 / 309.2 |

Isp agrees within 0.6 %. T<sub>c</sub> and c\* agree within 1.5 %.

The tests (`pytest`) also check:
- the Mach–area relation against tabulated values
- that the contour volume reproduces L\*·A<sub>t</sub>
- that the thrust equation closes both ways (ṁ·Isp·g₀ and C<sub>f</sub>·P<sub>c</sub>·A<sub>t</sub>)
- the orifice flow round trip
- that the coolant energy balance matches the integrated wall heat load

## Limitations

This is a preliminary-design tool. Known simplifications:
- The Rao angles θn and θe are read from charts to about ±1°. A method-of-characteristics contour is planned.
- Bartz generally over-predicts throat heat flux for small engines, which makes these results conservative.
- The model leaves out film cooling, radiation, soot or carbon deposits, and axial conduction in the wall.
- The coolant is modelled as single-phase with constant properties except viscosity. Boiling and supercritical behaviour are not modelled.
- Combustion efficiency and nozzle efficiency are user inputs, not predicted.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python examples/design_5kN_lox_ethanol.py
pytest
```
