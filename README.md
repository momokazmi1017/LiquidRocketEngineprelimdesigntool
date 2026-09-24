# Liquid Rocket Engine Preliminary Design Tool

A Python toolkit that takes top-level requirements (thrust, chamber pressure,
propellants) and produces a complete preliminary thrust-chamber design:
equilibrium combustion performance, chamber and bell-nozzle geometry,
a method-of-characteristics (MOC) supersonic nozzle solver, injector
orifice sizing, and a 1-D regenerative cooling analysis with fuel film
cooling.

The combustion and MOC solvers are validated against published reference data
and exact solutions (see [Validation](#validation)), and every physical
relation is covered by tests.

## Example: E5-75, a 5 kN LOX / 75 % ethanol engine

`python examples/design_5kN_lox_ethanol.py` → [`outputs/E5-75/`](outputs/E5-75)

| Requirement | Value |
|---|---|
| Thrust (sea level) | 5.0 kN |
| Chamber pressure | 25 bar |
| Propellants | LOX / 75 % ethanol + 25 % water |
| Mixture ratio | 1.30 (fuel-rich of the 1.38 Isp peak, for cooling margin) |
| Combustion efficiency | η<sub>c\*</sub> = 0.94 (assumed) |
| Nozzle | Rao 80 % bell; η<sub>Cf</sub> computed by MOC (see [Nozzle](#nozzle-contour-method-of-characteristics)) |
| Cooling | Regenerative (all fuel) + 5 % of fuel as wall film |

| Result | Value |
|---|---|
| Chamber temperature (overall / core) | 3081 K / 3110 K |
| Isp, ideal / delivered (sea level) | 245.2 s / 223.7 s |
| Isp, vacuum (ideal, no film) | 278.2 s |
| Nozzle efficiency η<sub>Cf</sub> | 0.971 = 0.981 MOC divergence × 0.99 boundary layer |
| Total mass flow | 2.28 kg/s (1.29 ox, 0.99 fuel) |
| Throat / chamber / exit diameter | 42.3 / 103.6 / 90.0 mm |
| Area ratio | 4.53 |
| Overall length | 287 mm |
| Peak heat flux | 15.7 MW/m² |
| Peak hot-wall temperature | 657 K (CuCrZr limit 800 K) |
| Coolant temperature rise | 298 → 391 K |
| Coolant pressure drop | 3.4 bar |

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

That left one problem: with regenerative cooling alone, the coolant leaves at ~478 K,
right at the estimated boiling onset of the ethanol/water blend at this pressure.

### Film cooling trade study

To fix the coolant temperature, part of the fuel is injected along the wall as a
film. It forms a cool, fuel-rich layer that gradually mixes with the hot core gas
([`film.py`](engine_design/film.py)).

The mixing rate is set by a turbulent mixing coefficient K<sub>t</sub> that is not
known before hot-fire testing. The design therefore has to pass across a
**16× range of K<sub>t</sub>** (0.0025–0.04), not just at a nominal value
(`python examples/film_trade_study.py`):

![Film cooling trade study](outputs/E5-75/film_trade.png)

| Film fuel | Coolant out, worst case | Hot wall, worst case | Isp penalty, worst case |
|---|---|---|---|
| 0 % | 478 K ✗ | 759 K | none |
| 3 % | 457 K ✗ | 752 K | 0.7 % |
| **5 %** | **447 K ✓** | **747 K ✓** | **1.2 %** |
| 8 % | 434 K ✓ | 737 K ✓ | 1.9 % |

**5 % film was selected**: the smallest fraction that keeps the coolant at least
30 K below boiling and the wall under its limit at every K<sub>t</sub> in the
range. At the nominal K<sub>t</sub> = 0.01, the hot wall drops from 759 K to
657 K and the regen heat load from 529 kW to 277 kW.

![Film cooling wall layer](outputs/E5-75/film.png)

**Follow-up:** the film is sized as 32 × 0.31 mm orifices, which is hard to drill
reliably. A continuous film slot, or fewer, larger orifices with a splash ring,
should be used instead.

### Nozzle contour: method of characteristics

The bell was first drawn with Rao's chart angles (read to about ±1°) and an
assumed nozzle efficiency. [`moc.py`](engine_design/moc.py) replaces both
assumptions with an axisymmetric method-of-characteristics solver that:

- **analyses** any nozzle wall, marching the supersonic flow from the throat
  and integrating wall pressure to get the real divergence loss;
- **designs** the ideal contour, the wall that turns the flow to perfectly
  uniform, axial exit flow, and truncates it to a practical length
  (a truncated ideal contour, TIC).

Every contour below was analysed with the same solver at the engine's area
ratio, so they can be compared directly (`python examples/nozzle_trade_study.py`):

![Mach number field in the bell](outputs/E5-75/moc_net.png)
![Nozzle efficiency vs length](outputs/E5-75/nozzle_efficiency.png)

| Contour | Length (% of 15° cone) | Exit angle | Sea-level divergence efficiency |
|---|---|---|---|
| Ideal contour, full length | 191 % | 0.0° | 1.000 |
| 15° cone | 101 % | 15.0° | 0.988 |
| Rao TOP bell, 90 % | 90 % | 11.0° | 0.989 |
| **Rao TOP bell, 80 % (engine)** | **80 %** | **13.5°** | **0.981** |
| MOC truncated ideal contour | 79 % | 14.7° | 0.980 |

Findings:
- **The chart-based Rao bell holds up.** At 80 % and 90 % length it performs
  within 0.1–0.2 % of an MOC-designed truncated ideal contour of the same
  length. That confirms the chart approximation for this engine.
- **Length trades directly against efficiency.** Shortening from 80 % to 60 %
  would save about 0.85 Rt (18 mm on this engine) but cost about 3 % of thrust.
- **The engine now uses a computed nozzle efficiency** (η<sub>Cf</sub> = 0.981 ×
  0.99 = 0.971) instead of the assumed 0.98. This lowered delivered Isp from
  225.9 s to 223.7 s: the earlier assumption was optimistic by about 1 %.
- The MOC analysis also finds weak internal compression (a weak shock) where
  the Rao bell's throat arc meets its parabola. Real Rao bells have this too.
  The truncated ideal contour does not.

The truncated ideal contour is available as an option (`bell="tic"`), and
`bell_fraction` can then be any length, not only the 80 % and 90 % that
Rao's charts cover.

![Nozzle contours](outputs/E5-75/nozzle_contours.png)

## Method

| Module | What it does |
|---|---|
| [`combustion.py`](engine_design/combustion.py) | Adiabatic HP equilibrium with Cantera + NASA thermo data, using liquid-propellant enthalpies. Isentropic expansion with shifting or frozen composition. The throat is found by maximising mass flux ρv, which avoids needing an equilibrium sound speed. The exit comes from area ratio or exit pressure. |
| [`nozzle.py`](engine_design/nozzle.py) | Chamber volume from L\*, a converging section (fillet, cone, 1.5 Rt arc), and the bell: either a Rao thrust-optimised parabola (0.382 Rt arc, then a quadratic Bézier between θn and θe) or an MOC truncated ideal contour. |
| [`moc.py`](engine_design/moc.py) | Axisymmetric method of characteristics with predictor-corrector unit processes (interior, axis, wall). **Analysis**: marches from the throat and integrates wall pressure for thrust. **Design**: a kernel through the throat arc, then a Goursat problem between the last kernel characteristic and the uniform-flow region. The wall is found as the streamline carrying the full mass flow. Crossing characteristics (weak shocks) are merged. |
| [`injector.py`](engine_design/injector.py) | Unlike-doublet orifice sizing from ṁ = C<sub>d</sub>A√(2ρΔp), plus the spray resultant angle from the momentum balance. |
| [`cooling.py`](engine_design/cooling.py) | Bartz gas-side coefficient with the σ correction and recovery-factor adiabatic wall temperature. 1-D wall conduction. Dittus-Boelter coolant side with rib fin efficiency. Counterflow march of coolant energy and pressure (Blasius friction). |
| [`film.py`](engine_design/film.py) | Fuel film cooling as a two-stream model. Core gas is entrained into the wall layer at d(ṁ<sub>e</sub>)/ds = 2K<sub>t</sub>ṁ<sub>core</sub>/r. The layer's temperature is the equilibrium flame temperature at its local O/F, and Isp is the mass average of the core and wall streams. Sizing is iterated because the film penalty changes the throat size. |
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

The method-of-characteristics solver is checked against exact solutions:

| Check | Reference | Result |
|---|---|---|
| Compatibility relations | Exact conical source flow | match to 10<sup>-4</sup> |
| Planar ideal nozzle, M = 2–3 | Wall angle θ = ν<sub>e</sub>/2; exit area = A/A\* | within 0.03° and 0.1 % |
| 10°, 15°, 20° conical nozzles | Source-flow thrust over the spherical exit cap | within 0.25 % |
| Axisymmetric ideal nozzles, ε = 4.5–25 | Designed contour, re-analysed by the direct march | efficiency 1.000–1.003, exit flow within 0.02° of axial |

The last check is the strongest one: the inverse design and the direct
analysis are separate algorithms, and they agree.

The tests (`pytest`) also check:
- the Mach–area relation against tabulated values
- that the contour volume reproduces L\*·A<sub>t</sub>
- that the thrust equation closes both ways (ṁ·Isp·g₀ and C<sub>f</sub>·P<sub>c</sub>·A<sub>t</sub>)
- the orifice flow round trip
- that the coolant energy balance matches the integrated wall heat load
- film cooling behaviour: the wall-layer O/F starts at zero and rises monotonically without exceeding the core O/F; film lowers wall temperature at an Isp cost; faster mixing weakens the film
- that the Rao bell and an MOC truncated ideal contour of the same length perform within 0.3 %

## Limitations

This is a preliminary-design tool. Known simplifications:
- The Rao angles θn and θe are read from charts to about ±1°. MOC analysis shows this costs under 0.2 % against a designed contour.
- The MOC solver assumes constant γ (the chamber's frozen value) and starts from uniform sonic flow across the throat plane, ignoring the curvature of the real sonic line. Weak shocks are merged, not fitted, so their entropy rise is neglected. The boundary-layer loss (0.99) is assumed.
- Bartz generally over-predicts throat heat flux for small engines, which makes these results conservative.
- The model leaves out radiation, soot or carbon deposits, and axial conduction in the wall.
- The film mixing coefficient K<sub>t</sub> is an assumed parameter, which is why the design is checked across a 16× range. The film is treated as immediately vaporised. Below O/F 0.1, the wall-layer temperature is interpolated linearly to the fuel's boiling point. The product species do not include condensed carbon (soot).
- The coolant is modelled as single-phase with constant properties except viscosity. Boiling and supercritical behaviour are not modelled.
- Combustion efficiency is a user input, not predicted.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python examples/design_5kN_lox_ethanol.py
python examples/film_trade_study.py
python examples/nozzle_trade_study.py
pytest
```
