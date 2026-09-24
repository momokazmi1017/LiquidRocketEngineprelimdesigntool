"""Example: 5 kN pressure-fed LOX / 75 % ethanol engine, regeneratively cooled.

Run from the repository root:
    python examples/design_5kN_lox_ethanol.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine_design import cooling
from engine_design.engine import EngineSpec, design
from engine_design.propellants import ETHANOL_75, LOX
from engine_design.report import write_reports

spec = EngineSpec(
    name="E5-75",
    oxidizer=LOX,
    fuel=ETHANOL_75,
    thrust=5_000.0,
    pc=25e5,
    pa=101_325.0,          # sea-level, optimum expansion
    of=1.3,                # slightly fuel-rich of peak Isp for cooling margin
    eta_cstar=0.94,
    eta_cf=0.98,
    L_star=1.1,
    contraction_ratio=6.0,
    coolant=cooling.ETHANOL_75,
    wall=cooling.CUCRZR,
    # Chosen from a channel trade study (see README): keeps the hot wall under the
    # CuCrZr limit with ~22 m/s coolant velocity at the throat.
    channels=cooling.Channels(n=72, height=0.8e-3, rib_width=1.0e-3, wall_thickness=0.7e-3),
    # 5 % of the fuel as a wall film keeps the coolant below boiling across the
    # whole plausible range of the mixing coefficient (see film_trade_study.py).
    film_fraction=0.05,
    film_mixing=0.01,
    n_elements=16,
)

if __name__ == "__main__":
    d = design(spec)
    summary = write_reports(d, ROOT / "outputs" / spec.name)
    width = max(len(k) for k in summary)
    for k, v in summary.items():
        print(f"{k.ljust(width)}  {v:,.3f}" if isinstance(v, float) else f"{k.ljust(width)}  {v}")
