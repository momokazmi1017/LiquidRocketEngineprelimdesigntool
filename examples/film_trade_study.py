"""Film-cooling trade study for the E5-75 engine.

Sweeps the fraction of fuel used as wall film against the uncertain turbulent
mixing coefficient K_t, and plots coolant outlet temperature, peak hot-wall
temperature and delivered Isp. The goal is a film fraction that passes at
every plausible K_t, not just the nominal one.

Run from the repository root:
    python examples/film_trade_study.py
"""
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import matplotlib.pyplot as plt
import numpy as np

from design_5kN_lox_ethanol import spec
from engine_design.engine import design
from engine_design.report import AXIS, CRITICAL, INK_2, SERIES, SURFACE

FILM = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10])
K_T = [("Kt = 0.0025 (slow mixing)", 0.0025), ("Kt = 0.01 (nominal)", 0.01), ("Kt = 0.04 (fast mixing)", 0.04)]
COOLANT_MARGIN = 30.0   # K below the coolant's boiling estimate


def run():
    results = {}
    for label, kt in K_T:
        rows = []
        for f in FILM:
            d = design(dataclasses.replace(spec, film_fraction=float(f), film_mixing=kt))
            c = d.cool.summary()
            rows.append((c["coolant_T_out_K"], c["max_hot_wall_T_K"], d.isp))
            print(f"{label:28s} film {f:4.0%}  T_out {rows[-1][0]:6.1f} K  "
                  f"T_wall {rows[-1][1]:6.1f} K  Isp {rows[-1][2]:6.1f} s")
        results[label] = np.array(rows)
    return results


def plot(results, path):
    chosen = spec.film_fraction
    limits = [spec.coolant.T_limit - COOLANT_MARGIN, spec.wall.T_limit, None]
    limit_names = [f"Coolant limit {spec.coolant.T_limit:.0f} K − {COOLANT_MARGIN:.0f} K margin",
                   f"{spec.wall.name} limit {spec.wall.T_limit:.0f} K", None]
    titles = ["Coolant outlet temperature", "Peak hot-wall temperature", "Delivered sea-level Isp"]
    ylabels = ["T (K)", "T (K)", "Isp (s)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for j, ax in enumerate(axes):
        for k, (label, _) in enumerate(K_T):
            ax.plot(FILM * 100, results[label][:, j], color=SERIES[k], marker="o", ms=5,
                    mec=SURFACE, mew=1.5, label=label)
        if limits[j] is not None:
            ax.axhline(limits[j], color=CRITICAL, lw=1.0, ls="--")
            ax.annotate(limit_names[j], (FILM[-1] * 100, limits[j]), xytext=(0, (5, -13)[j]),
                        textcoords="offset points", ha="right", fontsize=9, color=INK_2)
        ax.axvline(chosen * 100, color=AXIS, lw=1.0)
        ax.annotate(f"chosen {chosen:.0%}", (chosen * 100, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", fontsize=9, color=INK_2)
        ax.set_title(titles[j])
        ax.set_ylabel(ylabels[j])
        ax.set_xlabel("Film fuel (% of fuel flow)")
    axes[2].legend(loc="lower left")
    fig.suptitle(f"{spec.name}: film-cooling trade study across mixing-coefficient uncertainty",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    out = ROOT / "outputs" / spec.name
    out.mkdir(parents=True, exist_ok=True)
    plot(run(), out / "film_trade.png")
    print(f"saved {out / 'film_trade.png'}")
