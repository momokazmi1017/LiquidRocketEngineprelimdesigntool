"""Plots and text/JSON reports for an EngineDesign."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .combustion import of_sweep

# Chart palette (validated categorical order) and chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.labelcolor": INK_2,
})


def plot_of_sweep(design, path, of_range=None):
    s = design.spec
    exp = {"eps": s.eps} if s.eps is not None else {"pe": s.pe if s.pe is not None else s.pa}
    lo, hi = of_range or s.of_bounds
    ofs = np.linspace(lo, hi, 40)
    perfs = of_sweep(s.oxidizer, s.fuel, s.pc, ofs, **exp)

    panels = [
        ("Ideal specific impulse at design ambient", "Isp (s)", [p.isp(s.pa) for p in perfs], design.isp_ideal),
        ("Characteristic velocity", "c* (m/s)", [p.cstar for p in perfs], design.perf.cstar),
        ("Chamber temperature", "Tc (K)", [p.Tc for p in perfs], design.perf.Tc),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, (title, ylabel, y, y_design) in zip(axes, panels):
        ax.plot(ofs, y, color=SERIES[0])
        ax.plot([design.of], [y_design], "o", ms=8, color=SERIES[0], mec=SURFACE, mew=2, zorder=3)
        ax.annotate(f"design  O/F {design.of:.2f}", (design.of, y_design), xytext=(8, -14),
                    textcoords="offset points", fontsize=9, color=INK_2)
        ax.set_title(title)
        ax.set_xlabel("Mixture ratio O/F")
        ax.set_ylabel(ylabel)
    fig.suptitle(f"{s.oxidizer.name} / {s.fuel.name}   Pc = {s.pc/1e5:.0f} bar   shifting equilibrium",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_contour(design, path):
    g = design.geom
    x, r = g.x * 1e3, g.r * 1e3
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    ax.fill_between(x, r, -r, color="#cde2fb", lw=0)
    ax.plot(x, r, color=SERIES[0])
    ax.plot(x, -r, color=SERIES[0])
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (6, 3, 1, 3)))
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("Axial position from throat (mm)")
    ax.set_ylabel("Radius (mm)")
    ax.set_title(f"{design.spec.name}: thrust chamber inner contour")
    notes = [
        (x[0] + 0.5 * g.L_cyl * 1e3, g.Rc * 1e3, 8, f"Dc {2*g.Rc*1e3:.1f} mm"),
        (0, g.Rt * 1e3, -16, f"Dt {2*g.Rt*1e3:.1f} mm"),
        (x[-1], g.Re * 1e3, 8, f"De {2*g.Re*1e3:.1f} mm"),
    ]
    for xx, rr, dy, t in notes:
        ax.annotate(t, (xx, rr), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=9,
                    color=INK_2, bbox=dict(boxstyle="square,pad=0.15", fc="#cde2fb" if dy < 0 else SURFACE, lw=0))
    ax.text(0.99, 0.04,
            f"ε = {design.perf.eps:.2f}   L* = {design.spec.L_star:.2f} m   "
            f"Rao {design.spec.bell_fraction:.0%} bell, θn {g.theta_n:.1f}°, θe {g.theta_e:.1f}°",
            transform=ax.transAxes, ha="right", fontsize=9, color=INK_2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cooling(design, path):
    c, s = design.cool, design.spec
    x = c.x * 1e3
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(x, c.q / 1e6, color=SERIES[0])
    ax.set_title("Gas-side heat flux (Bartz)")
    ax.set_ylabel("q (MW/m²)")

    ax = axes[1]
    # Direct labels placed where each line has clear space: hot wall above the
    # converging section, coolant-side wall below the chamber, coolant above its inlet.
    i_pk = int(np.argmin(np.abs(x - 0.3 * x[0])))
    i_ch = len(x) // 4
    series = [("Hot-gas wall", c.T_wg, SERIES[1], i_pk, 8),
              ("Coolant-side wall", c.T_wc, SERIES[0], i_ch, -14),
              ("Coolant", c.T_cool, SERIES[2], len(x) - 1, 8)]
    for name, y, col, i, dy in series:
        ax.plot(x, y, color=col, label=name)
        ax.annotate(name, (x[i], y[i]), xytext=(0, dy), textcoords="offset points",
                    ha="right" if i == len(x) - 1 else "center", fontsize=9, color=INK_2)
    ax.axhline(s.wall.T_limit, color=CRITICAL, lw=1.0, ls="--")
    ax.annotate(f"{s.wall.name} limit {s.wall.T_limit:.0f} K", (x[0], s.wall.T_limit), xytext=(2, 4),
                textcoords="offset points", fontsize=9, color=INK_2)
    ax.set_title("Temperatures")
    ax.set_ylabel("T (K)")
    ax.set_ylim(top=s.wall.T_limit + 60)
    ax.legend(loc="center left", bbox_to_anchor=(0, 0.42), ncols=3)
    ax.set_xlim(x[0], x[-1])

    ax = axes[2]
    ax.plot(x, c.p_cool / 1e5, color=SERIES[0])
    ax.set_title("Coolant pressure (flows from nozzle exit toward injector)")
    ax.set_ylabel("p (bar)")
    ax.set_xlabel("Axial position from throat (mm)")

    for a in axes:
        a.axvline(0, color=AXIS, lw=0.8)
    fig.suptitle(f"{s.name}: regenerative cooling, {s.channels.n} channels, {s.coolant.name} coolant, "
                 f"{s.wall.name} liner", x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fmt(v):
    return f"{v:,.3f}" if isinstance(v, float) else str(v)


def write_reports(design, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = design.summary()
    (out / "design_summary.json").write_text(json.dumps(summary, indent=2))
    width = max(len(k) for k in summary)
    lines = [f"{k.ljust(width)}  {_fmt(v)}" for k, v in summary.items()]
    (out / "design_summary.txt").write_text("\n".join(lines) + "\n")

    np.savetxt(out / "contour.csv", np.column_stack([design.geom.x * 1e3, design.geom.r * 1e3]),
               delimiter=",", header="x_mm,r_mm", comments="", fmt="%.4f")

    plot_of_sweep(design, out / "of_sweep.png")
    plot_contour(design, out / "contour.png")
    if design.cool is not None:
        plot_cooling(design, out / "cooling.png")
    return summary
