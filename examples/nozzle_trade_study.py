"""Nozzle contour trade study for the E5-75 engine, using the method of characteristics.

Compares, at the engine's area ratio:
- a 15 deg conical nozzle,
- the Rao thrust-optimised parabola (chart angles) used by the engine,
- truncated ideal contours (TIC) of different lengths, designed by MOC,
- the full-length ideal contour (uniform, axial exit flow).
Every contour is analysed with the same MOC solver, so the comparison is like-for-like.

Run from the repository root:
    python examples/nozzle_trade_study.py
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from design_5kN_lox_ethanol import spec
from engine_design import moc, nozzle
from engine_design.combustion import performance
from engine_design.report import AXIS, INK_2, MUTED, SERIES, SURFACE

perf = performance(spec.oxidizer, spec.fuel, spec.of, spec.pc, pe=spec.pa)
EPS, G = perf.eps, perf.gamma_c
L_CONE = moc.cone_length(EPS)


def cone(alpha_deg=15.0, Rd=0.382):
    a = math.radians(alpha_deg)
    t = np.linspace(0, a, 60)
    x, r = Rd * np.sin(t), 1 + Rd * (1 - np.cos(t))
    x_e = x[-1] + (math.sqrt(EPS) - r[-1]) / math.tan(a)
    xl = np.linspace(x[-1], x_e, 200)[1:]
    return np.concatenate([x, xl]), np.concatenate([r, r[-1] + math.tan(a) * (xl - x[-1])])


def run():
    rows = {}
    x, r = cone()
    rows["15° cone"] = (x, r, moc.analyze(x, r, G))
    x, r, _, _ = nozzle.diverging_contour(EPS, 0.8, "rao")
    rows["Rao TOP 80%"] = (x, r, moc.analyze(x, r, G))
    x, r, _, _ = nozzle.diverging_contour(EPS, 0.9, "rao")
    rows["Rao TOP 90%"] = (x, r, moc.analyze(x, r, G))

    tic = []
    M0 = moc.mach_from_area(EPS, G)
    for M in np.linspace(M0, M0 + 1.1, 12):
        c = moc.ideal_contour(M, G)
        x, r = (c.x, c.r) if M == M0 else moc.truncate(c, EPS)
        tic.append((x, r, moc.analyze(x, r, G)))
    rows["Ideal (full length)"] = tic[0]
    return rows, tic


def report(rows, tic):
    print(f"Area ratio {EPS:.2f}, gamma {G:.3f}, 15 deg cone length {L_CONE:.2f} Rt\n")
    print(f"{'Contour':22s} {'Length':>8s} {'Length/cone':>12s} {'Exit angle':>11s} "
          f"{'Momentum eff.':>14s} {'Merges':>7s}")
    for name, (x, r, f) in rows.items():
        exit_angle = math.degrees(math.atan((r[-1] - r[-2]) / (x[-1] - x[-2])))
        print(f"{name:22s} {x[-1]:7.2f}R {x[-1]/L_CONE:11.0%} {exit_angle:10.1f}° "
              f"{f.momentum_efficiency:14.4f} {f.merges:7d}")
    print("\nTruncated ideal contours:")
    for x, r, f in tic:
        print(f"  length {x[-1]/L_CONE:5.0%}  momentum eff. {f.momentum_efficiency:.4f}")


def plot_contours(rows, path):
    fig, ax = plt.subplots(figsize=(10, 3.0), constrained_layout=True)
    # (name, label offset, label alignment)
    order = [("15° cone", (6, 0), "left"), ("Rao TOP 80%", (0, 9), "center"),
             ("Ideal (full length)", (0, 9), "center")]
    for k, (name, offset, ha) in enumerate(order):
        x, r, f = rows[name]
        ax.plot(x, r, color=SERIES[k], label=f"{name}  (η = {f.momentum_efficiency:.4f})")
        ax.annotate(name, (x[-1], r[-1]), xytext=offset, textcoords="offset points",
                    ha=ha, va="center", fontsize=9, color=INK_2)
    ax.set_aspect("equal")
    ax.set_xlim(0, rows["Ideal (full length)"][0][-1] * 1.08)
    ax.set_ylim(0.9, 2.5)
    ax.set_xlabel("Axial distance from throat (throat radii)")
    ax.set_ylabel("Radius (throat radii)")
    ax.set_title(f"Nozzle contours at area ratio {EPS:.2f}")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_efficiency(rows, tic, path):
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    L = np.array([t[0][-1] / L_CONE for t in tic]) * 100
    eff = np.array([t[2].momentum_efficiency for t in tic])
    ax.plot(L, eff, color=SERIES[0], marker="o", ms=5, mec=SURFACE, mew=1.5,
            label="Truncated ideal contour (MOC design)")
    ax.annotate("Truncated ideal contours", (L[-4], eff[-4]), xytext=(10, -4), textcoords="offset points",
                fontsize=9, color=INK_2)
    # (name, marker, series slot, label offset)
    points = [("Rao TOP 80%", "D", 1, (10, -4)), ("Rao TOP 90%", "D", 1, (-78, 6)),
              ("15° cone", "s", 2, (8, -14))]
    for name, marker, slot, offset in points:
        x, r, f = rows[name]
        ax.plot([x[-1] / L_CONE * 100], [f.momentum_efficiency], marker=marker, ms=8, color=SERIES[slot],
                mec=SURFACE, mew=1.5, linestyle="none",
                label={"Rao TOP 80%": "Rao TOP bell (chart angles)", "15° cone": "15° cone"}.get(name))
        ax.annotate(name, (x[-1] / L_CONE * 100, f.momentum_efficiency), xytext=offset,
                    textcoords="offset points", fontsize=9, color=INK_2)
    ax.axhline(1.0, color=AXIS, lw=0.8)
    ax.set_xlabel("Nozzle length (% of 15° cone)")
    ax.set_ylabel("Divergence (momentum) efficiency")
    ax.set_title("Shorter nozzles lose more thrust to flow divergence")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_net(rows, path):
    """Mach number field inside the engine's Rao bell, from the characteristic net."""
    x, r, f = rows["Rao TOP 80%"]
    p = f.points
    inside = (p[:, moc.X] <= x[-1]) & (p[:, moc.R] <= np.interp(p[:, moc.X], x, r) + 1e-9)
    p = p[inside]
    tri = mtri.Triangulation(p[:, moc.X], p[:, moc.R])
    cx = p[tri.triangles, moc.X].mean(axis=1)
    cr = p[tri.triangles, moc.R].mean(axis=1)
    tri.set_mask(cr > np.interp(cx, x, r))            # drop triangles bridging outside the wall

    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    levels = np.linspace(1.0, np.ceil(p[:, moc.MA].max() * 10) / 10, 13)
    cf = ax.tricontourf(tri, p[:, moc.MA], levels=levels, cmap="Blues")
    ax.plot(x, r, color=INK_2, lw=1.5)
    ax.plot([0, x[-1]], [0, 0], color=MUTED, lw=0.8, ls=(0, (6, 3, 1, 3)))
    cb = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.01, aspect=25)
    cb.set_label("Mach number", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, r[-1] * 1.05)
    ax.set_xlabel("Axial distance from throat (throat radii)")
    ax.set_ylabel("Radius (throat radii)")
    ax.set_title(f"Mach number in the E5-75 bell\n(method of characteristics, {len(f.points):,}-point net)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    out = ROOT / "outputs" / spec.name
    out.mkdir(parents=True, exist_ok=True)
    rows, tic = run()
    report(rows, tic)
    plot_contours(rows, out / "nozzle_contours.png")
    plot_efficiency(rows, tic, out / "nozzle_efficiency.png")
    plot_net(rows, out / "moc_net.png")
    print(f"\nsaved plots to {out}")
