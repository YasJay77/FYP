"""Overlays the pore-free and pore-resolved Pareto fronts in the stiffness/absorption plane.

Needs matplotlib (not in the sandbox), run locally.
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
FREE = HERE / "pareto_front_porefree_repro.csv"
RES  = HERE / "pareto_front_pore_resolved.csv"

# project lattice colour map
COLORS = {"SC": "#1f77b4", "FCC": "#ff7f0e", "FCC_face": "#2ca02c"}
LABEL  = {"SC": "SC", "FCC": "FCC nodal-pore", "FCC_face": "FCC face-pore"}
MARKER = {"SC": "o", "FCC": "s", "FCC_face": "^"}
ALPHA_USEFUL = 0.55


def load_fronts():
    """load the pore-free and pore-resolved fronts, sorted by absorption"""
    free = pd.read_csv(FREE).sort_values("band_mean_alpha", ascending=False)
    res  = pd.read_csv(RES).sort_values("band_mean_alpha", ascending=False)
    return free, res


def main():
    """plot the two fronts overlaid in the stiffness/absorption plane"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 12, "legend.fontsize": 8.5, "figure.dpi": 150,
    })

    free, res = load_fronts()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # shade the useful-absorber band
    ax.axhspan(ALPHA_USEFUL, 1.0, color="#2ca02c", alpha=0.06, zorder=0)
    ax.axhline(ALPHA_USEFUL, color="#2ca02c", lw=0.8, ls=":", alpha=0.6, zorder=1)
    ax.text(0.015, ALPHA_USEFUL + 0.012, r"useful absorber band  $\langle\alpha\rangle \gtrsim 0.55$",
            color="#1b5e20", fontsize=8, alpha=0.9)

    # pore-free reference, grey open circles + line
    fs = free.sort_values("C11_norm")
    ax.plot(fs["C11_norm"], fs["band_mean_alpha"], color="0.55", lw=1.0, ls="--", zorder=2)
    ax.scatter(free["C11_norm"], free["band_mean_alpha"], s=55, facecolors="none",
               edgecolors="0.45", linewidths=1.1, zorder=3, label="pore-free front (reported)")

    # pore-resolved, coloured by family
    for fam in ["SC", "FCC_face", "FCC"]:
        sub = res[res["lattice_type"] == fam]
        if len(sub):
            ax.scatter(sub["C11_norm"], sub["band_mean_alpha"], s=42, c=COLORS[fam],
                       marker=MARKER[fam], edgecolors="black", linewidths=0.4, zorder=5,
                       label=f"{LABEL[fam]} (pore-resolved)")

    ax.set_xlabel(r"Normalised axial stiffness  $C_{11}/C_{11,\max}$")
    ax.set_ylabel(r"Band-mean absorption  $\langle\alpha\rangle$  [500--2000\,Hz]")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 0.80)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.95)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig_pareto_front_pore_sensitivity.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_pareto_front_pore_sensitivity.png/.pdf to", HERE)
    print(f"pore-free: {len(free)} pts {free.lattice_type.value_counts().to_dict()}")
    print(f"pore-resolved: {len(res)} pts {res.lattice_type.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
