import os
import sys
import csv
import numpy as np
import matplotlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

OUTDIR = "results/property_datasets"
os.makedirs(OUTDIR, exist_ok=True)


C0   = 343.0      # speed of sound in air
RHO0 = 1.2        # air density
MU   = 1.81e-5    # dynamic viscosity of air
Z0   = RHO0 * C0  # characteristic impedance of air

# Geometry

def sc_geometric_params(t_over_L, d_pore, L):
    """SC dims -> MLHR params (neck length, cavity depth, surface porosity).

    Sound in z: the z-plate is the perforated panel, gap between z-plates is the cavity.
    """
    t_over_L = float(np.clip(t_over_L, 1e-4, 0.499))
    t     = t_over_L * L
    D     = L - t
    # 4 pores, one per L/2 quadrant, area referenced to (L-t)^2
    sigma = 4.0 * np.pi * (d_pore / 2.0)**2 / (L - t)**2
    sigma = float(np.clip(sigma, 1e-6, 0.99))
    return t, D, sigma



def fcc_geometric_params(t_over_L, d_pore, L):
    """MLHR params for the FCC nodal-pore lattice (4 pores, porosity ref L^2).

    {111} plates bound octahedral cavities; cavity depth is the cube-root-equivalent
    D_eff = (L^3/6)^(1/3) ~= 0.55 L.
    """
    t_over_L = float(np.clip(t_over_L, 1e-4, 0.499))
    t_eff = t_over_L * L * np.sqrt(3)          # neck projected along z: 1/cos(54.74 deg) = sqrt(3)
    V_oct = L**3 / 6.0                          # octahedral cavity volume
    D     = V_oct ** (1.0 / 3.0)               # effective cavity depth = L / cbrt(6) ~= 0.55L
    # D and t_eff stay independent: inclined plates form a neck but the wave still
    # sees the full octahedral cavity, so don't subtract t_eff from D. fed to T_C and Z_P separately.
    sigma = 4.0 * np.pi * (d_pore / 2.0)**2 / L**2
    sigma = float(np.clip(sigma, 1e-6, 0.99))
    return t_eff, D, sigma


def fcc_facepore_geometric_params(t_over_L, d_pore, L):
    """MLHR params for the FCC face-pore lattice (Type II): 16 pores, porosity ref L^2.

    Each {111} triangle splits into 4 sub-triangles with a pore at each centroid;
    4 upward plates per cell -> 16 active pores.
    """
    t_over_L = float(np.clip(t_over_L, 1e-4, 0.499))
    t_plate  = t_over_L * L
    t_eff    = t_plate * np.sqrt(3)          # projected neck along z for 54.74 deg plates
    V_oct    = L**3 / 6.0                    # octahedral cavity volume
    D_eff    = V_oct ** (1.0 / 3.0)         # effective cavity depth = L / cbrt(6) ~= 0.55L
    sigma    = 16.0 * np.pi * (d_pore / 2.0)**2 / L**2
    sigma    = float(np.clip(sigma, 1e-6, 0.99))
    return t_eff, D_eff, sigma


def helmholtz_freq(t_over_L, d_pore, L):
    """Helmholtz resonance: fh = c0/(2pi) * sqrt(A / (V*t))"""
    t, D, sigma = sc_geometric_params(t_over_L, d_pore, L)
    A  = 4.0 * np.pi * (d_pore / 2.0)**2
    V  = D * (L - t)**2
    return (C0 / (2.0 * np.pi)) * np.sqrt(A / (V * t))


# Pore impedance Z_P

def pore_impedance(f, t, d_pore, sigma):
    """normalised pore impedance Z_P(f), with end corrections"""
    omega = 2.0 * np.pi * f

    # perforation constant, defined on diameter d (not radius)
    kd = d_pore * np.sqrt(RHO0 * omega / (4.0 * MU))   # dimensionless

    delta = 0.85    # mass reactance end correction for circular pores

    # real part (resistance). end correction sqrt(2)/16 * kd * d/t is dimensionless
    # (a '2*eps*Rs' form would carry units and be inconsistent with sqrt(1+kd^2/32))
    end_corr_r = np.sqrt(2) / 16.0 * kd * d_pore / t   # dimensionless
    ZP_real = (32.0 * MU * t) / (d_pore**2 * sigma) \
              * (np.sqrt(1.0 + kd**2 / 32.0) + end_corr_r)

    # imag part (mass reactance)
    ZP_imag = (omega * RHO0 * t / sigma) \
              * (1.0 + (9.0 + kd**2 / 2.0)**(-0.5) + delta * d_pore / t)

    ZP = ZP_real + 1j * ZP_imag
    return ZP


# Pore transfer matrix T_P

def pore_matrix(ZP):
    """pore (neck) transfer matrix T_P = [[1, Z_P], [0, 1]]"""
    Nf = len(ZP)
    TP = np.zeros((Nf, 2, 2), dtype=np.complex128)
    TP[:, 0, 0] = 1.0
    TP[:, 0, 1] = ZP
    TP[:, 1, 0] = 0.0
    TP[:, 1, 1] = 1.0
    return TP


# Cavity transfer matrix T_C

def cavity_matrix(f, D):
    """cavity propagation matrix: T_C = [[cos(k0D), jZ0 sin(k0D)], [j sin(k0D)/Z0, cos(k0D)]]"""
    omega = 2.0 * np.pi * f
    k0    = omega / C0
    k0D   = k0 * D

    Nf = len(f)
    TC = np.zeros((Nf, 2, 2), dtype=np.complex128)
    TC[:, 0, 0] = np.cos(k0D)
    TC[:, 0, 1] = 1j * Z0 * np.sin(k0D)
    TC[:, 1, 0] = 1j * np.sin(k0D) / Z0
    TC[:, 1, 1] = np.cos(k0D)
    return TC


# Unit cell matrix T_T = T_P * T_C

def unit_cell_matrix(f, t, D, d_pore, sigma):
    """combine pore and cavity into one unit-cell matrix T_T = T_P * T_C"""
    ZP = pore_impedance(f, t, d_pore, sigma)
    TP = pore_matrix(ZP)
    TC = cavity_matrix(f, D)
    TT = TP @ TC
    return TT


# Slab matrix: N identical unit cells

def matrix_power_batch(T, n):
    """t^n for a batch of (Nf,2,2) matrices via binary exponentiation — O(log n) products"""
    if n < 1:
        raise ValueError("n must be >= 1")
    result = np.zeros_like(T)
    result[:, 0, 0] = 1.0
    result[:, 1, 1] = 1.0
    base = T.copy()
    exp  = n
    while exp > 0:
        if exp & 1:
            result = result @ base
        base = base @ base
        exp >>= 1
    return result


# Heterogeneous slab: N cells with different L values

def heterogeneous_slab(f, t_over_L, d_pore, L_list):
    """Total matrix for a heterogeneous slab — each layer i has its own L_i (distinct fh_i).

    Cells differ so we multiply in sequence rather than taking a matrix power.
    """
    Nf = len(f)
    Tslab = np.zeros((Nf, 2, 2), dtype=np.complex128)
    Tslab[:, 0, 0] = 1.0
    Tslab[:, 1, 1] = 1.0
    for L_i in L_list:
        t_i, D_i, sigma_i = sc_geometric_params(t_over_L, d_pore, float(L_i))
        TT_i  = unit_cell_matrix(f, t_i, D_i, d_pore, sigma_i)
        Tslab = Tslab @ TT_i
    return Tslab


# Impedance and absorption

def input_impedance_rigid(Tslab):
    """surface impedance for a rigid backing: Z_in = T11 / T21"""
    return Tslab[:, 0, 0] / Tslab[:, 1, 0]


def alpha_from_Zin(Zin):
    """absorption from surface impedance: R = (Zin-Z0)/(Zin+Z0), alpha = 1 - |R|^2"""
    R = (Zin - Z0) / (Zin + Z0)
    return np.clip(1.0 - np.abs(R)**2, 0.0, 1.0)


# Helper utilities

def band_mean(alpha, f, flo=500.0, fhi=2000.0):
    """mean absorption over the EV tonal band (500-2000 Hz)"""
    mask = (f >= flo) & (f <= fhi)
    return float(np.mean(alpha[mask]))

def alpha_at_frequency(alpha, f, f_target):
    """pick out alpha at the frequency closest to f_target"""
    idx = np.argmin(np.abs(f - f_target))
    return float(alpha[idx])

def compute_objectives(f, t_over_L, d_pore, L, N, f_target=800.0):
    """returns the two optimiser objectives: band-mean alpha and alpha at f_target"""
    alpha = compute_alpha(f, t_over_L, d_pore, L, N)
    obj_A = band_mean(alpha, f)
    obj_B = alpha_at_frequency(alpha, f, f_target)
    return obj_A, obj_B

def shade_ev(ax):
    """shade the EV tonal band on an axis"""
    ax.axvspan(500, 2000, color="steelblue", alpha=0.07,
               label="EV tonal band (500–2000 Hz)")
    for v in [500, 2000]:
        ax.axvline(v, color="steelblue", lw=0.7, ls="--", alpha=0.5)

def fmt_ax(ax, title=""):
    """set the standard alpha-vs-frequency axis limits, labels and grid"""
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Absorption coefficient  α")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

def compute_alpha(f, t_over_L, d_pore, L, N):
    """SC pipeline: geometry -> Z_P -> T_P*T_C -> T^N -> Z_in -> alpha"""
    t, D, sigma = sc_geometric_params(t_over_L, d_pore, L)
    TT    = unit_cell_matrix(f, t, D, d_pore, sigma)
    Tslab = matrix_power_batch(TT, N)
    Zin   = input_impedance_rigid(Tslab)
    return alpha_from_Zin(Zin)


def compute_alpha_fcc(f, t_over_L, d_pore, L, N):
    """FCC nodal-pore full pipeline"""
    t_eff, D, sigma = fcc_geometric_params(t_over_L, d_pore, L)
    TT    = unit_cell_matrix(f, t_eff, D, d_pore, sigma)
    Tslab = matrix_power_batch(TT, N)
    Zin   = input_impedance_rigid(Tslab)
    return alpha_from_Zin(Zin)


def compute_alpha_fcc_face(f, t_over_L, d_pore, L, N):
    """FCC face-pore (Type II) pipeline"""
    t_eff, D_eff, sigma = fcc_facepore_geometric_params(t_over_L, d_pore, L)
    TT    = unit_cell_matrix(f, t_eff, D_eff, d_pore, sigma)
    Tslab = matrix_power_batch(TT, N)
    Zin   = input_impedance_rigid(Tslab)
    return alpha_from_Zin(Zin)


def compute_alpha_explicit(f, t_neck, D_cav, d_pore, sigma, N):
    """TMM straight from explicit geometry params (skips the derivation) — for validation"""
    TT    = unit_cell_matrix(f, t_neck, D_cav, d_pore, sigma)
    Tslab = matrix_power_batch(TT, N)
    Zin   = input_impedance_rigid(Tslab)
    return alpha_from_Zin(Zin)


def compute_alpha_heterogeneous(f, t_over_L, d_pore, L_list):
    """heterogeneous MLHR pipeline (SC geometry, variable L per layer)"""
    Tslab = heterogeneous_slab(f, t_over_L, d_pore, L_list)
    Zin   = input_impedance_rigid(Tslab)
    return alpha_from_Zin(Zin)


# Study 1: alpha(f) vs t/L at fixed d and N

def run_study1(f, tL_list, d_pore, N, L):
    """compute alpha(f) sweeping t/L at fixed d and N"""
    results = {}
    for tL in tL_list:
        a = compute_alpha(f, tL, d_pore, L, N)
        t, D, sigma = sc_geometric_params(tL, d_pore, L)
        results[tL] = dict(alpha=a, bm=band_mean(a, f), t=t, D=D, sigma=sigma)
    return results

def plot_study1(f, res, d_pore, N, L, outdir):
    """plot alpha(f) vs t/L at fixed d and N"""
    tls    = list(res.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(tls)))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, tL in enumerate(tls):
        d = res[tL]
        ax.plot(f, d["alpha"], color=colors[i],
                label=f"t/L={tL:.3f}  σ={d['sigma']:.3f}")
    shade_ev(ax)
    fmt_ax(ax, f"α(f) vs plate thickness  [d={d_pore*1e3:.1f} mm, N={N}]")
    ax.legend(ncol=2, fontsize=8)

    for i, tL in enumerate(tls):
        d = res[tL]
        ax2.plot(f, d["alpha"], color=colors[i],
                 label=f"t/L={tL:.3f}  ⟨α⟩={d['bm']:.3f}")
    shade_ev(ax2)
    ax2.set_xlim(300, 2600)
    fmt_ax(ax2, "Zoom: EV tonal band (500–2000 Hz)")
    ax2.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    p = os.path.join(outdir, "fig1_alpha_vs_thickness.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# Study 2: alpha(f) vs d at fixed t/L and N

def run_study2(f, tL_ref, d_list, N, L):
    """compute alpha(f) sweeping pore diameter d at fixed t/L and N"""
    results = {}
    for d in d_list:
        a = compute_alpha(f, tL_ref, d, L, N)
        _, _, sigma = sc_geometric_params(tL_ref, d, L)
        results[d] = dict(alpha=a, bm=band_mean(a, f), sigma=sigma)
    return results

def plot_study2(f, res, tL_ref, N, L, outdir):
    """plot alpha(f) vs pore diameter d at fixed t/L and N"""
    d_list = list(res.keys())
    colors = plt.cm.plasma(np.linspace(0.10, 0.88, len(d_list)))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, d in enumerate(d_list):
        r = res[d]
        ax.plot(f, r["alpha"], color=colors[i],
                label=f"d={d*1e3:.2f} mm  σ={r['sigma']:.3f}")
    shade_ev(ax)
    fmt_ax(ax, f"α(f) vs pore diameter  [t/L={tL_ref:.3f}, N={N}]")
    ax.legend(ncol=2, fontsize=8)

    da  = np.array(d_list) * 1e3
    bma = np.array([res[d]["bm"] for d in d_list])
    ax2.plot(da, bma, "o-", color="darkorange", lw=2, ms=7)
    ax2.set_xlabel("Pore diameter  d  (mm)")
    ax2.set_ylabel("Band-mean α  [500–2000 Hz]")
    ax2.set_title(f"Effect of pore diameter  [t/L={tL_ref:.3f}, N={N}]")
    ax2.grid(True, alpha=0.3)
    for x, y in zip(da, bma):
        ax2.annotate(f"{y:.2f}", (x, y), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    p = os.path.join(outdir, "fig2_alpha_vs_pore_diameter.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# Study 3: alpha(f) vs N at fixed t/L and d

def run_study3(f, tL_ref, d_ref, N_list, L):
    """compute alpha(f) sweeping number of layers N at fixed t/L and d"""
    results = {}
    for N in N_list:
        a = compute_alpha(f, tL_ref, d_ref, L, N)
        results[N] = dict(alpha=a, bm=band_mean(a, f), slab_mm=N*L*1e3)
    return results

def plot_study3(f, res, tL_ref, d_ref, L, outdir):
    """plot alpha(f) vs number of layers N at fixed t/L and d"""
    N_list = list(res.keys())
    colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(N_list)))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, N in enumerate(N_list):
        r = res[N]
        ax.plot(f, r["alpha"], color=colors[i],
                label=f"N={N}  ({r['slab_mm']:.0f} mm)")
    shade_ev(ax)
    fmt_ax(ax, f"α(f) vs number of layers  [t/L={tL_ref:.3f}, d={d_ref*1e3:.1f} mm]")
    ax.legend(ncol=2, fontsize=8)

    slab_mm = np.array([res[N]["slab_mm"] for N in N_list])
    bma     = np.array([res[N]["bm"]      for N in N_list])
    ax2.plot(slab_mm, bma, "D-", color="crimson", lw=2, ms=7)
    ax2.set_xlabel("Total slab thickness  N·L  (mm)")
    ax2.set_ylabel("Band-mean α  [500–2000 Hz]")
    ax2.set_title("Diminishing returns with slab thickness")
    ax2.grid(True, alpha=0.3)
    for x, y in zip(slab_mm, bma):
        ax2.annotate(f"{y:.2f}", (x, y), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    p = os.path.join(outdir, "fig3_alpha_vs_N_layers.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# Study 4: 2D design map  t/L x d  at fixed N

def plot_study4(f, tL_arr, d_arr, N, L, outdir):
    """plot the 2D band-mean alpha design map over t/L x d at fixed N"""
    alpha_map = np.zeros((len(d_arr), len(tL_arr)))
    for j, tL in enumerate(tL_arr):
        for i, d in enumerate(d_arr):
            a = compute_alpha(f, tL, d, L, N)
            alpha_map[i, j] = band_mean(a, f)

    d_mm = d_arr * 1e3

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.contourf(tL_arr, d_mm, alpha_map, levels=25,
                     cmap="RdYlGn", vmin=0, vmax=1)
    cs = ax.contour(tL_arr, d_mm, alpha_map,
                    levels=[0.20, 0.35, 0.50, 0.65, 0.80],
                    colors="k", linewidths=0.9)
    ax.clabel(cs, fmt="α=%.2f", fontsize=8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Band-mean absorption  ⟨α⟩  [500–2000 Hz]")
    ax.set_xlabel("Plate thickness ratio  t/L")
    ax.set_ylabel("Pore diameter  d  (mm)")
    ax.set_title(f"Absorption design map  (SC plate lattice, MLHR, N={N})")
    fig.tight_layout()
    p = os.path.join(outdir, "fig4_design_map_tL_d.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# Study 5: alpha(f) vs L

def run_study5(f, tL_ref, d_ref, N, L_list):
    """sweep L at fixed t/L,d,N — shows fh moving in/out of the EV band (larger L -> lower fh)"""
    results = {}
    for L in L_list:
        a  = compute_alpha(f, tL_ref, d_ref, L, N)
        fh = helmholtz_freq(tL_ref, d_ref, L)
        results[L] = dict(alpha=a, bm=band_mean(a, f), fh=fh)
    return results

def plot_study5(f, res, tL_ref, d_ref, N, outdir):
    """plot alpha(f) vs unit cell size L and show fh tuning into the EV band"""
    L_list = list(res.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_list)))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for i, L in enumerate(L_list):
        r = res[L]
        ax.plot(f, r["alpha"], color=colors[i],
                label=f"L={L*1e3:.0f}mm  fh={r['fh']:.0f}Hz  ⟨α⟩={r['bm']:.2f}")
    shade_ev(ax)
    fmt_ax(ax, f"α(f) vs unit cell size L  [t/L={tL_ref:.3f}, d={d_ref*1e3:.1f}mm, N={N}]")
    ax.legend(fontsize=8)

    L_mm   = np.array(L_list) * 1e3
    fh_arr = np.array([res[L]["fh"] for L in L_list])
    ax2.plot(L_mm, fh_arr, "o-", color="steelblue", lw=2, ms=7)
    ax2.axhspan(500, 2000, color="steelblue", alpha=0.10, label="EV tonal band 500–2000 Hz")
    ax2.axhline(500,  color="steelblue", lw=0.8, ls="--")
    ax2.axhline(2000, color="steelblue", lw=0.8, ls="--")
    ax2.set_xlabel("Unit cell size  L  (mm)")
    ax2.set_ylabel("Helmholtz resonant frequency  fh  (Hz)")
    ax2.set_title("Use L to tune fh into EV band")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)
    for x, y in zip(L_mm, fh_arr):
        ax2.annotate(f"{y:.0f} Hz", (x, y), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    p = os.path.join(outdir, "fig5_alpha_vs_L.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# Study 6: Heterogeneous MLHR -- spread L per layer for flat broadband alpha

def run_study6(f, tL_ref, d_ref, N):
    """Heterogeneous slab (L spread 15-25 mm) vs homogeneous (L=20 mm).

    Spreading L scatters each fh_i so the response fills the band instead of one sharp peak.
    """
    L_seq  = np.linspace(0.015, 0.025, N)   # 15 mm to 25 mm across N layers
    L_homo = 0.020                           # reference homogeneous cell

    alpha_hetero = compute_alpha_heterogeneous(f, tL_ref, d_ref, L_seq)
    alpha_homo   = compute_alpha(f, tL_ref, d_ref, L_homo, N)

    return dict(
        alpha_hetero = alpha_hetero,
        alpha_homo   = alpha_homo,
        L_seq        = L_seq,
        L_homo       = L_homo,
        bm_hetero    = band_mean(alpha_hetero, f),
        bm_homo      = band_mean(alpha_homo, f),
        fh_seq       = [helmholtz_freq(tL_ref, d_ref, float(Li)) for Li in L_seq],
    )

def plot_study6(f, res, tL_ref, d_ref, outdir):
    """plot heterogeneous vs homogeneous MLHR and the spread-L sequence with its fh"""
    N   = len(res["L_seq"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: alpha(f) comparison
    ax.plot(f, res["alpha_homo"], color="steelblue", lw=2, ls="--",
            label=(f"Homogeneous  L={res['L_homo']*1e3:.0f} mm"
                   f"  ⟨α⟩={res['bm_homo']:.3f}"))
    ax.plot(f, res["alpha_hetero"], color="darkorange", lw=2,
            label=(f"Heterogeneous  L∈[{res['L_seq'][0]*1e3:.0f},"
                   f"{res['L_seq'][-1]*1e3:.0f}] mm"
                   f"  ⟨α⟩={res['bm_hetero']:.3f}"))
    shade_ev(ax)
    fmt_ax(ax, f"Heterogeneous vs Homogeneous  [t/L={tL_ref:.3f}, d={d_ref*1e3:.1f} mm, N={N}]")
    ax.legend(fontsize=9)

    # Right: L sequence with resonant frequencies on secondary y-axis
    idx = np.arange(1, N + 1)
    ax2r = ax2.twinx()
    ax2.bar(idx, res["L_seq"] * 1e3, color="steelblue", alpha=0.55,
            label="L_i  (mm)")
    ax2r.plot(idx, res["fh_seq"], "ro-", ms=6, lw=1.5, label="fh_i  (Hz)")
    ax2r.axhspan(500, 2000, color="steelblue", alpha=0.08)
    ax2r.axhline(500,  color="steelblue", lw=0.7, ls="--")
    ax2r.axhline(2000, color="steelblue", lw=0.7, ls="--")
    ax2.set_xlabel("Layer index  i")
    ax2.set_ylabel("Unit cell size  L_i  (mm)")
    ax2r.set_ylabel("Resonant frequency  fh_i  (Hz)")
    ax2.set_title("Heterogeneous L sequence — fh spread across EV band")
    lines1, lbl1 = ax2.get_legend_handles_labels()
    lines2, lbl2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=9)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    p = os.path.join(outdir, "fig6_heterogeneous_mlhr.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# Study 7: SC vs FCC nodal-pore acoustic comparison

def run_study7(f, tL_ref, d_ref, L, N):
    """SC vs FCC absorption at matched t/L,d,L,N — only the geometry model differs"""
    alpha_sc  = compute_alpha(f, tL_ref, d_ref, L, N)
    alpha_fcc = compute_alpha_fcc(f, tL_ref, d_ref, L, N)

    return {
        "SC":  dict(alpha=alpha_sc,  bm=band_mean(alpha_sc,  f)),
        "FCC": dict(alpha=alpha_fcc, bm=band_mean(alpha_fcc, f)),
    }

def plot_study7(f, res, tL_ref, d_ref, L, N, outdir):
    """plot SC vs FCC nodal-pore alpha(f) and band-mean comparison"""
    colors = {"SC": "steelblue", "FCC": "forestgreen"}
    styles = {"SC": "-", "FCC": "-."}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: alpha(f)
    for lat in ["SC", "FCC"]:
        r = res[lat]
        ax.plot(f, r["alpha"], color=colors[lat], ls=styles[lat], lw=2,
                label=f"{lat}  ⟨α⟩={r['bm']:.3f}")
    shade_ev(ax)
    fmt_ax(ax, f"SC vs FCC  [t/L={tL_ref:.3f}, d={d_ref*1e3:.1f} mm, L={L*1e3:.0f} mm, N={N}]")
    ax.legend(fontsize=9)

    # Right: band-mean alpha bar chart with geometric annotation
    lattice_names = ["SC", "FCC"]
    bm_vals = [res[lat]["bm"] for lat in lattice_names]
    bar_colors = [colors[lat] for lat in lattice_names]
    bars = ax2.bar(lattice_names, bm_vals, color=bar_colors, alpha=0.75, width=0.4)
    for bar, val in zip(bars, bm_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Band-mean absorption  ⟨α⟩  [500–2000 Hz]")
    ax2.set_title("Lattice type comparison")
    ax2.grid(True, alpha=0.25, axis="y")

    # Geometric summary in text box
    _, D_sc, sig_sc = sc_geometric_params(tL_ref, d_ref, L)
    te_fcc, D_fcc, sig_fcc = fcc_geometric_params(tL_ref, d_ref, L)
    info = (
        f"SC:   t={tL_ref*L*1e3:.2f}mm  D={D_sc*1e3:.2f}mm  σ={sig_sc:.4f}\n"
        f"FCC:  t_eff={te_fcc*1e3:.2f}mm  D={D_fcc*1e3:.2f}mm  σ={sig_fcc:.4f}"
    )
    ax2.text(0.03, 0.03, info, transform=ax2.transAxes, fontsize=8,
             va="bottom", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    fig.tight_layout()
    p = os.path.join(outdir, "fig7_sc_fcc_comparison.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# Study 8: Property-space dataset

def build_property_dataset(f, outdir):
    """sweep the full SC+FCC geometry grid and write the CSV for the optimiser and inversion net"""
    tL_grid  = np.linspace(0.03, 0.18, 15)
    d_grid   = np.linspace(0.0005, 0.003, 15)
    L_grid   = np.linspace(0.010, 0.040, 10)
    N_grid   = [5, 10, 15, 20]
    lattices = ["SC", "FCC"]
    alpha_fns = {
        "SC":  compute_alpha,
        "FCC": compute_alpha_fcc,
    }

    total     = len(tL_grid) * len(d_grid) * len(L_grid) * len(N_grid) * len(lattices)
    milestone = max(1, total // 10)
    print(f"  Total combinations: {total:,}")

    rows  = []
    count = 0
    for lattice in lattices:
        fn = alpha_fns[lattice]
        for N in N_grid:
            for L in L_grid:
                for d in d_grid:
                    for tL in tL_grid:
                        alpha = fn(f, tL, d, L, N)
                        bm    = band_mean(alpha, f)
                        a1k   = alpha_at_frequency(alpha, f, 1000.0)
                        rows.append({
                            "t_over_L":        round(float(tL), 6),
                            "d_mm":            round(float(d * 1e3), 4),
                            "L_mm":            round(float(L * 1e3), 4),
                            "N":               int(N),
                            "lattice_type":    lattice,
                            "band_mean_alpha": round(float(bm), 6),
                            "alpha_1000hz":    round(float(a1k), 6),
                        })
                        count += 1
                        if count % milestone == 0:
                            print(f"    {count:>6}/{total}  ({100*count/total:.0f}%)")

    csv_path = os.path.join(outdir, "property_dataset.csv")
    fieldnames = ["t_over_L", "d_mm", "L_mm", "N", "lattice_type",
                  "band_mean_alpha", "alpha_1000hz"]
    with open(csv_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV: {csv_path}")
    return rows

def plot_study8_histogram(rows, outdir):
    """property-space coverage: band-mean alpha and alpha(1000 Hz) distributions"""
    bm_all  = np.array([r["band_mean_alpha"] for r in rows])
    a1k_all = np.array([r["alpha_1000hz"]    for r in rows])

    colors = {"SC": "steelblue", "FCC": "forestgreen"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Overall band-mean distribution
    ax = axes[0]
    ax.hist(bm_all, bins=60, color="steelblue", edgecolor="white", lw=0.4)
    ax.set_xlabel("Band-mean absorption  ⟨α⟩  [500–2000 Hz]")
    ax.set_ylabel("Count")
    ax.set_title(f"Overall distribution  ({len(rows):,} points)")
    ax.grid(True, alpha=0.25)

    # Per-lattice band-mean
    ax2 = axes[1]
    for lat, col in colors.items():
        vals = [r["band_mean_alpha"] for r in rows if r["lattice_type"] == lat]
        ax2.hist(vals, bins=50, color=col, alpha=0.65,
                 label=f"{lat}  (n={len(vals):,})", edgecolor="white", lw=0.3)
    ax2.set_xlabel("Band-mean absorption  ⟨α⟩")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution by lattice type")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # alpha(1000 Hz) distribution
    ax3 = axes[2]
    ax3.hist(a1k_all, bins=60, color="darkorange", edgecolor="white", lw=0.4)
    ax3.set_xlabel("Absorption at 1000 Hz  α(1000)")
    ax3.set_ylabel("Count")
    ax3.set_title("Target-frequency coverage")
    ax3.grid(True, alpha=0.25)

    fig.suptitle(
        f"Acoustic property-space coverage  "
        f"(⟨α⟩ range: {bm_all.min():.3f}–{bm_all.max():.3f},  "
        f"mean={bm_all.mean():.3f})",
        fontsize=11
    )
    fig.tight_layout()
    p = os.path.join(outdir, "fig8_property_space_histogram.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# Study 9: Validation against published FCC-plate data
#
# Reference geometry: L=6mm, d=0.68mm, t=1.56mm, sigma=5.14%, D_eff=3mm.
# Two paths side by side: (A) the exact published params straight in (tests TMM physics),
# (B) fcc_geometric_params from the dimensions (tests the geometry derivation).

VALID_L     = 0.006      # 6 mm unit cell
VALID_d     = 0.00068    # 0.68 mm pore diameter (viewed from (001) face)
VALID_t     = 0.00156    # 1.56 mm neck length (effective, along wave path)
VALID_sigma = 0.0514     # 5.14 % porosity
VALID_D     = 0.003      # 3.0 mm effective cavity depth
VALID_tL    = VALID_t / (VALID_L * np.sqrt(3))   # back out t/L from t_eff, sqrt(3) factor


def run_study9(f):
    """alpha(f) for N=1,2,3 via two paths (exact params vs derived geometry); prints a compare table"""
    N_list = [1, 2, 3]

    # path A: exact params
    res_paper = {}
    for N in N_list:
        a = compute_alpha_explicit(f, VALID_t, VALID_D, VALID_d, VALID_sigma, N)
        res_paper[N] = dict(alpha=a, bm=band_mean(a, f))

    # path B: geometry derived from the dimensions
    t_model, D_model, sigma_model = fcc_geometric_params(VALID_tL, VALID_d, VALID_L)
    res_model = {}
    for N in N_list:
        a = compute_alpha_explicit(f, t_model, D_model, VALID_d, sigma_model, N)
        res_model[N] = dict(alpha=a, bm=band_mean(a, f))

    geom = dict(
        paper  = dict(t=VALID_t,   D=VALID_D,   sigma=VALID_sigma),
        model  = dict(t=t_model,   D=D_model,   sigma=sigma_model),
    )

    print("  Geometry comparison (paper vs our fcc_geometric_params):")
    print(f"  {'Parameter':<18}  {'Paper':>10}  {'Our model':>10}  {'Error %':>8}")
    print("  " + "-" * 55)
    for attr, label in [("t", "t  (mm)"), ("D", "D  (mm)"), ("sigma", "sigma (%)")]:
        pv  = geom["paper"][attr]
        mv  = geom["model"][attr]
        scl = 1e3 if attr in ("t", "D") else 100.0
        err = 100.0 * (mv - pv) / max(abs(pv), 1e-12)
        print(f"  {label:<18}  {pv*scl:>10.3f}  {mv*scl:>10.3f}  {err:>+8.1f}")

    return res_paper, res_model, geom


def plot_study9(f, res_paper, res_model, geom, outdir):
    """plot the FCC-plate validation: alpha(f) for both paths vs published data"""
    N_list = [1, 2, 3]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_list)))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # left: exact params -- the curves to validate against
    for i, N in enumerate(N_list):
        r = res_paper[N]
        ax.plot(f * 1e-3, r["alpha"], color=colors[i], ls="-", lw=2,
                label=f"N={N}  (paper params)  ⟨α⟩={r['bm']:.3f}")
    shade_ev(ax)
    ax.set_xlim(0.2, 8.0)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Absorption coefficient  α")
    ax.set_title(
        "Study 9A — Paper's exact params\n"
        f"L={VALID_L*1e3:.0f}mm  d={VALID_d*1e3:.2f}mm  "
        f"t={VALID_t*1e3:.2f}mm  D={VALID_D*1e3:.1f}mm  σ={VALID_sigma*100:.2f}%\n"
        "Compare against Fig. 4C of validation paper"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # right: exact (solid) vs derived (dashed) -- geometry accuracy check
    for i, N in enumerate(N_list):
        rp = res_paper[N]
        rm = res_model[N]
        ax2.plot(f * 1e-3, rp["alpha"], color=colors[i], ls="-", lw=2,
                 label=f"N={N} paper  ⟨α⟩={rp['bm']:.3f}")
        ax2.plot(f * 1e-3, rm["alpha"], color=colors[i], ls="--", lw=1.5,
                 label=f"N={N} model  ⟨α⟩={rm['bm']:.3f}")
    shade_ev(ax2)
    ax2.set_xlim(0.2, 8.0)
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Absorption coefficient  α")

    # Geometry delta annotation
    gp = geom["paper"]
    gm = geom["model"]
    err_D = 100.0 * (gm["D"] - gp["D"]) / gp["D"]
    err_t = 100.0 * (gm["t"] - gp["t"]) / gp["t"]
    err_s = 100.0 * (gm["sigma"] - gp["sigma"]) / gp["sigma"]
    ann = (
        f"Geometry delta (model - paper):\n"
        f"  t:     {gm['t']*1e3:.2f} vs {gp['t']*1e3:.2f} mm  ({err_t:+.1f}%)\n"
        f"  D:     {gm['D']*1e3:.2f} vs {gp['D']*1e3:.2f} mm  ({err_D:+.1f}%)\n"
        f"  sigma: {gm['sigma']*100:.2f} vs {gp['sigma']*100:.2f} %  ({err_s:+.1f}%)"
    )
    ax2.text(0.02, 0.97, ann, transform=ax2.transAxes, fontsize=8,
             va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
    ax2.set_title(
        "Study 9B — Paper exact (—) vs our model (- -)\n"
        "Geometry derivation accuracy check"
    )
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    p = os.path.join(outdir, "fig9_fcc_validation.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# Study 10: Three-lattice acoustic comparison

def study_10_three_lattice_comparison():
    """SC vs FCC nodal-pore vs FCC face-pore at matched params, swept over three L values"""
    import pandas as pd

    freq = np.arange(500, 2001, 10)

    configs = [
        dict(L=0.015, t_over_L=0.09, d=0.0015, N=5, label="L=15mm"),
        dict(L=0.020, t_over_L=0.09, d=0.0015, N=5, label="L=20mm"),
        dict(L=0.025, t_over_L=0.09, d=0.0015, N=5, label="L=25mm"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = {"SC": "#2196F3", "FCC nodal": "#4CAF50", "FCC face": "#FF9800"}

    summary_rows = []
    for ax, cfg in zip(axes, configs):
        L, tL, d, N = cfg["L"], cfg["t_over_L"], cfg["d"], cfg["N"]

        a_sc    = compute_alpha(freq, tL, d, L, N)
        a_nodal = compute_alpha_fcc(freq, tL, d, L, N)
        a_face  = compute_alpha_fcc_face(freq, tL, d, L, N)

        ax.plot(freq, a_sc,    lw=2, color=colors["SC"],        label="SC")
        ax.plot(freq, a_nodal, lw=2, color=colors["FCC nodal"], label="FCC nodal-pore", ls="--")
        ax.plot(freq, a_face,  lw=2, color=colors["FCC face"],  label="FCC face-pore",  ls=":")
        ax.axvspan(500, 2000, alpha=0.06, color="grey")
        ax.set_title(f"{cfg['label']}, t/L={tL}, d={d*1000:.1f}mm, N={N}", fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        band = (freq >= 500) & (freq <= 2000)
        for name, alpha in [("SC", a_sc), ("FCC_nodal", a_nodal), ("FCC_face", a_face)]:
            summary_rows.append({
                "L_mm": L * 1000, "lattice": name,
                "band_mean_alpha": float(np.mean(alpha[band])),
                "peak_freq_hz": float(freq[np.argmax(alpha)]),
            })

    axes[0].set_ylabel("Absorption coefficient α")
    fig.suptitle(
        "Study 10: Three-Lattice Acoustic Comparison\n"
        "SC vs FCC nodal-pore vs FCC face-pore",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = os.path.join(OUTDIR, "fig10_three_lattice_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    df = pd.DataFrame(summary_rows)
    print("\n  Study 10 Summary:")
    print(df.to_string(index=False))
    return df


# Main

def main():
    """run all the studies and write out the figures + dataset"""
    print("=" * 65)
    print("  SC/FCC Plate Lattice — MLHR Transfer Matrix Method")
    print("  Sections 6.2 and 8.1.1 of the paper")
    print("=" * 65)

    # Fixed model parameters
    L      = 0.020
    N_ref  = 10
    tL_ref = 0.09
    d_ref  = 0.001

    # Sweep ranges
    f       = np.linspace(200.0, 5000.0, 1000)
    tL_list = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18]
    d_list  = np.array([0.0005, 0.001, 0.002, 0.003, 0.004, 0.005])
    N_list  = [2, 5, 10, 20, 40, 80]
    L_list  = [0.010, 0.020, 0.030, 0.040, 0.050, 0.060]
    tL_map  = np.linspace(0.03, 0.18, 20)
    d_map   = np.linspace(0.0005, 0.005, 20)

    # fh targeting table
    print("\n  fh targeting table  (t/L={:.3f}, d={:.1f}mm)".format(tL_ref, d_ref*1e3))
    print(f"  {'L (mm)':>8}  {'t (mm)':>8}  {'D (mm)':>8}  {'fh (Hz)':>10}  {'in band?':>10}")
    print("  " + "-" * 55)
    for L_val in np.linspace(0.005, 0.100, 20):
        fh = helmholtz_freq(tL_ref, d_ref, L_val)
        t, D, _ = sc_geometric_params(tL_ref, d_ref, L_val)
        in_band = "YES" if 500 <= fh <= 2000 else ""
        print(f"  {L_val*1e3:8.1f}  {t*1e3:8.2f}  {D*1e3:8.2f}  {fh:10.1f}  {in_band:>10}")

    # Studies 1-5
    print(f"\n[Study 1] α(f) vs t/L  (d={d_ref*1e3:.1f}mm, N={N_ref}, L={L*1e3:.0f}mm)...")
    res1 = run_study1(f, tL_list, d_ref, N_ref, L)
    plot_study1(f, res1, d_ref, N_ref, L, OUTDIR)

    print(f"\n[Study 2] α(f) vs d  (t/L={tL_ref:.3f}, N={N_ref}, L={L*1e3:.0f}mm)...")
    res2 = run_study2(f, tL_ref, d_list, N_ref, L)
    plot_study2(f, res2, tL_ref, N_ref, L, OUTDIR)

    print(f"\n[Study 3] α(f) vs N  (t/L={tL_ref:.3f}, d={d_ref*1e3:.1f}mm, L={L*1e3:.0f}mm)...")
    res3 = run_study3(f, tL_ref, d_ref, N_list, L)
    plot_study3(f, res3, tL_ref, d_ref, L, OUTDIR)

    print(f"\n[Study 4] 2D design map t/L × d  (N={N_ref}, L={L*1e3:.0f}mm)...")
    plot_study4(f, tL_map, d_map, N_ref, L, OUTDIR)

    print(f"\n[Study 5] α(f) vs L  (t/L={tL_ref:.3f}, d={d_ref*1e3:.1f}mm, N={N_ref})...")
    res5 = run_study5(f, tL_ref, d_ref, N_ref, L_list)
    plot_study5(f, res5, tL_ref, d_ref, N_ref, OUTDIR)

    # Study 6: Heterogeneous MLHR
    print(f"\n[Study 6] Heterogeneous MLHR  "
          f"(t/L={tL_ref:.3f}, d={d_ref*1e3:.1f}mm, N={N_ref}, L∈[15,25]mm)...")
    res6 = run_study6(f, tL_ref, d_ref, N_ref)
    plot_study6(f, res6, tL_ref, d_ref, OUTDIR)
    print(f"  Homogeneous ⟨α⟩ = {res6['bm_homo']:.4f}")
    print(f"  Heterogeneous ⟨α⟩ = {res6['bm_hetero']:.4f}")

    # Study 7: SC vs FCC
    print(f"\n[Study 7] SC vs FCC  "
          f"(t/L={tL_ref:.3f}, d={d_ref*1e3:.1f}mm, L={L*1e3:.0f}mm, N={N_ref})...")
    res7 = run_study7(f, tL_ref, d_ref, L, N_ref)
    plot_study7(f, res7, tL_ref, d_ref, L, N_ref, OUTDIR)
    for lat in ["SC", "FCC"]:
        print(f"  {lat:<4} ⟨α⟩ = {res7[lat]['bm']:.4f}")

    # Study 8: Property-space dataset
    print(f"\n[Study 8] Building property-space dataset (18,000 points)...")
    dataset = build_property_dataset(f, OUTDIR)
    plot_study8_histogram(dataset, OUTDIR)

    # Study 9: FCC validation
    print(f"\n[Study 9] FCC validation against paper (L={VALID_L*1e3:.0f}mm, "
          f"d={VALID_d*1e3:.2f}mm, t={VALID_t*1e3:.2f}mm, D={VALID_D*1e3:.1f}mm, "
          f"sigma={VALID_sigma*100:.2f}%, N=1,2,3)...")
    res9_pape