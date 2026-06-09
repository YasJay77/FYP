"""Refreshes the Appendix-A table and the round-trip numbers
with the v4 ensemble net, and writes fig_inversion_demo.png.

Builds the 19-D property vector per Pareto point (3 stiffness + 16-point spectrum,
spectrum recomputed by TMM not from a cache), runs the v4 inference (refine K
candidates, keep the best-fitting one), TMM-verifies, then prints stats + the
LaTeX table body. FCC nodal-pore is dropped — SC and FCC face dominate it.
"""
import sys, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

ROOT       = Path(__file__).resolve().parent.parent
OUT_DIR    = ROOT / "results" / "inversion_results"
PARETO_CSV = ROOT / "results" / "property_datasets" / "pareto_front.csv"
sys.path.insert(0, str(ROOT / "scripts"))
from tmm_script import compute_alpha, compute_alpha_fcc, compute_alpha_fcc_face

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# must match the v4 notebook
FREQ_HZ      = np.arange(500, 2001, 100, dtype=float)
MECH_COLS    = ["C11", "C12", "C44"]
LT_LABELS    = ["SC", "FCC", "FCC_face"]
N_VALUES     = [5, 10, 15, 20]
K            = 4
REFINE_STEPS = 200
REFINE_LR    = 0.05
SEED         = 42

# nets, must match the v4 notebook
class ForwardNet(nn.Module):
    """plain MLP forward surrogate (geom -> mech or geom -> alpha)"""
    def __init__(self, in_dim, out_dim, hidden, depth, use_layernorm):
        """build the stacked linear/gelu trunk"""
        super().__init__()
        layers = [nn.Linear(in_dim, hidden)]
        if use_layernorm: layers.append(nn.LayerNorm(hidden))
        layers.append(nn.GELU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            if use_layernorm: layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        """run the net"""
        return self.net(x)

class InversionNetV4(nn.Module):
    """v4 inversion net: shared trunk + continuous/lattice/N heads"""
    def __init__(self, in_dim=19, trunk_hidden=512, depth=6, K=K,
                 n_lt=3, n_N=4, dropout=0.1):
        """build trunk, projection and the three output heads"""
        super().__init__()
        layers = []
        d_in = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, trunk_hidden), nn.LayerNorm(trunk_hidden),
                       nn.GELU(), nn.Dropout(dropout)]
            d_in = trunk_hidden
        self.trunk = nn.Sequential(*layers)
        feat = trunk_hidden // 2
        self.proj = nn.Sequential(nn.Linear(trunk_hidden, feat), nn.LayerNorm(feat), nn.GELU())
        self.head_cont = nn.Sequential(
            nn.Linear(feat, 192), nn.GELU(), nn.Linear(192, K * 3), nn.Sigmoid())
        self.head_N  = nn.Sequential(nn.Linear(feat, 64), nn.GELU(), nn.Linear(64, n_N))
        self.head_lt = nn.Sequential(nn.Linear(feat, 64), nn.GELU(), nn.Linear(64, n_lt))
        self.K = K
    def forward(self, x):
        """return (cont, lattice-logits, N-logits)"""
        f = self.proj(self.trunk(x))
        cont = self.head_cont(f).view(-1, self.K, 3)
        lt   = self.head_lt(f)
        Nl   = self.head_N(f)
        return cont, lt, Nl

# models + scalers
with open(OUT_DIR / "forward_scalers.pkl", "rb") as f:
    scalers = pickle.load(f)
scaler_mech, scaler_X, scaler_cont = scalers["mech"], scalers["X"], scalers["cont"]

fwd_mech  = ForwardNet(7, 3, 128, 3, use_layernorm=False).to(DEVICE)
fwd_alpha = ForwardNet(7, 16, 512, 6, use_layernorm=True).to(DEVICE)
fwd_mech.load_state_dict(torch.load(OUT_DIR / "forward_net_mech.pt", map_location=DEVICE))
fwd_alpha.load_state_dict(torch.load(OUT_DIR / "forward_net_alpha.pt", map_location=DEVICE))
fwd_mech.eval(); fwd_alpha.eval()
for p in fwd_mech.parameters():  p.requires_grad_(False)
for p in fwd_alpha.parameters(): p.requires_grad_(False)

inv_models = []
for s in [42, 123, 7]:
    m = InversionNetV4().to(DEVICE)
    m.load_state_dict(torch.load(OUT_DIR / f"inversion_net_seed{s}.pt", map_location=DEVICE))
    m.eval()
    inv_models.append(m)
print(f"Loaded {len(inv_models)} inversion-net seeds and 2 forward surrogates.")

# 19-D property vector per Pareto point
pf = pd.read_csv(PARETO_CSV)
pf = pf.sort_values("band_mean_alpha", ascending=False).reset_index(drop=True)

# greedy non-dominance on (band_mean_alpha, C11_norm)
keep = []
seen = []
for i, row in pf.iterrows():
    dominated = any(s[0] >= row["band_mean_alpha"] and s[1] >= row["C11_norm"]
                    and (s[0] > row["band_mean_alpha"] or s[1] > row["C11_norm"]) for s in seen)
    if not dominated:
        keep.append(i); seen.append((row["band_mean_alpha"], row["C11_norm"]))
pf = pf.iloc[keep].reset_index(drop=True)

# drop FCC nodal-pore (front is SC + FCC face only)
pf = pf[pf["lattice_type"] != "FCC"].reset_index(drop=True)
print(f"Pareto points after dominance filter + FCC-nodal exclusion: {len(pf)}")

def tmm_spectrum(lt, t_over_L, d_mm, L_mm, N, freqs=FREQ_HZ):
    """tmm absorption spectrum for one geometry, dispatched by lattice type"""
    d_m, L_m = d_mm / 1000.0, L_mm / 1000.0
    if lt == "SC":       return compute_alpha(freqs, t_over_L, d_m, L_m, int(N))
    if lt == "FCC":      return compute_alpha_fcc(freqs, t_over_L, d_m, L_m, int(N))
    if lt == "FCC_face": return compute_alpha_fcc_face(freqs, t_over_L, d_m, L_m, int(N))
    raise ValueError(lt)

# 16-point spectrum per row
spectra = np.stack([
    tmm_spectrum(row["lattice_type"], row["t_over_L"], row["d_mm"], row["L_mm"], row["N"])
    for _, row in pf.iterrows()
])
X_raw = np.column_stack([pf[MECH_COLS].values, spectra]).astype(np.float32)
X_n   = scaler_X.transform(X_raw).astype(np.float32)
X_te  = torch.tensor(X_n).to(DEVICE)
B = X_te.shape[0]

# inference helpers, mirror the v4 notebook
N_VALUES_T = torch.tensor(N_VALUES, dtype=torch.float32, device=DEVICE)
FREQ_HZ_T  = torch.tensor(FREQ_HZ,  dtype=torch.float32, device=DEVICE)
F_RANGE    = float(FREQ_HZ.max() - FREQ_HZ.min())

def denorm_cont(c):
    """undo the cont-scaler min-max back to physical geom"""
    lo = torch.tensor(scaler_cont.data_min_, dtype=torch.float32, device=c.device)
    hi = torch.tensor(scaler_cont.data_max_, dtype=torch.float32, device=c.device)
    return c * (hi - lo) + lo

def fwd_input_from_geom(geom_phys, lt_soft, N_soft):
    """pack physical geom + soft lattice/N into the 7-d forward-net input"""
    N_eff  = (N_soft * N_VALUES_T.unsqueeze(0)).sum(dim=1)
    N_norm = (N_eff - float(min(N_VALUES))) / (float(max(N_VALUES)) - float(min(N_VALUES)))
    return torch.stack([
        geom_phys[:, 0], geom_phys[:, 1] / 3.0, geom_phys[:, 2] / 40.0, N_norm,
        lt_soft[:, 0], lt_soft[:, 1], lt_soft[:, 2]
    ], dim=1)

def x_split(Xb):
    """split the 19-d input into mech and alpha targets for the surrogates"""
    x_mech_n = Xb[:, :3]; x_alpha = Xb[:, 3:]
    lo = torch.tensor(scaler_X.data_min_[:3], dtype=torch.float32, device=Xb.device)
    sc = torch.tensor(scaler_X.scale_[:3],    dtype=torch.float32, device=Xb.device)
    mech_phys = (x_mech_n / sc) + lo
    lo_m = torch.tensor(scaler_mech.data_min_, dtype=torch.float32, device=Xb.device)
    sc_m = torch.tensor(scaler_mech.scale_,    dtype=torch.float32, device=Xb.device)
    mech_target_for_surr = (mech_phys - lo_m) * sc_m
    lo_a = torch.tensor(scaler_X.data_min_[3:], dtype=torch.float32, device=Xb.device)
    sc_a = torch.tensor(scaler_X.scale_[3:],    dtype=torch.float32, device=Xb.device)
    alpha_target = (x_alpha / sc_a) + lo_a
    return mech_target_for_surr, alpha_target

def centroid_loss(ap, at, eps=1e-3):
    """squared error between predicted and target spectral centroids"""
    num_p = (FREQ_HZ_T.unsqueeze(0) * ap).sum(1); den_p = ap.sum(1) + eps
    num_t = (FREQ_HZ_T.unsqueeze(0) * at).sum(1); den_t = at.sum(1) + eps
    return ((num_p/den_p - num_t/den_t) / F_RANGE).pow(2).mean()

def refine_one(geom_init, lt_oh, N_oh, mech_tgt, alpha_tgt):
    """gradient-refine one candidate geom against mech+alpha targets in logit space"""
    lo = torch.tensor(scaler_cont.data_min_, dtype=torch.float32, device=geom_init.device)
    hi = torch.tensor(scaler_cont.data_max_, dtype=torch.float32, device=geom_init.device)
    geom_n = ((geom_init - lo) / (hi - lo)).clamp(1e-4, 1-1e-4)
    z = torch.log(geom_n / (1 - geom_n)).detach().requires_grad_(True)
    opt = optim.Adam([z], lr=REFINE_LR)
    for _ in range(REFINE_STEPS):
        opt.zero_grad()
        geom_p = torch.sigmoid(z) * (hi - lo) + lo
        fi = fwd_input_from_geom(geom_p, lt_oh, N_oh)
        ap = fwd_alpha(fi)
        L = nn.functional.mse_loss(fwd_mech(fi), mech_tgt) \
          + nn.functional.mse_loss(ap, alpha_tgt) \
          + 0.1 * centroid_loss(ap, alpha_tgt)
        L.backward(); opt.step()
    with torch.no_grad():
        return (torch.sigmoid(z) * (hi - lo) + lo).detach()

# ensemble inference: refine every candidate, then pick
cont_list, lt_list, N_list = [], [], []
for m in inv_models:
    with torch.no_grad():
        c, l, N = m(X_te)
    cont_list.append(c); lt_list.append(l); N_list.append(N)
cont_all = torch.cat(cont_list, dim=1)
lt_avg   = torch.stack(lt_list, dim=0).mean(dim=0)
N_avg    = torch.stack(N_list, dim=0).mean(dim=0)
lt_pred  = lt_avg.argmax(dim=1)
N_pred_idx = N_avg.argmax(dim=1)
N_pred_int = N_VALUES_T.gather(0, N_pred_idx).long()

lt_oh = torch.zeros_like(lt_avg); lt_oh.scatter_(1, lt_pred.unsqueeze(1), 1.0)
N_oh  = torch.zeros_like(N_avg);  N_oh.scatter_(1, N_pred_idx.unsqueeze(1), 1.0)
mech_tgt, alpha_tgt = x_split(X_te)
MK = cont_all.shape[1]
cont_phys_all = denorm_cont(cont_all.reshape(B*MK, 3)).reshape(B, MK, 3)

refined = torch.zeros_like(cont_phys_all)
for k in range(MK):
    refined[:, k, :] = refine_one(cont_phys_all[:, k, :], lt_oh, N_oh, mech_tgt, alpha_tgt)

# score and keep the best
lt_rep = lt_oh.unsqueeze(1).expand(-1, MK, -1).reshape(B*MK, 3)
N_rep  = N_oh.unsqueeze(1).expand(-1, MK, -1).reshape(B*MK, 4)
fi = fwd_input_from_geom(refined.reshape(B*MK, 3), lt_rep, N_rep)
with torch.no_grad():
    mp = fwd_mech(fi).reshape(B, MK, 3)
    ap = fwd_alpha(fi).reshape(B, MK, 16)
err = ((mp - mech_tgt.unsqueeze(1))**2).mean(dim=2) + ((ap - alpha_tgt.unsqueeze(1))**2).mean(dim=2)
win = err.argmin(dim=1)
geom_final = refined[torch.arange(B, device=DEVICE), win].cpu().numpy()
N_final    = N_pred_int.cpu().numpy()
lt_final   = lt_pred.cpu().numpy()

# TMM-verify each recovered geometry, build the table
rows = []
band = np.arange(500, 2001, 10)
for i in range(B):
    tgt = pf.iloc[i]
    lt  = LT_LABELS[lt_final[i]]
    tL  = float(geom_final[i, 0]); d = float(geom_final[i, 1]); L = float(geom_final[i, 2])
    N   = int(N_final[i])
    if lt == "SC":       a_band = compute_alpha(band, tL, d/1000., L/1000., N)
    elif lt == "FCC":    a_band = compute_alpha_fcc(band, tL, d/1000., L/1000., N)
    else:                a_band = compute_alpha_fcc_face(band, tL, d/1000., L/1000., N)
    a_tmm = float(a_band.mean())
    sp_v  = tmm_spectrum(lt, tL, d, L, N)
    rows.append({
        "P":      i + 1,
        "lt_t":   tgt["lattice_type"], "tL_t": float(tgt["t_over_L"]),
        "d_t":    float(tgt["d_mm"]),   "alpha_t": float(tgt["band_mean_alpha"]),
        "C11n_t": float(tgt["C11_norm"]),
        "lt_p":   lt, "tL_p": tL, "d_p": d, "L_p": L, "N_p": N, "alpha_tmm": a_tmm,
        "alpha_err": a_tmm - float(tgt["band_mean_alpha"]),
        "spectrum_target": spectra[i].tolist(),
        "spectrum_verified": sp_v.tolist(),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "appendix_pareto.csv", index=False)

print("\n" + "="*70)
print("Summary stats for §5.4 / §6.3 / Conclusions (1)(6)")
print("="*70)
abs_err = df["alpha_err"].abs()
print(f"  Band-mean α MAE       : {abs_err.mean():.4f}")
print(f"  Band-mean α median    : {abs_err.median():.4f}")
print(f"  Band-mean α max       : {abs_err.max():.4f}  (P{int(df.loc[abs_err.idxmax(), 'P'])}, lattice={df.loc[abs_err.idxmax(), 'lt_p']})")
# per-freq R²
target_sp = np.array(df["spectrum_target"].tolist())
ver_sp    = np.array(df["spectrum_verified"].tolist())
per_freq_r2 = [r2_score(target_sp[:, j], ver_sp[:, j]) for j in range(16)]
print(f"  Per-freq α R² (mean)  : {np.mean(per_freq_r2):.4f}")
print(f"  Per-freq α R² (min)   : {np.min(per_freq_r2):.4f} @ {int(FREQ_HZ[np.argmin(per_freq_r2)])} Hz")

print("\n" + "="*70)
print("LaTeX table body for Appendix A (paste between \\midrule and \\bottomrule)")
print("="*70)
for _, r in df.iterrows():
    lt_short = {"SC": "SC", "FCC": "FCC", "FCC_face": "FCC face"}[r["lt_t"]]
    print(f"        P{int(r['P']):<3d}& {lt_short:<8s} & {r['tL_t']:.3f} & {r['d_t']:.3f} & "
          f"{r['alpha_t']:.3f} & {r['C11n_t']:.3f} & "
          f"{r['tL_p']:.3f} & {r['d_p']:.3f} & {r['L_p']:.2f} & {r['N_p']:>2d} & {r['alpha_tmm']:.3f} \\\\")
print(f"\nSaved per-row CSV → {OUT_DIR / 'appendix_pareto.csv'}")

# 3-panel demo figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

print("\n" + "="*70)
print("Generating fig_inversion_demo.png ...")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# left: front, targets vs round-trip, error lines
ax = axes[0]
x_t = df["C11n_t"].values
y_t = df["alpha_t"].values
y_p = df["alpha_tmm"].values
for xi, yt, yp in zip(x_t, y_t, y_p):
    ax.plot([xi, xi], [yt, yp], color="grey", lw=0.7, alpha=0.6, zorder=1)
ax.scatter(x_t, y_t, s=55, c="#1f77b4", marker="o", label="Pareto target",
           zorder=3, edgecolors="white", linewidths=0.5)
ax.scatter(x_t, y_p, s=55, c="#ff7f0e", marker="^", label="InversionNet round-trip (TMM)",
           zorder=3, edgecolors="white", linewidths=0.5)
ax.set_xlabel(r"Normalised axial stiffness $C_{11}/C_{11,\max}$")
ax.set_ylabel(r"Band-mean absorption $\langle\alpha\rangle$")
ax.set_title("Pareto front round-trip")
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.25)

# middle: recovered (t/L, d), colour = target alpha, marker = predicted lattice
ax = axes[1]
norm = Normalize(vmin=float(df["alpha_t"].min()), vmax=float(df["alpha_t"].max()))
cmap = plt.get_cmap("viridis")
shape_map = {"SC": "o", "FCC": "s", "FCC_face": "^"}
for lt_p, marker in shape_map.items():
    mask = (df["lt_p"] == lt_p).values
    if mask.sum() == 0:
        continue
    sub = df[mask]
    ax.scatter(sub["tL_p"], sub["d_p"], c=sub["alpha_t"],
               cmap=cmap, norm=norm, s=80, marker=marker,
               edgecolors="black", linewidths=0.5,
               label=lt_p.replace("_", " "))
ax.set_xlabel(r"Recovered $t/L$")
ax.set_ylabel(r"Recovered $d$ (mm)")
ax.set_title(r"Recovered geometries (colour $=$ target $\langle\alpha\rangle$)")
ax.legend(loc="best", fontsize=9, framealpha=0.95, title="Predicted lattice")
ax.grid(True, alpha=0.25)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label(r"Target $\langle\alpha\rangle$")

# right: per-point relative error in band-mean alpha
ax = axes[2]
rel_err = (df["alpha_err"].abs() / df["alpha_t"].abs().clip(lower=1e-9) * 100.0).values
colours_by_family = {"SC": "#1f77b4", "FCC": "#2ca02c", "FCC_face": "#d62728"}
bar_colours = [colours_by_family[lt] for lt in df["lt_p"].values]
ax.bar(df["P"].values, rel_err, color=bar_colours, edgecolor="black", linewidth=0.4)
ax.set_xlabel("Pareto-point index")
ax.set_ylabel(r"Relative error in $\langle\alpha\rangle$ (%)")
ax.set_title("Per-point round-trip error")
ax.set_xticks(df["P"].values[::2])
ax.grid(True, axis="y", alpha=0.25)
ax.legend(handles=[Patch(facecolor=c, edgecolor="black", label=k.replace("_", " "))
                   for k, c in colours_by_family.items()],
          loc="best", fontsize=9, framealpha=0.95, title="Predicted lattice")

fig.tight_layout()
demo_path = OUT_DIR / "fig_inversion_demo.png"
fig.savefig(demo_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved demo figure → {demo_path}")
