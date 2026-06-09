"""Production InversionNet (v4): target property vector (stiffness + 16-point
absorption spectrum) -> plate-lattice geometry, with TMM verification.

Pipeline: forward surrogates (small MLP for C11/C12/C44, larger one for the
16-point spectrum) -> InversionNet (shared trunk, three heads: K geometry
candidates, N classifier, lattice-type classifier; best-of-K + physics losses
after a geometry-only pre-warm) -> refine every candidate through the frozen
surrogates and keep the best -> push predicted geometries back through the TMM.

Trained in Colab (~2 h on a T4). Reads property_dataset_spectral.csv, writes
models/figures/verification CSVs to results/inversion_results/.
"""

import sys, json, pickle, time, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)
if DEVICE.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_CSV = ROOT / "results" / "property_datasets" / "property_dataset_spectral.csv"
OUT_DIR  = ROOT / "results" / "inversion_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
from tmm_script import compute_alpha, compute_alpha_fcc, compute_alpha_fcc_face

assert DATA_CSV.exists(), f"Missing spectral dataset: {DATA_CSV}"
print(f"Data : {DATA_CSV}")
print(f"Out  : {OUT_DIR}")

FREQ_HZ    = np.arange(500, 2001, 100, dtype=float)              # 16 points
FREQ_COLS  = [f'alpha_{int(f)}hz' for f in FREQ_HZ]
MECH_COLS  = ['C11', 'C12', 'C44']
INPUT_COLS = MECH_COLS + FREQ_COLS                                # 19-D
GEOM_COLS  = ['t_over_L', 'd_mm', 'L_mm']                         # 3-D, no T
LT_LABELS  = ['SC', 'FCC', 'FCC_face']
N_VALUES   = [5, 10, 15, 20]
BOUNDS     = {'t_over_L': (0.03, 0.18), 'd_mm': (0.5, 3.0), 'L_mm': (10.0, 40.0)}

SEED               = 42
FWD_MECH_EPOCHS    = 200
FWD_ALPHA_EPOCHS   = 600
FWD_LR             = 3e-4
FWD_BS             = 256
INV_EPOCHS         = 1500
INV_LR             = 3e-4
INV_BS             = 256
INV_PATIENCE       = 300
K_CANDIDATES       = 4
PREWARM_EPOCHS     = 100              # geometry-only phase before physics kicks in
LAM_MECH_MAX       = 0.3
LAM_ALPHA_MAX      = 0.3
LAM_RAMP_EPOCHS    = 100              # ramp length after pre-warm
LAM_CENTROID       = 0.1              # spectrum-centroid aux loss weight
CE_LT_WEIGHT       = 0.5
CE_N_WEIGHT        = 1.0
REFINE_STEPS       = 200
REFINE_LR          = 0.05

torch.manual_seed(SEED); np.random.seed(SEED)

df_all = pd.read_csv(DATA_CSV)
print('Rows:', len(df_all), '| Cols:', df_all.shape[1])
df = df_all.dropna(subset=MECH_COLS + FREQ_COLS).copy().reset_index(drop=True)
df['lt_class'] = df['lattice_type'].map({lt: i for i, lt in enumerate(LT_LABELS)})
df['N_class']  = df['N'].map({n: i for i, n in enumerate(N_VALUES)})

idx = np.arange(len(df))
idx_tv, idx_te  = train_test_split(idx,    test_size=0.15,        random_state=SEED)
idx_tr, idx_val = train_test_split(idx_tv, test_size=0.15/0.85,   random_state=SEED)
print(f'Split: train={len(idx_tr)} val={len(idx_val)} test={len(idx_te)}')
for split, s in [('train', idx_tr), ('val', idx_val), ('test', idx_te)]:
    counts = df.iloc[s]['lattice_type'].value_counts().to_dict()
    n_counts = df.iloc[s]['N'].value_counts().sort_index().to_dict()
    print(f'  {split} — lattice: {counts}  |  N: {n_counts}')

# N class weights: inverse frequency, normalised to sum to len(N_VALUES)
n_train_counts = df.iloc[idx_tr]['N_class'].value_counts().sort_index().values.astype(np.float32)
w_N = (1.0 / n_train_counts); w_N = w_N * len(N_VALUES) / w_N.sum()
print(f'N class weights for CE: {w_N.tolist()}')
w_N_t = torch.tensor(w_N, dtype=torch.float32, device=DEVICE)

X_raw = df[INPUT_COLS].values.astype(np.float32)                   # 19-D
Y_raw = df[GEOM_COLS].values.astype(np.float32)                    # 3-D geom
Y_lt  = df['lt_class'].values.astype(np.int64)
Y_N   = df['N_class'].values.astype(np.int64)

scaler_X    = MinMaxScaler().fit(X_raw[idx_tr])
scaler_cont = MinMaxScaler().fit(Y_raw[idx_tr])
X_n = scaler_X.transform(X_raw).astype(np.float32)
Y_n = scaler_cont.transform(Y_raw).astype(np.float32)

def ds(i):
    """wrap the rows at index i into a tensor dataset (X, geom, lt, N)"""
    return TensorDataset(torch.tensor(X_n[i]), torch.tensor(Y_n[i]),
                                 torch.tensor(Y_lt[i]), torch.tensor(Y_N[i]))
ds_tr_inv, ds_val_inv, ds_te_inv = ds(idx_tr), ds(idx_val), ds(idx_te)
df_test = df.iloc[idx_te].reset_index(drop=True)

def build_fwd_arrays(df):
    """build the 7-D forward-net inputs + mech/alpha targets from a dataframe"""
    lt_sc       = (df['lattice_type'] == 'SC').astype(float).values
    lt_fcc      = (df['lattice_type'] == 'FCC').astype(float).values
    lt_fcc_face = (df['lattice_type'] == 'FCC_face').astype(float).values
    N_norm = ((df['N'].values - min(N_VALUES)) / (max(N_VALUES) - min(N_VALUES))).astype(np.float32)
    X = np.column_stack([
        df['t_over_L'].values, df['d_mm'].values/3., df['L_mm'].values/40.,
        N_norm, lt_sc, lt_fcc, lt_fcc_face
    ]).astype(np.float32)
    Y_mech  = df[MECH_COLS].values.astype(np.float32)
    Y_alpha = df[FREQ_COLS].values.astype(np.float32)
    return X, Y_mech, Y_alpha

X_fwd, Y_mech_fwd, Y_alpha_fwd = build_fwd_arrays(df)
scaler_mech = MinMaxScaler().fit(Y_mech_fwd[idx_tr])
Y_mech_n    = scaler_mech.transform(Y_mech_fwd).astype(np.float32)

class ForwardNet(nn.Module):
    """small MLP surrogate mapping geom+lt/N to mech or alpha-spectrum"""
    def __init__(self, in_dim=7, out_dim=3, hidden=128, depth=3, use_layernorm=False):
        """build the GELU MLP with optional layernorm and given depth/width"""
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
        """run the input through the MLP"""
        return self.net(x)

def train_fwd(model, X, Y, epochs, lr, bs, name='Fwd', patience=120):
    """train a forward surrogate with Adam + cosine LR and early stopping"""
    Xtr, Xvl, Ytr, Yvl = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    ld_tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)), batch_size=bs, shuffle=True)
    ld_vl = DataLoader(TensorDataset(torch.tensor(Xvl), torch.tensor(Yvl)), batch_size=bs)
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/100)
    mse = nn.MSELoss()
    best = math.inf; best_st = None; pat = 0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in ld_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); l = mse(model(xb), yb); l.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([mse(model(xb.to(DEVICE)), yb.to(DEVICE)).item() for xb, yb in ld_vl]))
        if v < best:
            best, best_st, pat = v, {k: v_.clone() for k, v_ in model.state_dict().items()}, 0
        else: pat += 1
        if ep % 50 == 0: print(f'  [{name}] ep {ep:4d}/{epochs}  val={v:.6f}')
        if pat >= patience: print(f'  [{name}] early stop @ ep {ep}'); break
    model.load_state_dict(best_st); return model, best

print('=== Stage 1a: ForwardNet (mechanical) ===')
t0 = time.time()
fwd_mech, best_mech = train_fwd(ForwardNet(7, 3, 128, 3, use_layernorm=False),
                                 X_fwd, Y_mech_n, FWD_MECH_EPOCHS, FWD_LR, FWD_BS, name='MECH')
print(f'  best val MSE = {best_mech:.5e}  ({time.time()-t0:.1f}s)')

print('\n=== Stage 1b: ForwardNet (alpha-spectrum, 16-D output, LARGER) ===')
t0 = time.time()
fwd_alpha, best_alpha = train_fwd(ForwardNet(7, 16, 512, 6, use_layernorm=True),
                                   X_fwd, Y_alpha_fwd, FWD_ALPHA_EPOCHS, FWD_LR, FWD_BS, name='ALPHA')
print(f'  best val MSE = {best_alpha:.5e}  ({time.time()-t0:.1f}s)')

torch.save(fwd_mech.state_dict(),  OUT_DIR/'forward_net_mech.pt')
torch.save(fwd_alpha.state_dict(), OUT_DIR/'forward_net_alpha.pt')
with open(OUT_DIR/'forward_scalers.pkl', 'wb') as f:
    pickle.dump({'mech': scaler_mech, 'X': scaler_X, 'cont': scaler_cont}, f)

# Pre-flight: per-freq R^2 of the alpha surrogate. If it's < 0.95 the surrogate
# is the bottleneck and caps inversion accuracy.
fwd_alpha.eval()
with torch.no_grad():
    Xtr, Xvl, Ytr, Yvl = train_test_split(X_fwd, Y_alpha_fwd, test_size=0.2, random_state=SEED)
    yhat = fwd_alpha(torch.tensor(Xvl).to(DEVICE)).cpu().numpy()
print('\n=== α surrogate per-frequency validation R² ===')
for i, f in enumerate(FREQ_HZ):
    r2 = r2_score(Yvl[:, i], yhat[:, i])
    flag = '' if r2 >= 0.95 else '  ← below 0.95'
    print(f'  {int(f):>5d} Hz:  R²={r2:.3f}{flag}')
alpha_surr_mean_r2 = float(np.mean([r2_score(Yvl[:, i], yhat[:, i]) for i in range(len(FREQ_HZ))]))
print(f'\n  Mean R² across 16 freqs: {alpha_surr_mean_r2:.3f}')
if alpha_surr_mean_r2 < 0.95:
    print('  WARNING: α surrogate is the bottleneck. Inversion accuracy is capped here.')
else:
    print('  α surrogate accuracy is healthy — pipeline can use its gradients.')

class InversionNetV4(nn.Module):
    """19-D properties -> K candidate geometries + N class + lattice class"""
    def __init__(self, in_dim=19, trunk_hidden=512, depth=6, K=K_CANDIDATES,
                 n_lt=3, n_N=4, dropout=0.1):
        """build the shared trunk plus geom-candidate, N and lattice heads"""
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
            nn.Linear(feat, 192), nn.GELU(), nn.Linear(192, K * 3), nn.Sigmoid()
        )
        self.head_N  = nn.Sequential(nn.Linear(feat, 64), nn.GELU(), nn.Linear(64, n_N))
        self.head_lt = nn.Sequential(nn.Linear(feat, 64), nn.GELU(), nn.Linear(64, n_lt))
        self.K = K
    def forward(self, x):
        """return K candidate geoms, lattice logits and N logits"""
        f = self.proj(self.trunk(x))
        cont = self.head_cont(f).view(-1, self.K, 3)   # [B,K,3] in [0,1]
        lt   = self.head_lt(f)                         # [B,3]
        Nl   = self.head_N(f)                          # [B,4]
        return cont, lt, Nl

def denorm_cont(cont_norm_flat):
    """[B,3] in [0,1] -> physical (t/L, d_mm, L_mm)"""
    lo = torch.tensor(scaler_cont.data_min_, dtype=torch.float32, device=cont_norm_flat.device)
    hi = torch.tensor(scaler_cont.data_max_, dtype=torch.float32, device=cont_norm_flat.device)
    return cont_norm_flat * (hi - lo) + lo

# soft N: prob-weighted sum over N values, so physics losses stay differentiable w.r.t. N
N_VALUES_T = torch.tensor(N_VALUES, dtype=torch.float32, device=DEVICE)
N_MN = float(min(N_VALUES)); N_MX = float(max(N_VALUES))

def fwd_input_from_geom(geom_phys, lt_soft, N_soft):
    """build 7-D forward-net input from geom + soft lt/N"""
    N_eff  = (N_soft * N_VALUES_T.unsqueeze(0)).sum(dim=1)
    N_norm = (N_eff - N_MN) / (N_MX - N_MN)
    return torch.stack([
        geom_phys[:, 0],
        geom_phys[:, 1] / 3.0,
        geom_phys[:, 2] / 40.0,
        N_norm,
        lt_soft[:, 0], lt_soft[:, 1], lt_soft[:, 2],
    ], dim=1)

def x_split(Xb):
    """split 19-D input"""
    x_mech_n = Xb[:, :3]
    x_alpha  = Xb[:, 3:]
    lo = torch.tensor(scaler_X.data_min_[:3], dtype=torch.float32, device=Xb.device)
    sc = torch.tensor(scaler_X.scale_[:3],    dtype=torch.float32, device=Xb.device)
    mech_phys = (x_mech_n / sc) + lo
    lo_m = torch.tensor(scaler_mech.data_min_, dtype=torch.float32, device=Xb.device)
    sc_m = torch.tensor(scaler_mech.scale_,    dtype=torch.float32, device=Xb.device)
    mech_target_for_surrogate = (mech_phys - lo_m) * sc_m
    lo_a = torch.tensor(scaler_X.data_min_[3:], dtype=torch.float32, device=Xb.device)
    sc_a = torch.tensor(scaler_X.scale_[3:],    dtype=torch.float32, device=Xb.device)
    alpha_target = (x_alpha / sc_a) + lo_a
    return mech_target_for_surrogate, alpha_target

# spectrum centroid sum(f*alpha)/sum(alpha)
FREQ_HZ_T = torch.tensor(FREQ_HZ, dtype=torch.float32, device=DEVICE)
F_RANGE   = float(FREQ_HZ.max() - FREQ_HZ.min())
def centroid_loss(alpha_pred, alpha_target, eps=1e-3):
    """penalise spectra with right area but wrong spectral centroid (peak)"""
    num_p = (FREQ_HZ_T.unsqueeze(0) * alpha_pred).sum(dim=1)
    den_p = alpha_pred.sum(dim=1) + eps
    num_t = (FREQ_HZ_T.unsqueeze(0) * alpha_target).sum(dim=1)
    den_t = alpha_target.sum(dim=1) + eps
    cp = num_p / den_p; ct = num_t / den_t
    return ((cp - ct) / F_RANGE).pow(2).mean()

def lam_sched(ep, prewarm, ramp, lmax):
    """ramp the physics-loss weight from 0 to lmax after a geometry-only pre-warm"""
    if ep <= prewarm: return 0.0
    return lmax * min(1.0, (ep - prewarm) / ramp)

def train_inversion_v4(seed, fwd_mech, fwd_alpha, w_N_t,
                       epochs=INV_EPOCHS, bs=INV_BS, lr=INV_LR,
                       lam_mech_max=LAM_MECH_MAX, lam_alpha_max=LAM_ALPHA_MAX,
                       prewarm=PREWARM_EPOCHS, ramp=LAM_RAMP_EPOCHS,
                       lam_centroid=LAM_CENTROID, verbose_every=50):
    """train one InversionNet with best-of-K geom + ramped physics losses"""
    torch.manual_seed(seed); np.random.seed(seed)
    for p in fwd_mech.parameters():  p.requires_grad_(False)
    for p in fwd_alpha.parameters(): p.requires_grad_(False)
    fwd_mech.eval(); fwd_alpha.eval()
    model = InversionNetV4().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/100)
    mse_per = nn.MSELoss(reduction='none')
    ce_lt   = nn.CrossEntropyLoss()
    ce_N    = nn.CrossEntropyLoss(weight=w_N_t)

    ld_tr = DataLoader(ds_tr_inv,  batch_size=bs, shuffle=True, drop_last=True)
    ld_vl = DataLoader(ds_val_inv, batch_size=bs)
    best = math.inf; best_st = None; pat = 0
    hist = {k: [] for k in ['train_total','train_geom','train_pmech','train_palpha',
                             'train_pcen','train_N_acc','val_geom','val_N_acc']}

    for ep in range(1, epochs+1):
        lam_m = lam_sched(ep, prewarm, ramp, lam_mech_max)
        lam_a = lam_sched(ep, prewarm, ramp, lam_alpha_max)
        lam_c = lam_sched(ep, prewarm, ramp, lam_centroid)
        model.train()
        et, eg, em, ea, ec, eN = [], [], [], [], [], []
        for Xb, Yc, Yl, YN in ld_tr:
            Xb, Yc, Yl, YN = Xb.to(DEVICE), Yc.to(DEVICE), Yl.to(DEVICE), YN.to(DEVICE)
            opt.zero_grad()
            cont, llt, lN = model(Xb)
            B, K, _ = cont.shape
            # best-of-K geometry loss + 0.25 mean-of-K
            Yc_exp = Yc.unsqueeze(1).expand(-1, K, -1)
            per_k  = mse_per(cont, Yc_exp).mean(dim=2)
            best_k_vals, best_k_idx = per_k.min(dim=1)
            Lg = best_k_vals.mean() + 0.25 * per_k.mean()
            Lg = Lg + CE_LT_WEIGHT * ce_lt(llt, Yl) + CE_N_WEIGHT * ce_N(lN, YN)
            with torch.no_grad():
                N_acc_train = (lN.argmax(dim=1) == YN).float().mean().item()
            # physics on the winning candidate
            if lam_m > 0 or lam_a > 0 or lam_c > 0:
                cont_phys_all = denorm_cont(cont.reshape(B*K, 3)).reshape(B, K, 3)
                cont_phys_w   = cont_phys_all.gather(1, best_k_idx.view(B,1,1).expand(-1,-1,3)).squeeze(1)
                lt_soft = torch.softmax(llt, dim=1)
                N_soft  = torch.softmax(lN,  dim=1)
                fi_w = fwd_input_from_geom(cont_phys_w, lt_soft, N_soft)
                mech_tgt, alpha_tgt = x_split(Xb)
                ap_w = fwd_alpha(fi_w)
                Lm = nn.functional.mse_loss(fwd_mech(fi_w), mech_tgt)
                La = nn.functional.mse_loss(ap_w, alpha_tgt)
                Lc = centroid_loss(ap_w, alpha_tgt)
            else:
                Lm = torch.tensor(0., device=DEVICE)
                La = torch.tensor(0., device=DEVICE)
                Lc = torch.tensor(0., device=DEVICE)
            Lt = Lg + lam_m*Lm + lam_a*La + lam_c*Lc
            Lt.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            et.append(Lt.item()); eg.append(Lg.item()); em.append(Lm.item()); ea.append(La.item()); ec.append(Lc.item())
            eN.append(N_acc_train)
        sch.step()
        model.eval()
        with torch.no_grad():
            vl, vN = [], []
            for Xb, Yc, Yl, YN in ld_vl:
                Xb, Yc, Yl, YN = Xb.to(DEVICE), Yc.to(DEVICE), Yl.to(DEVICE), YN.to(DEVICE)
                cont, llt, lN = model(Xb); K = cont.shape[1]
                per_k = mse_per(cont, Yc.unsqueeze(1).expand(-1,K,-1)).mean(dim=2)
                vl.append((per_k.min(dim=1)[0].mean() + CE_LT_WEIGHT*ce_lt(llt,Yl) + CE_N_WEIGHT*ce_N(lN,YN)).item())
                vN.append((lN.argmax(dim=1) == YN).float().mean().item())
            v   = float(np.mean(vl))
            v_N = float(np.mean(vN))
        for k, arr in zip(['train_total','train_geom','train_pmech','train_palpha','train_pcen','train_N_acc','val_geom','val_N_acc'],
                          [np.mean(et), np.mean(eg), np.mean(em), np.mean(ea), np.mean(ec), np.mean(eN), v, v_N]):
            hist[k].append(float(arr))
        if v < best:
            best, best_st, pat = v, {k: v_.clone() for k, v_ in model.state_dict().items()}, 0
        else: pat += 1
        if ep % verbose_every == 0:
            phase = 'pre-warm' if ep <= prewarm else 'physics-on'
            print(f'  ep {ep:4d}/{epochs} [{phase:>9s}]  train={np.mean(et):.4f}  '
                  f'val={v:.4f}  N_acc(tr/vl)={np.mean(eN)*100:.1f}/{v_N*100:.1f}%  '
                  f'Lm={np.mean(em):.4f} La={np.mean(ea):.4f} Lc={np.mean(ec):.4f}  '
                  f'λm={lam_m:.2f}')
        if pat >= INV_PATIENCE:
            print(f'  early stop @ ep {ep}'); break
    model.load_state_dict(best_st)
    print(f'  seed {seed}: best val = {best:.6f}')
    return model, hist, best

print('=== Stage 2: InversionNet v4 (seed 42, base run) ===')
t0 = time.time()
inv_main, hist_main, best_main = train_inversion_v4(seed=42, fwd_mech=fwd_mech, fwd_alpha=fwd_alpha, w_N_t=w_N_t)
print(f'  total time: {(time.time()-t0)/60:.1f} min')
torch.save(inv_main.state_dict(), OUT_DIR/'inversion_net_seed42.pt')
with open(OUT_DIR/'hist_seed42.json', 'w') as f: json.dump(hist_main, f)

def refine_one_candidate(geom_phys_init, lt_soft, N_soft, mech_tgt, alpha_tgt,
                         steps=REFINE_STEPS, lr=REFINE_LR):
    """refine geom with Adam in sigmoid pre-image of bounds; lt/N held fixed"""
    lo = torch.tensor(scaler_cont.data_min_, dtype=torch.float32, device=geom_phys_init.device)
    hi = torch.tensor(scaler_cont.data_max_, dtype=torch.float32, device=geom_phys_init.device)
    geom_n = ((geom_phys_init - lo) / (hi - lo)).clamp(1e-4, 1-1e-4)
    z = torch.log(geom_n / (1 - geom_n)).detach().requires_grad_(True)
    opt = optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        geom_p = torch.sigmoid(z) * (hi - lo) + lo
        fi = fwd_input_from_geom(geom_p, lt_soft, N_soft)
        ap = fwd_alpha(fi)
        L = nn.functional.mse_loss(fwd_mech(fi), mech_tgt) \
          + nn.functional.mse_loss(ap, alpha_tgt) \
          + 0.1 * centroid_loss(ap, alpha_tgt)
        L.backward(); opt.step()
    with torch.no_grad():
        return (torch.sigmoid(z) * (hi - lo) + lo).detach()

def evaluate_full_pipeline_v4(inv_models, df_te, refine=True, tag='main', chunk=256):
    """run the ensemble, refine all K candidates and pick the best per sample"""
    X_te_all = torch.tensor(scaler_X.transform(df_te[INPUT_COLS].values.astype(np.float32))).to(DEVICE)
    final_geom = []; final_N = []; final_lt = []
    for s in range(0, len(df_te), chunk):
        Xb = X_te_all[s:s+chunk]
        B  = Xb.shape[0]
        # avg lt/N logits across ensemble, gather K candidates from each model
        cont_list, lt_list, N_list = [], [], []
        for m in inv_models:
            m.eval()
            with torch.no_grad():
                c, l, N = m(Xb)
            cont_list.append(c); lt_list.append(l); N_list.append(N)
        cont_all = torch.cat(cont_list, dim=1)            # [B, M*K, 3] normalised
        lt_avg   = torch.stack(lt_list, dim=0).mean(dim=0)
        N_avg    = torch.stack(N_list,  dim=0).mean(dim=0)
        lt_soft  = torch.softmax(lt_avg, dim=1)
        N_soft   = torch.softmax(N_avg,  dim=1)
        lt_pred  = lt_avg.argmax(dim=1)
        N_pred_idx = N_avg.argmax(dim=1)
        N_pred_int = N_VALUES_T.gather(0, N_pred_idx).long()
        # one-hot the chosen N/lt for refinement + scoring
        lt_oh = torch.zeros_like(lt_soft); lt_oh.scatter_(1, lt_pred.unsqueeze(1), 1.0)
        N_oh  = torch.zeros_like(N_soft);  N_oh.scatter_(1, N_pred_idx.unsqueeze(1), 1.0)
        mech_tgt, alpha_tgt = x_split(Xb)
        MK = cont_all.shape[1]
        cont_phys_all = denorm_cont(cont_all.reshape(B*MK, 3)).reshape(B, MK, 3)
        if refine:
            refined_phys = torch.zeros_like(cont_phys_all)
            for k in range(MK):
                refined_phys[:, k, :] = refine_one_candidate(
                    cont_phys_all[:, k, :], lt_oh, N_oh, mech_tgt, alpha_tgt)
            geom_pool = refined_phys
        else:
            geom_pool = cont_phys_all
        # score every candidate via both surrogates, pick smallest combined error
        lt_rep = lt_oh.unsqueeze(1).expand(-1, MK, -1).reshape(B*MK, 3)
        N_rep  = N_oh.unsqueeze(1).expand(-1, MK, -1).reshape(B*MK, 4)
        fi = fwd_input_from_geom(geom_pool.reshape(B*MK, 3), lt_rep, N_rep)
        with torch.no_grad():
            mp = fwd_mech(fi).reshape(B, MK, 3)
            ap = fwd_alpha(fi).reshape(B, MK, len(FREQ_HZ))
        err = ((mp - mech_tgt.unsqueeze(1))**2).mean(dim=2) + ((ap - alpha_tgt.unsqueeze(1))**2).mean(dim=2)
        win = err.argmin(dim=1)
        final_geom.append(geom_pool[torch.arange(B, device=DEVICE), win].detach().cpu().numpy())
        final_N.append(N_pred_int.detach().cpu().numpy())
        final_lt.append(lt_pred.detach().cpu().numpy())
    return (np.concatenate(final_geom, axis=0),
            np.concatenate(final_N,    axis=0),
            np.concatenate(final_lt,   axis=0))

print('=== Stage 3: Inference + refine-all-K + select  (single seed) ===')
t0 = time.time()
geom_pred, N_pred, lt_pred = evaluate_full_pipeline_v4([inv_main], df_test, refine=True, tag='single')
print(f'  done in {time.time()-t0:.1f}s | shape={geom_pred.shape}')

FREQ_BAND = np.arange(500, 2001, 10)

def tmm_alpha_for(lt, t_over_L, d_mm, L_mm, N):
    """run the TMM over the fine band for a given lattice + geometry"""
    d_m = d_mm/1000.; L_m = L_mm/1000.
    if lt == 'SC':       return compute_alpha(FREQ_BAND, t_over_L, d_m, L_m, int(N))
    if lt == 'FCC':      return compute_alpha_fcc(FREQ_BAND, t_over_L, d_m, L_m, int(N))
    if lt == 'FCC_face': return compute_alpha_fcc_face(FREQ_BAND, t_over_L, d_m, L_m, int(N))
    raise ValueError(lt)

def tmm_spectrum_for(lt, t_over_L, d_mm, L_mm, N, freqs=FREQ_HZ):
    """run the TMM at the 16 spectrum freqs for a given lattice + geometry"""
    d_m = d_mm/1000.; L_m = L_mm/1000.
    if lt == 'SC':       return compute_alpha(freqs, t_over_L, d_m, L_m, int(N))
    if lt == 'FCC':      return compute_alpha_fcc(freqs, t_over_L, d_m, L_m, int(N))
    if lt == 'FCC_face': return compute_alpha_fcc_face(freqs, t_over_L, d_m, L_m, int(N))
    raise ValueError(lt)

def verify_pipeline(geom_pred, N_pred, lt_pred, df_te, tag='main'):
    """TMM-check the predicted geometries against the target spectra/metrics"""
    rows = []
    for i in range(len(df_te)):
        lt = LT_LABELS[lt_pred[i]]
        tL = float(geom_pred[i, 0]); d = float(geom_pred[i, 1]); L = float(geom_pred[i, 2])
        N  = int(N_pred[i])
        try:
            sp_pred = tmm_spectrum_for(lt, tL, d, L, N)
            bm_pred = float(np.mean(tmm_alpha_for(lt, tL, d, L, N)))
        except Exception:
            sp_pred = np.full(len(FREQ_HZ), np.nan); bm_pred = np.nan
        r = df_te.iloc[i]
        rec = {'t_over_L_pred': tL, 'd_mm_pred': d, 'L_mm_pred': L, 'N_pred': N, 'lt_pred': lt,
               't_over_L_true': r['t_over_L'], 'd_mm_true': r['d_mm'], 'L_mm_true': r['L_mm'],
               'N_true': int(r['N']), 'lt_true': r['lattice_type'],
               'band_mean_target': r['band_mean_alpha'], 'band_mean_verified': bm_pred}
        for j, freq in enumerate(FREQ_HZ):
            rec[f'alpha_{int(freq)}hz_target']   = r[f'alpha_{int(freq)}hz']
            rec[f'alpha_{int(freq)}hz_verified'] = sp_pred[j]
        rows.append(rec)
    df_r = pd.DataFrame(rows)
    print(f'\n── Verification ({tag}) ───────────────────────────────────────')
    valid = df_r.dropna(subset=['band_mean_verified'])
    bm_err = np.abs(valid['band_mean_target'] - valid['band_mean_verified'])
    bm_r2  = r2_score(valid['band_mean_target'], valid['band_mean_verified'])
    print(f'  band-mean α:  MAE={bm_err.mean():.4f}  MRE={(bm_err/valid["band_mean_target"].abs().clip(1e-9)).mean()*100:.1f}%  R²={bm_r2:.3f}')
    # per-frequency
    print('  Per-frequency α (TMM of pred geom vs target):')
    for f in FREQ_HZ:
        tcol, vcol = f'alpha_{int(f)}hz_target', f'alpha_{int(f)}hz_verified'
        m = df_r[[tcol, vcol]].dropna()
        if len(m):
            mae = (m[tcol] - m[vcol]).abs().mean()
            r2  = r2_score(m[tcol], m[vcol])
            print(f'    {int(f):>5d} Hz: MAE={mae:.4f}  R²={r2:.3f}  (n={len(m)})')
    print(f'  Lattice acc:  {(df_r["lt_pred"] == df_r["lt_true"]).mean()*100:.1f}%')
    print(f'  N exact acc:  {(df_r["N_pred"] == df_r["N_true"]).mean()*100:.1f}%')
    return df_r

df_ver_single = verify_pipeline(geom_pred, N_pred, lt_pred, df_test, tag='single-seed + refine-all')
df_ver_single.to_csv(OUT_DIR/'inversion_verification_single.csv', index=False)

RUN_ENSEMBLE = True

if RUN_ENSEMBLE:
    print('=== Stage 4: Ensemble training (seeds 123, 7) ===')
    inv_seeds = [inv_main]
    for s in (123, 7):
        t0 = time.time()
        m, h, b = train_inversion_v4(seed=s, fwd_mech=fwd_mech, fwd_alpha=fwd_alpha, w_N_t=w_N_t)
        print(f'  seed {s}: best={b:.4f} ({(time.time()-t0)/60:.1f} min)')
        torch.save(m.state_dict(), OUT_DIR/f'inversion_net_seed{s}.pt')
        with open(OUT_DIR/f'hist_seed{s}.json', 'w') as f: json.dump(h, f)
        inv_seeds.append(m)
    print('=== Stage 5: Ensemble inference + refine-all ===')
    geom_pred_e, N_pred_e, lt_pred_e = evaluate_full_pipeline_v4(inv_seeds, df_test, refine=True, tag='ensemble')
    df_ver_ens = verify_pipeline(geom_pred_e, N_pred_e, lt_pred_e, df_test, tag='3-seed ensemble + refine-all')
    df_ver_ens.to_csv(OUT_DIR/'inversion_verification_ensemble.csv', index=False)
else:
    inv_seeds = [inv_main]
    df_ver_ens = df_ver_single.copy()
    geom_pred_e, N_pred_e, lt_pred_e = geom_pred, N_pred, lt_pred

def parity_plot(df_ver, geom_pred, N_pred, lt_pred, df_te, tag, save_path):
    """plot geom/T/alpha parity and N/lattice confusion matrices to a figure"""
    tp = df_te[['t_over_L', 'd_mm', 'L_mm']].values
    pp = np.column_stack([geom_pred[:, 0], geom_pred[:, 1], geom_pred[:, 2]])
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)
    labs = ['t/L', 'd (mm)', 'L (mm)']; cols = ['#2196F3','#4CAF50','#FF9800']
    for i in range(3):
        ax = fig.add_subplot(gs[0, i]); t2, p2 = tp[:, i], pp[:, i]
        mn, mx = min(t2.min(), p2.min()), max(t2.max(), p2.max())
        ax.scatter(t2, p2, s=3, alpha=0.25, color=cols[i], rasterized=True)
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
        ax.set_xlabel(f'True {labs[i]}'); ax.set_ylabel(f'Pred {labs[i]}')
        ax.set_title(f'{labs[i]}\nMAE={np.mean(np.abs(t2-p2)):.3f}  R²={r2_score(t2,p2):.3f}', fontsize=9)
    # T derived (compare against v3 plot)
    ax = fig.add_subplot(gs[0, 3])
    T_true = df_te['N'].values * df_te['L_mm'].values
    T_pred = N_pred * geom_pred[:, 2]
    ax.scatter(T_true, T_pred, s=3, alpha=0.25, color='#9C27B0', rasterized=True)
    mn, mx = T_true.min(), T_true.max()
    ax.plot([mn, mx],[mn, mx], 'k--', lw=1)
    ax.set_xlabel('True T = N·L (mm)'); ax.set_ylabel('Pred T (mm)')
    ax.set_title(f'T = N·L (derived)\nMAE={np.mean(np.abs(T_true-T_pred)):.1f}  R²={r2_score(T_true, T_pred):.3f}', fontsize=9)
    # alpha @ 1000 Hz
    ax = fig.add_subplot(gs[1, 0])
    valid = df_ver.dropna(subset=['alpha_1000hz_verified'])
    ax.scatter(valid['alpha_1000hz_target'], valid['alpha_1000hz_verified'], s=3, alpha=0.3, color='#7E57C2', rasterized=True)
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel('True α @ 1000 Hz'); ax.set_ylabel('Pred α @ 1000 Hz (TMM-verified)')
    r2_a = r2_score(valid['alpha_1000hz_target'], valid['alpha_1000hz_verified'])
    ax.set_title(f'α @ 1000 Hz\nR²={r2_a:.3f}', fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1)
    # band-mean alpha
    ax = fig.add_subplot(gs[1, 1])
    valid_bm = df_ver.dropna(subset=['band_mean_verified'])
    ax.scatter(valid_bm['band_mean_target'], valid_bm['band_mean_verified'], s=3, alpha=0.3, color='#26A69A', rasterized=True)
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel('True band-mean α'); ax.set_ylabel('Pred band-mean α (TMM-verified)')
    r2_bm = r2_score(valid_bm['band_mean_target'], valid_bm['band_mean_verified'])
    ax.set_title(f'band-mean α (TMM)\nR²={r2_bm:.3f}', fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1)
    # N confusion matrix
    ax = fig.add_subplot(gs[1, 2])
    n_idx_true = np.array([N_VALUES.index(int(x)) for x in df_te['N'].values])
    n_idx_pred = np.array([N_VALUES.index(int(x)) for x in N_pred])
    cmN = confusion_matrix(n_idx_true, n_idx_pred, labels=list(range(len(N_VALUES))))
    ax.imshow(cmN, cmap='Blues')
    ax.set_xticks(range(len(N_VALUES))); ax.set_yticks(range(len(N_VALUES)))
    ax.set_xticklabels(N_VALUES); ax.set_yticklabels(N_VALUES)
    ax.set_xlabel('Predicted N'); ax.set_ylabel('True N')
    accN = (n_idx_true == n_idx_pred).mean() * 100
    ax.set_title(f'N classification\nacc={accN:.1f}%', fontsize=9)
    for r in range(len(N_VALUES)):
        for c in range(len(N_VALUES)):
            ax.text(c, r, str(cmN[r,c]), ha='center', va='center',
                    color='white' if cmN[r,c]>cmN.max()/2 else 'black', fontsize=10)
    # lattice confusion matrix
    ax = fig.add_subplot(gs[1, 3])
    tlt = df_te['lt_class'].values; plt2 = lt_pred
    cm = confusion_matrix(tlt, plt2, labels=[0,1,2])
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LT_LABELS, rotation=15); ax.set_yticklabels(LT_LABELS)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Lattice type\nacc={(tlt==plt2).mean()*100:.1f}%', fontsize=9)
    for r in range(3):
        for c in range(3):
            ax.text(c, r, str(cm[r,c]), ha='center', va='center',
                    color='white' if cm[r,c]>cm.max()/2 else 'black', fontsize=10)
    fig.suptitle(f'InversionNet v4 — {tag}', fontsize=13, fontweight='bold', y=0.99)
    fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  saved → {save_path}')

parity_plot(df_ver_single, geom_pred, N_pred, lt_pred, df_test,
            tag='single seed + refine-all', save_path=OUT_DIR/'fig_inversion_parity_single.png')
if RUN_ENSEMBLE:
    parity_plot(df_ver_ens, geom_pred_e, N_pred_e, lt_pred_e, df_test,
                tag='3-seed ensemble + refine-all', save_path=OUT_DIR/'fig_inversion_parity_ensemble.png')

# loss curves
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].semilogy(hist_main['train_total'], label='train total')
axes[0].semilogy(hist_main['val_geom'],    label='val', ls='--')
axes[0].axvline(PREWARM_EPOCHS, color='gray', ls=':', label='pre-warm end')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Total / Val'); axes[0].legend()
axes[1].semilogy(hist_main['train_geom'],    label='geom')
axes[1].semilogy(hist_main['train_pmech'],   label='physics-mech')
axes[1].semilogy(hist_main['train_palpha'],  label='physics-alpha')
axes[1].semilogy(hist_main['train_pcen'],    label='centroid')
axes[1].axvline(PREWARM_EPOCHS, color='gray', ls=':')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].set_title('Components'); axes[1].legend()
axes[2].plot([100*x for x in hist_main['train_N_acc']], label='train N acc')
axes[2].plot([100*x for x in hist_main['val_N_acc']],   label='val N acc', ls='--')
axes[2].axvline(PREWARM_EPOCHS, color='gray', ls=':')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Accuracy %'); axes[2].set_title('N classification'); axes[2].legend(); axes[2].set_ylim(0, 105)
fig.tight_layout(); fig.savefig(OUT_DIR/'fig_training_loss.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('  loss curves saved')

# v2/v3 baselines, hard-coded from earlier runs
v2_baseline = {
    'tL_R2': 0.998, 'd_R2': 0.451, 'L_R2': 0.507, 'T_R2': 0.296,
    'N_acc_pct': 33.9, 'lt_acc_pct': 83.8,
    'alpha_R2_bandmean': 0.828, 'alpha_MRE_bandmean_pct': 28.8,
    'alpha_R2_per_freq_mean': float('nan'),
}
v3_baseline = {
    'tL_R2': 0.9994, 'd_R2': 0.794, 'L_R2': 0.903, 'T_R2': 0.244,
    'N_acc_pct': 54.2, 'lt_acc_pct': 99.6,
    'alpha_R2_bandmean': 0.984, 'alpha_MRE_bandmean_pct': 7.6,
    'alpha_R2_per_freq_mean': 0.801,
}

def summarise_v4(df_ver, geom_pred, N_pred, lt_pred, df_te):
    """collect the v4 geom R-squared, N/lattice acc and alpha metrics into a dict"""
    tp = df_te[['t_over_L', 'd_mm', 'L_mm']].values
    pp = np.column_stack([geom_pred[:, 0], geom_pred[:, 1], geom_pred[:, 2]])
    T_true = df_te['N'].values * df_te['L_mm'].values
    T_pred = N_pred * geom_pred[:, 2]
    out = {
        'tL_R2': r2_score(tp[:,0], pp[:,0]),
        'd_R2':  r2_score(tp[:,1], pp[:,1]),
        'L_R2':  r2_score(tp[:,2], pp[:,2]),
        'T_R2':  r2_score(T_true, T_pred),
        'N_acc_pct':  (df_te['N'].values == N_pred).mean() * 100,
        'lt_acc_pct': (df_te['lt_class'].values == lt_pred).mean() * 100,
    }
    valid = df_ver.dropna(subset=['band_mean_verified'])
    out['alpha_R2_bandmean']      = r2_score(valid['band_mean_target'], valid['band_mean_verified'])
    out['alpha_MRE_bandmean_pct'] = float((np.abs(valid['band_mean_target']-valid['band_mean_verified'])
                                           / valid['band_mean_target'].abs().clip(1e-9)).mean() * 100)
    # per-freq mean R^2
    pf = []
    for f in FREQ_HZ:
        tcol, vcol = f'alpha_{int(f)}hz_target', f'alpha_{int(f)}hz_verified'
        m = df_ver[[tcol, vcol]].dropna()
        if len(m): pf.append(r2_score(m[tcol], m[vcol]))
    out['alpha_R2_per_freq_mean'] = float(np.mean(pf)) if pf else float('nan')
    return out

summary_single = summarise_v4(df_ver_single, geom_pred, N_pred, lt_pred, df_test)
rows = [['metric', 'v2 baseline', 'v3 (corrected)', 'v4 single+refine']]
for k in v3_baseline:
    rows.append([k, v2_baseline.get(k, ''), v3_baseline[k], summary_single.get(k, '')])
if RUN_ENSEMBLE:
    summary_ens = summarise_v4(df_ver_ens, geom_pred_e, N_pred_e, lt_pred_e, df_test)
    rows[0].append('v4 ensemble+refine')
    for r in rows[1:]:
        r.append(summary_ens.get(r[0], ''))

comp_df = pd.DataFrame(rows[1:], columns=rows[0])
print(comp_df.to_string(index=False))
comp_df.to_csv(OUT_DIR/'comparison_summary.csv', index=False)

print(f"\nDone. Results written to {OUT_DIR}")
