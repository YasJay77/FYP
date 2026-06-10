# MEng Plate Lattice Inverse Design Framework

This repository contains the code and datasets to reproduce the project's results.

---

## 1. Overview

The project builds an inverse-design framework for **plate lattices** that couples:

1. an acoustic **Transfer Matrix Method (TMM)** model of micro-perforated plate-lattice
   absorbers (`tmm_script.py`),
2. a **numerical homogenisation** of the mechanical response (effective stiffness tensor
   `C11, C12, C44`) from voxel models (`hompy/`), distilled into a fast **surrogate**
   (`train_mechanical_surrogate.py`), and
3. a neural **inversion network** that maps a target absorption spectrum back to a lattice
   geometry (`train_inversion.py`).

Three plate-lattice families are studied: **SC** (simple-cubic, 4 pores/cell, porosity
referenced to (L−t)²), **FCC** (nodal-pore, 4 pores/cell, porosity referenced to L²), and
**FCC_face** (face-pore, 16 pores/cell, porosity referenced to L²). The optimiser operates
in **property space** (stiffness vs absorption), not geometry space.

Primary references: Li et al. (2021), *Small* **17**, 2100336; Liu et al. (2022),
*Materials & Design* **223**, 111122.

---

## 2. Directory layout

```
.
├── README.md                     (this file)
├── requirements.txt              pinned dependencies
├── hompy/                        numerical homogenisation package
├── lattice/                      voxel generators (SC, FCC, FCC-face)
├── scripts/                      pipeline + analysis scripts (see §3)
│
└── results/
    ├── data/                     RAW mechanical-FEA output  (§4.1)
    ├── property_datasets/        PROCESSED property, Pareto & spectral datasets  (§4.2)
    ├── inversion_results/        inversion verification + comparison  (§4.3)
    └── pore_investigation/       pore-validity sub-study (analysis scripts + knockdown-fit inputs)
```

---

## 3. Code

### Core engine and models
| File | Purpose |
|------|---------|
| `scripts/tmm_script.py` | TMM acoustic engine (canonical Maa micro-perforated-plate formulation; absorption spectra). |
| `lattice/*.py` | Voxel generators for SC, FCC and FCC-face plate lattices. |
| `hompy/` | Numerical homogenisation — effective elastic tensor (`C11, C12, C44`) from a voxel unit cell. |

### Pipeline (recommended execution order)

1. `scripts/run_sc_fea_exact_tL.py` — mechanical homogenisation FEA for SC → `results/data/sc_mech_res100_dilation.csv`
2. `scripts/run_fcc_fea_exact_tL.py` — mechanical homogenisation FEA for FCC → `results/data/fcc_mech_res120_dilation.csv`
3. `scripts/train_mechanical_surrogate.py` — fit Gibson–Ashby / GPR / XGBoost surrogates
4. `scripts/regenerate_dataset.py` — SC+FCC acoustic dataset + merge with mechanical
5. `scripts/generate_fcc_facepore_dataset.py` — FCC_face acoustic + all-lattices dataset
6. `scripts/populate_fcc_facepore_mechanical.py` — fill `C11/C12/C44` for FCC_face
7. `scripts/run_optimisation.py` — two-objective (stiffness vs absorption) Pareto front
8. `scripts/augment_dataset_with_spectrum.py` — add the 16-point spectrum → spectral dataset (§4.2)
9. `scripts/train_inversion.py` — train the inversion network: forward surrogates, InversionNet, refine-and-select inference, and TMM verification (writes `results/inversion_results/`)

### Supporting scripts
`generate_sc_dataset.py` / `generate_fcc_dataset.py` / `run_mechanical_datasets.py`
(an alternate, higher-resolution dilation-based FEA generator),
`pore_correction.py` (pore-knockdown correction for the mechanical surrogate),
`run_optimisation_continuous.py` and `run_optimisation_heterogeneous.py` (the continuous
gradient-based search used for the multi-layer / heterogeneous Pareto extension),
`regenerate_appendix_pareto.py` (appendix Pareto table → `results/inversion_results/appendix_pareto.csv`),
`run_convergence_study.py` (HomPy mesh-convergence study → `results/data/hompy_convergence.csv`).
`results/pore_investigation/*.py` hold the pore-validity sub-study analysis.

---

## 4. Data

All data are CSV. The raw mechanical FEA (§4.1) is the only expensive-to-regenerate input;
every other table can be rebuilt from it via the §3 pipeline.

### 4.1 Raw computational data — mechanical FEA (`results/data/`)
Homogenised stiffness from HomPy voxel models. Columns: `t_over_L, rho, voxel_hash,
C11, C12, C44, time_s, direct_solution, resolution, E, nu, lx, ly, lz, rho_(liu|approx),
discretization_warning`.

| File | Rows | Description |
|------|-----:|-------------|
| `sc_mech_res100_dilation.csv` | 20 | SC stiffness vs t/L, resolution 100, dilation correction. |
| `fcc_mech_res120_dilation.csv` | 20 | FCC stiffness vs t/L, resolution 120. |
| `fcc_mech_N30_complete_ref.csv` | 13 | FCC reference run, N=30 voxels (convergence check). |
| `fcc_mech_N60_partial_ref.csv` | 8 | FCC reference run, N=60 voxels (partial, convergence check). |
| `hompy_convergence.csv` | 10 | Mesh-convergence study: C11/C12/C44 vs voxel resolution {30,50,70,90,110} for SC and FCC at fixed t/L=0.10. Produced by `run_convergence_study.py`; own schema (`lattice, res, n_iter, eff_tL, rho, C11, C12, C44, time_s, dC11, dC12, dC44, max_d`). The `dC*`/`max_d` columns are the per-step relative changes quoted in the report. |

### 4.2 Processed data — property, Pareto & spectral datasets (`results/property_datasets/`)
Build chain: base acoustic (18,000) → validity-filtered (16,200) → + mechanical (16,200)
→ + FCC_face (5,520) → **all-lattices master (21,720)** → + 16-point spectrum (spectral set).

| File | Rows | Description |
|------|-----:|-------------|
| `property_dataset.csv` | 18,000 | SC+FCC acoustic only (`band_mean_alpha`, `alpha_1000hz`). |
| `property_dataset_filtered.csv` | 16,200 | Same, after geometric-validity filtering. |
| `property_dataset_with_mechanical.csv` | 16,200 | SC+FCC with surrogate `C11/C12/C44`. |
| `property_dataset_fcc_face.csv` | 5,520 | FCC-face acoustic. |
| `property_dataset_all_lattices.csv` | **21,720** | **Master dataset** — all three families, acoustic + mechanical. |
| `pareto_front.csv` | 22 | Pareto-optimal designs (stiffness `C11` vs `band_mean_alpha`, normalised). |
| `property_dataset_spectral.csv` | 21,720 | Master dataset + full 16-point spectrum (`alpha_500hz` … `alpha_2000hz`); the **input to the inversion network**. Produced by `augment_dataset_with_spectrum.py`. |

Master-dataset columns: `t_over_L` (t/L), `d_mm` (pore diameter), `L_mm` (cell size),
`N` (layers), `lattice_type` (SC/FCC/FCC_face), `band_mean_alpha` (mean absorption 500–2000 Hz),
`alpha_1000hz`, `C11/C12/C44` (homogenised stiffness), `surrogate_method` (surrogate used).

### 4.3 Inversion results (`results/inversion_results/`)
Produced by `scripts/train_inversion.py` (verification CSVs) and `regenerate_appendix_pareto.py`
(appendix table).

| File | Rows | Description |
|------|-----:|-------------|
| `inversion_verification_single.csv` | 3,258 | Single-model inversion: predicted vs true geometry + target vs TMM-verified spectra. |
| `inversion_verification_ensemble.csv` | 3,258 | Ensemble inversion: predicted vs true + verified spectra. |
| `comparison_summary.csv` | 9 | Headline metrics across model variants (single, ensemble). |
| `appendix_pareto.csv` | 23 | Pareto targets inverted and TMM-verified (appendix table). |

---

## 5. Environment & reproduction

Python ≥ 3.11. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
# PyTorch CPU build is sufficient:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Key packages: `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-learn`, `xgboost` (optional,
surrogate comparison), `torch` (inversion network). `pypardiso` is an optional faster sparse
solver for the homogenisation FEA at high resolution. Physical constants used throughout:
`C0 = 343.0 m/s`, `RHO0 = 1.2 kg/m³`, `MU = 1.81e-5 Pa·s`, `Z0 = 411.6 Pa·s/m`.

Run the scripts from the repository root. Every script reads and writes under `results/` — raw
FEA in `results/data/`, datasets and intermediate surrogate models in
`results/property_datasets/`, and inversion outputs in `results/inversion_results/` — so the
pipeline chains end-to-end without any extra path setup.

`train_inversion.py` was developed on a GPU (~2 h on a T4); it runs on CPU but training is
slower. It reads `results/property_datasets/property_dataset_spectral.csv` and writes its
models, histories, figures and verification CSVs into `results/inversion_results/`.
