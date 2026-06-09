"""FCC face-pore (Type II) plate lattice voxel generator.

Same stellated-octahedron skeleton as fcc_plate_voxels, but each {111} plate is
split into 4 sub-triangles with a pore at each sub-triangle centroid -> 32 pores
per cell. For z-transmission the 4 upward-facing plates give 16 acoustically
active pores (TMM sigma = 16*pi*(d/2)^2 / L^2). Pores drilled along the plate
normal; all 32 centres sit inside [1/6, 5/6]^3 so no wrapping needed.
"""

import numpy as np
from lattice.fcc_plate_voxels import (
    stellated_plates,
    _plate_normal,
    _rasterize_plate,
    _drill_cylinder,
)

_PLATES = stellated_plates()    # same skeleton as the nodal-pore variant

# barycentric coords of the 4 sub-triangle centroids per plate
_BARY = np.array([
    [4.0/6, 1.0/6, 1.0/6],   # near vA
    [1.0/6, 4.0/6, 1.0/6],   # near vB
    [1.0/6, 1.0/6, 4.0/6],   # near vC
    [1.0/3, 1.0/3, 1.0/3],   # plate centroid
], dtype=np.float64)


def gen_fcc_facepore(
    resolution: int   = 100,
    t_over_L:   float = 0.10,
    d_over_L:   float = 0.075,
    dtype             = np.uint8,
) -> np.ndarray:
    """voxelise one FCC face-pore unit cell: skeleton + 32 pores at the sub-triangle centroids"""
    N    = int(resolution)
    tL   = float(t_over_L)
    dL   = float(d_over_L)

    # voxel centres in [0,1]^3
    ii, jj, kk = np.mgrid[0:N, 0:N, 0:N]
    pos = (np.stack([ii, jj, kk], axis=-1).astype(np.float64) + 0.5) / N

    t_half = tL / 2.0

    vox = np.zeros((N, N, N), dtype=bool)
    for plate_verts in _PLATES:
        vox |= _rasterize_plate(pos, plate_verts, t_half)

    # drill the 32 pores
    r_pore   = np.float32(dL / 2.0)
    half_len = np.float32(0.6 * tL)
    pos_f    = pos.astype(np.float32)

    for plate_verts in _PLATES:
        vA = plate_verts[0].astype(np.float32)
        vB = plate_verts[1].astype(np.float32)
        vC = plate_verts[2].astype(np.float32)

        e1     = vB - vA;  e2 = vC - vA
        normal = np.cross(e1, e2).astype(np.float32)
        normal /= float(np.linalg.norm(normal))

        for bc in _BARY:
            centre = (bc[0]*vA + bc[1]*vB + bc[2]*vC).astype(np.float32)
            _drill_cylinder(vox, pos_f, centre, normal, r_pore, half_len)

    return vox.astype(dtype)


def pore_centres(t_over_L: float = 0.10) -> np.ndarray:
    """all 32 pore centres in [0,1]^3 (t_over_L unused, kept for API parity)"""
    centres = []
    for plate_verts in _PLATES:
        vA, vB, vC = plate_verts
        for bc in _BARY:
            c = bc[0]*vA + bc[1]*vB + bc[2]*vC
            centres.append(c)
    return np.array(centres)    # (32, 3)


def plate_normals() -> np.ndarray:
    """unit normals for all 8 plates, (8,3)"""
    return np.array([_plate_normal(p) for p in _PLATES])


def upward_facing_mask() -> np.ndarray:
    """true where plate centroid z > 0.5; these carry the 16 z-active pores"""
    return np.array([p[:, 2].mean() > 0.5 for p in _PLATES])


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("FCC face-pore plate lattice (corrected stellated-octahedron skeleton)")
    print("=" * 68)

    normals  = plate_normals()
    uf_mask  = upward_facing_mask()
    centres  = pore_centres()

    print(f"\n8 {{111}} plates (corrected cube-corner vertices):")
    labels = ['T1-F1','T1-F2','T1-F3','T1-F4','T2-G1','T2-G2','T2-G3','T2-G4']
    for i, (pv, n, uf, lbl) in enumerate(zip(_PLATES, normals, uf_mask, labels)):
        centroid  = pv.mean(axis=0)
        direction = "up  " if uf else "down"
        print(f"  {lbl}: centroid=({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f})"
              f"  normal=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})  [{direction}]")

    up_plates = int(uf_mask.sum())
    up_pores  = up_plates * 4
    print(f"\nUpward-facing plates: {up_plates}  -> {up_pores} active pores")
    print(f"TMM expects 16 active pores -> {'OK' if up_pores == 16 else 'MISMATCH'}")

    print(f"\n32 pore centres (all should be in [1/6, 5/6]^3):")
    c_all = pore_centres()
    print(f"  min coord: {c_all.min():.4f}  (expect >= 1/6 = {1/6:.4f})")
    print(f"  max coord: {c_all.max():.4f}  (expect <= 5/6 = {5/6:.4f})")
    all_interior = (c_all.min() >= 1/6 - 1e-9) and (c_all.max() <= 5/6 + 1e-9)
    print(f"  All pore centres interior (no wrap needed): {all_interior}")

    print("\nSolid fractions at resolution=100:")
    from lattice.fcc_plate_voxels import gen_fcc
    print(f"  {'t/L':>6}  {'rho_face':>10}  {'rho_nodal':>11}  {'delta_rho':>11}")
    print("  " + "-" * 45)
    for tL in [0.03, 0.06, 0.10, 0.14, 0.18]:
        dL = 0.075
        v_face  = gen_fcc_facepore(resolution=100, t_over_L=tL, d_over_L=dL)
        v_nodal = gen_fcc(resolution=100, t_over_L=tL, d_over_L=dL)
        rf = float(v_face.mean()); rn = float(v_nodal.mean())
        print(f"  {tL:>6.2f}  {rf:>10.4f}  {rn:>11.4f}  {rn-rf:>+11.4f}")

    print("\nLiu 2022 Eq.6 sanity check (no pores, rR=0):")
    print(f"  {'t/L':>6}  {'rho_vox':>10}  {'rho_Liu':>10}  {'error_%':>9}")
    print("  " + "-" * 40)
    for tL in [0.03, 0.06, 0.10, 0.14, 0.18]:
        v   = gen_fcc(resolution=100, t_over_L=tL, d_over_L=0.0)
        rv  = float(v.mean())
        liu = 6.867*tL - 15.79*tL**2
        err = 100.0*(rv - liu)/liu
        flag = "  <-- STOP" if abs(err) > 10.0 else ""
        print(f"  {tL:>6.2f}  {rv:>10.4f}  {liu:>10.4f}  {err:>+9.2f}%{flag}")
