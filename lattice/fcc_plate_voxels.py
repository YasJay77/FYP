"""FCC nodal-pore plate lattice voxel generator.

Skeleton is the stellated octahedron (two interpenetrating tetrahedra -> 8 {111}
triangular plates), rasterised exactly via signed-distance + barycentric tests
(no dilation). One circular pore per plate, drilled along the plate normal at the
centroid.
"""

import numpy as np


def stellated_plates():
    """8 {111} triangular plates as (3,3) vertex arrays in [0,1]^3 (cube corners)"""
    A = np.array([0., 0., 0.]);  B = np.array([1., 1., 0.])
    C = np.array([1., 0., 1.]);  D = np.array([0., 1., 1.])
    E = np.array([1., 0., 0.]);  F = np.array([0., 1., 0.])
    G = np.array([0., 0., 1.]);  H = np.array([1., 1., 1.])
    return [
        # T1 faces
        np.array([A, B, C]),
        np.array([A, B, D]),
        np.array([A, C, D]),
        np.array([B, C, D]),
        # T2 faces
        np.array([E, F, G]),
        np.array([E, F, H]),
        np.array([E, G, H]),
        np.array([F, G, H]),
    ]


_PLATES = stellated_plates()   # reused by fcc_facepore_voxels


def _plate_normal(verts):
    """unit outward normal of a triangular plate"""
    e1 = verts[1] - verts[0]
    e2 = verts[2] - verts[0]
    n  = np.cross(e1, e2)
    return n / np.linalg.norm(n)


def _rasterize_plate(pos, verts, t_half):
    """mask of voxels inside both the plate slab and its triangular extent"""
    vA = verts[0];  vB = verts[1];  vC = verts[2]
    e1 = vB - vA;  e2 = vC - vA
    n  = np.cross(e1, e2);  n = n / np.linalg.norm(n)

    # signed distance from plane, then slab test
    d_plane = np.einsum('...i,i->...', pos - vA, n)   # (N,N,N)
    in_slab = np.abs(d_plane) <= t_half

    if not in_slab.any():
        return in_slab

    # project residual onto the plate plane
    w = pos - vA - d_plane[..., np.newaxis] * n        # (N,N,N,3)

    # gram matrix
    d00   = float(np.dot(e1, e1))
    d01   = float(np.dot(e1, e2))
    d11   = float(np.dot(e2, e2))
    denom = d00 * d11 - d01 * d01

    d20 = np.einsum('...i,i->...', w, e1)              # (N,N,N)
    d21 = np.einsum('...i,i->...', w, e2)              # (N,N,N)

    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom

    in_tri = (u >= 0) & (v >= 0) & (u + v <= 1)
    return in_slab & in_tri


def _drill_cylinder(vox, pos, centre, normal, r_pore, half_len):
    """carve a cylindrical void from vox in-place (solid = True)"""
    disp   = pos - centre                                      # (N,N,N,3)
    axial  = np.einsum('...i,i->...', disp, normal)            # (N,N,N)
    v_rad  = disp - axial[..., np.newaxis] * normal            # (N,N,N,3)
    radial = np.sqrt((v_rad * v_rad).sum(axis=-1))             # (N,N,N)
    inside = (radial <= r_pore) & (np.abs(axial) <= half_len)
    vox[inside] = False


def gen_fcc(resolution=60, t_over_L=0.10, d_over_L=0.0,
            dtype=np.uint8) -> np.ndarray:
    """voxelise one FCC unit cell; with d_over_L>0 drill one pore per plate at the centroid"""
    N   = int(resolution)
    tL  = float(t_over_L)
    dL  = float(d_over_L)

    # voxel centres in [0,1]^3
    ii, jj, kk = np.mgrid[0:N, 0:N, 0:N]
    pos = (np.stack([ii, jj, kk], axis=-1).astype(np.float64) + 0.5) / N

    t_half = tL / 2.0
    vox    = np.zeros((N, N, N), dtype=bool)

    for plate_verts in _PLATES:
        vox |= _rasterize_plate(pos, plate_verts, t_half)

    # nodal pores at plate centroids
    if dL > 0.0:
        r_pore   = np.float32(dL / 2.0)
        half_len = np.float32(0.6 * tL)     # >0.5
        pos_f    = pos.astype(np.float32)

        for plate_verts in _PLATES:
            vA, vB, vC = plate_verts
            centre = ((vA + vB + vC) / 3.0).astype(np.float32)

            e1 = vB - vA;  e2 = vC - vA
            n  = np.cross(e1, e2).astype(np.float32)
            n /= float(np.linalg.norm(n))

            _drill_cylinder(vox, pos_f, centre, n, r_pore, half_len)

    return vox.astype(dtype)


# Convenience functions

def void_fraction(resolution: int, t_over_L: float,
                  d_over_L: float = 0.0) -> float:
    """air void fraction (1 - solid)"""
    return 1.0 - float(gen_fcc(resolution, t_over_L, d_over_L).mean())


def unique_geometries(resolution: int,
                      t_min: float = 0.03,
                      t_max: float = 0.18,
                      n_samples: int = 100) -> list:
    """unique t/L values; exact rasteriser means every value is distinct, kept for API parity"""
    return sorted(set(np.linspace(t_min, t_max, n_samples)))


def plate_normals() -> np.ndarray:
    """unit normals for all 8 plates, (8,3)"""
    return np.array([_plate_normal(p) for p in _PLATES])


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("FCC plate lattice (stellated octahedron) — corrected geometry")
    print("=" * 65)

    print("\nPlate normals and centroids:")
    A=np.array([0.,0.,0.]); B=np.array([1.,1.,0.])
    C=np.array([1.,0.,1.]); D=np.array([0.,1.,1.])
    E=np.array([1.,0.,0.]); F=np.array([0.,1.,0.])
    G=np.array([0.,0.,1.]); H=np.array([1.,1.,1.])
    labels = ['T1-F1','T1-F2','T1-F3','T1-F4','T2-G1','T2-G2','T2-G3','T2-G4']
    for label, plate in zip(labels, _PLATES):
        n = _plate_normal(plate)
        c = plate.mean(axis=0)
        edge = np.linalg.norm(plate[1] - plate[0])
        print(f"  {label}: centroid={c.round(4)}  normal={n.round(4)}  edge={edge:.4f}")

    print("\nSolid fractions at resolution=100 (no pores):")
    print(f"  {'t/L':>6}  {'rho_vox':>10}  {'rho_Liu2022':>13}  {'error_%':>9}")
    print("  " + "-" * 45)
    for tL in [0.03, 0.06, 0.09, 0.10, 0.12, 0.15, 0.18]:
        v    = gen_fcc(resolution=100, t_over_L=tL)
        rho  = float(v.mean())
        liu  = 6.867 * tL - 15.79 * tL**2
        err  = 100.0 * (rho - liu) / liu
        flag = "  <-- STOP" if abs(err) > 10.0 else ""
        print(f"  {tL:>6.3f}  {rho:>10.4f}  {liu:>13.4f}  {err:>+9.2f}%{flag}")

    print("\nWith pores (d/L=0.075):")
    for tL in [0.06, 0.10, 0.15]:
        v_n = gen_fcc(100, tL, 0.075)
        v_s = gen_fcc(100, tL, 0.0)
        print(f"  t/L={tL:.2f}  rho_nodal={v_n.mean():.4f}  "
              f"rho_skeleton={v_s.mean():.4f}  "
              f"delta={v_s.mean()-v_n.mean():.4f}")
