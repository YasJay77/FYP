"""Simple-cubic (SC) plate lattice voxel generator.

Three orthogonal mid-plane plates, grown from a 1-voxel skeleton by binary
dilation (each iteration is a distinct geometry, so it doubles as the dataset's
thickness knob).
"""

import numpy as np
import scipy.ndimage


def gen_sc(resolution: int = 60, t_over_L: float = 0.10,
           dtype=np.uint8) -> np.ndarray:
    """voxelise one SC unit cell by dilating a 1-voxel mid-plane skeleton to thickness t/L"""
    N = resolution

    # 1-voxel skeleton: three orthogonal mid-planes
    vox = np.zeros((N, N, N), dtype=bool)
    mid = N // 2

    vox[mid, :, :] = True   # perp to x
    vox[:, mid, :] = True   # perp to y
    vox[:, :, mid] = True   # perp to z

    n_iter = max(1, round(t_over_L * N / 2))

    # 6-connected element grows evenly along each axis
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    vox = scipy.ndimage.binary_dilation(vox, structure=struct,
                                        iterations=n_iter)

    return vox.astype(dtype)


def effective_t_over_L(resolution: int, t_over_L: float) -> float:
    """actual t/L the dilation lands on at this resolution (it's quantised)"""
    N = resolution
    n_iter = max(1, round(t_over_L * N / 2))
    # central plate is 2*n_iter + 1 voxels thick
    return (2 * n_iter + 1) / N


def unique_geometries(resolution: int,
                      t_min: float = 0.03,
                      t_max: float = 0.18,
                      n_samples: int = 100) -> list:
    """distinct n_iter values over t/L in [t_min, t_max] at this resolution"""
    N = resolution
    seen = set()
    for t in np.linspace(t_min, t_max, n_samples):
        n = max(1, round(t * N / 2))
        seen.add(n)
    return sorted(seen)


if __name__ == "__main__":
    import hashlib

    print("SC plate lattice — dilation generator")
    print("=" * 50)

    for res in [60, 100, 150]:
        geoms = unique_geometries(res)
        print(f"  resolution={res:3d}  unique geometries: {len(geoms)}"
              f"  n_iter range: {geoms[0]}–{geoms[-1]}")

    print()
    print("Geometry hashes at resolution=100:")
    for t in [0.03, 0.06, 0.10, 0.14, 0.18]:
        v = gen_sc(resolution=100, t_over_L=t)
        h = hashlib.md5(v.tobytes()).hexdigest()[:8]
        n = max(1, round(t * 100 / 2))
        actual_t = effective_t_over_L(100, t)
        print(f"  t/L={t:.2f}  n_iter={n:2d}  actual_t/L={actual_t:.3f}  "
              f"solid_frac={v.mean():.4f}  hash={h}")