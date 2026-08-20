"""
Stage C: animation choreography data for the portrait layer.

- intro groups: ~60 random (non-spatial) groups for the once-off shimmer-in,
  verified with a spatial-evenness metric so groups read as scattered, not
  patch-by-patch.
- drift bands: ~94 bands for the loop's portrait phase, built from a noisy
  projection so the band boundary is organic rather than grid-like.
- travelers: ~900 dots per logo, id-matched across react -> flutter -> code
  -> react via optimal transport (Hungarian assignment) so each one takes the
  shortest individual path when it morphs.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

RNG = np.random.default_rng(619)

N_INTRO_GROUPS = 60
N_DRIFT_BANDS = 94
N_TRAVELERS = 900
DRIFT_FRACTION = 0.42
NOISE_SIGMA = 4.0


def build_intro_groups(dots):
    n = len(dots)
    order = RNG.permutation(n)
    group_id = np.empty(n, dtype=np.int32)
    group_id[order] = np.arange(n) % N_INTRO_GROUPS
    return group_id


def evenness_metric(dots, group_id):
    """Normalized distance of each group's centroid from the whole-set
    centroid, relative to the whole-set spread. A group scattered evenly
    across the portrait lands its centroid near the global one (small
    value); a group confined to one region (patch-by-patch reveal) shifts
    its centroid noticeably (large value)."""
    dots = dots.astype(np.float64)
    global_centroid = dots.mean(axis=0)
    global_scale = np.linalg.norm(dots.std(axis=0))
    offsets = []
    for g in range(group_id.max() + 1):
        mask = group_id == g
        if mask.sum() < 2:
            continue
        gc = dots[mask].mean(axis=0)
        offsets.append(np.linalg.norm(gc - global_centroid) / global_scale)
    return float(np.mean(offsets))


def build_drift_bands(dots, target_centroid):
    n = len(dots)
    centroid = dots.mean(axis=0)
    axis = target_centroid - centroid
    axis_norm = axis / (np.linalg.norm(axis) + 1e-9)

    noise = RNG.normal(0, NOISE_SIGMA, size=dots.shape)
    noisy = dots.astype(np.float64) + noise
    proj = noisy @ axis_norm

    edges = np.linspace(proj.min(), proj.max() + 1e-6, N_DRIFT_BANDS + 1)
    band_id = np.clip(np.digitize(proj, edges) - 1, 0, N_DRIFT_BANDS - 1)

    drift_vec = axis_norm * np.linalg.norm(axis) * DRIFT_FRACTION
    return band_id, drift_vec, axis_norm


def straight_boundary_metric(dots, band_id, axis_norm, n_bands):
    """Lower = organic boundary, higher = a straight (grid-like) edge.
    For each internal boundary, take dots within one band-width of it and
    measure how tightly the along-axis crossing point is pinned as a
    function of cross-axis position -- a hard linear cut pins it exactly
    (near-zero residual after detrending); noise smears it out."""
    proj = dots.astype(np.float64) @ axis_norm
    perp = np.array([-axis_norm[1], axis_norm[0]])
    cross = dots.astype(np.float64) @ perp

    residuals = []
    for b in range(n_bands - 1):
        near = np.isin(band_id, [b, b + 1])
        if near.sum() < 10:
            continue
        c, p = cross[near], proj[near]
        order = np.argsort(c)
        c, p = c[order], p[order]
        bins = np.linspace(c.min(), c.max() + 1e-6, 12)
        idx = np.clip(np.digitize(c, bins) - 1, 0, 10)
        local_std = []
        for i in range(11):
            vals = p[idx == i]
            if len(vals) > 3:
                local_std.append(vals.std())
        if local_std:
            residuals.append(np.mean(local_std))
    span = proj.max() - proj.min()
    return float(np.mean(residuals) / span) if residuals and span > 0 else 0.0


def sample_fixed(cloud, k=N_TRAVELERS):
    idx = RNG.choice(len(cloud), size=min(k, len(cloud)), replace=False)
    return cloud[idx].astype(np.float64)


def match_next(current, dst_cloud, k=N_TRAVELERS):
    """current: (k,2) traveler positions at the previous keyframe, index i is
    traveler i. Returns (k,2) positions at the next keyframe, still indexed
    by the same traveler i, chosen via optimal transport (Hungarian) against
    a fresh sample from dst_cloud so each traveler takes its shortest path."""
    dst = sample_fixed(dst_cloud, k)
    cost = np.linalg.norm(current[:, None, :] - dst[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    ordered = np.empty_like(dst)
    ordered[row] = dst[col]
    return ordered


def main():
    dark_dots = np.load("data/dark_dots.npy")
    react = np.load("data/logo_khanda.npy")
    flutter = np.load("data/logo_eternal.npy")
    code = np.load("data/logo_code.npy")

    group_id = build_intro_groups(dark_dots)
    evenness = evenness_metric(dark_dots, group_id)
    # sanity check: a deliberately patchy (spatially-clustered) grouping should score much worse
    order_by_x = np.argsort(dark_dots[:, 0] + dark_dots[:, 1] * 0.7)
    patchy_id = np.empty(len(dark_dots), dtype=np.int32)
    patchy_id[order_by_x] = np.arange(len(dark_dots)) * N_INTRO_GROUPS // len(dark_dots)
    patchy_evenness = evenness_metric(dark_dots, patchy_id)
    print(f"intro groups: {N_INTRO_GROUPS}, evenness = {evenness:.4f} (patchy control = {patchy_evenness:.4f})")
    np.save("data/intro_group_id.npy", group_id)

    react_centroid = react.mean(axis=0)
    band_id, drift_vec, axis_norm = build_drift_bands(dark_dots, react_centroid)
    straightness = straight_boundary_metric(dark_dots, band_id, axis_norm, N_DRIFT_BANDS)
    print(f"drift bands: {N_DRIFT_BANDS}, straight-boundary metric = {straightness:.4f} (organic ~0.01, grid ~0.17)")
    np.save("data/drift_band_id.npy", band_id)
    np.save("data/drift_vec.npy", drift_vec)

    t_react = sample_fixed(react)
    t_flutter = match_next(t_react, flutter)
    t_code = match_next(t_flutter, code)
    np.save("data/traveler_khanda.npy", t_react)
    np.save("data/traveler_eternal.npy", t_flutter)
    np.save("data/traveler_code.npy", t_code)
    d1 = np.linalg.norm(t_flutter - t_react, axis=1).mean()
    d2 = np.linalg.norm(t_code - t_flutter, axis=1).mean()
    print(f"travelers: {N_TRAVELERS}, mean hop react->flutter {d1:.1f}px, flutter->code {d2:.1f}px (grid units)")


if __name__ == "__main__":
    main()
