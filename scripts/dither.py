"""
Stage A: portrait -> 1-bit dithered dot grid (dark mode + light mode).

Source of truth: writes .npy files under data/. The SVG is a derived artifact.
"""
import math
from collections import deque
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy.ndimage import (binary_closing, binary_fill_holes, label,
                           binary_erosion, binary_opening)

GRID_W, GRID_H = 300, 340
SUPER = 3  # supersample factor for enhancement + segmentation quality


def load_and_crop(path):
    im = Image.open(path).convert("RGB")
    # The source photo carries its own pale rounded border; left in, it survives
    # segmentation as a straight dotted rectangle framing the subject.
    inset = int(min(im.size) * 0.035)
    im = im.crop((inset, inset, im.size[0] - inset, im.size[1] - inset))
    w, h = im.size
    target_aspect = GRID_W / GRID_H  # 0.882
    if w / h > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        im = im.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        im = im.crop((0, top, w, top + new_h))
    return im.resize((GRID_W * SUPER, GRID_H * SUPER), Image.LANCZOS)


def enhance(im):
    im = ImageOps.autocontrast(im, cutoff=1)
    im = im.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    im = ImageEnhance.Contrast(im).enhance(1.3)
    return im


def segment_foreground(rgb_small, tol=14.0):
    """rgb_small: (H,W,3) float, at grid resolution. Background varies smoothly
    (gradient) so we region-grow it from the border by LOCAL pixel-to-pixel
    similarity rather than distance-from-one-reference-color -- that's what
    correctly separates a near-black suit/hair from a lit gray background
    that a single global threshold cannot."""
    h, w, _ = rgb_small.shape
    visited = np.zeros((h, w), dtype=bool)
    bg = np.zeros((h, w), dtype=bool)
    tol2 = tol * tol

    # Seed ONLY from border pixels that actually look like the backdrop.
    # The subject's near-black suit runs off the bottom edge, so seeding every
    # border pixel lets the fill walk straight up through the suit and eat the
    # body. Reference color comes from the top corners, which are always backdrop.
    patch = max(3, min(h, w) // 20)
    ref = np.median(
        np.concatenate([
            rgb_small[:patch, :patch].reshape(-1, 3),
            rgb_small[:patch, -patch:].reshape(-1, 3),
        ], axis=0),
        axis=0,
    )
    seed_tol2 = (tol * 4.0) ** 2

    dq = deque()

    def try_seed(y, x):
        dr = rgb_small[y, x] - ref
        if dr @ dr < seed_tol2:
            dq.append((y, x))

    for x in range(w):
        try_seed(0, x)
        try_seed(h - 1, x)
    for y in range(h):
        try_seed(y, 0)
        try_seed(y, w - 1)

    while dq:
        y, x = dq.popleft()
        if visited[y, x]:
            continue
        visited[y, x] = True
        bg[y, x] = True
        cr, cg, cb = rgb_small[y, x]
        if x + 1 < w and not visited[y, x + 1]:
            nr, ng, nb = rgb_small[y, x + 1]
            if (nr - cr) ** 2 + (ng - cg) ** 2 + (nb - cb) ** 2 < tol2:
                dq.append((y, x + 1))
        if x - 1 >= 0 and not visited[y, x - 1]:
            nr, ng, nb = rgb_small[y, x - 1]
            if (nr - cr) ** 2 + (ng - cg) ** 2 + (nb - cb) ** 2 < tol2:
                dq.append((y, x - 1))
        if y + 1 < h and not visited[y + 1, x]:
            nr, ng, nb = rgb_small[y + 1, x]
            if (nr - cr) ** 2 + (ng - cg) ** 2 + (nb - cb) ** 2 < tol2:
                dq.append((y + 1, x))
        if y - 1 >= 0 and not visited[y - 1, x]:
            nr, ng, nb = rgb_small[y - 1, x]
            if (nr - cr) ** 2 + (ng - cg) ** 2 + (nb - cb) ** 2 < tol2:
                dq.append((y - 1, x))

    fg = ~bg
    fg = binary_closing(fg, structure=np.ones((5, 5)))
    fg = binary_fill_holes(fg)
    # Opening knocks off the single-pixel staircase jags the flood fill leaves
    # along the shoulder line, which otherwise show up as blocky steps.
    fg = binary_opening(fg, structure=np.ones((3, 3)))
    lbl, n = label(fg)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        largest = sizes.argmax()
        fg = lbl == largest
    return fg


def serpentine_floyd_steinberg(gray):
    """gray: (H,W) float in [0,1], 1=white/bright, 0=black/dark.
    Returns ink mask (H,W) bool: True where a dot is drawn (dark pixel after threshold)."""
    h, w = gray.shape
    buf = gray.astype(np.float64).copy()
    ink = np.zeros((h, w), dtype=bool)
    for y in range(h):
        left_to_right = (y % 2 == 0)
        xs = range(w) if left_to_right else range(w - 1, -1, -1)
        for x in xs:
            old = buf[y, x]
            new = 1.0 if old > 0.5 else 0.0
            ink[y, x] = (new == 0.0)
            err = old - new
            if left_to_right:
                if x + 1 < w:
                    buf[y, x + 1] += err * 7 / 16
                if y + 1 < h:
                    if x - 1 >= 0:
                        buf[y + 1, x - 1] += err * 3 / 16
                    buf[y + 1, x] += err * 5 / 16
                    if x + 1 < w:
                        buf[y + 1, x + 1] += err * 1 / 16
            else:
                if x - 1 >= 0:
                    buf[y, x - 1] += err * 7 / 16
                if y + 1 < h:
                    if x + 1 < w:
                        buf[y + 1, x + 1] += err * 3 / 16
                    buf[y + 1, x] += err * 5 / 16
                    if x - 1 >= 0:
                        buf[y + 1, x - 1] += err * 1 / 16
    return ink


def downsample_gray(im_rgb_super):
    gray = im_rgb_super.convert("L").resize((GRID_W, GRID_H), Image.LANCZOS)
    return np.asarray(gray, dtype=np.float64) / 255.0


def dots_to_coords(ink_mask):
    ys, xs = np.nonzero(ink_mask)
    return np.stack([xs, ys], axis=1).astype(np.int16)  # (N,2) grid coords


def main():
    src = load_and_crop("portrait-source.jpg")
    enhanced = enhance(src)

    rgb_small = np.asarray(
        enhanced.resize((GRID_W, GRID_H), Image.LANCZOS), dtype=np.float64
    )
    fg_mask = segment_foreground(rgb_small)

    # Kill the halo. The flood fill stops at the first pixel that differs from
    # the backdrop, so the anti-aliased transition ring -- pixels that are
    # mostly backdrop -- stays inside the mask. In dark mode those are the
    # BRIGHTEST pixels present, so they dither to near-solid ink and trace a
    # glowing outline around the whole silhouette. Dropping a 2px rim removes
    # the blend zone; anything still backdrop-coloured just inside it goes too.
    rim = fg_mask & ~binary_erosion(fg_mask, structure=np.ones((5, 5)))
    bg_ref = np.median(
        np.concatenate([
            rgb_small[:12, :12].reshape(-1, 3),
            rgb_small[:12, -12:].reshape(-1, 3),
        ], axis=0),
        axis=0,
    )
    bg_like = np.linalg.norm(rgb_small - bg_ref, axis=2) < 62.0
    fg_mask = fg_mask & ~(rim | (bg_like & ~binary_erosion(fg_mask, structure=np.ones((11, 11)))))
    fg_mask_eroded = binary_erosion(fg_mask, structure=np.ones((3, 3)))

    gray = downsample_gray(enhanced)

    # LIGHT MODE: keep background, dots on dark parts of the photo.
    # Compress the shadow end first -- at full range the black suit dithers to
    # 100% ink, which reads as a flat purple slab and roughly doubles the file
    # size. Capping it keeps the suit as dense texture with tone still visible.
    light_input = np.clip(gray * 0.72 + 0.24, 0.0, 1.0)
    light_ink = serpentine_floyd_steinberg(light_input)
    light_dots = dots_to_coords(light_ink)

    # DARK MODE: background segmented out, dots draw the LIT subject.
    # Standard FS below inks pixels that threshold dark, so invert brightness
    # first (bright subject -> "dark" input -> gets ink); force background to
    # white AFTER inversion so it never inks, then hard-clear any bleed past
    # the mask edge from error diffusion.
    # Lift shadows before inverting: the black suit would otherwise fall below
    # the dither floor and vanish entirely, leaving a head floating in space.
    # A small floor keeps it as sparse texture that still reads as "dark".
    lifted = np.clip(gray * 0.90 + 0.10, 0.0, 1.0)
    dark_input = 1.0 - lifted
    dark_input[~fg_mask] = 1.0
    dark_ink = serpentine_floyd_steinberg(dark_input)
    dark_ink &= fg_mask_eroded
    dark_dots = dots_to_coords(dark_ink)

    print(f"grid: {GRID_W}x{GRID_H}")
    print(f"foreground coverage: {fg_mask.mean()*100:.1f}%")
    print(f"light mode dots: {len(light_dots)}")
    print(f"dark mode dots: {len(dark_dots)}")

    np.save("data/light_dots.npy", light_dots)
    np.save("data/dark_dots.npy", dark_dots)
    np.save("data/fg_mask.npy", fg_mask)

    # preview PNGs for visual sanity check
    def render_preview(dots, path, size=(GRID_W, GRID_H), invert_bg=False):
        canvas = np.zeros((GRID_H, GRID_W), dtype=np.uint8) if not invert_bg else np.full((GRID_H, GRID_W), 255, dtype=np.uint8)
        val = 255 if not invert_bg else 0
        canvas[dots[:, 1], dots[:, 0]] = val
        Image.fromarray(canvas).save(path)

    render_preview(dark_dots, "data/preview_dark.png", invert_bg=False)
    render_preview(light_dots, "data/preview_light.png", invert_bg=True)


if __name__ == "__main__":
    main()
