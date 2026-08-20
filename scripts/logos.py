"""
Stage B: three logo marks, rendered to the same dot-matrix style as the
portrait so the loop animation can morph between them.

Khanda and the eternal knot are TRACED from the user's own reference files
(khanda.svg, eternal.png) -- personal/religious marks whose proportions must
come from the real artwork, not an approximation. The </> glyph is drawn
geometrically since it is a typographic mark with no canonical source.

All three go through the same serpentine Floyd-Steinberg pass as the portrait
so density and texture match across the morph.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from dither import GRID_W, GRID_H, serpentine_floyd_steinberg, dots_to_coords

SUPER = 4
W, H = GRID_W * SUPER, GRID_H * SUPER
CX, CY = W // 2, H // 2
FILL_VALUE = 0.32  # -> ~68% local ink density inside the shape, matches portrait midtones


def _new_canvas():
    return Image.new("L", (W, H), 255), None


def render_khanda(scale=1.0):
    """Traced from the user-supplied khanda.svg (rasterized by Chrome), not
    hand-drawn -- the mark's proportions are religious/identity-significant
    and must come from the real vector, not an approximation."""
    return _trace_raster("assets/khanda_raster.png", 0.52, scale)


def _trace_raster(path, height_frac, scale=1.0, threshold=140):
    """Shared loader for user-supplied reference marks: flatten alpha onto
    white, trim to the ink bbox, scale to a fixed height, hard-threshold."""
    src = Image.open(path)
    if src.mode in ("RGBA", "LA") or "transparency" in src.info:
        src = src.convert("RGBA")
        flat = Image.new("RGBA", src.size, (255, 255, 255, 255))
        flat.alpha_composite(src)
        src = flat.convert("L")
    else:
        src = src.convert("L")
    bbox = ImageOps.invert(src).getbbox()
    src = src.crop(bbox)
    target_h = int(height_frac * H * scale)
    target_w = max(1, int(target_h * src.size[0] / src.size[1]))
    src = src.resize((target_w, target_h), Image.LANCZOS)
    src = src.point(lambda v: 0 if v < threshold else 255)
    im = Image.new("L", (W, H), 255)
    im.paste(src, (CX - target_w // 2, CY - target_h // 2))
    return im


def render_eternal(scale=1.0):
    """Traced from the user-supplied eternal-knot reference."""
    return _trace_raster("assets/eternal.png", 0.50, scale)


def render_code_glyph(scale=1.0):
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    s = 0.20 * W * scale
    stroke = max(6, int(0.06 * W * scale))
    gap = 0.95 * s
    cx1, cx2 = CX - gap, CX + gap
    r = stroke / 2
    # '<'
    pts_l = [(cx1 + s * 0.65, CY - s), (cx1 - s * 0.65, CY), (cx1 + s * 0.65, CY + s)]
    d.line(pts_l, fill=0, width=stroke, joint="curve")
    for p in pts_l:
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=0)
    # '>'
    pts_r = [(cx2 - s * 0.65, CY - s), (cx2 + s * 0.65, CY), (cx2 - s * 0.65, CY + s)]
    d.line(pts_r, fill=0, width=stroke, joint="curve")
    for p in pts_r:
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=0)
    # '/'
    slash_stroke = max(5, int(stroke * 0.85))
    d.line([(CX + s * 0.22, CY - s * 1.3), (CX - s * 0.22, CY + s * 1.3)], fill=0, width=slash_stroke)
    return im


RENDERERS = {"khanda": render_khanda, "eternal": render_eternal, "code": render_code_glyph}


def dither_logo(im_super):
    small = im_super.resize((GRID_W, GRID_H), Image.LANCZOS)
    gray = np.asarray(small, dtype=np.float64) / 255.0
    shape_mask = gray < 0.5
    dith_input = np.where(shape_mask, FILL_VALUE, 1.0)
    ink = serpentine_floyd_steinberg(dith_input)
    return dots_to_coords(ink), shape_mask


def main():
    for name, fn in RENDERERS.items():
        im = fn()
        dots, mask = dither_logo(im)
        np.save(f"data/logo_{name}.npy", dots)
        canvas = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
        canvas[dots[:, 1], dots[:, 0]] = 255
        Image.fromarray(canvas).save(f"data/preview_logo_{name}.png")
        print(f"{name}: {len(dots)} dots")


if __name__ == "__main__":
    main()
