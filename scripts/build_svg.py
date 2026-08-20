"""
Stage D: assemble the final animated banner SVG (dark + light) from the
.npy source-of-truth data. This file is generated -- edit the data/scripts,
never the SVG by hand.
"""
import numpy as np
from groups import build_intro_groups, build_drift_bands

GRID_W, GRID_H = 300, 340
CANVAS_W, CANVAS_H = 1180, 610
TITLEBAR_H = 34
PAD = 18
PORTRAIT_FRAC = 0.38

PALETTE = {
    "dark": {
        "bg": "#0A101F",
        "panel_bg": "#0A101F",
        "portrait": "#A78BFA",
        "chrome": "#10B981",
        "accent": "#22D3EE",
        "text": "#E5E7EB",
        "text_dim": "#6B7280",
        "leader": "#374151",
        "frame_stroke": "#1F2937",
    },
    "light": {
        "bg": "#F8FAFC",
        "panel_bg": "#FFFFFF",
        "portrait": "#7C3AED",
        "chrome": "#059669",
        "accent": "#0891B2",
        "text": "#111827",
        "text_dim": "#6B7280",
        "leader": "#D1D5DB",
        "frame_stroke": "#E5E7EB",
    },
}
LIVE_RED = "#EF4444"

FONT_MONO = "ui-monospace, 'SF Mono', 'Cascadia Code', Consolas, monospace"
CHAR_W_RATIO = 0.605

ROWS = [
    ("Subject", "Harvey"),
    ("Role", "Full Stack Engineer (Web & Mobile)"),
    ("Origin", "Hong Kong"),
    ("Education", "BSc Info. Management, HKU"),
    ("Status", "Engineering . Teaching . Shipping"),
    ("ToolChain", "Git, Figma, Postman, VS Code"),
    ("Core.Lang", "JavaScript, TypeScript, PHP, Python, Dart"),
    ("Core.Frontend", "React, Vue, Next.js, Tailwind"),
    ("Core.Backend", "Node.js, Laravel, Symfony, Django"),
    ("Core.Database", "MySQL, PostgreSQL, MongoDB, Firebase"),
    ("Core.Infra", "AWS, Docker, Nginx"),
    ("Grid.Mail", "harveyworkhk@gmail.com"),
    ("Grid.Portfolio", "harveyss.netlify.app"),
    ("Grid.LinkedIn", "linkedin.com/in/harveyss"),
    ("Grid.GitHub", "harvey619"),
    ("Grid.YouTube", "@harveyreturn"),
]

RNG = np.random.default_rng(619)


def char_w(font_size):
    return font_size * CHAR_W_RATIO


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- dot paths
def dots_to_path(dots, ox, oy, scale, dot_px):
    """Run-length encode horizontally-adjacent dots into path 'h' segments."""
    if len(dots) == 0:
        return ""
    order = np.lexsort((dots[:, 0], dots[:, 1]))
    d = dots[order]
    segs = []
    run_start = d[0, 0]
    run_x = d[0, 0]
    run_y = d[0, 1]
    for i in range(1, len(d)):
        x, y = d[i]
        if y == run_y and x == run_x + 1:
            run_x = x
            continue
        segs.append((run_start, run_y, run_x - run_start + 1))
        run_start, run_x, run_y = x, x, y
    segs.append((run_start, run_y, run_x - run_start + 1))

    # 1dp is ~0.1px at this scale -- below what crispEdges can resolve, and it
    # cuts roughly a quarter off the path data across ~90k segments.
    parts = []
    for (sx, sy, rlen) in segs:
        px = ox + sx * scale
        py = oy + sy * scale
        w = rlen * scale
        parts.append(f"M{px:.1f},{py:.1f}h{w:.1f}")
    return "".join(parts)


def dots_bbox_center(dots):
    return dots[:, 0].mean(), dots[:, 1].mean()


# ---------------------------------------------------------------- info panel
def build_info_panel(pal, x0, y0, width, mode):
    font_row = 14
    font_header = 13
    line_h = 23
    cw = char_w(font_row)
    label_w = max(len(l) for l, _ in ROWS) * cw
    value_max_w = width - label_w - 40

    out = []
    out.append(
        f'<text x="{x0}" y="{y0}" font-family="{FONT_MONO}" font-size="{font_header}" '
        f'letter-spacing="2" fill="{pal["chrome"]}" font-weight="600">SYSTEM.INFO</text>'
    )

    live_x = x0 + width - 118
    out.append(f'<g transform="translate({live_x},{y0 - 10})">')
    out.append(
        f'<circle cx="5" cy="0" r="4" fill="{LIVE_RED}">'
        f'<animate attributeName="opacity" values="1;0.25;1" dur="1.6s" repeatCount="indefinite"/>'
        f"</circle>"
    )
    out.append(
        f'<text x="15" y="4" font-family="{FONT_MONO}" font-size="12" letter-spacing="1.5" '
        f'fill="{LIVE_RED}" font-weight="600">LIVE</text>'
    )
    out.append("</g>")

    pill_w = (len("@harvey619") * char_w(14)) + 22
    pill_x = x0 + width - pill_w
    pill_y = y0 + 14
    out.append(
        f'<rect x="{pill_x:.1f}" y="{pill_y:.1f}" width="{pill_w:.1f}" height="24" rx="12" '
        f'fill="{pal["accent"]}" opacity="0.16" stroke="{pal["accent"]}" stroke-width="1"/>'
    )
    out.append(
        f'<text x="{pill_x + pill_w / 2:.1f}" y="{pill_y + 16.5:.1f}" text-anchor="middle" '
        f'font-family="{FONT_MONO}" font-size="14" fill="{pal["accent"]}" font-weight="600">@harvey619</text>'
    )

    rows_top = y0 + 56
    label_x = x0
    value_right_x = x0 + width

    for i, (label, value) in enumerate(ROWS):
        ry = rows_top + i * line_h
        l_natural = len(label) * cw
        out.append(
            f'<text x="{label_x:.1f}" y="{ry:.1f}" font-family="{FONT_MONO}" font-size="{font_row}" '
            f'textLength="{l_natural:.1f}" lengthAdjust="spacingAndGlyphs" fill="{pal["text_dim"]}">{esc(label)}</text>'
        )
        v_natural = min(len(value) * cw, value_max_w)
        v_x = value_right_x
        out.append(
            f'<text x="{v_x:.1f}" y="{ry:.1f}" text-anchor="end" font-family="{FONT_MONO}" '
            f'font-size="{font_row}" textLength="{v_natural:.1f}" lengthAdjust="spacingAndGlyphs" '
            f'fill="{pal["text"]}">{esc(value)}</text>'
        )
        leader_start = label_x + l_natural + 8
        leader_end = v_x - v_natural - 8
        gap = leader_end - leader_start
        if gap > 4:
            dot_spacing = 5.2
            n_dots = max(1, int(gap / dot_spacing))
            leader_str = "." * n_dots
            out.append(
                f'<text x="{leader_start:.1f}" y="{ry:.1f}" font-family="{FONT_MONO}" font-size="{font_row}" '
                f'textLength="{gap:.1f}" lengthAdjust="spacingAndGlyphs" fill="{pal["leader"]}">{leader_str}</text>'
            )
    return "\n".join(out), rows_top + len(ROWS) * line_h


# ---------------------------------------------------------------- portrait / logos
def build_intro_layer(pal, dots, group_id, ox, oy, scale, dot_px):
    """One-shot shimmer-in: ~60 random groups fade in over ~2s, hold to 3.2s,
    then this whole duplicate layer permanently hides (the loop layer takes over)."""
    n_groups = int(group_id.max()) + 1
    starts = np.linspace(0, 1.6, n_groups)  # interleaved starts across ~1.6s so all fade complete by ~2.0s
    order = RNG.permutation(n_groups)
    parts = [f'<g id="intro-layer" fill="{pal["portrait"]}">']
    for g in range(n_groups):
        mask = group_id == g
        gd = dots[mask]
        if len(gd) == 0:
            continue
        path = dots_to_path(gd, ox, oy, scale, dot_px)
        b = float(starts[order[g]])
        parts.append(
            f'<path d="{path}" stroke="{pal["portrait"]}" stroke-width="{dot_px:.2f}" '
            f'stroke-linecap="butt" shape-rendering="crispEdges" opacity="0">'
            f'<animate attributeName="opacity" begin="{b:.3f}s" dur="0.55s" '
            f'values="0;1" fill="freeze"/>'
            f"</path>"
        )
    parts.append(
        f'<animate xlink:href="#intro-layer" attributeName="opacity" '
        f'begin="3.2s" dur="0.01s" values="1;0" fill="freeze"/>'
    )
    parts.append("</g>")
    return "\n".join(parts)


def kt(times, total):
    return ";".join(f"{t / total:.4f}" for t in times)


LOOP_DUR = 14.2
# portrait 3.0s | logos 2.0s each | transitions 1.3s each -> 3.0 + 3*2.0 + 4*1.3 = 14.2
T_HOLD_END = 3.0     # portrait hold ends, transition 1 begins
T_TR1_END = 4.3      # react fully in
T_REACT_END = 6.3    # react hold ends, transition 2 begins
T_TR2_END = 7.6      # flutter fully in
T_FLUTTER_END = 9.6  # flutter hold ends, transition 3 begins
T_TR3_END = 10.9     # code fully in
T_CODE_END = 12.9    # code hold ends, transition 4 begins
T_TR4_END = 14.2     # portrait fully back


def build_loop_portrait_layer(pal, dots, band_id, drift_vec, ox, oy, scale, dot_px):
    n_bands = int(band_id.max()) + 1
    jitter = RNG.normal(0, 0.28, n_bands)  # per-band stagger for an organic dissolve, not lockstep
    parts = [f'<g id="loop-portrait" opacity="0">']
    parts.append(
        f'<animate attributeName="opacity" begin="3.2s" dur="{LOOP_DUR}s" repeatCount="indefinite" '
        f'keyTimes="{kt([0, 0.01, T_HOLD_END, T_TR4_END], LOOP_DUR)}" values="0;1;1;1"/>'
    )
    dx, dy = float(drift_vec[0]) * scale, float(drift_vec[1]) * scale
    for b in range(n_bands):
        mask = band_id == b
        bd = dots[mask]
        if len(bd) == 0:
            continue
        path = dots_to_path(bd, ox, oy, scale, dot_px)
        j = float(jitter[b])
        t1 = max(T_HOLD_END, min(T_TR1_END, T_HOLD_END + (T_TR1_END - T_HOLD_END) * (0.5 + j)))
        t4 = max(T_CODE_END, min(T_TR4_END, T_CODE_END + (T_TR4_END - T_CODE_END) * (0.5 + j)))
        times = [0, T_HOLD_END, t1, T_CODE_END, t4, LOOP_DUR]
        op_vals = "1;1;0;0;0;1"
        tx_vals = f"0;0;{dx:.2f};{dx:.2f};{dx:.2f};0"
        ty_vals = f"0;0;{dy:.2f};{dy:.2f};{dy:.2f};0"
        parts.append(
            f'<path d="{path}" stroke="{pal["portrait"]}" stroke-width="{dot_px:.2f}" '
            f'stroke-linecap="butt" shape-rendering="crispEdges">'
            f'<animate attributeName="opacity" begin="3.2s" dur="{LOOP_DUR}s" repeatCount="indefinite" '
            f'keyTimes="{kt(times, LOOP_DUR)}" values="{op_vals}"/>'
            f'<animateTransform attributeName="transform" attributeType="XML" type="translate" '
            f'begin="3.2s" dur="{LOOP_DUR}s" repeatCount="indefinite" '
            f'keyTimes="{kt(times, LOOP_DUR)}" values="0,0;0,0;{dx:.2f},{dy:.2f};{dx:.2f},{dy:.2f};{dx:.2f},{dy:.2f};0,0"/>'
            f"</path>"
        )
    parts.append("</g>")
    return "\n".join(parts)


def build_logo_layer(pal, dots, ox, oy, scale, dot_px, t_in_start, t_in_end, t_out_start, t_out_end, layer_id):
    path = dots_to_path(dots, ox, oy, scale, dot_px)
    times = [0, t_in_start, t_in_end, t_out_start, t_out_end, LOOP_DUR]
    vals = "0;0;1;1;0;0"
    return (
        f'<g id="{layer_id}">'
        f'<path d="{path}" stroke="{pal["portrait"]}" stroke-width="{dot_px:.2f}" '
        f'stroke-linecap="butt" shape-rendering="crispEdges" opacity="0">'
        f'<animate attributeName="opacity" begin="3.2s" dur="{LOOP_DUR}s" repeatCount="indefinite" '
        f'keyTimes="{kt(times, LOOP_DUR)}" values="{vals}"/>'
        f"</path></g>"
    )


def build_travelers_layer(pal, t_react, t_flutter, t_code, ox, oy, scale, dot_px):
    times = [0, T_HOLD_END, T_TR1_END, T_REACT_END, T_TR2_END, T_FLUTTER_END, T_TR3_END, T_CODE_END, LOOP_DUR]
    # hidden through the portrait phase -- their heavier dots would crowd the fine dither
    op_vals = "0;0;1;1;1;1;1;1;0"
    kt_str = kt(times, LOOP_DUR)
    r = dot_px * 0.9
    parts = [f'<defs><circle id="tdot" r="{r:.2f}" fill="{pal["accent"]}"/></defs>']
    parts.append(f'<g id="travelers" opacity="0">')
    parts.append(
        f'<animate attributeName="opacity" begin="3.2s" dur="{LOOP_DUR}s" repeatCount="indefinite" '
        f'keyTimes="{kt_str}" values="{op_vals}"/>'
    )
    n = len(t_react)
    for i in range(n):
        rx, ry = ox + t_react[i, 0] * scale, oy + t_react[i, 1] * scale
        fx, fy = ox + t_flutter[i, 0] * scale, oy + t_flutter[i, 1] * scale
        cx, cy = ox + t_code[i, 0] * scale, oy + t_code[i, 1] * scale
        # keyframes align with: portrait | tr1 | REACT | tr2 | FLUTTER | tr3 | CODE | tr4
        xs = [rx, rx, rx, rx, fx, fx, cx, cx, cx]
        ys = [ry, ry, ry, ry, fy, fy, cy, cy, cy]
        # (react held through tr1+react-hold, morph to flutter across tr2,
        #  hold, morph to code across tr3, hold, fade out across tr4)
        xvals = ";".join(f"{v:.2f}" for v in xs)
        yvals = ";".join(f"{v:.2f}" for v in ys)
        parts.append(
            f'<use href="#tdot">'
            f'<animate attributeName="cx" begin="3.2s" dur="{LOOP_DUR}s" repeatCount="indefinite" '
            f'keyTimes="{kt_str}" values="{xvals}"/>'
            f'<animate attributeName="cy" begin="3.2s" dur="{LOOP_DUR}s" repeatCount="indefinite" '
            f'keyTimes="{kt_str}" values="{yvals}"/>'
            f"</use>"
        )
    parts.append("</g>")
    return "\n".join(parts)


# ---------------------------------------------------------------- chrome
def build_chrome(pal, mode):
    out = []
    out.append(
        f'<rect x="0.5" y="0.5" width="{CANVAS_W - 1}" height="{CANVAS_H - 1}" rx="14" '
        f'fill="{pal["bg"]}" stroke="{pal["frame_stroke"]}"/>'
    )
    out.append(
        f'<path d="M0.5,{TITLEBAR_H} h{CANVAS_W - 1}" stroke="{pal["frame_stroke"]}" stroke-width="1"/>'
    )
    for i, c in enumerate(["#EF4444", "#F59E0B", "#10B981"]):
        out.append(f'<circle cx="{22 + i * 18}" cy="{TITLEBAR_H / 2}" r="5" fill="{c}" opacity="0.85"/>')
    out.append(
        f'<text x="{CANVAS_W / 2}" y="{TITLEBAR_H / 2 + 4}" text-anchor="middle" '
        f'font-family="{FONT_MONO}" font-size="12.5" fill="{pal["text_dim"]}">profile.sh --live</text>'
    )
    return "\n".join(out)


def build(mode):
    pal = PALETTE[mode]
    dark_dots = np.load("data/dark_dots.npy")
    light_dots = np.load("data/light_dots.npy")
    react = np.load("data/logo_khanda.npy")
    flutter = np.load("data/logo_eternal.npy")
    code = np.load("data/logo_code.npy")
    t_react = np.load("data/traveler_khanda.npy")
    t_flutter = np.load("data/traveler_eternal.npy")
    t_code = np.load("data/traveler_code.npy")

    portrait_dots = dark_dots if mode == "dark" else light_dots
    group_id = build_intro_groups(portrait_dots)
    band_id, drift_vec, _axis = build_drift_bands(portrait_dots, react.mean(axis=0))

    panel_w = CANVAS_W * PORTRAIT_FRAC
    panel_h = CANVAS_H - TITLEBAR_H
    panel_x0, panel_y0 = 0, TITLEBAR_H
    scale = min((panel_w - 2 * PAD) / GRID_W, (panel_h - 2 * PAD) / GRID_H)
    ox = panel_x0 + (panel_w - GRID_W * scale) / 2
    oy = panel_y0 + (panel_h - GRID_H * scale) / 2
    dot_px = scale * 1.05

    # re-center logo dot clouds on the portrait's own centroid so they occupy the same visual anchor
    pcx, pcy = dots_bbox_center(portrait_dots)

    # Normalize each mark to the same on-screen extent, then re-center on the
    # portrait's centroid, so the three logos read as one size in the loop.
    LOGO_TARGET_EXTENT = 150.0  # grid units, max(width, height)

    def norm_params(cloud):
        cx, cy = dots_bbox_center(cloud)
        extent = max(cloud[:, 0].max() - cloud[:, 0].min(),
                     cloud[:, 1].max() - cloud[:, 1].min())
        return np.array([cx, cy]), LOGO_TARGET_EXTENT / max(extent, 1e-6)

    def apply_norm(cloud, center, k):
        return (cloud.astype(np.float64) - center) * k + np.array([pcx, pcy])

    c_r, k_r = norm_params(react)
    c_f, k_f = norm_params(flutter)
    c_c, k_c = norm_params(code)

    react_c = apply_norm(react, c_r, k_r)
    flutter_c = apply_norm(flutter, c_f, k_f)
    code_c = apply_norm(code, c_c, k_c)
    # travelers were sampled from the same clouds, so they take the same transform
    t_react_c = apply_norm(t_react, c_r, k_r)
    t_flutter_c = apply_norm(t_flutter, c_f, k_f)
    t_code_c = apply_norm(t_code, c_c, k_c)

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" width="{CANVAS_W}" height="{CANVAS_H}" '
        f'font-family="{FONT_MONO}">'
    )
    svg.append(f'<rect width="{CANVAS_W}" height="{CANVAS_H}" fill="{pal["bg"]}"/>')
    svg.append(build_chrome(pal, mode))

    fx, fy = panel_x0 + PAD, panel_y0 + PAD
    fw, fh = panel_w - PAD, panel_h - 2 * PAD
    svg.append(
        f'<rect x="{fx:.1f}" y="{fy:.1f}" width="{fw:.1f}" height="{fh:.1f}" rx="8" '
        f'fill="none" stroke="{pal["frame_stroke"]}" stroke-width="1"/>'
    )
    corner = 14
    for (cxp, cyp, sx, sy) in [
        (fx, fy, 1, 1), (fx + fw, fy, -1, 1), (fx, fy + fh, 1, -1), (fx + fw, fy + fh, -1, -1),
    ]:
        svg.append(
            f'<path d="M{cxp:.1f},{cyp + sy * corner:.1f} L{cxp:.1f},{cyp:.1f} L{cxp + sx * corner:.1f},{cyp:.1f}" '
            f'fill="none" stroke="{pal["chrome"]}" stroke-width="1.5"/>'
        )
    svg.append(
        f'<rect x="{fx + 14:.1f}" y="{fy - 8:.1f}" width="94" height="16" fill="{pal["bg"]}"/>'
    )
    svg.append(
        f'<text x="{fx + 20:.1f}" y="{fy + 4:.1f}" font-family="{FONT_MONO}" font-size="11" '
        f'letter-spacing="1.6" fill="{pal["chrome"]}" font-weight="600">VISUAL.MAP</text>'
    )
    svg.append(f'<clipPath id="portrait-clip-{mode}"><rect x="{fx:.1f}" y="{fy:.1f}" width="{fw:.1f}" height="{fh:.1f}" rx="8"/></clipPath>')
    svg.append(f'<g clip-path="url(#portrait-clip-{mode})">')
    svg.append(build_intro_layer(pal, portrait_dots, group_id, ox, oy, scale, dot_px))
    svg.append(build_loop_portrait_layer(pal, portrait_dots, band_id, drift_vec, ox, oy, scale, dot_px))
    svg.append(build_logo_layer(pal, react_c, ox, oy, scale, dot_px, T_HOLD_END, T_TR1_END, T_REACT_END, T_TR2_END, "logo-react"))
    svg.append(build_logo_layer(pal, flutter_c, ox, oy, scale, dot_px, T_REACT_END, T_TR2_END, T_FLUTTER_END, T_TR3_END, "logo-flutter"))
    svg.append(build_logo_layer(pal, code_c, ox, oy, scale, dot_px, T_FLUTTER_END, T_TR3_END, T_CODE_END, T_TR4_END, "logo-code"))
    svg.append(build_travelers_layer(pal, t_react_c, t_flutter_c, t_code_c, ox, oy, scale, dot_px))
    svg.append("</g>")

    info_x0 = panel_w + PAD * 2
    info_w = CANVAS_W - info_x0 - PAD
    info_y0 = panel_y0 + 34
    panel_svg, _ = build_info_panel(pal, info_x0, info_y0, info_w, mode)
    svg.append(panel_svg)

    svg.append("</svg>")
    return "\n".join(svg)


def main():
    import os
    os.makedirs("dist", exist_ok=True)
    for mode in ("dark", "light"):
        content = build(mode)
        path = f"dist/{mode}.svg"
        with open(path, "w") as f:
            f.write(content)
        size_kb = len(content.encode()) / 1024
        print(f"{path}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
