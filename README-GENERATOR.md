# Dithered Profile Banner Generator

Generates the animated SVG banner on [github.com/harvey619](https://github.com/harvey619) —
a 1-bit Floyd–Steinberg dithered portrait that shimmers in, then morphs through
three marks on a 14.2s loop.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| Portrait → dot grid | `scripts/dither.py` | `data/{dark,light}_dots.npy` |
| Logo marks → dot grid | `scripts/logos.py` | `data/logo_*.npy` |
| Animation choreography | `scripts/groups.py` | intro groups, drift bands, traveller paths |
| SVG assembly | `scripts/build_svg.py` | `dist/{dark,light}.svg` |

The `.npy` files are the source of truth. The SVGs are generated artifacts —
regenerate, never hand-edit.

```bash
python3 scripts/dither.py && python3 scripts/groups.py && python3 scripts/build_svg.py
```

## Technique notes

**Background segmentation** — a flood fill seeded only from border pixels matching
the backdrop reference. Seeding every border pixel lets the fill walk up through a
dark suit that touches the frame edge and eat the subject.

**Halo removal** — anti-aliased edge pixels are partly backdrop, so in dark mode
they are the brightest pixels present and dither to a glowing outline. A 2px rim
drop plus a backdrop-colour test removes it.

**Intro shimmer** — ~60 randomly interleaved groups, verified with a centroid-offset
evenness metric (0.05 scattered vs 0.62 for a deliberately spatial control).

**Drift bands** — drift is a linear function of position, so naive quantisation
recreates a square grid. Per-dot Gaussian noise before banding keeps boundaries
organic, verified at 0.014 (a grid artifact scores ~0.17).

**Traveller morph** — ~900 dots matched between marks by optimal transport
(Hungarian assignment) so each takes its shortest path.
