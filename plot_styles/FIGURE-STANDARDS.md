# Figure Standards (Manuscript)

Canonical rules for producing publication figures. This file is the source of
truth; `figure_style.py` is the implementation. When generating figure code,
follow these rules and use `figure_style.py` rather than re-deriving styling.

Drop this file (or a link to it) into `CLAUDE.md` so figure code in any script
is consistent, whether written by hand or by Claude Code.

## How to use

```python
from figure_style import set_style, PALETTE, COL_SINGLE, COL_DOUBLE, save_figure

set_style("tufte")          # "tufte" (default) or "box"
fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.8))
# ... plot, colouring from PALETTE ...
save_figure(fig, "figure_01_short_descriptive_name")
```

## Visual style — toggle

Two styles, chosen with one argument to `set_style()`:

- `"tufte"` (default) — minimal: top and right spines removed, no grid. Best
  for scatter, calibration, ROC/PR, SHAP, most main-text figures.
- `"box"` — `theme_bw` look: full box around the axes with a faint dashed
  grid. Best for forest plots, faceted panels, and figures where the reader
  reads values off a grid.

Pick one style per manuscript and keep it consistent across all figures.

## Colour palette

Primary 6-colour academic palette, used by semantic role (not by position):

| Key         | Hex       | Role                              |
|-------------|-----------|-----------------------------------|
| `highlight` | `#C0392B` | primary / critical (ruby red)     |
| `base1`     | `#5D6D7E` | main comparator (steel blue)      |
| `base2`     | `#BDC3C7` | background / secondary (pale grey)|
| `accent1`   | `#27AE60` | positive class (emerald)          |
| `accent2`   | `#E67E22` | third group (matte orange)        |
| `accent3`   | `#8E44AD` | errors / mixed (amethyst)         |
| `text`      | `#333333` | text and axes                     |
| `ci_grey`   | `#7F7F7F` | confidence-interval ranges        |

Rules:
- Assign colours by meaning and keep the mapping fixed across every figure
  (e.g. the proposed model is always `highlight`).
- The `highlight` (ruby) / `accent1` (emerald) pairing is **not** safe for
  red-green colour vision deficiency. If a figure relies on distinguishing
  those two, switch to the colour-blind-safe set: `set_style(palette="okabe_ito")`
  (Okabe-Ito), or separate them with shape/linetype as well as colour.
- Two-group reference-interval work (refineR vs Direct) uses the carried-over
  mapping `METHOD_COLORS`: refineR `#FF7F0E`, Direct `#0072B2`.

## Typography

- Sans-serif, Arial first (fallbacks Helvetica, DejaVu Sans).
- Base 12 pt; axis titles bold; tick labels one step smaller; legend frameless.
- Text colour `#333333` by default. **Some journals require pure black
  (`#000000`)** — switch if the target journal's guide demands it.
- **Editable/embedded text in vector output is mandatory**: `pdf.fonttype = 42`,
  `ps.fonttype = 42` (set automatically by `set_style(vector_editable=True)`).
  This keeps text as text in Illustrator/Inkscape and satisfies journal font
  rules. Do not flatten text to paths.

## Resolution and file formats

Every figure is exported in three forms by `save_figure()`:

- **PNG, 300 dpi** — for submission/review (RGB, small, screen-readable).
- **TIFF, 600 dpi, LZW-compressed** — for print/production (lossless).
  LZW is applied by re-saving with PIL, because matplotlib's native TIFF
  compression is unreliable.
- **PDF, vector** — resolution-independent line art; preferred by many journals
  for plots. Disable with `include_vector=False` only if the journal forbids it.

Confirm the target journal's exact requirements before final submission (some
want ≥600 dpi even for review, some want CMYK, some specify min/max widths).
Adjust `png_dpi` / `tiff_dpi` / `include_vector` accordingly.

## Figure dimensions

Build figures at final print width so fonts scale correctly:

- Single column ≈ 89 mm → `COL_SINGLE`
- Double column ≈ 183 mm → `COL_DOUBLE`

Do not design at arbitrary sizes and rescale later; that distorts text size.

## Output layout and naming

`save_figure()` writes into role-named subfolders under the output dir:

```
figures/
  PNG_300DPI/figure_01_....png
  TIFF_600DPI/figure_01_....tif
  PDF_VECTOR/figure_01_....pdf
```

- Name files `figure_NN_short_descriptive_name` (zero-padded, snake_case);
  supplementary figures `supp_figure_SN_...`.
- Panel letters (A, B, C) go inside the figure; the caption/legend lives in the
  manuscript, not embedded in the image.

## Reminders

- Code and comments in English.
- Do not embed titles/subtitles meant for the caption inside the figure; keep
  captions in the manuscript.
- Keep one style and one colour mapping per manuscript.
