"""Publication figure style for manuscript figures (matplotlib).

Canonical rules live in FIGURE-STANDARDS.md; this module is the implementation.
Ported and consolidated from two established sources:
  - the CASCADE master script (6-colour academic palette, Tufte look, 300/600
    dpi + TIFF-LZW export via PIL), and
  - plots_final.R (theme_bw box look, orange/blue two-group mapping).

Two visual styles are selectable via one argument:
  set_style("tufte")  -> minimal: top/right spines off, no grid (default)
  set_style("box")    -> theme_bw: full box + faint dashed grid

Usage:
    import matplotlib.pyplot as plt
    from figure_style import set_style, PALETTE, COL_SINGLE, save_figure

    set_style("tufte")                 # or "box"
    fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.8))
    ax.scatter(x, y, color=PALETTE["highlight"])
    save_figure(fig, "figure_01_reference_intervals")
    # -> figures/PNG_300DPI/figure_01_reference_intervals.png   (300 dpi)
    #    figures/TIFF_600DPI/figure_01_reference_intervals.tif  (600 dpi, LZW)
    #    figures/PDF_VECTOR/figure_01_reference_intervals.pdf   (vector)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# Allow large TIFFs to be re-saved by PIL for LZW compression.
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:  # PIL absent -> TIFF still written, just uncompressed
    Image = None

# --- Colour palette ------------------------------------------------------
# Primary 6-colour academic palette (from the CASCADE design system). Colours
# carry semantic roles so figures stay consistent across a manuscript.
PALETTE = {
    "highlight": "#C0392B",  # ruby red     - primary/critical (e.g. Cascade)
    "base1":     "#5D6D7E",  # steel blue   - main comparator
    "base2":     "#BDC3C7",  # pale grey    - background/secondary
    "accent1":   "#27AE60",  # emerald      - positive class
    "accent2":   "#E67E22",  # matte orange - third group
    "accent3":   "#8E44AD",  # amethyst     - errors/mixed
    "text":      "#333333",  # soft black   - text/axes (see md: some journals
                             #                 require pure #000000 instead)
    "ci_grey":   "#7F7F7F",  # ~grey50      - confidence-interval ranges
}

# Colour-blind-safe fallback (Okabe-Ito). Use this when a figure must be safe
# for red-green colour vision deficiency: the ruby/emerald pairing in PALETTE
# is NOT cvd-safe. `#0072B2` here is also the plots_final.R "Direct" blue.
OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

# Two-group method mapping carried over from plots_final.R.
METHOD_COLORS = {"refineR": "#FF7F0E", "Direct": "#0072B2"}

# --- Journal column widths ----------------------------------------------
# Most journals: single column ~= 89 mm, double column ~= 183 mm.
# Build figures at final print width so text scales correctly.
_MM = 1 / 25.4
COL_SINGLE = 89 * _MM   # inches
COL_DOUBLE = 183 * _MM  # inches


# --- Global style --------------------------------------------------------
def set_style(style: str = "tufte", base_size: int = 12,
              vector_editable: bool = True, palette: str = "academic") -> None:
    """Apply publication rcParams.

    Parameters
    ----------
    style : {"tufte", "box"}
        "tufte" -> top/right spines off, no grid (minimal, default).
        "box"   -> full box around axes + faint dashed grid (theme_bw look).
    base_size : int
        Base font size in points (12 matches both source files).
    vector_editable : bool
        If True, set pdf/ps fonttype 42 and svg fonttype "none" so text stays
        editable/embedded in vector output (Illustrator/Inkscape; required by
        many journals). Leave True unless a specific reason not to.
    palette : {"academic", "okabe_ito"}
        Colour cycle for automatic colouring. "academic" = 6-colour PALETTE;
        "okabe_ito" = colour-blind-safe set.
    """
    if style not in ("tufte", "box"):
        raise ValueError("style must be 'tufte' or 'box'")

    if palette == "okabe_ito":
        cycle_colors = OKABE_ITO
    else:
        cycle_colors = [PALETTE["base1"], PALETTE["accent2"], PALETTE["accent1"],
                        PALETTE["highlight"], PALETTE["base2"], PALETTE["accent3"]]

    common = {
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base_size,
        "axes.titlesize": base_size + 1,
        "axes.titleweight": "bold",
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 2,
        "legend.frameon": False,
        # Colours
        "text.color": PALETTE["text"],
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "axes.prop_cycle": cycler(color=cycle_colors),
        # Background + save
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    }

    if vector_editable:
        common.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                       "svg.fonttype": "none"})

    if style == "tufte":
        common.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        })
    else:  # box (theme_bw)
        common.update({
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
        })

    mpl.rcParams.update(common)


# --- Multi-format / multi-DPI export ------------------------------------
def save_figure(fig, basename, outdir="figures",
                png_dpi=300, tiff_dpi=600, include_vector=True):
    """Export a figure for review (PNG 300 dpi) and print (TIFF 600 dpi + PDF).

    Files are written into role-named subfolders, matching the CASCADE
    convention:
        {outdir}/PNG_300DPI/{basename}.png
        {outdir}/TIFF_600DPI/{basename}.tif   (LZW-compressed if PIL present)
        {outdir}/PDF_VECTOR/{basename}.pdf    (vector; resolution-independent)

    TIFF LZW compression is applied by re-saving with PIL, because matplotlib's
    native TIFF compression is unreliable across backends.
    """
    outdir = Path(outdir)
    png_dir = outdir / "PNG_300DPI"
    tiff_dir = outdir / "TIFF_600DPI"
    pdf_dir = outdir / "PDF_VECTOR"
    for d in (png_dir, tiff_dir, pdf_dir):
        d.mkdir(parents=True, exist_ok=True)

    # PNG 300 dpi (review / RGB)
    png_path = png_dir / f"{basename}.png"
    fig.savefig(png_path, dpi=png_dpi, format="png")

    # TIFF 600 dpi (print), LZW-compressed. Prefer one-step LZW via pil_kwargs
    # (delegates to PIL at write time); fall back to a temp-file re-save. The old
    # in-place re-save (open then save to the same path) fails on Windows/OneDrive
    # with Errno 22, so it is avoided.
    tiff_path = tiff_dir / f"{basename}.tif"
    try:
        fig.savefig(tiff_path, dpi=tiff_dpi, format="tiff",
                    pil_kwargs={"compression": "tiff_lzw"})
    except Exception:
        fig.savefig(tiff_path, dpi=tiff_dpi, format="tiff")
        if Image is not None:
            try:
                with Image.open(tiff_path) as img:
                    img.load()
                    tmp = tiff_path.with_suffix(".tmp.tif")
                    img.save(tmp, compression="tiff_lzw")
                tmp.replace(tiff_path)
            except Exception as exc:  # keep the uncompressed TIFF on failure
                print(f"  ! TIFF LZW compression skipped for {basename}: {exc}")

    # Vector PDF (editable line art)
    if include_vector:
        fig.savefig(pdf_dir / f"{basename}.pdf", format="pdf")

    print(f"saved: {basename}  (PNG {png_dpi} / TIFF {tiff_dpi} LZW"
          f"{' / PDF' if include_vector else ''})")


if __name__ == "__main__":
    # Self-test: renders both styles so the toggle is visible at a glance.
    import numpy as np

    rng = np.random.default_rng(0)
    for style in ("tufte", "box"):
        set_style(style)
        fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.8))
        for key in ("base1", "highlight"):
            x = rng.normal(size=40)
            ax.scatter(x, x * 0.5 + rng.normal(size=40),
                       s=18, alpha=0.75, color=PALETTE[key], label=key)
        ax.set_xlabel("Reference")
        ax.set_ylabel("Predicted")
        ax.set_title(f"A. Example ({style})")
        ax.legend()
        save_figure(fig, f"figure_style_selftest_{style}")
