# -*- coding: utf-8 -*-
"""Keep the six manuscript main figures at the root of each format folder and move
every other figure into a `supplementary/` subfolder. Re-runnable: save_figure()
always writes to the format-folder root, so run this again after regenerating figures.

Main-figure -> manuscript number map (files keep their generator names):
  figure_12_study_overview          -> Figure 1
  figure_M2_discrimination          -> Figure 2
  figure_M4_calibration_shap        -> Figure 3
  figure_M3_vs_previous             -> Figure 4
  figure_15_phase2_wave1            -> Figure 5
  figure_14_implementation_fidelity -> Figure 6
Everything else (individual panels + genuine supplement figures: by-sex, ATB,
confusion matrices, Boruta) goes to supplementary/.
"""
import os
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(BASE, "figures")
FORMAT_DIRS = ["PNG_300DPI", "TIFF_600DPI", "PDF_VECTOR"]
SUB = "supplementary"

# stems (filename without extension) that stay at the root
MAIN_STEMS = {
    "figure_12_study_overview",             # Figure 1
    "figure_M2_discrimination",             # Figure 2
    "figure_M4_calibration_shap",           # Figure 3
    "figure_M3_vs_previous",                # Figure 4
    "figure_15_phase2_evaluation",          # Figure 5 (2x2 pre/post ROC + confusion)
    "figure_16_postimpl_calibration",       # Figure 6 (post-implementation calibration)
}
# figure_14_implementation_fidelity, figure_17_postimpl_confusion and the old
# figure_15_phase2_wave1 / figure_16_postimpl_roc_calibration are obsolete (deleted).

moved, kept = 0, 0
for d in FORMAT_DIRS:
    src_dir = os.path.join(FIG, d)
    if not os.path.isdir(src_dir):
        print(f"(skip, missing) {d}")
        continue
    sub_dir = os.path.join(src_dir, SUB)
    os.makedirs(sub_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        fp = os.path.join(src_dir, name)
        if not os.path.isfile(fp):
            continue  # skip the supplementary/ subfolder itself
        stem = os.path.splitext(name)[0]
        if stem in MAIN_STEMS:
            kept += 1
            continue
        dst = os.path.join(sub_dir, name)
        if os.path.exists(dst):
            os.remove(dst)  # re-run: replace the older copy
        shutil.move(fp, dst)
        moved += 1

print(f"Main files kept at root: {kept}   |   moved to {SUB}/: {moved}")
for d in FORMAT_DIRS:
    root = os.path.join(FIG, d)
    if os.path.isdir(root):
        roots = sorted(f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)))
        print(f"\n[{d}] root ({len(roots)}): " + ", ".join(roots))
