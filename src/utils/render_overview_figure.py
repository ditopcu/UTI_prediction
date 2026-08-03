# -*- coding: utf-8 -*-
"""Render paper/figure_study_overview.html to figure_12 (PNG/TIFF/PDF) via Chrome headless."""
import os
import subprocess
from PIL import Image, ImageChops

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HTML = os.path.join(BASE, "paper", "figure_study_overview.html")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
TMP = os.path.join(BASE, "figures", "_ov_raw.png")

url = "file:///" + HTML.replace("\\", "/").replace(" ", "%20")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--force-device-scale-factor=3", "--default-background-color=FFFFFFFF",
                f"--screenshot={TMP}", "--window-size=960,1500", url],
               check=True, capture_output=True)

im = Image.open(TMP).convert("RGB")
bb = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
l, t, r, b = bb
im = im.crop((max(0, l - 18), max(0, t - 18), min(im.width, r + 18), min(im.height, b + 18)))
im.save(os.path.join(BASE, "figures", "PNG_300DPI", "figure_12_study_overview.png"))
im.save(os.path.join(BASE, "figures", "TIFF_600DPI", "figure_12_study_overview.tif"), compression="tiff_lzw")
im.save(os.path.join(BASE, "figures", "PDF_VECTOR", "figure_12_study_overview.pdf"), "PDF", resolution=300)
os.remove(TMP)
print("figure_12 re-rendered", im.size)
