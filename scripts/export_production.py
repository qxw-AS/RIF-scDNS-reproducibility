#!/usr/bin/env python3
"""Production export: 170-mm-wide vector PDF + 300-dpi RGB TIFF (LZW).

Every figure PDF is scaled to a 170 mm page width with Ghostscript
(pdfwrite + PDFFitPage, vector content preserved).  The same page is then
rasterised at 300 dpi and saved as an RGB TIFF with LZW compression.

Because all in-figure text is >= 5.4 pt at the original 183-mm layout,
scaling to 170 mm leaves an effective minimum of ~5.0 pt (5.4 * 170/183).
"""
import os
import re
import subprocess

from PIL import Image

# Paths resolve relative to this package; override with RIF_DATA / RIF_DATA2 / RIF_DATA3 / RIF_FIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
ROOT = _ROOT
SRC = os.environ.get("RIF_FIG", os.path.join(ROOT, "output"))
OUT = os.environ.get("RIF_PROD", os.path.join(ROOT, "production"))
MM170_PT = 170 / 25.4 * 72  # 481.89 pt


def page_size_pt(pdf):
    b = open(pdf, "rb").read()
    boxes = re.findall(
        rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", b)
    x0, y0, x1, y1 = [float(v) for v in boxes[-1]]
    return x1 - x0, y1 - y0


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".pdf"):
            continue
        src = os.path.join(SRC, name)
        stem = name[:-4]
        w_pt, h_pt = page_size_pt(src)
        w_mm = w_pt / 72 * 25.4
        h_mm = h_pt / 72 * 25.4
        scale = MM170_PT / w_pt
        out_pdf = f"{OUT}/{stem}_170mm.pdf"
        run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER",
             "-sDEVICE=pdfwrite",
             "-dFIXEDMEDIA",
             f"-dDEVICEWIDTHPOINTS={MM170_PT:.2f}",
             f"-dDEVICEHEIGHTPOINTS={h_pt * scale:.2f}",
             "-dPDFFitPage", "-dAutoRotatePages=/None",
             f"-sOutputFile={out_pdf}", src])
        pw, ph = page_size_pt(out_pdf)
        # 300-dpi raster, RGB TIFF (LZW)
        tmp = f"{OUT}/_tmp"
        run(["pdftoppm", "-r", "300", "-png", "-singlefile", out_pdf, tmp])
        im = Image.open(tmp + ".png").convert("RGB")
        tif = f"{OUT}/{stem}_170mm_300dpi.tif"
        im.save(tif, compression="tiff_lzw", dpi=(300, 300))
        os.remove(tmp + ".png")
        rows.append((stem, f"{w_mm:.1f}", f"{h_mm:.1f}",
                     f"{pw / 72 * 25.4:.1f}", f"{ph / 72 * 25.4:.1f}",
                     f"{5.4 * scale:.2f}", f"{im.size[0]}x{im.size[1]}"))
        print(f"{stem:44s} {w_mm:6.1f} -> 170.0 mm  "
              f"min font {5.4 * scale:.2f} pt  {im.size[0]}x{im.size[1]} px",
              flush=True)

    with open(f"{OUT}/production_manifest.tsv", "w") as fh:
        fh.write("figure\twidth_orig_mm\theight_orig_mm\twidth_170mm\t"
                 "height_170mm\tmin_font_pt_after_scale\tpixels_300dpi\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print("PRODUCTION_DONE", len(rows))


if __name__ == "__main__":
    main()
