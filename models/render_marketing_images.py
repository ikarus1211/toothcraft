"""Renders the static comparison PNGs under assets/generated/ from the same
anonymised sample set used by build_cases.py, replacing the real-patient
renders that used to live there.

Usage: python models/render_marketing_images.py
"""
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont

from lib_mesh import face_shade_colors, load_volume, mesh_from_volume

SRC_BASE = "/home/dejvax/PHD/ToothCraft/exps/normal/exps/saves/stellar-plant-183"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "generated")

ELEV, AZIM = 28, 105
PANEL_PX = 720  # render square, then crop to content with shared margins per case

FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
TEXT_COLOR = (21, 32, 46)
RULE_COLOR = (219, 227, 238)
ARROW_COLOR = (150, 163, 181)

COLS = {
    "en": ["Input scan", "ToothCraft output", "Ground truth"],
    "cs": ["Vstupní sken", "Výstup ToothCraft", "Ground truth"],
}

HERO_CASE = "N43_0768_L_Final_36.npy_98_samp0"
GRID_CASES = [
    "N43_0532_L_Final_32.npy_121_samp0",
    "N43_0042_L_Final_33.npy_113_samp0",
    "N43_0607_L_Final_46.npy_62_samp0",
]
MESH_FILES = ["incomplete.npy", "sample.npy", "gt.npy"]


def render_panel_rgba(vol, ctr, rng):
    verts, faces = mesh_from_volume(vol)
    fig = plt.figure(figsize=(PANEL_PX / 100, PANEL_PX / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    color = face_shade_colors(verts, faces, ELEV, AZIM)
    poly = Poly3DCollection(verts[faces], facecolor=color, edgecolor="none", antialiased=True)
    ax.add_collection3d(poly)
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), verts


def case_bounds(vol0):
    """Shared center/radius for all three panels of a case, taken from the
    incomplete scan so scan/output/design line up like they do in the live viewer."""
    verts, _ = mesh_from_volume(vol0)
    mn, mx = verts.min(axis=0), verts.max(axis=0)
    return (mn + mx) / 2, (mx - mn).max() / 2 * 1.08


def common_crop_box(images, pad_frac=0.06):
    """Bounding box of non-white content, unioned across a case's panels so
    every column crops identically (no per-column jitter)."""
    boxes = []
    for im in images:
        arr = np.asarray(im)
        mask = (arr < 250).any(axis=2)
        ys, xs = np.where(mask)
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    w, h = x1 - x0, y1 - y0
    pad = int(max(w, h) * pad_frac)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(images[0].width, x1 + pad), min(images[0].height, y1 + pad)
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    return (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)


def render_case_panels(case):
    vols = [load_volume(os.path.join(SRC_BASE, case, f)) for f in MESH_FILES]
    ctr, rng = case_bounds(vols[0])
    images = [render_panel_rgba(v, ctr, rng)[0] for v in vols]
    box = common_crop_box(images)
    return [im.crop(box) for im in images]


def draw_titles(draw, lang, col_x, title_y, col_w):
    font = ImageFont.truetype(FONT_BOLD, 21)
    for x, text in zip(col_x, COLS[lang]):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + col_w / 2 - tw / 2, title_y), text, font=font, fill=TEXT_COLOR)


def draw_arrow(draw, cx, cy, size=11):
    pts = [(cx - size * 0.5, cy - size), (cx + size * 0.6, cy), (cx - size * 0.5, cy + size)]
    draw.line(pts, fill=ARROW_COLOR, width=4, joint="curve")


def compose_row(canvas, panels, left, top, col_w, col_h, gutter):
    for i, panel in enumerate(panels):
        x = left + i * (col_w + gutter)
        resized = panel.resize((col_w, col_h), Image.LANCZOS)
        canvas.paste(resized, (x, top))
        if i < len(panels) - 1:
            draw_arrow(ImageDraw.Draw(canvas), x + col_w + gutter // 2, top + col_h // 2)


def build_hero(lang):
    W, H = 976, 316
    margin, gutter = 20, 36
    title_y, rule_y, top = 14, 54, 66
    col_w = (W - 2 * margin - 2 * gutter) // 3
    col_h = H - top - 16
    col_x = [margin + i * (col_w + gutter) for i in range(3)]

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    draw_titles(draw, lang, col_x, title_y, col_w)
    draw.line([(0, rule_y), (W, rule_y)], fill=RULE_COLOR, width=1)

    panels = render_case_panels(HERO_CASE)
    compose_row(canvas, panels, margin, top, col_w, col_h, gutter)
    out = os.path.join(OUT_DIR, f"result_crown_{lang}.png")
    canvas.save(out)
    print("wrote", out)


def build_grid(lang):
    W = 976
    margin, gutter, row_gap = 20, 36, 26
    title_y, rule_y, top0 = 14, 54, 66
    col_w = (W - 2 * margin - 2 * gutter) // 3
    row_h = col_w  # square-ish panels like the original grid
    H = top0 + 3 * row_h + 2 * row_gap + 16
    col_x = [margin + i * (col_w + gutter) for i in range(3)]

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    draw_titles(draw, lang, col_x, title_y, col_w)
    draw.line([(0, rule_y), (W, rule_y)], fill=RULE_COLOR, width=1)

    for r, case in enumerate(GRID_CASES):
        panels = render_case_panels(case)
        top = top0 + r * (row_h + row_gap)
        compose_row(canvas, panels, margin, top, col_w, row_h, gutter)

    out = os.path.join(OUT_DIR, f"result_cases_{lang}.png")
    canvas.save(out)
    print("wrote", out)


def main():
    for lang in ("en", "cs"):
        build_hero(lang)
        build_grid(lang)


if __name__ == "__main__":
    main()
