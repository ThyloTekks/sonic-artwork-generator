"""Post-Effekte. Alle alpha-bewusst und aufloesungsrelativ.

Alpha-bewusst: jeder Effekt trennt RGB und Alpha, damit transparente PNGs nicht
mit Grain oder Halo auf den leeren Flaechen zurueckkommen.
Aufloesungsrelativ: Korngroesse und Radien skalieren mit der Bildbreite, damit
die 700-px-Vorschau aussieht wie der 3000-px-Export.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .palette import hex_to_rgb


def _split(img):
    arr = np.asarray(img).astype(float)
    return (arr[..., :3], arr[..., 3:4]) if arr.shape[2] == 4 else (arr, None)


def _merge(rgb, a):
    rgb = np.clip(rgb, 0, 255)
    if a is None:
        return Image.fromarray(rgb.astype("uint8"), "RGB")
    return Image.fromarray(
        np.concatenate([rgb, np.clip(a, 0, 255)], 2).astype("uint8"), "RGBA")


def add_grain(img, amount, seed=None, cell_ref=2.2):
    if amount <= 0:
        return img
    rgb, a = _split(img)
    H, W = rgb.shape[:2]
    scale = W / 1000.0
    cell = max(1, int(round(cell_ref * scale)))          # Korngroesse ~ konstant
    rng = np.random.default_rng(seed)
    sh, sw = -(-H // cell), -(-W // cell)                # Aufrunden -> deckt Bild ab
    small = rng.standard_normal((sh, sw))
    noise = np.kron(small, np.ones((cell, cell)))[:H, :W, None] * 255 * amount * 0.42
    lum = rgb.mean(2, keepdims=True) / 255.0
    mask = 0.15 + 0.85 * lum
    if a is not None:
        mask = mask * (a / 255.0)                        # kein Grain auf Transparenz
    return _merge(rgb + noise * mask, a)


def add_bloom(img, amount):
    if amount <= 0:
        return img
    rgb, a = _split(img)
    lum = rgb.mean(2)
    thr = np.percentile(lum, 85)
    m = np.clip((lum - thr) / (255 - thr + 1e-9), 0, 1)
    bright = np.clip(rgb * m[..., None], 0, 255).astype("uint8")
    rad = (6 + 22 * amount) * (rgb.shape[1] / 1000.0)      # aufloesungsrelativ
    blur = np.asarray(Image.fromarray(bright).filter(
        ImageFilter.GaussianBlur(radius=rad))).astype(float)
    rgb = 255 - (255 - rgb) * (255 - blur * amount) / 255.0     # Screen
    if a is not None:
        amask = np.asarray(Image.fromarray((m * 255).astype("uint8")).filter(
            ImageFilter.GaussianBlur(radius=rad))).astype(float)[..., None]
        a = np.maximum(a, amask * amount)                        # Halo weitet Alpha
    return _merge(rgb, a)


def chroma_ab(img, px):
    if px <= 0:
        return img
    rgb, a = _split(img)
    r = np.roll(rgb[:, :, 0], px, 1)
    b = np.roll(rgb[:, :, 2], -px, 1)
    return _merge(np.stack([r, rgb[:, :, 1], b], 2), a)


def add_vignette(img, amount):
    if amount <= 0:
        return img
    rgb, a = _split(img)
    H, W = rgb.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    v = 1 - amount * np.clip((r - 0.4) / 0.6, 0, 1) ** 2
    return _merge(rgb * v[..., None], a)


def posterize(img, levels):
    if levels < 2:
        return img
    rgb, a = _split(img)
    q = np.round(rgb / 255 * (levels - 1)) / (levels - 1) * 255
    return _merge(q, a)


def add_halftone(img, cell, bg):
    if cell < 2:
        return img
    rgb, a = _split(img)
    H, W = rgb.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    dx = (xx % cell) / cell - 0.5
    dy = (yy % cell) / cell - 0.5
    d = np.sqrt(dx ** 2 + dy ** 2) / 0.707
    dot = (rgb.mean(2) / 255.0) > d           # heller -> groesserer Punkt
    m = dot[..., None]
    out = np.where(m, rgb, np.array(hex_to_rgb(bg)))
    if a is not None:
        a = np.where(m, a, 0)
    return _merge(out, a)


def add_streaks(img, amount):
    if amount <= 0:
        return img
    from scipy.ndimage import uniform_filter1d
    rgb, a = _split(img)
    W = rgb.shape[1]
    lum = rgb.mean(2)
    thr = np.percentile(lum, 88)
    mask = np.clip((lum - thr) / (255 - thr + 1e-9), 0, 1)
    bright = rgb * mask[..., None]
    k = int(W * 0.16 * amount) + 3
    streak = uniform_filter1d(bright, size=k, axis=1, mode="constant")
    out = 255 - (255 - rgb) * (255 - streak * 1.6) / 255      # Screen
    if a is not None:
        am = uniform_filter1d(mask * 255, size=k, axis=1, mode="constant")[..., None]
        a = np.maximum(a, am * amount)
    return _merge(out, a)


def add_depth(img, amount):
    if amount <= 0:
        return img
    rgb, a = _split(img)
    H, W = rgb.shape[:2]
    rad = 8 * W / 1000 * amount
    blur = np.asarray(Image.fromarray(np.clip(rgb, 0, 255).astype("uint8"))
                      .filter(ImageFilter.GaussianBlur(radius=rad))).astype(float)
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / np.sqrt((W / 2) ** 2 + (H / 2) ** 2)
    w = (np.clip((r - 0.5) / 0.5, 0, 1) ** 1.5 * amount)[..., None]   # Raender weich
    return _merge(rgb * (1 - w) + blur * w, a)


DEFAULTS = dict(intensity=1.0, grain=True, bloom=True, chroma=True, seed=None,
                vignette=0.0, posterize_levels=0, halftone=0, streaks=0.0,
                depth=0.0)


def apply_effects(img, feat, bg="#000000", intensity=1.0, grain=True, bloom=True,
                  chroma=True, seed=None, vignette=0.0, posterize_levels=0,
                  halftone=0, streaks=0.0, depth=0.0):
    """Effektkette. feat sind die Timbre-Features aus Analysis.features."""
    scale = img.width / 1000.0
    if depth > 0:
        img = add_depth(img, depth)
    if bloom:
        img = add_bloom(img, feat["energy"] * intensity)
    if streaks > 0:
        img = add_streaks(img, streaks * (0.4 + 0.6 * feat["energy"]))
    if chroma:
        img = chroma_ab(img, int(feat["spread"] * intensity * 8 * scale))
    if posterize_levels and posterize_levels >= 2:
        img = posterize(img, posterize_levels)
    if halftone and halftone >= 2:
        img = add_halftone(img, int(halftone * scale), bg)
    if vignette > 0:
        img = add_vignette(img, vignette)
    if grain:
        img = add_grain(img, feat["roughness"] * intensity, seed=seed)
    return img
