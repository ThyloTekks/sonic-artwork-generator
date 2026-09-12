"""Zusammensetzen: Modi mischen, Bild einsetzen, Form auf der Flaeche platzieren.

Die Platzierung ist der Grund, warum bisher jedes Ergebnis verwandt aussah:
alle Modi zeichnen eine Scheibe, und die sass immer mittig. LAYOUTS bricht das
auf, ohne die Modi anzufassen.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from .analysis import Analysis
from .palette import hex_to_rgb
from .render import MODES, fig_to_pil, mode_params, render


# ----------------------------------------------------------------------
# Modi mischen
# ----------------------------------------------------------------------
def render_pil(an: Analysis, mode, cmap, bg, size_px, params, transparent=False):
    return fig_to_pil(render(an, mode, cmap, bg, size_px, params, transparent),
                      size_px, transparent=transparent)


MIX_BLENDS = ["Ueberlagern", "Aufhellen"]


def _coverage(layer: np.ndarray, bg_arr: np.ndarray) -> np.ndarray:
    """Wo deckt eine Ebene? Abstand zum Hintergrund, 0..1."""
    d = np.abs(layer - bg_arr).max(axis=2)
    return np.clip(d / (np.percentile(d, 99.5) + 1e-9), 0, 1)


def mix_modes(an: Analysis, weights, cmap, bg, size_px, base_params,
              transparent=False, blend="Aufhellen"):
    """Mehrere Modi uebereinanderlegen. weights: {mode: 0..1}.

    "Aufhellen" nimmt je Pixel den helleren Wert. Das funktioniert nur,
    solange die Ebenen duenn sind: ein flaechenfuellender Modus wie Gitter
    deckt ueber die Haelfte des Bildes und schluckt dann alles andere — eine
    Rose kommt daneben nur noch auf wenigen Prozent der Flaeche durch.
    "Ueberlagern" malt die Ebenen stattdessen der Reihe nach uebereinander,
    jeweils nur dort, wo sie wirklich decken. Damit bleibt auch eine duenne
    Form ueber einem vollflaechigen Grund sichtbar.
    """
    active = [(m, w) for m, w in weights.items() if w > 0 and m in MODES]
    if transparent and blend == "Ueberlagern":
        out_rgb = None
        out_a = np.zeros((size_px, size_px, 1))
        for mode, w in active:
            img = render_pil(an, mode, cmap, bg, size_px,
                             mode_params(mode, base_params), transparent=True)
            arr = np.asarray(img).astype(float) / 255.0
            rgb, a = arr[..., :3], arr[..., 3:] * w
            if out_rgb is None:
                out_rgb = np.zeros_like(rgb)
            out_rgb = out_rgb * (1 - a) + rgb * a      # klassisches Over
            out_a = out_a + a * (1 - out_a)
        if out_rgb is None:
            out_rgb = np.zeros((size_px, size_px, 3))
        rgba = np.concatenate([out_rgb, out_a], axis=2)
        return Image.fromarray(np.clip(rgba * 255, 0, 255).astype("uint8"), "RGBA")
    if transparent:
        out_rgb = np.zeros((size_px, size_px, 3))
        out_a = np.zeros((size_px, size_px, 1))
        for mode, w in active:
            img = render_pil(an, mode, cmap, bg, size_px,
                             mode_params(mode, base_params), transparent=True)
            arr = np.asarray(img).astype(float) / 255.0
            rgb, a = arr[..., :3], arr[..., 3:] * w
            out_rgb = np.maximum(out_rgb, rgb * a)     # praemultipliziertes Lighten
            out_a = np.maximum(out_a, a)
        rgb = np.where(out_a > 1e-6, out_rgb / np.clip(out_a, 1e-6, 1), 0)
        rgba = np.concatenate([rgb, out_a], axis=2)
        return Image.fromarray(np.clip(rgba * 255, 0, 255).astype("uint8"), "RGBA")

    bg_arr = np.array(hex_to_rgb(bg)) / 255.0
    out = None
    for mode, w in active:
        img = render_pil(an, mode, cmap, bg, size_px,
                         mode_params(mode, base_params))
        la = np.asarray(img).astype(float) / 255.0
        if blend == "Ueberlagern":
            if out is None:
                out = np.tile(bg_arr, (size_px, size_px, 1))
            a = (_coverage(la, bg_arr) * w)[..., None]
            out = out * (1 - a) + la * a
        else:
            lw = bg_arr + w * (la - bg_arr)
            out = lw if out is None else np.maximum(out, lw)
    if out is None:
        out = np.tile(bg_arr, (size_px, size_px, 1))
    return Image.fromarray(np.clip(out * 255, 0, 255).astype("uint8"))


# ----------------------------------------------------------------------
# Bild / Textur in die Form
# ----------------------------------------------------------------------
IMAGE_MODES = ["—", "Masking", "Textur-Blend"]


def apply_image(art, image, mode, bg, transparent=False):
    """Masking = Bild NUR innerhalb der Audio-Form; Textur-Blend = Form x Textur."""
    W = art.width
    im = np.asarray(ImageOps.fit(image.convert("RGB"), (W, W),
                                 method=Image.LANCZOS)).astype(float)
    ar = np.asarray(art).astype(float)
    lum = (ar[..., 3] / 255.0 if ar.shape[2] == 4 else ar[..., :3].mean(2) / 255.0)
    lum = np.clip(lum, 0, 1)
    bg_rgb = np.array(hex_to_rgb(bg))

    if mode == "Masking":
        if transparent:
            out = np.concatenate([im, (lum * 255)[..., None]], 2)
            return Image.fromarray(np.clip(out, 0, 255).astype("uint8"), "RGBA")
        m = lum[..., None]
        return Image.fromarray(
            np.clip(im * m + bg_rgb * (1 - m), 0, 255).astype("uint8"), "RGB")

    art_rgb = ar[..., :3]
    blended = art_rgb * (im / 255.0)                    # Multiply -> Textur faerbt
    m3 = (lum > 0.03)[..., None]
    out_rgb = np.where(m3, blended, art_rgb)
    if ar.shape[2] == 4:
        out = np.concatenate([out_rgb, ar[..., 3:4]], 2)
        return Image.fromarray(np.clip(out, 0, 255).astype("uint8"), "RGBA")
    return Image.fromarray(np.clip(out_rgb, 0, 255).astype("uint8"), "RGB")


# ----------------------------------------------------------------------
# Platzierung
# ----------------------------------------------------------------------
#  scale ist relativ zur kuerzeren Kante, anchor der Mittelpunkt der Form in
#  Anteilen der Flaeche. scale > 1 heisst: die Form laeuft aus dem Bild.
LAYOUTS = {
    "Zentriert":        dict(scale=0.90, anchor=(0.500, 0.500)),
    "Vollflaechig":     dict(scale=1.00, anchor=(0.500, 0.500)),
    "Goldener Schnitt": dict(scale=0.68, anchor=(0.382, 0.382)),
    "Drittel oben":     dict(scale=0.74, anchor=(0.500, 0.330)),
    "Drittel unten":    dict(scale=0.74, anchor=(0.500, 0.670)),
    "Angeschnitten":    dict(scale=1.38, anchor=(0.400, 0.440)),
    "Eckig links":      dict(scale=0.86, anchor=(0.280, 0.520)),
    "Raster 2x2":       dict(grid=2),
    "Raster 3x3":       dict(grid=3),
}


def compose(art: Image.Image, W: int, H: int, layout: str = "Zentriert",
            bg: str = "#000000", transparent: bool = False,
            scale_mul: float = 1.0, anchor: tuple | None = None,
            rotate_tiles: bool = False) -> Image.Image:
    """Quadratisches Artwork auf eine Flaeche W x H setzen."""
    spec = LAYOUTS.get(layout, LAYOUTS["Zentriert"])
    if transparent:
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    else:
        canvas = Image.new("RGB", (W, H), hex_to_rgb(bg))

    def paste(im, box):
        rgba = im.convert("RGBA")
        if transparent:
            canvas.alpha_composite(rgba, box)
        else:
            canvas.paste(rgba, box, rgba)

    if "grid" in spec:
        n = spec["grid"]
        cw, ch = W // n, H // n
        s = int(min(cw, ch) * 0.92 * scale_mul)
        tile = art.resize((s, s), Image.LANCZOS)
        for r in range(n):
            for c in range(n):
                t = tile.rotate(90 * ((r + c) % 4), expand=False) if rotate_tiles else tile
                paste(t, (c * cw + (cw - s) // 2, r * ch + (ch - s) // 2))
        return canvas

    s = max(1, int(min(W, H) * spec["scale"] * scale_mul))
    ax, ay = anchor or spec["anchor"]
    resized = art.resize((s, s), Image.LANCZOS)
    paste(resized, (int(ax * W - s / 2), int(ay * H - s / 2)))
    return canvas
