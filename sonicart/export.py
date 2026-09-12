"""Export: Social-Formate, SVG, PNG mit eingebettetem Rezept, Druckpruefung."""

from __future__ import annotations

import io
import json
import zipfile

from PIL import Image, PngImagePlugin

from .analysis import Analysis
from .artwork import Recipe
from .compose import compose
from .palette import print_check
from .render import MODES, fig_to_svg, render

PNG_KEY = "sonicart-recipe"

#  Zielformate. Die Kantenlaengen folgen den ueblichen Plattformvorgaben.
SIZES = {
    "1x1_Quadrat":            (1080, 1080),
    "4x5_IG-Feed":            (1080, 1350),
    "9x16_Story_Reel_Canvas": (1080, 1920),
    "16x9_YouTube":           (1920, 1080),
}

#  Anforderungen der Plattformen, gegen die geprueft wird.
PLATFORM_SPECS = {
    "Spotify Cover":  dict(min_px=640, ideal_px=3000, square=True,
                           max_bytes=10 * 1024 * 1024, fmt="JPEG"),
    "Bandcamp Cover": dict(min_px=1400, ideal_px=3000, square=True,
                           max_bytes=10 * 1024 * 1024, fmt="PNG"),
    "Vinyl 12\" Print": dict(min_px=3543, ideal_px=3543, square=True,
                             max_bytes=None, fmt="PNG"),
}


# ----------------------------------------------------------------------
# Rasterformate
# ----------------------------------------------------------------------
def export_formats(square_img: Image.Image, bg: str, transparent: bool = False,
                   layout: str = "Zentriert", scale_mul: float = 1.0,
                   sizes: dict | None = None) -> dict[str, Image.Image]:
    """Das quadratische Artwork auf alle Social-Seitenverhaeltnisse setzen."""
    out = {}
    for name, (W, H) in (sizes or SIZES).items():
        out[name] = compose(square_img, W, H, layout, bg, transparent,
                            scale_mul=scale_mul)
    return out


def formats_zip(images: dict[str, Image.Image], recipe: Recipe | None = None) -> bytes:
    """Formate als ZIP; das Rezept liegt als JSON daneben und steckt in jedem PNG."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, im in images.items():
            z.writestr(f"{name}.png", save_png(im, recipe))
        if recipe is not None:
            z.writestr("rezept.json", recipe.to_json())
    return buf.getvalue()


# ----------------------------------------------------------------------
# PNG mit Rezept
# ----------------------------------------------------------------------
def save_png(img: Image.Image, recipe: Recipe | None = None) -> bytes:
    """PNG-Bytes; das Rezept wandert in einen tEXt-Chunk."""
    buf = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    if recipe is not None:
        info.add_text(PNG_KEY, recipe.to_json(indent=None))
        info.add_text("Software", "Sonic Artwork")
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def save_jpeg(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality,
                            subsampling=0, optimize=True)
    return buf.getvalue()


def read_recipe(src) -> Recipe | None:
    """Rezept aus einem exportierten PNG zurueckholen. None, wenn keins drin ist."""
    try:
        im = src if isinstance(src, Image.Image) else Image.open(src)
        raw = (im.text.get(PNG_KEY) if hasattr(im, "text") else None) \
            or im.info.get(PNG_KEY)
        return Recipe.from_json(raw) if raw else None
    except Exception:
        return None


# ----------------------------------------------------------------------
# Vektor
# ----------------------------------------------------------------------
#  Spektrogramm-Modi erzeugen ein Viereck je Frequenzband und Zeitspalte.
#  Massgeblich ist also das Produkt der beiden, nicht die Spaltenzahl allein:
#  bei kurzen Stuecken hat das Spektrogramm ohnehin wenige Spalten, und die
#  Datei wird trotzdem zweistellig megabyteschwer. Deshalb ein Budget ueber
#  beide Achsen — sonst oeffnet kein Zeichenprogramm das Ergebnis fluessig.
#  Richtwert aus der Messung: matplotlib braucht rund 350 Byte je Viereck,
#  15 000 landen also bei etwa 5 MB. Das oeffnet Affinity noch fluessig.
SVG_MAX_QUADS = 15_000
SVG_MAX_MELS = 64


def svg_detail(params: dict, mode: str, max_quads: int = SVG_MAX_QUADS) -> dict:
    """Aufloesung eines Modus auf das Vektorbudget herunterrechnen."""
    p = dict(params)
    known = MODES[mode].params
    if "n_mels" in known:
        p["n_mels"] = min(int(p.get("n_mels", 110)), SVG_MAX_MELS)
    if "cols" in known:
        mels = int(p.get("n_mels", SVG_MAX_MELS))
        p["cols"] = max(120, min(int(p.get("cols", 10 ** 6)), max_quads // max(1, mels)))
    return p


def export_svg(an: Analysis, recipe: Recipe, mode: str | None = None,
               size_px: int = 1000, cmap=None,
               max_quads: int = SVG_MAX_QUADS) -> bytes:
    """Ein Modus als SVG. Ohne Rastereffekte und ohne Typografie."""
    mode = mode or recipe.mode
    if not MODES[mode].vector:
        raise ValueError(f"{mode} ist rasterbasiert und taugt nicht als SVG.")
    params = svg_detail(recipe.params, mode, max_quads)
    fig = render(an.slice(recipe.start, recipe.end), mode,
                 cmap if cmap is not None else recipe.cmap(), recipe.bg,
                 size_px, params, recipe.transparent)
    return fig_to_svg(fig, recipe.transparent)


def export_svg_layers(an: Analysis, recipe: Recipe, size_px: int = 1000,
                      cmap=None, max_quads: int = SVG_MAX_QUADS) -> dict[str, bytes]:
    """Aktive Mix-Ebenen je als eigene SVG — zum Stapeln in Affinity/Illustrator."""
    modes = [m for m, w in (recipe.mix or {}).items() if w > 0] or [recipe.mode]
    out = {}
    for m in modes:
        if not MODES[m].vector:
            continue
        out[m] = export_svg(an, recipe, m, size_px, cmap, max_quads)
    return out


def svg_zip(layers: dict[str, bytes], recipe: Recipe | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in layers.items():
            z.writestr(f"{name.replace(' ', '_')}.svg", data)
        if recipe is not None:
            z.writestr("rezept.json", recipe.to_json())
    return buf.getvalue()


# ----------------------------------------------------------------------
# Abgabepruefung
# ----------------------------------------------------------------------
def platform_check(img: Image.Image, platform: str, data: bytes | None = None) -> dict:
    """Prueft ein fertiges Bild gegen die Vorgaben einer Plattform."""
    spec = PLATFORM_SPECS[platform]
    W, H = img.size
    issues = []
    if spec["square"] and W != H:
        issues.append(f"nicht quadratisch ({W}x{H})")
    if min(W, H) < spec["min_px"]:
        issues.append(f"zu klein: {min(W, H)} px, gefordert sind {spec['min_px']}")
    elif min(W, H) < spec["ideal_px"]:
        issues.append(f"unter der empfohlenen Kantenlaenge {spec['ideal_px']} px")
    if data is not None and spec["max_bytes"] and len(data) > spec["max_bytes"]:
        issues.append(f"Datei zu gross: {len(data)/1e6:.1f} MB, "
                      f"Grenze {spec['max_bytes']/1e6:.0f} MB")
    return {"platform": platform, "ok": not issues, "issues": issues,
            "size": (W, H), "empfohlenes_format": spec["fmt"]}


def print_report(recipe: Recipe) -> dict:
    """Druckvorschau der Palette: was bricht im Offsetdruck weg?"""
    rows = print_check(list(recipe.stops) + [recipe.bg])
    bad = [r for r in rows if not r["printable"]]
    return {
        "rows": rows,
        "kritisch": len(bad),
        "hinweis": ("Alle Farben liegen im Druckraum."
                    if not bad else
                    f"{len(bad)} Farbe(n) ausserhalb des Offset-Farbraums — "
                    "sie kommen im Druck matter zurueck als am Bildschirm."),
    }
