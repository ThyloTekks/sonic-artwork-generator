"""Album-Modus: eine Serie statt eines Einzelbildes.

Der reale Anwendungsfall ist selten ein Cover, sondern ein Release mit acht
bis zwoelf Titeln, die zusammengehoeren muessen. Die Palettenstruktur bleibt
deshalb ueber das Album konstant, nur der Farbton wandert pro Titel — in einem
Rahmen, den man vorgibt (spread_deg). Die Form kommt weiter aus dem Audio,
also bleibt jeder Titel unterscheidbar.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
from PIL import Image

from .analysis import Analysis, load
from .artwork import Recipe, build
from .export import export_formats, formats_zip, save_jpeg, save_png
from .palette import PRESETS, generate_palette, key_to_hue, oklch_to_hex

AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff")


def find_audio(folder: str) -> list[str]:
    """Audiodateien eines Ordners, alphabetisch — das ist meist die Trackfolge."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.lower().endswith(AUDIO_EXT) and not f.startswith(".")]
    if not files:
        raise FileNotFoundError(f"Keine Audiodateien in {folder}")
    return files


def _circular_mean(angles, weights=None) -> float:
    w = np.ones_like(angles) if weights is None else np.asarray(weights)
    if w.sum() < 1e-9:
        w = np.ones_like(angles)
    return float(np.arctan2(np.sum(w * np.sin(angles)), np.sum(w * np.cos(angles))))


def album_hues(analyses: list[Analysis], spread_deg: float = 40.0) -> dict:
    """Albumfarbton und die Abweichung je Titel.

    Der Albumton ist das nach Erkennungssicherheit gewichtete Kreismittel der
    Tonarten. Titel weichen davon nur innerhalb von spread_deg ab, sonst
    zerfaellt die Serie optisch.
    """
    hues, conf = [], []
    for a in analyses:
        k = a.key
        hues.append(key_to_hue(k["tonic"], k["mode"]))
        conf.append(max(k["confidence"], 0.05))
    hues = np.array(hues)
    base = _circular_mean(hues, conf)
    d = np.arctan2(np.sin(hues - base), np.cos(hues - base))    # nach [-pi, pi)
    lim = np.deg2rad(spread_deg) / 2
    dmax = np.abs(d).max()
    scaled = d * (lim / dmax) if dmax > lim else d              # in den Rahmen stauchen
    return {"base": base, "offsets": scaled,
            "per_track": [base + o for o in scaled]}


def album_palettes(analyses: list[Analysis], n: int = 4,
                   harmony: str = "Profil-Struktur",
                   template: list | None = None, spread_deg: float = 40.0,
                   chroma: float = 0.14) -> dict:
    """Palette fuers Album und je eine pro Titel."""
    tmpl = template or PRESETS["Korrend"]
    hues = album_hues(analyses, spread_deg)
    complexity = float(np.mean([a.harmonic_complexity for a in analyses]))
    spread = 0.6 + 0.8 * complexity          # dichte Harmonik -> weitere Palette

    def pal(h, a=None):
        L = 0.58
        C = chroma * (0.78 if a is not None and a.key["mode"] == "moll" else 1.0)
        return generate_palette(oklch_to_hex(L, C, h), n=n, mode=harmony,
                                template=tmpl, hue_spread=spread)

    return {
        "album": pal(hues["base"]),
        "tracks": [pal(h, a) for h, a in zip(hues["per_track"], analyses)],
        "spread": spread,
        "complexity": complexity,
    }


def contact_sheet(images: list[Image.Image], labels: list[str] | None = None,
                  cols: int = 4, cell: int = 420, pad: int = 18,
                  bg: str = "#0b0b0e", label_h: int = 34) -> Image.Image:
    """Kontaktbogen: das ganze Release auf einen Blick."""
    from .palette import hex_to_rgb
    from .typography import TypeSpec, apply_typography
    n = len(images)
    cols = max(1, min(cols, n))
    rows = -(-n // cols)
    lh = label_h if labels else 0
    W = cols * cell + (cols + 1) * pad
    H = rows * (cell + lh) + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), hex_to_rgb(bg))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + lh + pad)
        sheet.paste(im.convert("RGB").resize((cell, cell), Image.LANCZOS), (x, y))
        if labels:
            tile = Image.new("RGB", (cell, lh), hex_to_rgb(bg))
            tile = apply_typography(tile, TypeSpec(
                title=labels[i][:38], anchor="mitte links", margin=0.02,
                size=0.055, color="#b9b9c4"))
            sheet.paste(tile, (x, y + cell))
    return sheet


def render_album(sources, recipe: Recipe, out_dir: str, size: int = 3000,
                 shared_palette: bool = True, spread_deg: float = 40.0,
                 formats: bool = False, jpeg: bool = False,
                 titles: list[str] | None = None, sheet_cols: int = 4,
                 progress=None) -> dict:
    """Ganzes Release rendern: ein Bild je Titel plus Kontaktbogen.

    sources: Ordnerpfad, Liste von Pfaden oder Liste fertiger Analysis-Objekte.
    """
    if isinstance(sources, str):
        sources = find_audio(sources)
    os.makedirs(out_dir, exist_ok=True)

    analyses, names = [], []
    for i, s in enumerate(sources):
        if isinstance(s, Analysis):
            an = s
            name = an.name or f"track_{i+1:02d}"
        else:
            an = load(s)
            name = os.path.splitext(os.path.basename(s))[0]
        analyses.append(an)
        names.append(name)
        if progress:
            progress(("analyse", i + 1, len(sources), name))

    pals = None
    if shared_palette:
        pals = album_palettes(analyses, n=len(recipe.stops),
                              template=list(recipe.stops), spread_deg=spread_deg)

    images, rows = [], []
    for i, (an, name) in enumerate(zip(analyses, names)):
        r = replace(recipe, size=size, audio_name=name)
        if pals:
            r.stops = pals["tracks"][i]
            r.bg = pals["tracks"][i][0]
        if titles and i < len(titles):
            r.typography = dict(r.typography or {}, title=titles[i])
        img = build(an, r, size)
        images.append(img)
        stem = f"{i+1:02d}_{name}".replace(" ", "_")
        if jpeg:
            path = os.path.join(out_dir, f"{stem}.jpg")
            with open(path, "wb") as fh:
                fh.write(save_jpeg(img))
        else:
            path = os.path.join(out_dir, f"{stem}.png")
            with open(path, "wb") as fh:
                fh.write(save_png(img, r))
        if formats:
            with open(os.path.join(out_dir, f"{stem}_formate.zip"), "wb") as fh:
                fh.write(formats_zip(export_formats(img, r.bg, r.transparent,
                                                    r.layout), r))
        rows.append({"track": name, "datei": os.path.basename(path),
                     **an.summary(), "palette": list(r.stops)})
        if progress:
            progress(("render", i + 1, len(analyses), name))

    sheet = contact_sheet(images, names, cols=sheet_cols,
                          bg=(pals["album"][0] if pals else recipe.bg))
    sheet_path = os.path.join(out_dir, "00_kontaktbogen.png")
    sheet.save(sheet_path)

    report = {"verzeichnis": out_dir, "titel": rows,
              "kontaktbogen": os.path.basename(sheet_path),
              "album_palette": pals["album"] if pals else list(recipe.stops),
              "farbspreizung_grad": spread_deg if shared_palette else None}
    import json
    with open(os.path.join(out_dir, "album.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report
