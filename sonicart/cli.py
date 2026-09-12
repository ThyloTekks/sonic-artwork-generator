"""Kommandozeile — damit das Projekt auch ohne Browser benutzbar ist.

Streamlit ist gut zum Ausprobieren und schlecht fuer alles Wiederholbare.
Ein Release neu rendern, ein Cover in CI bauen, zwoelf Titel durchlaufen
lassen: das geht hier.

    python -m sonicart render track.wav -o cover.png --preset Korrend
    python -m sonicart album ./tracks -o ./out --artist "Thylo Tekks"
    python -m sonicart video track.wav -o canvas.mp4 --duration 8 --pulse 0.6
    python -m sonicart inspect track.wav
    python -m sonicart recipe cover.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .album import render_album
from .analysis import load
from .artwork import DEFAULT_PARAMS, Recipe, build
from .export import (PLATFORM_SPECS, export_formats, export_svg, formats_zip,
                     platform_check, print_report, read_recipe, save_jpeg,
                     save_png)
from .palette import PRESETS, PROFILE_BG, palette_from_key
from .riso import INK_SETS
from .typography import CONTRAST_MODES, FONT_PAIRS, TYPEFACES, available_faces
from .compose import LAYOUTS
from .render import MODES


def _add_common(ap):
    ap.add_argument("--recipe", help="Rezept-JSON als Ausgangspunkt")
    ap.add_argument("--preset", choices=list(PRESETS), help="Farbpreset")
    ap.add_argument("--mode", choices=list(MODES), help="Bildmodus")
    ap.add_argument("--layout", choices=list(LAYOUTS), help="Platzierung")
    ap.add_argument("--size", type=int, help="Kantenlaenge in px")
    ap.add_argument("--bg", help="Hintergrundfarbe (#rrggbb)")
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--key-palette", action="store_true",
                    help="Palette aus der erkannten Tonart ableiten")
    ap.add_argument("--artist", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--catalog", default="")
    ap.add_argument("--font", help="Pfad zu einer eigenen TTF/OTF-Datei")
    ap.add_argument("--pair", choices=list(FONT_PAIRS),
                    help="Schriftpaarung (Titel / Artist / Meta)")
    ap.add_argument("--face-title", choices=available_faces())
    ap.add_argument("--face-artist", choices=available_faces())
    ap.add_argument("--face-meta", choices=available_faces())
    ap.add_argument("--contrast", choices=CONTRAST_MODES,
                    default="Farbe umschalten",
                    help="Lesbarkeitspruefung des Textsatzes")
    ap.add_argument("--contrast-min", type=float, default=4.5)
    ap.add_argument("--riso", choices=list(INK_SETS),
                    help="Als Risodruck ausgeben, mit diesem Farbsatz")
    ap.add_argument("--matte", action="store_true",
                    help="Matter Ausgangspunkt statt Neon auf Schwarz")
    ap.add_argument("--effects", action="store_true", help="Effektkette anwenden")
    ap.add_argument("--seed", type=int, default=0)
    for k in ("thickness", "gate", "gamma", "inner"):
        ap.add_argument(f"--{k}", type=float)
    ap.add_argument("--turns", type=float, help="Windungen der Spirale")
    ap.add_argument("--n-mels", type=int)
    ap.add_argument("--symmetry", type=int,
                    help="Zaehligkeit der Rose; 0 = aus der Taktart ableiten")


def _recipe_from_args(a, an=None) -> Recipe:
    if a.recipe:
        r = Recipe.from_json(open(a.recipe).read())
    elif getattr(a, "matte", False) or getattr(a, "riso", None):
        r = Recipe.matte(getattr(a, "riso", None) or "Zinnober",
                         mode=a.mode or "Gitter")
    else:
        r = Recipe()
    if a.preset:
        r.stops = list(PRESETS[a.preset])
        r.bg = PROFILE_BG[a.preset]
    if a.key_palette and an is not None:
        k = an.key
        r.stops = palette_from_key(k["tonic"], k["mode"], n=len(r.stops),
                                   hue_spread=0.6 + 0.8 * an.harmonic_complexity)
        r.bg = r.stops[0]
    for name, val in (("mode", a.mode), ("layout", a.layout),
                      ("size", a.size), ("bg", a.bg)):
        if val:
            setattr(r, name, val)
    if a.transparent:
        r.transparent = True
    p = dict(DEFAULT_PARAMS)
    p.update(r.params or {})
    for k in ("thickness", "gate", "gamma", "inner", "turns"):
        v = getattr(a, k, None)
        if v is not None:
            p[k] = v
    if a.n_mels:
        p["n_mels"] = a.n_mels
    if a.symmetry is not None:
        p["symmetry"] = a.symmetry if a.symmetry > 0 else (an.meter if an else 1)
    r.params = p
    typo = dict(r.typography or {})
    if any([a.artist, a.title, a.label, a.catalog, a.font]):
        typo.update(artist=a.artist, title=a.title, label=a.label,
                    catalog=a.catalog, font_path=a.font)
    if a.pair:
        t, ar, m = FONT_PAIRS[a.pair]
        typo.update(face_title=t, face_artist=ar, face_meta=m)
    for feld, wert in (("face_title", a.face_title),
                       ("face_artist", a.face_artist),
                       ("face_meta", a.face_meta)):
        if wert:
            typo[feld] = wert
    typo["contrast_mode"] = a.contrast
    typo["contrast_min"] = a.contrast_min
    if typo:
        r.typography = typo
    if a.effects:
        from .effects import DEFAULTS
        r.effects = dict(DEFAULTS, seed=a.seed or None)
    return r


def cmd_inspect(a) -> int:
    an = load(a.audio)
    print(json.dumps(an.summary(), indent=2, ensure_ascii=False))
    print("Abschnitte (s):",
          ", ".join(f"{t:.1f}" for t in an.segment_times()))
    return 0


def cmd_fonts(a) -> int:
    """Mitgelieferte Schriften auflisten."""
    for name, tf in TYPEFACES.items():
        vorhanden = "ja " if tf.path or name == "System" else "FEHLT"
        schnitte = ", ".join(tf.weights) if tf.weights else "einer"
        print(f"  {name:18s} [{vorhanden}] {tf.kind:8s} Schnitte: {schnitte}")
        print(f"      {tf.note}")
    print("\nPaarungen:")
    for n, (t, ar, m) in FONT_PAIRS.items():
        print(f"  {n:26s} Titel {t} / Artist {ar} / Meta {m}")
    return 0


def cmd_render(a) -> int:
    an = load(a.audio)
    r = _recipe_from_args(a, an)
    r.audio_name = os.path.basename(a.audio)
    befund = {}
    img = build(an, r, r.size, report=befund)
    if befund:
        grund = "#%02x%02x%02x" % tuple(befund["grund"])
        zeichen = "ok" if befund["ok"] else "ZU GERING"
        zusatz = ""
        if befund.get("platte"):
            zusatz = f", Feld {befund['platte']} unterlegt"
        elif befund.get("geaendert"):
            zusatz = f", Farbe auf {befund['farbe']} umgeschaltet"
        print(f"Textkontrast: {befund['kontrast']:.1f}:1 gegen {grund} "
              f"[{zeichen}]{zusatz}")
    out = a.out or os.path.splitext(a.audio)[0] + "_cover.png"
    if out.lower().endswith((".jpg", ".jpeg")):
        data = save_jpeg(img, a.quality)
    else:
        data = save_png(img, r)
    with open(out, "wb") as fh:
        fh.write(data)
    print(f"{out}  {img.size[0]}x{img.size[1]}  {len(data)/1e6:.2f} MB")
    if a.formats:
        z = formats_zip(export_formats(img, r.bg, r.transparent, r.layout), r)
        zp = os.path.splitext(out)[0] + "_formate.zip"
        with open(zp, "wb") as fh:
            fh.write(z)
        print(f"{zp}  {len(z)/1e6:.2f} MB")
    if a.svg:
        sp = os.path.splitext(out)[0] + ".svg"
        with open(sp, "wb") as fh:
            fh.write(export_svg(an, r, size_px=min(r.size, 1200)))
        print(f"{sp}")
    if a.check:
        print(json.dumps(platform_check(img, a.check, data), indent=2,
                         ensure_ascii=False))
    rep = print_report(r)
    print("Druck:", rep["hinweis"])
    return 0


def cmd_album(a) -> int:
    r = _recipe_from_args(a)
    titles = None
    if a.titles:
        titles = [t.strip() for t in open(a.titles).read().splitlines() if t.strip()]
    def prog(ev):
        kind, i, n, name = ev
        print(f"  [{i}/{n}] {kind}: {name}", file=sys.stderr)
    rep = render_album(a.folder, r, a.out, size=a.size or 3000,
                       shared_palette=not a.no_shared_palette,
                       spread_deg=a.spread, formats=a.formats, jpeg=a.jpeg,
                       titles=titles, progress=prog)
    print(f"{len(rep['titel'])} Titel -> {rep['verzeichnis']}")
    print("Album-Palette:", " ".join(rep["album_palette"]))
    for t in rep["titel"]:
        print(f"  {t['datei']:34s} {t['tonart']:8s} {t['tempo_bpm']:6.1f} bpm")
    return 0


def cmd_video(a) -> int:
    from .video import animate
    an = load(a.audio)
    r = _recipe_from_args(a, an)
    r.audio_name = os.path.basename(a.audio)
    out = a.out or os.path.splitext(a.audio)[0] + "_canvas.mp4"
    info = animate(an, r, out, audio=a.audio, mode=a.video_mode,
                   duration=a.duration, fps=a.fps, aspect=a.aspect,
                   rotate_turns=a.rotate_turns, pulse=a.pulse, effects=a.effects,
                   loop_safe=not a.no_loop, with_audio=not a.no_audio)
    print(f"{info['path']}  {info['duration']:.2f}s  {info['frames']} Frames  "
          f"Ton={'ja' if info['audio'] else 'nein'}  "
          f"Canvas-tauglich={'ja' if info['canvas_ok'] else 'nein'}")
    return 0


def cmd_recipe(a) -> int:
    r = read_recipe(a.png)
    if r is None:
        print("Kein Rezept in dieser Datei.", file=sys.stderr)
        return 1
    print(r.to_json())
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("sonicart", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="Analyse eines Stuecks ausgeben")
    p.add_argument("audio")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("render", help="Einzelcover rendern")
    p.add_argument("audio")
    p.add_argument("-o", "--out")
    p.add_argument("--formats", action="store_true", help="Social-Formate als ZIP")
    p.add_argument("--svg", action="store_true", help="zusaetzlich als SVG")
    p.add_argument("--quality", type=int, default=92)
    p.add_argument("--check", choices=list(PLATFORM_SPECS))
    _add_common(p)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("album", help="ganzen Ordner als Serie rendern")
    p.add_argument("folder")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--spread", type=float, default=40.0,
                   help="Farbtonspreizung ueber das Album in Grad")
    p.add_argument("--no-shared-palette", action="store_true")
    p.add_argument("--formats", action="store_true")
    p.add_argument("--jpeg", action="store_true")
    p.add_argument("--titles", help="Textdatei, eine Zeile je Titel")
    _add_common(p)
    p.set_defaults(fn=cmd_album)

    p = sub.add_parser("video", help="Canvas/Reel rendern")
    p.add_argument("audio")
    p.add_argument("-o", "--out")
    p.add_argument("--video-mode", default="Live-Kreisspektrum")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--aspect", default="9x16_Story_Reel_Canvas")
    p.add_argument("--rotate-turns", type=float, default=1.0,
                   help="Umdrehungen des Bildes waehrend des Videos")
    p.add_argument("--pulse", type=float, default=0.0)
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    _add_common(p)
    p.set_defaults(fn=cmd_video)

    p = sub.add_parser("fonts", help="mitgelieferte Schriften auflisten")
    p.set_defaults(fn=cmd_fonts)

    p = sub.add_parser("recipe", help="Rezept aus einem exportierten PNG lesen")
    p.add_argument("png")
    p.set_defaults(fn=cmd_recipe)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
