"""Footage: Videoclips ueber die Palette eines Tracks einfaerben.

Handyclips und Screen-Recordings passen farblich nie zum Artwork desselben
Tracks. Hier entsteht aus der Palette eine Hald-CLUT (Gradient Map): jeder
Eingangsfarbwert wird durch die Palettenfarbe seiner Helligkeit ersetzt, der
Farbton des Originals faellt also komplett weg, Struktur und Helligkeits-
verlauf bleiben. ffmpeg wendet die Tabelle mit `haldclut` auf den Clip an.

Interpoliert wird in OKLCH, nicht in sRGB, damit die Rampe dieselben
Zwischentoene trifft wie der Palettengenerator.
"""

from __future__ import annotations

import tempfile

import numpy as np
from PIL import Image

from . import ffmpeg
from .export import SIZES
from .palette import hex_to_oklch, hex_to_rgb, oklab_L, oklch_to_hex

HALD_LEVEL = 8                 # 8 -> 64 Stufen je Kanal -> 512x512 px

#  Nur die drei Hochformate/Breitformate, die fuer Footage gefragt sind.
FOOTAGE_SIZES = ["9x16_Story_Reel_Canvas", "1x1_Quadrat", "16x9_YouTube"]
AUDIO_MODES = ["Originalton", "Track-Audio", "stumm"]


# ----------------------------------------------------------------------
# Palette -> Rampe -> CLUT
# ----------------------------------------------------------------------
def oklch_ramp(stops, n: int = 256) -> np.ndarray:
    """Palettenstops -> n Farbstufen (uint8), interpoliert in OKLCH.

    Der Farbton laeuft ueber np.unwrap den kuerzesten Weg; sonst dreht ein
    Uebergang wie Violett -> Orange einmal falsch herum durch den Farbkreis.
    Die Rueckrechnung fittet die Chroma ins Gamut, statt Kanaele zu klemmen.
    """
    lch = np.array([hex_to_oklch(s) for s in stops], float)
    lch[:, 2] = np.unwrap(lch[:, 2])
    t_src = np.linspace(0, 1, len(lch))
    t = np.linspace(0, 1, n)
    L, C, H = (np.interp(t, t_src, lch[:, i]) for i in range(3))
    return np.array([hex_to_rgb(oklch_to_hex(*p, fit=True))
                     for p in zip(L, C, H)], dtype=np.uint8)


def map_luma(L, ramp: np.ndarray) -> np.ndarray:
    """Helligkeiten 0..1 auf Rampenfarben abbilden, linear zwischen den Stufen."""
    xs = np.linspace(0, 1, len(ramp))
    L = np.clip(L, 0, 1)
    return np.stack([np.interp(L, xs, ramp[:, c]) for c in range(3)], -1)


def hald_clut(stops, level: int = HALD_LEVEL, n_ramp: int = 256) -> Image.Image:
    """Hald-CLUT als Gradient Map: Eingangshelligkeit -> Palettenfarbe.

    Das Bild ist die Identitaetstabelle in Hald-Anordnung — Rot laeuft am
    schnellsten, dann Gruen, dann Blau, zeilenweise —, jeder Eintrag ersetzt
    durch die Palettenfarbe zur OKLab-Helligkeit seines Farbwerts. ffmpegs
    haldclut erwartet genau diese Anordnung.
    """
    n = level ** 2
    side = level ** 3
    i = np.arange(n ** 3)
    rgb = np.stack([i % n, i // n % n, i // (n * n)], 1) / (n - 1)
    out = map_luma(oklab_L(rgb), oklch_ramp(stops, n_ramp))
    return Image.fromarray(out.round().astype("uint8").reshape(side, side, 3),
                           "RGB")


def ramp_strip(stops, w: int = 512, h: int = 40, n_ramp: int = 256) -> Image.Image:
    """Die Rampe als flaches Band — zeigt im UI, worauf abgebildet wird."""
    row = map_luma(np.linspace(0, 1, w), oklch_ramp(stops, n_ramp))
    return Image.fromarray(np.tile(row.round().astype("uint8"), (h, 1, 1)), "RGB")


def clut_file(stops, path: str | None = None) -> str:
    """CLUT als PNG auf Platte; ffmpeg liest sie als zweiten Eingang."""
    if path is None:
        p = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        p.close()
        path = p.name
    hald_clut(stops).save(path)
    return path


# ----------------------------------------------------------------------
# Filtergraph
# ----------------------------------------------------------------------
def graph(w: int, h: int, contrast: float = 1.0, strength: float = 1.0,
          grain: float = 0.0) -> str:
    """Center-Crop/Scale -> Kontrast -> haldclut -> Blend -> Korn.

    Vorschau und Export teilen sich diesen Graphen, damit das Standbild nicht
    anders aussehen kann als das fertig gerenderte Video.
    """
    chain = [f"crop='min(iw,ih*{w}/{h})':'min(ih,iw*{h}/{w})'",
             f"scale={w}:{h}", "setsar=1", "format=rgb24"]
    if abs(contrast - 1.0) > 1e-3:
        chain.append(f"eq=contrast={contrast:.3f}")          # vor dem Mapping
    head = "[0:v]" + ",".join(chain)
    post = f",noise=alls={int(round(grain * 40))}:allf=t+u" if grain > 0 else ""
    if strength >= 0.999:
        return f"{head}[b];[b][1:v]haldclut{post}[v]"
    #  Bei blend ist der ERSTE Eingang die obere Ebene. Die eingefaerbte
    #  Fassung gehoert daher nach vorn, sonst ist der Staerkeregler invertiert.
    return (f"{head},split[a][b];[b][1:v]haldclut[g];"
            f"[g][a]blend=all_mode=normal:all_opacity={strength:.3f}{post}[v]")


def _audio_args(audio_mode: str, has_track: bool) -> list[str]:
    """Originalton aus dem Clip, Track-Audio (Eingang 2) oder stumm.

    Track-Audio ohne Track faellt auf den Originalton zurueck statt auf
    stumm: Ton war gewuenscht, und ein lautloses Video ist das Einzige, was
    in dem Fall sicher falsch ist.
    """
    if audio_mode == "Track-Audio" and has_track:
        return ["-map", "2:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    if audio_mode != "stumm":
        return ["-map", "0:a?", "-c:a", "aac", "-b:a", "192k"]
    return ["-an"]


# ----------------------------------------------------------------------
# Vorschau und Export
# ----------------------------------------------------------------------
def preview_frame(video: str, clut: str, t: float = 1.0,
                  size: tuple[int, int] = (1080, 1920), contrast: float = 1.0,
                  strength: float = 1.0, grain: float = 0.0) -> bytes:
    """Ein einzelner gemappter Frame als PNG. Kein Encoding, deshalb billig.

    Das volle Rendern kostet Minuten; die Regler lassen sich nur sinnvoll
    einstellen, wenn man vorher ein Standbild sieht.
    """
    out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    out.close()
    ffmpeg.run(["-y", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}",
                "-i", video, "-i", clut,
                "-filter_complex", graph(*size, contrast, strength, grain),
                "-map", "[v]", "-frames:v", "1", out.name])
    with open(out.name, "rb") as f:
        return f.read()


def render_clip(video: str, clut: str, out_path: str,
                size: tuple[int, int] = (1080, 1920), contrast: float = 1.0,
                strength: float = 1.0, grain: float = 0.0,
                audio_mode: str = "Originalton", track: str | None = None,
                crf: int = 20, preset: str = "medium",
                maxrate_mbit: int = 12) -> str:
    """Den ganzen Clip einfaerben: mp4, H.264, yuv420p, faststart.

    maxrate deckelt die Bitrate. Ohne Korn aendert das nichts, mit Korn
    verhindert es, dass x264 das Rauschen originalgetreu kodiert: ungebremst
    waechst eine Minute damit auf ueber ein Gigabyte, und die App laedt das
    fertige Video zum Ausliefern komplett in den Speicher.
    """
    args = ["-y", "-loglevel", "error", "-i", video, "-i", clut]
    if audio_mode == "Track-Audio" and track:
        args += ["-i", track]
    args += ["-filter_complex", graph(*size, contrast, strength, grain),
             "-map", "[v]"]
    args += _audio_args(audio_mode, bool(track))
    args += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-maxrate", f"{maxrate_mbit}M", "-bufsize", f"{maxrate_mbit * 2}M",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    ffmpeg.run(args)
    return out_path
