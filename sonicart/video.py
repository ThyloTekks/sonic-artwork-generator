"""Bewegtbild: Canvas und Reel.

Zwei Dinge, die vorher fehlten und bei einer Endlosschleife auffallen:
loopfaehige Laenge (das Video endet auf einem Taktanfang und die Drehung auf
einer ganzen Umdrehung) und Beat-Bindung (die Form pulsiert auf den Schlag,
statt gleichmaessig zu rotieren).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile

import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

from . import ffmpeg
from .analysis import Analysis, punch
from .artwork import Recipe
from .effects import apply_effects
from .export import SIZES
from .palette import hex_to_rgb
from .render import cart_fig, fig_to_pil, polar_fig, resample_time

VIDEO_MODES = ["Rose gespiegelt", "Rose roh", "Live-Kreisspektrum",
               "Live-Oszilloskop", "Spirale (Aufbau)", "Lissajous (Stereo)"]

SPOTIFY_CANVAS = (3.0, 8.0)      # zulaessige Laenge in Sekunden


# ----------------------------------------------------------------------
# Loop-Hilfen
# ----------------------------------------------------------------------
def loop_length(an: Analysis, target_s: float,
                bounds: tuple[float, float] | None = None) -> float:
    """Naechstliegende Laenge, die auf einem Taktanfang endet.

    Ein Canvas laeuft in Endlosschleife: endet das Video mitten im Takt,
    stolpert jede Wiederholung hoerbar und sichtbar.
    """
    bars = an.bar_times()
    lo, hi = bounds or (0.5, an.duration)
    cand = [b for b in bars if lo <= b <= min(hi, an.duration)]
    if not cand:
        return float(np.clip(target_s, lo, min(hi, an.duration)))
    return float(min(cand, key=lambda b: abs(b - target_s)))


def beat_pulse(an: Analysis, times: np.ndarray, decay: float = 6.0) -> np.ndarray:
    """Huellkurve, die auf jedem Beat auf 1 springt und dazwischen abfaellt."""
    bt = an.beat_times
    if len(bt) == 0:
        return np.zeros_like(times)
    idx = np.searchsorted(bt, times, "right") - 1
    before = idx < 0                       # noch kein Schlag gewesen -> kein Puls
    idx = np.clip(idx, 0, len(bt) - 1)
    since = np.clip(times - bt[idx], 0, None)
    return np.where(before, 0.0, np.exp(-decay * since))


# ----------------------------------------------------------------------
# Einzelbilder
# ----------------------------------------------------------------------
def _rose_frame(a, cmap, bg, size, mirror, thickness, inner, rotation,
                data_thickness=True, symmetry=1):
    from .render.modes import _rose_layout
    aa = np.clip(a, 0, 1)
    N = len(aa)
    sym = max(1, int(symmetry))
    ang, base_w = _rose_layout(N, sym, mirror)
    vals = np.tile(aa, 2 * sym if mirror else sym)
    w = base_w * thickness * ((0.4 + 1.2 * vals) if data_thickness else 1.0)
    fig, ax = polar_fig(size, bg, rotation=rotation)
    ax.bar(ang, vals, width=w, bottom=inner, color=cmap(vals), lw=0)
    return fig_to_pil(fig, size)


def _scope_frame(w, cmap, bg, size, thickness, inner, rotation):
    """Aktuelles Wellenform-Fenster als geschlossener Ring (Oszilloskop)."""
    n = len(w)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base = inner + 0.45
    amp = np.clip(w, -1, 1) * 0.3 * thickness
    th = np.append(theta, theta[0])
    rr = np.append(base + amp, base + amp[0])
    fig, ax = polar_fig(size, bg, rotation=rotation, rmax=1.1)
    pts = np.column_stack([th, rr])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    norm = np.abs(np.append(amp, amp[0]))[:-1] / (np.abs(amp).max() + 1e-9)
    ax.add_collection(LineCollection(segs, colors=cmap(norm),
                                     linewidths=2.6 * thickness))
    ax.fill_between(th, base, rr, color=cmap(0.5), alpha=0.28, lw=0)
    return fig_to_pil(fig, size)


def _spiral_frame(M, upto, cmap, bg, size, thickness, inner, rotation, turns):
    """Spirale, bis zum aktuellen Zeitpunkt gezeichnet — sie baut sich auf."""
    nf, nt = M.shape
    k = max(2, int(upto * nt))
    p_full = np.linspace(0, 1, nt + 1)
    theta = 2 * np.pi * turns * p_full[:k + 1]
    band = min((1 - inner) / turns * 0.85 * thickness, (1 - inner) * 0.9)
    r_base = inner + (1 - inner - band) * p_full[:k + 1]
    fi = np.linspace(0, 1, nf + 1)
    TH = np.tile(theta, (nf + 1, 1))
    R = r_base[None, :] + band * fi[:, None]
    fig, ax = polar_fig(size, bg, rotation=rotation, rmax=1.06)
    ax.pcolormesh(TH, R, M[:, :k], cmap=cmap, shading="flat")
    return fig_to_pil(fig, size)


def _lissajous_frame(l, r, cmap, bg, size, thickness, rotation, res=420):
    m = max(np.abs(l).max(), np.abs(r).max()) + 1e-9
    x = (l - r) / np.sqrt(2) / m
    y = (l + r) / np.sqrt(2) / m
    if rotation:
        a = np.deg2rad(rotation)
        x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    H, _, _ = np.histogram2d(x, y, bins=res, range=[[-1, 1], [-1, 1]])
    H = np.log1p(H * (30 * thickness))
    H = H / (H.max() + 1e-9)
    fig, ax = cart_fig(size, bg, lim=1.0)
    ax.imshow(H.T, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap,
              interpolation="bilinear")
    return fig_to_pil(fig, size)


# ----------------------------------------------------------------------
# Ton
# ----------------------------------------------------------------------
def _mux_audio(audio, silent_path, out_path, duration, start=0.0):
    """Originalton (nativ, auf die Videolaenge geschnitten) unter das Video legen."""
    try:
        import librosa
        import soundfile as sf
        if hasattr(audio, "seek"):
            audio.seek(0)
        ay, asr = librosa.load(audio, sr=None, mono=False,
                               offset=start, duration=duration)
        awav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(awav, ay.T if ay.ndim > 1 else ay, asr)
        subprocess.run([ffmpeg.exe(), "-y", "-i", silent_path, "-i", awav,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", out_path], check=True, capture_output=True)
        return True
    except Exception:
        shutil.copy(silent_path, out_path)
        return False


def _canvas_setup(aspect):
    W, H = SIZES[aspect]
    W -= W % 2
    H -= H % 2
    return W, H, int(min(W, H) * 0.96)


# ----------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------
def animate(an: Analysis, recipe: Recipe, out_path: str, audio=None,
            mode: str = "Live-Kreisspektrum", duration: float = 5.0,
            fps: int = 24, aspect: str = "9x16_Story_Reel_Canvas",
            rotate_turns: float = 0.25, spin: bool = True, react: float = 1.0,
            smoothing: float = 0.35, with_audio: bool = True,
            effects: bool = True, loop_safe: bool = True,
            loop_blend: float = 0.15, pulse: float = 0.0,
            pulse_decay: float = 6.0, progress=None) -> dict:
    """Video rendern. Dreht immer rechts herum (Uhrzeigersinn).

    loop_safe zwingt die Laenge auf einen Taktanfang und die Drehung auf ganze
    Umdrehungen — sonst springt das Bild bei jeder Wiederholung. Das allein
    glaettet nur die Geometrie; der Inhalt springt weiter, weil die Musik am
    Ende anders klingt als am Anfang. loop_blend blendet die letzten Bilder
    deshalb zurueck auf das erste (0 = aus).
    pulse bindet Dicke und Innenradius an den Beat (0 = aus).
    """
    import imageio

    p_ = recipe.params
    thick0 = p_.get("thickness", 1.0)
    rot0 = p_.get("rotation", 0.0)
    inner0 = p_.get("inner", 0.15)
    nm = int(p_.get("n_mels", 110))
    sym = int(p_.get("symmetry", 1))
    turns_spiral = float(p_.get("turns", 5.0))
    cmap = recipe.cmap()
    bg = recipe.bg

    if loop_safe:
        duration = loop_length(an, duration, (1.0, an.duration))
        rotate_turns = round(rotate_turns) if spin else 0.0
        if spin and rotate_turns == 0:
            rotate_turns = 1
    duration = float(min(duration, an.duration))
    frames = max(1, int(round(duration * fps)))
    turns = rotate_turns if spin else 0.0

    mirror = (mode != "Rose roh")
    is_scope = mode == "Live-Oszilloskop"
    is_liss = mode == "Lissajous (Stereo)"
    is_spiral = mode == "Spirale (Aufbau)"

    an_v = an.slice(0.0, duration)
    S = None
    if not (is_scope or is_liss):
        S = punch(an_v.mel_norm(nm), p_.get("gate", 0.15), p_.get("gamma", 1.4))
        if is_spiral:
            S = resample_time(S, 1600)
        a_mean = S.mean(axis=1)
        T = S.shape[1]

    feat = an_v.features if effects else None
    fx = recipe.seeded(recipe.audio_name or an.name or "sonic") if effects else None
    W, H, side = _canvas_setup(aspect)
    bg_rgb = hex_to_rgb(bg)

    #  Bei einer Schleife darf das letzte Bild nicht schon wieder das erste sein:
    #  p laeuft dann ueber [0, 1), sonst ueber [0, 1].
    span = float(frames) if loop_safe else float(max(1, frames - 1))
    times = np.linspace(0, duration, frames, endpoint=False)
    pulses = beat_pulse(an_v, times, pulse_decay) * pulse if pulse > 0 else np.zeros(frames)

    target = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name \
        if with_audio else out_path
    writer = imageio.get_writer(target, fps=fps, codec="libx264",
                                quality=8, macro_block_size=None)
    prev = None
    first_art = None
    #  Der Aufbau der Spirale ist von sich aus nicht zyklisch — da wuerde eine
    #  Rueckblende den ganzen Verlauf wieder einreissen.
    blend = 0.0 if (is_spiral or not loop_safe) else float(np.clip(loop_blend, 0, 0.5))
    win = int(an_v.sr / max(1, fps) * 2)
    LR = an_v.lissajous if is_liss else None
    try:
        for f in range(frames):
            p = f / span
            pu = pulses[f]
            rot = rot0 - turns * 360.0 * p                # MINUS = rechts herum
            thick = thick0 * (1 + 0.55 * pu)
            inner = inner0 * (1 - 0.35 * pu)

            if is_liss:
                c = int(p * max(1, LR.shape[1] - win))
                art = _lissajous_frame(LR[0, c:c + win], LR[1, c:c + win],
                                       cmap, bg, side, thick, rot)
            elif is_scope:
                c = int(p * max(1, len(an_v.y) - win))
                seg = an_v.y[c:c + win]
                if len(seg) < win:
                    seg = np.pad(seg, (0, win - len(seg)))
                step = max(1, len(seg) // 720)
                wv = seg[::step][:720] / (np.abs(an_v.y).max() + 1e-9)
                if prev is not None and len(prev) == len(wv):
                    wv = smoothing * prev + (1 - smoothing) * wv
                prev = wv
                art = _scope_frame(wv, cmap, bg, side, thick, inner, rot)
            elif is_spiral:
                art = _spiral_frame(S, max(0.02, p), cmap, bg, side, thick,
                                    inner, rot, turns_spiral)
            else:
                c1 = int(p * (T - 1))
                c0 = int((f - 1) / span * (T - 1)) if f > 0 else c1
                live = S[:, min(c0, c1):max(c0, c1) + 1].mean(axis=1)
                a = (1 - react) * a_mean + react * live
                if prev is not None and len(prev) == len(a):
                    a = smoothing * prev + (1 - smoothing) * a
                prev = a
                if mode == "Live-Kreisspektrum":
                    art = _rose_frame(a, cmap, bg, side, False, thick, inner,
                                      rot, True, sym)
                else:
                    art = _rose_frame(a, cmap, bg, side, mirror, thick, inner,
                                      rot, p_.get("data_thickness", True), sym)

            if blend > 0:
                if f == 0:
                    first_art = art.copy()
                elif p > 1 - blend:
                    w = (p - (1 - blend)) / blend
                    art = Image.blend(art.convert("RGB"),
                                      first_art.convert("RGB"), float(w))
            if fx:
                art = apply_effects(art, feat, bg=bg, **fx)
            canvas = Image.new("RGB", (W, H), bg_rgb)
            canvas.paste(art.convert("RGB"), ((W - side) // 2, (H - side) // 2))
            writer.append_data(np.asarray(canvas))
            if progress:
                progress((f + 1) / frames)
    finally:
        writer.close()

    muxed = False
    if with_audio and audio is not None:
        muxed = _mux_audio(audio, target, out_path, duration, recipe.start)
    elif with_audio:
        shutil.copy(target, out_path)
    return {"path": out_path, "duration": duration, "frames": frames,
            "turns": turns, "audio": muxed, "loop_safe": loop_safe,
            "loop_blend": blend,
            "canvas_ok": SPOTIFY_CANVAS[0] <= duration <= SPOTIFY_CANVAS[1]}
