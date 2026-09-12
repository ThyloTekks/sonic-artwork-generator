"""Die Bildmodi. Jeder bekommt eine fertige Analysis und gibt eine Figure zurueck.

Registrierung ueber MODES: dort steht auch, welche Regler ein Modus ueberhaupt
auswertet. Das UI blendet danach aus — vorher nahm z. B. die Kreis-Wellenform
'n_mels' entgegen und ignorierte es stillschweigend.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from ..analysis import Analysis, punch
from .base import cart_fig, cmap_to_alpha, flat_fig, polar_fig, resample_time

# ----------------------------------------------------------------------
# Rose — gemitteltes Spektrum als Balkenkranz
# ----------------------------------------------------------------------
def _rose_layout(N, sym, mirror):
    """Winkel und Grundbreite der Balken bei sym-zaehliger Symmetrie."""
    seg = 2 * np.pi / sym
    if mirror:
        span = seg / 2
        h = np.linspace(0, span, N)
        parts = []
        for k in range(sym):
            b = np.pi / 2 + k * seg
            parts += [b - h, b + h]
        return np.concatenate(parts), span / N
    parts = [k * seg + np.linspace(0, seg, N, endpoint=False) for k in range(sym)]
    return np.concatenate(parts), seg / N


def render_rose(an: Analysis, cmap, bg, size_px=3000, mirror=True,
                thickness=1.0, gate=0.15, gamma=1.4, inner=0.15,
                n_mels=110, data_thickness=True, transparent=False,
                rotation=0.0, symmetry=1, source="mix", tilt=0.0):
    """Klangfingerabdruck: das ueber die Zeit gemittelte Spektrum.

    Zeitblind — zwei Stuecke mit gleichem Frequenzhaushalt sehen aehnlich aus.
    Fuer eine Abbildung des Verlaufs 'Spirale', 'HPSS-Zeit' oder 'Segmente'.
    """
    a = an.spectrum(n_mels, gate, gamma, source, tilt)
    N = len(a)
    sym = max(1, int(symmetry))
    ang, base_w = _rose_layout(N, sym, mirror)
    vals = np.tile(a, 2 * sym if mirror else sym)
    w = base_w * thickness * ((0.4 + 1.2 * vals) if data_thickness else 1.0)
    fig, ax = polar_fig(size_px, bg, transparent=transparent, rotation=rotation)
    ax.bar(ang, vals, width=w, bottom=inner, color=cmap(vals), lw=0)
    return fig


# ----------------------------------------------------------------------
# HPSS-Zeit — Spektrogramm als Ring, perkussive Akzente obenauf
# ----------------------------------------------------------------------
def render_hpss_time(an: Analysis, cmap, bg, size_px=3000,
                     thickness=1.0, gate=0.5, gamma=1.7, inner=0.15,
                     onset=0.25, n_mels=120, transparent=False, rotation=0.0,
                     spoke_len=0.35, cols=1800, whiten_amount=0.0):
    Sh = resample_time(punch(an.mel_norm(n_mels, "harmonic", whiten_amount),
                             gate, gamma), cols)
    Sp = resample_time(an.mel_norm(n_mels, "percussive", whiten_amount), cols)
    fig, ax = polar_fig(size_px, bg, transparent=transparent, rotation=rotation)
    nf, nt = Sh.shape
    th = np.linspace(0, 2 * np.pi, nt + 1)
    r = np.linspace(inner, 1, nf + 1)
    base_cmap = cmap_to_alpha(cmap) if transparent else cmap
    ax.pcolormesh(th, r, Sh, cmap=base_cmap, shading="flat")
    acc = LinearSegmentedColormap.from_list(
        "a", [(0.95, 0.9, 1, 0), (1.0, 0.85, 1, 1)], N=256)
    ax.pcolormesh(th, r, np.ma.masked_less(Sp, 0.45), cmap=acc, shading="flat")
    if onset > 0:
        oenv = an.onset_env
        om = oenv.max() + 1e-9
        outer = 1.03
        for f in an.onsets:
            s = oenv[min(f, len(oenv) - 1)] / om
            if s < 0.30:                               # nur markante Onsets
                continue
            L = spoke_len * (0.4 + 0.6 * s)            # Laenge ~ Staerke
            ang = 2 * np.pi * f / len(oenv)
            ax.plot([ang, ang], [outer - L, outer], color="#f4ecff",
                    lw=(1.8 + 2.6 * s) * thickness,
                    alpha=onset * (0.55 + 0.45 * s), solid_capstyle="round")
    return fig


# ----------------------------------------------------------------------
# Kreis-Wellenform
# ----------------------------------------------------------------------
def render_wave_ring(an: Analysis, cmap, bg, size_px=3000, thickness=1.0,
                     gamma=1.4, inner=0.15, transparent=False, rotation=0.0,
                     cols=1400):
    y = an.y
    hop = max(1, len(y) // cols)
    usable = hop * cols
    env = np.abs(y[:usable]).reshape(cols, hop).max(axis=1)
    env = (env / (env.max() + 1e-9)) ** gamma        # Kontrast der Welle
    theta = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    base = inner + 0.45
    amp = env * 0.24 * thickness
    th = np.append(theta, theta[0])
    ro = np.append(base + amp, base + amp[0])
    ri = np.append(base - amp, base - amp[0])
    fig, ax = polar_fig(size_px, bg, transparent=transparent, rotation=rotation,
                        rmax=1.1)
    ax.fill_between(th, ri, ro, color=cmap(0.55), alpha=0.5, lw=0)
    for pts in (np.column_stack([th, ro]), np.column_stack([th, ri])):
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        ax.add_collection(LineCollection(
            segs, colors=cmap(th[:-1] / (2 * np.pi)), linewidths=2.2 * thickness))
    return fig


# ----------------------------------------------------------------------
# Spirale — Zeit als Radius. Der ganze Song in einem Bild, wie Jahresringe.
# ----------------------------------------------------------------------
def render_spiral(an: Analysis, cmap, bg, size_px=3000, thickness=1.0,
                  gate=0.15, gamma=1.4, inner=0.10, n_mels=110,
                  transparent=False, rotation=0.0, turns=5.0, source="mix",
                  cols=2600, mark_segments=True, whiten_amount=0.0):
    """Anfang innen, Ende aussen; die Frequenz liegt quer im Band.

    Loest die Zeitblindheit der Rose: Aufbau, Drop und Ausklang sind als
    Ringabschnitte sichtbar, und ein Remix sieht anders aus als das Original.
    """
    M = resample_time(punch(an.mel_norm(n_mels, source, whiten_amount),
                            gate, gamma), int(cols))
    nf, nt = M.shape
    turns = max(0.5, float(turns))
    p = np.linspace(0, 1, nt + 1)
    theta = 2 * np.pi * turns * p
    band = (1 - inner) / turns * 0.85 * thickness
    band = min(band, (1 - inner) * 0.9)
    r_base = inner + (1 - inner - band) * p
    fi = np.linspace(0, 1, nf + 1)
    TH = np.tile(theta, (nf + 1, 1))
    R = r_base[None, :] + band * fi[:, None]
    fig, ax = polar_fig(size_px, bg, transparent=transparent, rotation=rotation,
                        rmax=1.06)
    cm = cmap_to_alpha(cmap) if transparent else cmap
    ax.pcolormesh(TH, R, M, cmap=cm, shading="flat")
    if mark_segments:
        bounds = an.segments()
        total = max(1, an.n_frames - 1)
        for b in bounds[1:-1]:
            q = np.clip(b / total, 0, 1)
            th0 = 2 * np.pi * turns * q
            r0 = inner + (1 - inner - band) * q
            ax.plot([th0, th0], [r0, r0 + band], color=cmap(0.98),
                    lw=1.6 * thickness, alpha=0.85, solid_capstyle="round")
    return fig


# ----------------------------------------------------------------------
# Segmente — die Songstruktur als Karte
# ----------------------------------------------------------------------
def render_segments(an: Analysis, cmap, bg, size_px=3000, thickness=1.0,
                    gate=0.15, gamma=1.4, inner=0.18, n_mels=110,
                    transparent=False, rotation=0.0, n_segments=0,
                    gap_deg=2.0, source="mix", label_ring=True,
                    whiten_amount=0.0):
    """Jeder Abschnitt ein Kreissektor, Sektorbreite = seine Dauer."""
    k = int(n_segments) or None
    bounds = an.segments(k)
    M = punch(an.mel_norm(n_mels, source, whiten_amount), gate, gamma)
    nseg = max(1, len(bounds) - 1)
    total = max(1, bounds[-1] - bounds[0])
    gap = np.deg2rad(gap_deg)
    fig, ax = polar_fig(size_px, bg, transparent=transparent, rotation=rotation,
                        rmax=1.06)
    cm = cmap_to_alpha(cmap) if transparent else cmap
    cursor = np.pi / 2
    for i in range(nseg):
        b0, b1 = int(bounds[i]), int(bounds[i + 1])
        frac = (b1 - b0) / total
        th1 = cursor + 2 * np.pi * frac
        piece = M[:, b0:max(b1, b0 + 2)]
        piece = resample_time(piece, max(8, int(2200 * frac)))
        nf, nt = piece.shape
        th = np.linspace(cursor + gap / 2, th1 - gap / 2, nt + 1)
        r = np.linspace(inner, 0.90, nf + 1)
        if nt >= 1 and th1 - cursor > gap:
            ax.pcolormesh(th, r, piece, cmap=cm, shading="flat")
            if label_ring:
                arc = np.linspace(cursor + gap / 2, th1 - gap / 2, 48)
                ax.plot(arc, np.full_like(arc, 0.97),
                        lw=7 * thickness, solid_capstyle="butt",
                        color=cmap(i / max(1, nseg - 1)))
        cursor = th1
    return fig


# ----------------------------------------------------------------------
# Lissajous — Stereobild als Goniometer
# ----------------------------------------------------------------------
def render_lissajous(an: Analysis, cmap, bg, size_px=3000, thickness=1.0,
                     gamma=1.4, transparent=False, rotation=0.0,
                     points=400000, res=900):
    """Links gegen Rechts. Die einzige Darstellung hier, die echte
    Stereoinformation zeigt statt sie vorher wegzumitteln.

    Mono-Quellen haetten nur eine Diagonale — dafuer nimmt Analysis.lissajous
    ersatzweise die Hilbert-Phase als zweite Achse.
    """
    LR = an.lissajous
    n = LR.shape[1]
    step = max(1, n // int(points))
    l = LR[0, ::step].astype(np.float64)
    r = LR[1, ::step].astype(np.float64)
    m = max(np.abs(l).max(), np.abs(r).max()) + 1e-9
    # Goniometer-Konvention: Mitte senkrecht, Seite waagerecht (45 Grad gedreht)
    x = (l - r) / np.sqrt(2) / m
    y = (l + r) / np.sqrt(2) / m
    if rotation:
        a = np.deg2rad(rotation)
        x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    H, _, _ = np.histogram2d(x, y, bins=int(res), range=[[-1, 1], [-1, 1]])
    H = np.log1p(H * (10 * thickness))
    H = (H / (H.max() + 1e-9)) ** (1 / max(0.2, gamma))
    fig, ax = cart_fig(size_px, bg, transparent=transparent, lim=1.0)
    cm = cmap_to_alpha(cmap, knee=0.02) if transparent else cmap
    ax.imshow(H.T, origin="lower", extent=(-1, 1, -1, 1), cmap=cm,
              interpolation="bilinear")
    return fig


# ----------------------------------------------------------------------
# Gitter — ein Feld je Takt. Kein Zentrum, keine Radialsymmetrie.
# ----------------------------------------------------------------------
def render_gitter(an: Analysis, cmap, bg, size_px=3000, n_mels=96,
                  stufen=5, gap=0.055, transparent=False, whiten_amount=0.85,
                  source="mix", spalten=0, gamma=1.0):
    """Der Bogen des Stuecks als Raster: ein Feld je Takt, flacher Tonwert.

    Rangnormiert — Takt-Mittelwerte liegen dicht beieinander, nach Min-Max
    landen fast alle in derselben Stufe und das Raster kippt tonwertlich
    zusammen. Erst der Rang nutzt den Umfang aus.
    """
    from matplotlib.patches import Rectangle
    v = an.bar_values(n_mels, whiten_amount, source) ** max(0.1, gamma)
    stufen = max(2, int(stufen))
    v = np.round(v * (stufen - 1)) / (stufen - 1)
    n = len(v)
    sp = int(spalten) if spalten and spalten > 0 else int(np.ceil(np.sqrt(n)))
    ze = int(np.ceil(n / sp))
    fig, ax = flat_fig(size_px, bg, transparent=transparent)
    zw, zh = 1.0 / sp, 1.0 / ze
    m = min(zw, zh) * float(np.clip(gap, 0, 0.45))
    for i, val in enumerate(v):
        r, c = divmod(i, sp)
        ax.add_patch(Rectangle((c * zw + m, r * zh + m),
                               zw - 2 * m, zh - 2 * m,
                               facecolor=cmap(float(val)), lw=0))
    return fig


# ----------------------------------------------------------------------
# Strata — waagerechte Schichten, vollflaechig
# ----------------------------------------------------------------------
def render_strata(an: Analysis, cmap, bg, size_px=3000, n_mels=96,
                  baender=7, amp=0.8, glaette=0.0, transparent=False,
                  whiten_amount=0.85, source="mix", gamma=1.0,
                  farbe_nach_band=False):
    """Je Frequenzband eine waagerechte Schicht, deren Dicke dem Verlauf folgt.

    Wenige dicke Baender statt vieler duenner: bei mehr als etwa zehn
    verschwimmt das Bild zu Moire, und der Aufbau des Stuecks ist nicht mehr
    lesbar.
    """
    K = an.band_curves(int(baender), n_mels, whiten_amount, glaette, source)
    K = np.clip(K, 0, 1) ** max(0.1, gamma)
    nb, nt = K.shape
    x = np.linspace(0, 1, nt)
    fig, ax = flat_fig(size_px, bg, transparent=transparent)
    for i in range(nb):
        k = K[i]
        basis = (i + 0.5) / nb
        dicke = (1.0 / nb) * 0.46 * (0.16 + amp * k)
        farbe = cmap(i / max(1, nb - 1)) if farbe_nach_band else cmap(float(k.mean()))
        ax.fill_between(x, basis - dicke, basis + dicke, color=farbe, lw=0)
    return fig


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
@dataclass
class Mode:
    """Ein Bildmodus samt der Auskunft, welche Regler er auswertet.

    'params' wird aus der Signatur der Zeichenfunktion gelesen statt von Hand
    gepflegt. Damit kann kein Regler mehr im UI stehen, den der Modus gar nicht
    kennt — und keiner mehr durchgereicht werden, den er nicht annimmt.
    """

    name: str
    fn: Callable
    time_aware: bool = False
    vector: bool = True          # taugt fuer SVG-Export
    stereo: bool = False         # wertet echte Stereoinformation aus
    full_bleed: bool = False     # fuellt die ganze Flaeche, nicht nur eine Scheibe
    hint: str = ""
    params: frozenset = field(init=False)

    def __post_init__(self):
        sig = inspect.signature(self.fn)
        names = {n for n, p in sig.parameters.items()
                 if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        bound = set(getattr(self.fn, "keywords", None) or ())   # per partial gesetzt
        self.params = frozenset(names - bound - {"an", "cmap", "bg", "size_px"})

    def knobs(self) -> frozenset:
        """Regler fuer das UI — ohne die technischen Durchreicher."""
        return self.params - {"transparent", "cols", "res", "points"}

    def call(self, an, **kw):
        allowed = self.params | {"cmap", "bg", "size_px"}
        return self.fn(an, **{k: v for k, v in kw.items() if k in allowed})


MODES: dict[str, Mode] = {
    m.name: m for m in [
        Mode("Rose gespiegelt", partial(render_rose, mirror=True),
             hint="Gemitteltes Spektrum, gespiegelt. Zeitblind."),
        Mode("Rose roh", partial(render_rose, mirror=False),
             hint="Gemitteltes Spektrum, voller Kreis. Zeitblind."),
        Mode("Spirale", render_spiral, time_aware=True,
             hint="Zeit als Radius: innen Anfang, aussen Ende."),
        Mode("Segmente", render_segments, time_aware=True,
             hint="Songstruktur als Sektoren, Breite = Dauer des Abschnitts."),
        Mode("HPSS-Zeit", render_hpss_time, time_aware=True,
             hint="Harmonisch als Flaeche, perkussiv als Akzent."),
        Mode("Kreis-Wellenform", render_wave_ring, time_aware=True,
             hint="Huellkurve als geschlossener Ring."),
        Mode("Lissajous (Stereo)", render_lissajous, time_aware=True,
             vector=False, stereo=True,
             hint="Goniometer aus L/R. Braucht eine Stereodatei."),
        Mode("Gitter", render_gitter, time_aware=True, full_bleed=True,
             hint="Ein Feld je Takt. Kein Zentrum, kein Kreis."),
        Mode("Strata", render_strata, time_aware=True, full_bleed=True,
             hint="Waagerechte Schichten, vollflaechig. Aufbau laeuft nach rechts."),
    ]
}

DEFAULT_GATE = {"HPSS-Zeit": 0.5}


def mode_params(mode: str, base: dict) -> dict:
    """Reglerwerte auf einen Modus zuschneiden und Modus-Eigenheiten setzen."""
    p = dict(base)
    if mode == "HPSS-Zeit":
        p.setdefault("onset", 0.25)
        p["gate"] = max(p.get("gate", 0.5), 0.4)     # HPSS braucht hoeheres Gate
    return {k: v for k, v in p.items() if k in MODES[mode].params}


def render(an: Analysis, mode: str, cmap, bg, size_px=3000, params=None,
           transparent=False):
    """Einzelnen Modus zeichnen -> matplotlib Figure."""
    p = mode_params(mode, params or {})
    p["transparent"] = transparent
    return MODES[mode].call(an, cmap=cmap, bg=bg, size_px=size_px, **p)
