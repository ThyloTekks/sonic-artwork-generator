"""Farbe: OKLCH-Konvertierung, Palettengenerator, LUT-Import, Druck-Gamut.

Alles Farbliche laeuft ueber OKLCH statt HSL: gleiche Helligkeitsschritte sehen
dort auch gleich aus, was fuer Paletten mit konstanter Anmutung noetig ist.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------
# Presets: nur Defaults. Die Palette ist im UI frei aenderbar.
# ----------------------------------------------------------------------
PRESETS = {
    "Korrend":    ["#08060d", "#3a1d6e", "#7b2ff7", "#c9a0ff"],
    "Type Drift": ["#1a1712", "#5c4a32", "#b08d57", "#e8d8b8"],
    "Seek":       ["#0d0606", "#7a1420", "#e23b2e", "#ffb37a"],
    "Monochrom":  ["#000000", "#666666", "#ffffff"],
}
PROFILE_BG = {"Korrend": "#08060d", "Type Drift": "#1a1712",
              "Seek": "#0d0606", "Monochrom": "#000000"}

HARMONIES = ["Monochrom-Ramp", "Analog", "Komplementaer-Akzent",
             "Triadisch", "Profil-Struktur"]


# ----------------------------------------------------------------------
# Hex <-> RGB
# ----------------------------------------------------------------------
def _expand_hex(h: str) -> str:
    """'#abc' -> 'aabbcc'. Akzeptiert 3- und 6-stellige Hex-Werte."""
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"Kein gueltiger Hex-Farbwert: {h!r}")
    return h


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = _expand_hex(h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_rgb01(h: str) -> tuple[float, float, float]:
    return tuple(v / 255 for v in hex_to_rgb(h))


def rgb01_to_hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def make_cmap(hex_colors):
    return LinearSegmentedColormap.from_list("c", list(hex_colors), N=512)


# ----------------------------------------------------------------------
# sRGB <-> OKLab/OKLCH  (Bjoern Ottosson)
# ----------------------------------------------------------------------
def _srgb_lin(c):
    c = np.asarray(c, float)                     # skalar- und arraytauglich
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_srgb(c):
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def linrgb_to_oklab(r, g, b):
    """Lineares sRGB -> OKLab. Einzige Quelle der Matrizen (skalar oder Array)."""
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def oklab_L(rgb01):
    """Perzeptuelle Helligkeit 0..1 zu sRGB-Werten; letzte Achse ist RGB.

    Nimmt ein einzelnes Tripel ebenso wie ein ganzes Bild. Gedacht fuer
    Gradient Maps, die nach Helligkeit einfaerben: die OKLab-Helligkeit
    trifft das Auge besser als eine gewichtete RGB-Summe.
    """
    lin = _srgb_lin(np.asarray(rgb01, float))
    return linrgb_to_oklab(*np.moveaxis(lin, -1, 0))[0]


def hex_to_oklch(hx: str) -> tuple[float, float, float]:
    L, a, bb = linrgb_to_oklab(*(_srgb_lin(v) for v in hex_to_rgb01(hx)))
    return float(L), float(np.hypot(a, bb)), float(np.arctan2(bb, a))


def _oklch_to_rgb01_raw(L, C, h):
    """Lineare sRGB-Werte ohne Begrenzung — negativ oder >1 heisst: ausserhalb."""
    a, b = C * np.cos(h), C * np.sin(h)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def oklch_to_rgb01(L, C, h):
    a, b = C * np.cos(h), C * np.sin(h)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return (_lin_srgb(r), _lin_srgb(g), _lin_srgb(bl))


def in_srgb(L, C, h, eps=1e-4) -> bool:
    return all(-eps <= v <= 1 + eps for v in _oklch_to_rgb01_raw(L, C, h))


def fit_srgb(L, C, h, steps: int = 24) -> float:
    """Groesste Chroma bei diesem L und h, die noch in sRGB passt.

    Ohne das klemmt die Rueckrechnung einzelne Kanaele auf 0 oder 1 ab und
    veraendert dabei Helligkeit und Farbton — genau das, was eine Palette mit
    fester Helligkeitskurve nicht darf.
    """
    if in_srgb(L, C, h):
        return C
    lo, hi = 0.0, C
    for _ in range(steps):
        mid = (lo + hi) / 2
        if in_srgb(L, mid, h):
            lo = mid
        else:
            hi = mid
    return lo


def oklch_to_hex(L, C, h, fit: bool = True) -> str:
    """OKLCH -> Hex. fit=True senkt bei Bedarf die Chroma statt hart zu klemmen."""
    if fit:
        C = fit_srgb(L, C, h)
    return rgb01_to_hex(oklch_to_rgb01(L, C, h))


def oklab_delta(hex_a: str, hex_b: str) -> float:
    """Wahrnehmungsabstand zweier Farben in OKLab (grob: <0.02 unsichtbar)."""
    La, Ca, ha = hex_to_oklch(hex_a)
    Lb, Cb, hb = hex_to_oklch(hex_b)
    aa, ba = Ca * np.cos(ha), Ca * np.sin(ha)
    ab, bb = Cb * np.cos(hb), Cb * np.sin(hb)
    return float(np.sqrt((La - Lb) ** 2 + (aa - ab) ** 2 + (ba - bb) ** 2))


# ----------------------------------------------------------------------
# Palettengenerator
# ----------------------------------------------------------------------
def generate_palette(anchor_hex, n=4, mode="Monochrom-Ramp", template=None,
                     hue_spread=1.0):
    """Erzeugt aus EINER Ankerfarbe eine stimmige Palette (dunkel -> hell).

    hue_spread skaliert die Farbtonspreizung (1.0 = Standard). Die Analyse
    kann darueber die harmonische Komplexitaet eines Stuecks einspeisen.
    """
    L0, C0, h0 = hex_to_oklch(anchor_hex)
    C0 = max(C0, 0.06)
    deg = np.deg2rad

    if mode == "Profil-Struktur" and template:
        tl = [hex_to_oklch(c) for c in template]
        h_ref = tl[-1][2]
        return [oklch_to_hex(L, C, h0 + (h - h_ref) * hue_spread)
                for (L, C, h) in tl]

    Ls = np.linspace(0.12, 0.95, n)
    chroma_env = 0.55 + 0.45 * (1 - np.abs(2 * np.linspace(0, 1, n) - 1))
    stops = []
    for i, L in enumerate(Ls):
        C = C0 * chroma_env[i]
        if mode == "Monochrom-Ramp":
            h = h0
        elif mode == "Analog":
            h = h0 + deg(35) * hue_spread * (i / max(1, n - 1) - 0.5) * 2
        elif mode == "Komplementaer-Akzent":
            h = h0 if i < n - 1 else h0 + np.pi * hue_spread
            if i == n - 1:
                C = C0
        elif mode == "Triadisch":
            h = h0 + [0, 2 * np.pi / 3, -2 * np.pi / 3][i % 3] * hue_spread
        else:
            h = h0
        stops.append(oklch_to_hex(L, C, h))
    return stops


# ----------------------------------------------------------------------
# Tonart -> Farbton (Quintenzirkel)
# ----------------------------------------------------------------------
#  Quintenzirkel statt chromatischer Skala: benachbarte Tonarten sind
#  harmonisch verwandt und bekommen damit auch verwandte Farben.
CIRCLE_OF_FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def key_to_hue(tonic: str, mode: str = "dur", offset_deg: float = 0.0) -> float:
    """Tonart -> OKLCH-Farbton (Radiant). Moll wird um 30 Grad versetzt."""
    try:
        pos = CIRCLE_OF_FIFTHS.index(tonic)
    except ValueError:
        pos = 0
    h = 2 * np.pi * pos / 12 + np.deg2rad(offset_deg)
    if mode.lower().startswith("moll") or mode.lower().startswith("min"):
        h += np.deg2rad(30)
    return float(h)


def palette_from_key(tonic, mode="dur", n=4, harmony="Monochrom-Ramp",
                     template=None, chroma=0.14, hue_spread=1.0,
                     offset_deg=0.0):
    """Palette direkt aus der erkannten Tonart. Moll wird dunkler/matter."""
    h = key_to_hue(tonic, mode, offset_deg)
    is_minor = mode.lower().startswith(("moll", "min"))
    L = 0.58 if not is_minor else 0.48
    C = chroma if not is_minor else chroma * 0.78
    anchor = oklch_to_hex(L, C, h)
    return generate_palette(anchor, n=n, mode=harmony, template=template,
                            hue_spread=hue_spread)


# ----------------------------------------------------------------------
# .cube-LUT
# ----------------------------------------------------------------------
def cmap_from_cube(path, n=9):
    size = None
    dim = 3
    data = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("#", "TITLE", "DOMAIN_")):
                continue
            if s.startswith("LUT_3D_SIZE"):
                size = int(s.split()[-1]); dim = 3; continue
            if s.startswith("LUT_1D_SIZE"):
                size = int(s.split()[-1]); dim = 1; continue
            if s.startswith("LUT_"):
                continue
            p = s.split()
            if len(p) == 3:
                try:
                    data.append([float(v) for v in p])
                except ValueError:
                    pass
    data = np.array(data)
    if size is None or len(data) == 0:
        raise ValueError("Kein gueltiges .cube LUT")
    if dim == 1:
        ramp = data[np.linspace(0, size - 1, n).round().astype(int)]
    else:
        ks = np.linspace(0, size - 1, n).round().astype(int)
        ramp = data[ks * (1 + size + size * size)]      # Diagonale des Wuerfels
    return LinearSegmentedColormap.from_list("lut", np.clip(ramp, 0, 1), N=512)


# ----------------------------------------------------------------------
# Druck-Gamut (Naeherung, ohne ICC-Profil)
# ----------------------------------------------------------------------
#  Volltoene eines typischen Bogenoffsets auf gestrichenem Papier
#  (FOGRA39-nah). Daraus wird eine Hue -> Max-Chroma-Huelle interpoliert.
_PRINT_PRIMARIES = ["#0086c8", "#2e2382", "#de0074",
                    "#e1000f", "#ffed00", "#009f4a"]
_PRINT_RING = None


def _print_ring():
    """(hue, chroma, lightness) der Druckprimaerfarben, nach Hue sortiert."""
    global _PRINT_RING
    if _PRINT_RING is None:
        ring = [hex_to_oklch(c) for c in _PRINT_PRIMARIES]
        ring = sorted(((h, C, L) for (L, C, h) in ring), key=lambda t: t[0])
        _PRINT_RING = ring
    return _PRINT_RING


def max_print_chroma(L: float, h: float) -> float:
    """Groesste im Offsetdruck erreichbare Chroma bei Helligkeit L, Farbton h.

    Naeherung ueber einen Gamut-Kegel: bei der Helligkeit der Primaerfarbe ist
    die volle Chroma erreichbar, gegen Schwarz und Weiss laeuft sie auf 0 zu.
    """
    ring = _print_ring()
    hs = np.array([r[0] for r in ring])
    cs = np.array([r[1] for r in ring])
    ls = np.array([r[2] for r in ring])
    hh = (h + np.pi) % (2 * np.pi) - np.pi                  # nach [-pi, pi)
    hs_w = np.concatenate([hs - 2 * np.pi, hs, hs + 2 * np.pi])
    cs_w = np.tile(cs, 3)
    ls_w = np.tile(ls, 3)
    C_ref = float(np.interp(hh, hs_w, cs_w))
    L_ref = float(np.interp(hh, hs_w, ls_w))
    if L <= L_ref:
        f = L / max(L_ref, 1e-6)
    else:
        f = (1 - L) / max(1 - L_ref, 1e-6)
    return C_ref * max(0.0, min(1.0, f))


def clip_to_print(hex_color: str) -> str:
    """Naechstliegende druckbare Farbe (Chroma kappen, Hue und L halten)."""
    L, C, h = hex_to_oklch(hex_color)
    cmax = max_print_chroma(L, h)
    return oklch_to_hex(L, min(C, cmax), h) if C > cmax else hex_color


#  Unterhalb dieser Chroma ist eine Farbe praktisch neutral. Schwarz, Weiss und
#  Grau haben rechnerisch eine Maximalchroma von 0 — ohne diese Schwelle waere
#  jeder Quotient dort unendlich und Weiss galte als nicht druckbar.
NEUTRAL_CHROMA = 0.01


def print_check(stops) -> list[dict]:
    """Pro Farbstufe: druckbar? wie stark uebersaettigt? wie sieht sie gedruckt aus?

    Naeherung ohne ICC-Profil — als Warnsignal gedacht, nicht als Proof.
    """
    out = []
    for c in stops:
        L, C, h = hex_to_oklch(c)
        cmax = max_print_chroma(L, h)
        neutral = C <= NEUTRAL_CHROMA
        clipped = c if neutral else clip_to_print(c)
        ratio = 0.0 if neutral else float(C / max(cmax, NEUTRAL_CHROMA))
        out.append({
            "hex": c,
            "printable": neutral or C <= cmax * 1.02,
            "ratio": ratio,
            "clipped": clipped,
            "delta": oklab_delta(c, clipped),
        })
    return out
