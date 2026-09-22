"""Risodruck als Ausgabeschicht.

Der Grund, warum die bisherige Ausgabe nach Bildschirmschoner aussieht, ist
nicht der Modus, sondern das Material: additives Leuchten auf Schwarz. Hier
wird dasselbe Bild stattdessen als Druck behandelt — flache Farben, ein
Halbtonraster je Farbe mit eigenem Winkel, Passerversatz zwischen den Platten,
Multiply statt Screen, Papier statt Schwarz.

Die Schicht arbeitet auf dem fertig gerenderten Bild und gilt deshalb fuer
jeden Modus, auch fuer einen Mix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image, ImageFilter

from .palette import hex_to_rgb, oklab_L

#  Erprobte Farbsaetze. Reihenfolge: hellste zuerst, dunkelste zuletzt —
#  die Platten drucken in dieser Reihenfolge uebereinander.
INK_SETS = {
    "Zinnober":   {"inks": ["#e2542c", "#f2b134", "#2b2b30"], "paper": "#efe9dd"},
    "Indigo":     {"inks": ["#e8dcc4", "#4b47a8", "#141420"], "paper": "#efe9dd"},
    "Flaschengruen": {"inks": ["#2f6b4f", "#1e2422"],         "paper": "#f2efe6"},
    "Ocker":      {"inks": ["#e8b04b", "#c4452f", "#1d2b3a"], "paper": "#efe9dd"},
    "Graphit":    {"inks": ["#8d8a84", "#2a2a2e"],            "paper": "#f0ede7"},
    "Nachtdruck": {"inks": ["#3a3f6b", "#c9c2b2"],            "paper": "#14141a"},
}

#  Klassische Rasterwinkel des Vierfarbdrucks. Gleiche Winkel auf zwei Platten
#  erzeugen Moire, deshalb je Platte ein eigener.
DEFAULT_ANGLES = (15.0, 75.0, 45.0, 0.0)
DEFAULT_OFFSETS = ((0, 0), (2, -1), (-1, 2), (1, 1))


@dataclass
class RisoSpec:
    """Einstellung der Druckschicht."""

    inks: list = field(default_factory=lambda: list(INK_SETS["Zinnober"]["inks"]))
    paper: str = "#efe9dd"
    cell: float = 5.0            # Rasterweite in px, bezogen auf 1000 px Breite
    angles: tuple = DEFAULT_ANGLES
    offsets: tuple = DEFAULT_OFFSETS
    misregister: float = 1.0     # Skalierung des Passerversatzes (0 = perfekter Passer)
    texture: float = 0.06        # Papierfaser
    gain: float = 1.0            # Tonwertzunahme: >1 druckt fetter
    invert: bool = False         # Tonwerte tauschen
    overprint: str = "auto"      # "multiply" | "screen" | "auto"
    seed: int = 3

    def effective_overprint(self) -> str:
        """Auf dunklem Papier deckt man auf, statt zu lasieren.

        Multiply kann nur abdunkeln — helle Farbe auf dunklem Karton ist aber
        Deckfarbe. Ohne diese Unterscheidung laufen dunkle Papiere zu Schwarz zu.
        """
        if self.overprint in ("multiply", "screen"):
            return self.overprint
        r, g, b = (v / 255 for v in hex_to_rgb(self.paper))
        papier_hell = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "screen" if papier_hell < 0.35 else "multiply"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "RisoSpec | None":
        if not d:
            return None
        known = set(cls.__dataclass_fields__)
        spec = cls(**{k: v for k, v in d.items() if k in known})
        spec.angles = tuple(spec.angles)
        spec.offsets = tuple(tuple(o) for o in spec.offsets)
        return spec


def density(img: Image.Image, bg: str) -> np.ndarray:
    """Wieviel Farbe liegt an jeder Stelle? 0 = Papier, 1 = Vollton.

    Gemessen als Abstand zur Hintergrundfarbe, damit helle Formen auf dunklem
    Grund und dunkle auf hellem gleich behandelt werden.
    """
    arr = np.asarray(img).astype(float) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3] if arr.shape[2] == 4 else None
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    bg_lum = float(np.array(hex_to_rgb(bg)) / 255.0 @ np.array([0.2126, 0.7152, 0.0722]))
    spann = max(bg_lum, 1.0 - bg_lum, 1e-6)
    d = np.abs(lum - bg_lum) / spann
    if alpha is not None:
        d = d * alpha + (1 - alpha) * 0.0        # ausserhalb der Form kein Auftrag
    hi = np.percentile(d, 99.5)
    return np.clip(d / (hi + 1e-9), 0, 1)


def halftone_mask(v: np.ndarray, cell: float, angle_deg: float) -> np.ndarray:
    """Punktraster unter einem Winkel: grosser Wert -> grosser Punkt."""
    H, W = v.shape
    a = np.deg2rad(angle_deg)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xr = xx * np.cos(a) - yy * np.sin(a)
    yr = xx * np.sin(a) + yy * np.cos(a)
    dx = (xr % cell) / cell - 0.5
    dy = (yr % cell) / cell - 0.5
    return v > (np.sqrt(dx * dx + dy * dy) / 0.7071)


def paper_texture(arr: np.ndarray, staerke: float, seed: int = 3) -> np.ndarray:
    """Papierfaser — leichte, weiche Helligkeitsschwankung."""
    if staerke <= 0:
        return arr
    H, W = arr.shape[:2]
    rng = np.random.default_rng(seed)
    faser = rng.normal(0.5, 0.25, (H, W))
    faser = np.asarray(Image.fromarray(np.clip(faser * 255, 0, 255).astype("uint8"))
                       .filter(ImageFilter.GaussianBlur(max(0.4, W / 1400)))
                       ).astype(float) / 255.0
    return np.clip(arr * (1 - staerke + 2 * staerke * faser[..., None]), 0, 1)


def _helligkeit(hex_farbe: str) -> float:
    """Wahrgenommene Helligkeit einer Druckfarbe, 0..1."""
    return float(oklab_L(np.array(hex_to_rgb(hex_farbe)) / 255.0))


def apply_riso(img: Image.Image, spec: RisoSpec, bg: str = "#000000",
               keep_alpha: bool = False) -> Image.Image:
    """Fertiges Bild -> Druck. Gibt RGB zurueck (oder RGBA, wenn gewuenscht)."""
    if not spec or not spec.inks:
        return img
    W = img.width
    skala = W / 1000.0
    zelle = max(2.0, spec.cell * skala)

    d = density(img, bg)
    if spec.invert:
        d = 1.0 - d
    d = np.clip(d * spec.gain, 0, 1)

    papier = np.array(hex_to_rgb(spec.paper)) / 255.0
    out = np.ones((*d.shape, 3)) * papier
    modus = spec.effective_overprint()
    n = len(spec.inks)
    #  Welches Tonwertband eine Farbe bekommt, entscheidet ihre Helligkeit,
    #  nicht ihre Position in der Liste. Farbsaetze fuehren die dunkle Farbe
    #  zuletzt, eine Palette dagegen zuerst (dunkel -> hell). Nach Position zu
    #  gehen legte deshalb bei jeder geladenen Palette das Fast-Schwarz auf das
    #  unterste Band — und weil jede Platte alles Dunklere voll ueberdruckt,
    #  zog Multiply die ganze Flaeche nach Schwarz: alle Paletten sahen gleich
    #  aus. Hellste Farbe auf die hellsten Toene, dunkelste auf die tiefsten.
    nach_helligkeit = sorted(range(n), key=lambda i: -_helligkeit(spec.inks[i]))
    for platz, i in enumerate(nach_helligkeit):
        farbe = spec.inks[i]
        lo, hi = platz / n, (platz + 1) / n
        # Tonwertbereich dieser Platte; alles Dunklere wird voll ueberdruckt.
        platte = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
        platte = np.maximum(platte, (d > hi).astype(float))
        #  Winkel und Versatz haengen am Band, nicht an der Listenposition.
        #  Die Winkel sollen sich nur voneinander unterscheiden (sonst Moire);
        #  welche Farbe welchen bekommt, ist gleichgueltig. So ist der Druck
        #  vollstaendig unabhaengig davon, in welcher Reihenfolge die Farben
        #  hereinkommen.
        maske = halftone_mask(platte, zelle,
                              spec.angles[platz % len(spec.angles)])
        dy, dx = spec.offsets[platz % len(spec.offsets)]
        versatz = spec.misregister * skala
        maske = np.roll(np.roll(maske, int(round(dy * versatz)), 0),
                        int(round(dx * versatz)), 1)
        tinte = np.array(hex_to_rgb(farbe)) / 255.0
        if modus == "screen":
            gedruckt = 1.0 - (1.0 - out) * (1.0 - tinte)
        else:
            gedruckt = out * tinte
        out = np.where(maske[..., None], gedruckt, out)

    out = paper_texture(out, spec.texture, spec.seed)
    bild = Image.fromarray(np.clip(out * 255, 0, 255).astype("uint8"), "RGB")
    if keep_alpha and img.mode == "RGBA":
        bild = bild.convert("RGBA")
        bild.putalpha(img.getchannel("A"))
    return bild


def ink_set(name: str) -> RisoSpec:
    s = INK_SETS[name]
    return RisoSpec(inks=list(s["inks"]), paper=s["paper"])
