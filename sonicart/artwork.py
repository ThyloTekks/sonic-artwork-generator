"""Das Rezept und die Pipeline.

Recipe ist die eine Darstellung einer Einstellung: als Preset speicherbar, in
den PNG-Export einbettbar und daraus wieder auslesbar. Aus einem exportierten
Bild laesst sich damit das exakte Setup rekonstruieren — wichtig, wenn ein
Release ein Jahr spaeter noch einmal in anderer Groesse gebraucht wird.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from .analysis import Analysis
from .compose import apply_image, compose, mix_modes, render_pil
from .effects import apply_effects
from .palette import PRESETS, PROFILE_BG, make_cmap
from .typography import TypeSpec, add_signet, apply_typography_report

#  3: Timbre-Features auf dBFS und Oktavskala umgestellt. Ein Rezept aus
#     Version <= 2 ergibt mit derselben Einstellung eine andere Effektstaerke.
RECIPE_VERSION = 3

DEFAULT_PARAMS = dict(thickness=1.0, gate=0.15, gamma=1.5, inner=0.15,
                      n_mels=110, rotation=0.0, data_thickness=True,
                      symmetry=1, source="mix", turns=5.0, onset=0.25,
                      spoke_len=0.35, n_segments=0, gap_deg=2.0,
                      mark_segments=True, label_ring=True,
                      whiten_amount=0.0, tilt=0.0, stufen=5, gap=0.055, spalten=0,
                      baender=7, amp=0.8, glaette=0.0, farbe_nach_band=False)

#  Matter Ausgangspunkt: Papier statt Schwarz, kein Bloom, Riso-Schicht an.
#  Die fruehere Voreinstellung (Neon auf Schwarz mit Bloom) erzeugt den
#  Eindruck eines Bildschirmschoners, bevor man einen Regler angefasst hat.
MATTE_EFFECTS = dict(intensity=0.5, grain=True, bloom=False, chroma=False,
                     seed=None, vignette=0.0, posterize_levels=0, halftone=0,
                     streaks=0.0, depth=0.0)


@dataclass
class Recipe:
    """Alles, was ein Bild eindeutig bestimmt — ausser dem Audiomaterial."""

    mode: str = "Rose gespiegelt"
    mix: dict = field(default_factory=dict)          # {Modus: Gewicht}
    #  "Aufhellen" ist das alte Verhalten und bleibt Vorgabe, damit aeltere
    #  Rezepte unveraendert aussehen. Neue Einstellungen starten auf
    #  "Ueberlagern" — siehe Recipe.matte().
    mix_blend: str = "Aufhellen"
    stops: list = field(default_factory=lambda: list(PRESETS["Korrend"]))
    bg: str = PROFILE_BG["Korrend"]
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    effects: dict | None = None
    layout: str = "Zentriert"
    layout_scale: float = 1.0
    typography: dict = field(default_factory=dict)
    riso: dict | None = None          # Druckschicht, siehe sonicart.riso
    transparent: bool = False
    size: int = 3000
    start: float = 0.0
    end: float | None = None
    version: int = RECIPE_VERSION
    audio_name: str = ""
    audio_sha1: str = ""
    note: str = ""

    # ---------------- Serialisierung ----------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in (d or {}).items() if k in known}
        r = cls(**clean)
        merged = dict(DEFAULT_PARAMS)
        merged.update(r.params or {})
        r.params = merged
        return r

    @classmethod
    def from_json(cls, s) -> "Recipe":
        return cls.from_dict(json.loads(s))

    # ---------------- Ableitungen ----------------
    @property
    def riso_spec(self):
        from .riso import RisoSpec
        return RisoSpec.from_dict(self.riso)

    @classmethod
    def matte(cls, ink_set: str = "Zinnober", mode: str = "Gitter", **kw) -> "Recipe":
        """Matter Ausgangspunkt statt Neon auf Schwarz.

        Jedes Feld laesst sich per Schluesselwort ueberschreiben, auch riso
        und params — die Vorgaben sind ein Startpunkt, keine Sperre.
        """
        from .riso import INK_SETS
        satz = INK_SETS[ink_set]
        p = dict(DEFAULT_PARAMS)
        p["whiten_amount"] = 0.85     # zeitaufgeloeste Modi
        p["tilt"] = 0.6               # Rose: Hoehen anheben statt normieren
        felder = dict(
            mode=mode, stops=list(satz["inks"]), bg="#0b0b0e", params=p,
            effects=dict(MATTE_EFFECTS),
            riso=dict(inks=list(satz["inks"]), paper=satz["paper"]),
            mix_blend="Ueberlagern",
        )
        felder.update(kw)
        return cls(**felder)

    @property
    def type_spec(self) -> TypeSpec:
        known = {f for f in TypeSpec.__dataclass_fields__}
        return TypeSpec(**{k: v for k, v in (self.typography or {}).items()
                           if k in known})

    def cmap(self):
        return make_cmap(self.stops)

    def seeded(self, audio_name: str) -> dict | None:
        """Effekt-Dict mit aufgeloestem Seed (0/None = aus dem Dateinamen)."""
        if not self.effects:
            return None
        fx = dict(self.effects)
        if not fx.get("seed"):
            fx["seed"] = int(hashlib.md5(audio_name.encode()).hexdigest(), 16) % (2 ** 32)
        return fx


def _strip_rgba_garbage(img: Image.Image) -> Image.Image:
    """RGB-Muell auf vollstaendig transparenten Pixeln entfernen."""
    if img.mode != "RGBA":
        return img
    arr = np.asarray(img).copy()
    arr[arr[:, :, 3] == 0, :3] = 0
    return Image.fromarray(arr, "RGBA")


def build(an: Analysis, recipe: Recipe, size_px: int | None = None,
          canvas: tuple[int, int] | None = None, cmap=None,
          image=None, image_mode=None, signet=None,
          with_text: bool = True, return_features: bool = False,
          report: dict | None = None):
    """Rezept + Analyse -> fertiges Bild.

    canvas gibt eine abweichende Zielflaeche vor (Social-Formate); ohne
    Angabe bleibt es quadratisch. Effekte laufen auf der quadratischen Form,
    bevor sie platziert wird — so muss die Kette pro Format nur einmal laufen.

    report nimmt, wenn ein dict uebergeben wird, den Kontrastbefund des
    Textsatzes auf. Die Messung entsteht ohnehin beim Setzen; sie dort
    abzugreifen ist billiger, als das Bild danach noch einmal auszuwerten.
    """
    size = int(size_px or recipe.size)
    cm = cmap if cmap is not None else recipe.cmap()
    an = an.slice(recipe.start, recipe.end)
    tr = recipe.transparent

    active_mix = {m: w for m, w in (recipe.mix or {}).items() if w > 0}
    if active_mix:
        img = mix_modes(an, active_mix, cm, recipe.bg, size, recipe.params, tr,
                        blend=recipe.mix_blend)
    else:
        img = render_pil(an, recipe.mode, cm, recipe.bg, size, recipe.params, tr)

    if image is not None and image_mode and image_mode != "—":
        pic = image if hasattr(image, "size") else Image.open(image)
        img = apply_image(img, pic, image_mode, recipe.bg, transparent=tr)

    feat = None
    fx = recipe.seeded(recipe.audio_name or an.name or "sonic")
    if fx:
        feat = an.features
        img = apply_effects(img, feat, bg=recipe.bg, **fx)

    W, H = canvas if canvas else (size, size)
    if canvas or recipe.layout != "Zentriert" or recipe.layout_scale != 1.0:
        img = compose(img, W, H, recipe.layout, recipe.bg, tr,
                      scale_mul=recipe.layout_scale)

    #  Der Druck kommt nach der Platzierung, damit die ganze Flaeche Papier
    #  wird und nicht nur das Quadrat — aber vor Signet und Satz, damit Text
    #  eine Volltonplatte bleibt und lesbar ist.
    spec = recipe.riso_spec
    if spec is not None:
        from .riso import apply_riso
        img = apply_riso(img, spec, recipe.bg, keep_alpha=tr)

    if signet:
        img = add_signet(img, signet)
    if with_text:
        #  Als Textfarben kommen die Farben dieser Arbeit in Frage, nicht
        #  irgendein Weiss: so bleibt eine umgeschaltete Farbe in der Palette.
        kandidaten = list(recipe.stops)
        if recipe.riso:
            kandidaten.append(recipe.riso.get("paper", recipe.bg))
        kandidaten.append(recipe.bg)
        kandidaten += ["#ffffff", "#111111"]     # letzte Rueckfallebene
        img, befund = apply_typography_report(img, recipe.type_spec, kandidaten)
        if report is not None and befund:
            report.update(befund)
    img = _strip_rgba_garbage(img)
    return (img, feat) if return_features else img


def audio_fingerprint(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]
