"""Textsatz fuers Cover.

Drei Dinge, die vorher fehlten:

* **Schriften.** Es gab nur Pillows eingebauten Default oder eine eigene Datei.
  Jetzt liegen fuenf mitgelieferte Schriften bei (SIL OFL, siehe fonts/), je
  nach Rolle verschieden: eine Display-Grotesk fuer den Titel, eine Grotesk
  fuer den Artist, eine Monospace fuer Katalognummer und Label.
* **Kontrastpruefung.** Text konnte auf gleichfarbigem Grund landen und war
  unlesbar, ohne dass irgendetwas gewarnt haette. Jetzt wird der tatsaechliche
  Untergrund gemessen und die Farbe umgeschaltet oder ein Feld unterlegt.
* **Satz statt Beschriftung.** Blockbreite mit automatischer Verkleinerung,
  Zeilenabstand, Laufweite, Haarlinie, Versalien je Rolle.

Raster- und Vektorausgabe teilen sich die Ankerlogik (text_anchor), damit sie
deckungsgleich bleiben.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .palette import hex_to_rgb

FONT_DIR = Path(__file__).parent / "fonts"


# ----------------------------------------------------------------------
# Schriften
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Typeface:
    name: str
    file: str
    kind: str                 # "display" | "grotesk" | "mono"
    weights: tuple = ()       # benannte Schnitte variabler Schriften
    default_weight: str = ""
    caps_only: bool = False
    note: str = ""

    @property
    def path(self) -> str | None:
        p = FONT_DIR / self.file
        return str(p) if p.exists() else None


TYPEFACES: dict[str, Typeface] = {
    t.name: t for t in [
        Typeface("Bebas Neue", "BebasNeue-Regular.ttf", "display",
                 caps_only=True,
                 note="Schmale Versalien. Grosse Titel, viel Wirkung, wenig Platz."),
        Typeface("Space Grotesk", "SpaceGrotesk-var.ttf", "grotesk",
                 weights=("Light", "Regular", "Medium", "Bold"),
                 default_weight="Medium",
                 note="Zeitgenoessische Grotesk mit Eigenheiten. Guter Allrounder."),
        Typeface("Archivo", "Archivo-var.ttf", "grotesk",
                 weights=("Thin", "ExtraLight", "Light", "Regular", "Medium",
                          "SemiBold", "Bold", "ExtraBold", "Black"),
                 default_weight="SemiBold",
                 note="Kraeftige Grotesk, neun Schnitte. Schweizer Anmutung."),
        Typeface("Space Mono", "SpaceMono-Regular.ttf", "mono",
                 note="Technische Anmutung. Katalognummern, Laufzeiten, Notizen."),
        Typeface("Space Mono Bold", "SpaceMono-Bold.ttf", "mono",
                 note="Wie Space Mono, fetter."),
        Typeface("System", "", "grotesk",
                 note="Pillows eingebaute Schrift. Nur als Rueckfallebene."),
    ]
}

#  Sinnvolle Paarungen als ein Klick statt drei Auswahlfelder.
FONT_PAIRS = {
    "Display / Grotesk / Mono": ("Bebas Neue", "Space Grotesk", "Space Mono"),
    "Durchgehend Grotesk":      ("Space Grotesk", "Space Grotesk", "Space Mono"),
    "Schweizer Raster":         ("Archivo", "Archivo", "Space Mono"),
    "Durchgehend technisch":    ("Space Mono Bold", "Space Mono", "Space Mono"),
}


@lru_cache(maxsize=256)
def load_font(face: str, px: int, weight: str = "", path: str | None = None):
    """Schrift laden. Eigene Datei schlaegt die mitgelieferte Auswahl.

    Gecacht, weil pro Bild mehrere Zeilen in mehreren Groessen entstehen und
    das Oeffnen einer Schriftdatei je Zeile spuerbar kostet.
    """
    px = max(6, int(px))
    ziel = path or (TYPEFACES.get(face).path if face in TYPEFACES else None)
    if ziel:
        try:
            f = ImageFont.truetype(ziel, px)
            tf = TYPEFACES.get(face)
            gewuenscht = weight or (tf.default_weight if tf else "")
            if gewuenscht:
                try:
                    f.set_variation_by_name(gewuenscht)
                except Exception:
                    pass          # statische Schrift oder unbekannter Schnitt
            return f
        except Exception:
            pass
    try:
        return ImageFont.load_default(px)
    except Exception:
        return ImageFont.load_default()


def available_faces(kind: str | None = None) -> list[str]:
    return [n for n, t in TYPEFACES.items()
            if (kind is None or t.kind == kind) and (t.path or n == "System")]


# ----------------------------------------------------------------------
# Kontrast (WCAG)
# ----------------------------------------------------------------------
def _lin(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb) -> float:
    r, g, b = (_lin(v) for v in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a, b) -> float:
    """WCAG-Kontrast, 1 (gleich) bis 21 (Schwarz auf Weiss)."""
    la, lb = relative_luminance(a), relative_luminance(b)
    hell, dunkel = max(la, lb), min(la, lb)
    return (hell + 0.05) / (dunkel + 0.05)


def _crop(img: Image.Image, box):
    import numpy as np
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(img.width, max(x1, x0 + 1))
    y1 = min(img.height, max(y1, y0 + 1))
    return np.asarray(img.convert("RGB").crop((x0, y0, x1, y1)))


def background_at(img: Image.Image, box) -> tuple[int, int, int]:
    """Vorherrschender Untergrund in einem Bereich — Median, nicht Mittelwert.

    Der Median ignoriert Rasterpunkte und einzelne Ausreisser. Fuer die
    Kontrastpruefung reicht er aber nicht: siehe sample_background.
    """
    import numpy as np
    aus = _crop(img, box)
    if aus.size == 0:
        return (0, 0, 0)
    return tuple(int(v) for v in np.median(aus.reshape(-1, 3), axis=0))


def sample_background(img: Image.Image, box, n: int = 1200):
    """Stichprobe der Untergrundpixel eines Bereichs.

    Ein einzelner Kennwert genuegt nicht: laeuft eine Zeile ueber die Kante
    einer Form, trifft der Median die Mehrheitsflaeche und meldet gute Werte,
    waehrend die andere Haelfte der Zeile unlesbar ist. Deshalb wird die
    Verteilung behalten und spaeter das schlechteste Perzentil bewertet.
    """
    import numpy as np
    aus = _crop(img, box).reshape(-1, 3)
    if aus.size == 0:
        return np.zeros((1, 3), dtype=np.uint8)
    if len(aus) > n:
        schritt = max(1, len(aus) // n)
        aus = aus[::schritt][:n]
    return aus


def _luminanz(px):
    """Relative Leuchtdichte fuer ein (n, 3)-Array, vektorisiert."""
    import numpy as np
    c = np.asarray(px, dtype=float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def worst_contrast(proben, hexfarbe: str, perzentil: float = 5.0) -> float:
    """Kontrast im unguenstigsten Teil des Untergrunds.

    Nicht das Minimum: einzelne Rasterpunkte oder Kantenpixel wuerden sonst
    jede Stelle durchfallen lassen. Das 5. Perzentil trifft 'der schlechte
    Teil der Zeile', ohne an Ausreissern zu haengen.
    """
    import numpy as np
    lt = relative_luminance(hex_to_rgb(hexfarbe))
    lg = _luminanz(proben)
    hell = np.maximum(lt, lg)
    dunkel = np.minimum(lt, lg)
    return float(np.percentile((hell + 0.05) / (dunkel + 0.05), perzentil))


EPS = 1e-6          # 4.5 darf nicht an der Fliesskommastelle scheitern


def best_ink(proben_je_zeile, kandidaten, minimum: float = 4.5):
    """Beste Textfarbe fuer eine Menge von Untergrund-Stichproben.

    Bewertet wird der *schlechteste* Fall je Farbe, ueber alle Zeilen: eine
    Farbe taugt nur, wenn sie auf jeder Zeile und in jedem Teil davon reicht.
    Der Block behaelt dabei eine Farbe — zeilenweise verschiedene Farben
    waeren typografisch Unfug.

    Reihenfolge der Vorlieben: der Wunsch des Nutzers, falls er reicht; sonst
    die erste Farbe der Palette, die reicht — so bleibt die Arbeit farblich
    geschlossen, statt auf Weiss zu springen, sobald es irgendwo enger wird;
    und erst wenn gar nichts reicht, der beste schlechteste Fall.
    """
    bewertet = [(min(worst_contrast(p, c) for p in proben_je_zeile), c)
                for c in kandidaten]
    if not bewertet:
        return "#ffffff", 1.0
    if bewertet[0][0] >= minimum - EPS:
        return bewertet[0][1], bewertet[0][0]
    for v, c in bewertet[1:]:
        if v >= minimum - EPS:
            return c, v
    v, c = max(bewertet, key=lambda t: t[0])
    return c, v


CONTRAST_MODES = ["Aus", "Farbe umschalten", "Feld unterlegen", "Nur warnen"]


# ----------------------------------------------------------------------
# Anker
# ----------------------------------------------------------------------
ANCHORS = {
    "oben links":   (0.0, 0.0, "left",   "top"),
    "oben mitte":   (0.5, 0.0, "center", "top"),
    "oben rechts":  (1.0, 0.0, "right",  "top"),
    "mitte links":  (0.0, 0.5, "left",   "middle"),
    "mitte":        (0.5, 0.5, "center", "middle"),
    "mitte rechts": (1.0, 0.5, "right",  "middle"),
    "unten links":  (0.0, 1.0, "left",   "bottom"),
    "unten mitte":  (0.5, 1.0, "center", "bottom"),
    "unten rechts": (1.0, 1.0, "right",  "bottom"),
}

#  Richtwerte, keine offizielle Spezifikation.
SAFE_AREAS = {
    "Aus": None,
    "Cover (Beschnitt)":   dict(top=0.05, bottom=0.05, left=0.05, right=0.05),
    "Spotify Canvas 9:16": dict(top=0.14, bottom=0.30, left=0.06, right=0.06),
    "Story 9:16":          dict(top=0.16, bottom=0.20, left=0.06, right=0.06),
}


def text_anchor(anchor: str, margin: float) -> tuple[float, float, str, str]:
    """Relative Position eines Textblocks. Von PNG- und SVG-Pfad genutzt."""
    ax, ay, ha, va = ANCHORS.get(anchor, ANCHORS["unten links"])
    x = margin if ax == 0.0 else (1 - margin if ax == 1.0 else 0.5)
    y = margin if ay == 0.0 else (1 - margin if ay == 1.0 else 0.5)
    return x, y, ha, va


# ----------------------------------------------------------------------
# Satzspiegel
# ----------------------------------------------------------------------
@dataclass
class TypeSpec:
    """Der Textblock. Vier Rollen mit eigener Schrift, Groesse und Auszeichnung."""

    artist: str = ""
    title: str = ""
    label: str = ""
    catalog: str = ""

    # Schrift je Rolle
    face_title: str = "Bebas Neue"
    face_artist: str = "Space Grotesk"
    face_meta: str = "Space Mono"
    weight_title: str = ""
    weight_artist: str = "Medium"
    font_path: str | None = None        # eigene Datei ueberschreibt alles

    # Satz
    anchor: str = "unten links"
    margin: float = 0.07
    size: float = 0.052                 # Grundgroesse relativ zur Bildbreite
    tracking: float = 0.0               # Laufweite in Anteilen der Schriftgroesse
    title_tracking: float | None = None
    line_gap: float = 0.38
    max_width: float = 0.86             # Blockbreite; darueber wird verkleinert
    scales: tuple = (1.00, 2.10, 0.52, 0.46)   # Artist, Titel, Label, Katalog
    upper: bool = False
    title_upper: bool = True
    rule: float = 0.0                   # Haarlinie ueber dem Block, 0 = aus

    # Farbe und Lesbarkeit
    color: str = "#ffffff"
    shadow: float = 0.0
    contrast_mode: str = "Farbe umschalten"
    contrast_min: float = 4.5
    plate_color: str | None = None      # None = automatisch
    plate_pad: float = 0.35             # Polster des Feldes, in Zeilenhoehen

    def rollen(self) -> list[tuple[str, str, str, float, bool, float | None]]:
        """(Rolle, Text, Schrift, Groessenfaktor, Versalien, Laufweite)."""
        aus = []
        paare = [
            ("artist", self.artist, self.face_artist, self.scales[0],
             self.upper, None),
            ("title", self.title, self.face_title, self.scales[1],
             self.title_upper or self.upper, self.title_tracking),
            ("label", self.label, self.face_meta, self.scales[2],
             self.upper, None),
            ("catalog", self.catalog, self.face_meta, self.scales[3],
             self.upper, None),
        ]
        for rolle, text, face, skala, gross, track in paare:
            if text and text.strip():
                tf = TYPEFACES.get(face)
                if tf and tf.caps_only:
                    gross = True
                aus.append((rolle, text.upper() if gross else text, face,
                            skala, gross, track))
        return aus

    def is_empty(self) -> bool:
        return not self.rollen()

    def weight_for(self, face: str, rolle: str) -> str:
        if rolle == "title":
            return self.weight_title
        if rolle == "artist":
            return self.weight_artist
        return ""


@dataclass
class _Zeile:
    text: str
    font: object
    breite: float
    hoehe: float
    oben: float          # y-Versatz der Glyphen (textbbox[1])
    tracking: float


def _text_width(draw, text, font, track_px) -> float:
    if not text:
        return 0.0
    return draw.textlength(text, font=font) + track_px * max(0, len(text) - 1)


def _draw_tracked(draw, xy, text, font, fill, track_px):
    """Zeichnet mit Laufweite. PIL kennt kein letter-spacing, also zeichenweise."""
    if track_px == 0:
        draw.text(xy, text, font=font, fill=fill)
        return
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + track_px


def line_boxes(zeilen, kasten, start, spec: TypeSpec, W: int):
    """Kasten je Zeile. Grundlage der zeilenweisen Kontrastmessung.

    Ueber den ganzen Block gemessen taeuscht der Median: liegt der Titel auf
    heller Flaeche und die Zeile darunter auf einem dunklen Band, meldet der
    Block glatte Werte, waehrend eine Zeile unlesbar ist.
    """
    x0, y, ha = start
    abstand = spec.line_gap * spec.size * W
    aus = []
    for z in zeilen:
        if ha == "right":
            x = x0 - z.breite
        elif ha == "center":
            x = x0 - z.breite / 2
        else:
            x = x0
        aus.append((x, y, x + max(z.breite, 1.0), y + max(z.hoehe, 1.0)))
        y += z.hoehe + abstand
    return aus


def layout(spec: TypeSpec, W: int, H: int):
    """Zeilen ausmessen und den Block platzieren.

    Gibt (Zeilen, Kasten, Startpunkt) zurueck. Getrennt vom Zeichnen, damit
    der Kontrast gegen den Untergrund gemessen werden kann, bevor gemalt wird.
    """
    rollen = spec.rollen()
    if not rollen:
        return [], (0, 0, 0, 0), (0, 0, "left")
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    grund_px = spec.size * W

    def bauen(faktor: float):
        zeilen = []
        for rolle, text, face, skala, _gross, track in rollen:
            px = grund_px * skala * faktor
            f = load_font(face, px, spec.weight_for(face, rolle), spec.font_path)
            tr = (spec.tracking if track is None else track) * px
            bb = d.textbbox((0, 0), text or "X", font=f)
            zeilen.append(_Zeile(text, f, _text_width(d, text, f, tr),
                                 bb[3] - bb[1], bb[1], tr))
        return zeilen

    zeilen = bauen(1.0)
    grenze = spec.max_width * W
    breiteste = max((z.breite for z in zeilen), default=0.0)
    if breiteste > grenze > 0:
        zeilen = bauen(grenze / breiteste)      # Block automatisch verkleinern

    abstand = spec.line_gap * grund_px
    gesamt = sum(z.hoehe for z in zeilen) + abstand * (len(zeilen) - 1)
    rx, ry, ha, va = text_anchor(spec.anchor, spec.margin)
    x0, y0 = rx * W, ry * H
    if va == "bottom":
        y = y0 - gesamt
    elif va == "middle":
        y = y0 - gesamt / 2
    else:
        y = y0

    breiteste = max((z.breite for z in zeilen), default=0.0)
    if ha == "right":
        links = x0 - breiteste
    elif ha == "center":
        links = x0 - breiteste / 2
    else:
        links = x0
    kasten = (links, y, links + breiteste, y + gesamt)
    return zeilen, kasten, (x0, y, ha)


def apply_typography(img: Image.Image, spec: TypeSpec,
                     kandidaten: list | None = None) -> Image.Image:
    """Textblock setzen. Prueft vorher den Kontrast gegen den echten Untergrund."""
    ergebnis, _ = apply_typography_report(img, spec, kandidaten)
    return ergebnis


def apply_typography_report(img: Image.Image, spec: TypeSpec,
                            kandidaten: list | None = None):
    """Wie apply_typography, gibt zusaetzlich den Kontrastbefund zurueck."""
    if spec.is_empty():
        return img, None
    W, H = img.size
    zeilen, kasten, start = layout(spec, W, H)
    x0, y, ha = start
    grund_px = spec.size * W

    polster = spec.plate_pad * grund_px
    mess_kasten = (kasten[0] - polster, kasten[1] - polster,
                   kasten[2] + polster, kasten[3] + polster)
    #  Je Zeile einzeln messen, nicht ueber den ganzen Block — und je Zeile
    #  die Verteilung behalten, nicht nur einen Kennwert.
    rand = polster * 0.35
    kaesten = [(a - rand, b - rand, c + rand, d + rand)
               for a, b, c, d in line_boxes(zeilen, kasten, start, spec, W)]
    proben = [sample_background(img, k) for k in kaesten]
    gruende = [background_at(img, k) for k in kaesten]
    grund = background_at(img, mess_kasten)

    farbe = spec.color
    platte = None
    wahl = [spec.color] + [c for c in (kandidaten or ["#ffffff", "#111111"])
                           if c != spec.color]
    werte = [worst_contrast(p, spec.color) for p in proben]
    ist = min(werte)
    schlechteste = int(min(range(len(werte)), key=lambda i: werte[i]))
    befund = {"grund": gruende[schlechteste], "kontrast": ist,
              "minimum": spec.contrast_min, "block_grund": grund,
              "ok": ist >= spec.contrast_min - EPS, "farbe": spec.color,
              "modus": spec.contrast_mode, "geaendert": False,
              "zeile": zeilen[schlechteste].text if zeilen else ""}

    if spec.contrast_mode == "Farbe umschalten" and not befund["ok"]:
        farbe, neu = best_ink(proben, wahl, spec.contrast_min)
        befund.update(farbe=farbe, kontrast=neu,
                      ok=neu >= spec.contrast_min - EPS,
                      geaendert=farbe != spec.color)
    elif spec.contrast_mode == "Feld unterlegen" and not befund["ok"]:
        import numpy as np
        eigen = np.array([hex_to_rgb(spec.color)], dtype=np.uint8)
        platte = spec.plate_color or best_ink(
            [eigen], [c for c in wahl if c != spec.color] or ["#111111"],
            spec.contrast_min)[0]
        neu = contrast_ratio(hex_to_rgb(farbe), hex_to_rgb(platte))
        befund.update(kontrast=neu, ok=neu >= spec.contrast_min - EPS,
                      geaendert=True, platte=platte)

    was_rgba = img.mode == "RGBA"
    canvas = img.convert("RGBA")
    ebene = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ebene)
    y = start[1]

    if platte:
        ImageDraw.Draw(canvas).rectangle(
            [int(mess_kasten[0]), int(mess_kasten[1]),
             int(mess_kasten[2]), int(mess_kasten[3])],
            fill=hex_to_rgb(platte) + (255,))

    if spec.rule > 0:
        dicke = max(1, int(spec.rule * grund_px * 0.12))
        oben = int(kasten[1] - polster * 0.7)
        d.rectangle([int(kasten[0]), oben, int(kasten[2]), oben + dicke],
                    fill=hex_to_rgb(farbe) + (255,))

    fill = hex_to_rgb(farbe) + (255,)
    abstand = spec.line_gap * grund_px
    for i, z in enumerate(zeilen):
        if ha == "right":
            x = x0 - z.breite
        elif ha == "center":
            x = x0 - z.breite / 2
        else:
            x = x0
        _draw_tracked(d, (x, y - z.oben), z.text, z.font, fill, z.tracking)
        y += z.hoehe + abstand

    if spec.shadow > 0:
        schatten = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        schatten.putalpha(ebene.getchannel("A"))
        schatten = schatten.filter(ImageFilter.GaussianBlur(radius=0.012 * W))
        schatten.putalpha(schatten.getchannel("A").point(
            lambda v: int(v * spec.shadow)))
        canvas.alpha_composite(schatten)
    canvas.alpha_composite(ebene)
    aus = canvas if was_rgba else canvas.convert("RGB")
    return aus, befund


def check_contrast(img: Image.Image, spec: TypeSpec) -> dict | None:
    """Nur messen, nichts zeichnen — fuer die Anzeige in der Oberflaeche."""
    if spec.is_empty():
        return None
    zeilen, kasten, start = layout(spec, img.width, img.height)
    polster = spec.plate_pad * spec.size * img.width
    rand = polster * 0.35
    kaesten = [(a - rand, b - rand, c + rand, d + rand)
               for a, b, c, d in line_boxes(zeilen, kasten, start, spec, img.width)]
    gruende = [background_at(img, k) for k in kaesten]
    werte = [worst_contrast(sample_background(img, k), spec.color)
             for k in kaesten]
    i = int(min(range(len(werte)), key=lambda k: werte[k]))
    return {"grund": gruende[i], "kontrast": werte[i],
            "minimum": spec.contrast_min, "zeile": zeilen[i].text,
            "ok": werte[i] >= spec.contrast_min - EPS, "farbe": spec.color}


def draw_safe_area(img: Image.Image, preset: str, color=(255, 80, 80)) -> Image.Image:
    """Sperrflaechen als Overlay — nur fuer die Vorschau, nie fuer den Export."""
    spec = SAFE_AREAS.get(preset)
    if not spec:
        return img
    W, H = img.size
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    tint = color + (55,)
    d.rectangle([0, 0, W, int(H * spec["top"])], fill=tint)
    d.rectangle([0, int(H * (1 - spec["bottom"])), W, H], fill=tint)
    d.rectangle([0, 0, int(W * spec["left"]), H], fill=tint)
    d.rectangle([int(W * (1 - spec["right"])), 0, W, H], fill=tint)
    d.rectangle([int(W * spec["left"]), int(H * spec["top"]),
                 int(W * (1 - spec["right"])), int(H * (1 - spec["bottom"]))],
                outline=color + (200,), width=max(1, W // 400))
    out = img.convert("RGBA")
    out.alpha_composite(over)
    return out


def add_signet(img, path, scale=0.16, margin=0.06, anchor="unten rechts"):
    """Logo/Signet einsetzen, an denselben Ankerpunkten wie der Text."""
    sig = Image.open(path).convert("RGBA")
    w = int(img.width * scale)
    sig = sig.resize((w, max(1, int(w * sig.height / sig.width))), Image.LANCZOS)
    rx, ry, ha, va = text_anchor(anchor, margin)
    x = rx * img.width - (sig.width if ha == "right" else
                          sig.width / 2 if ha == "center" else 0)
    y = ry * img.height - (sig.height if va == "bottom" else
                           sig.height / 2 if va == "middle" else 0)
    was_rgba = img.mode == "RGBA"
    base = img.convert("RGBA")
    base.alpha_composite(sig, (int(x), int(y)))
    return base if was_rgba else base.convert("RGB")
