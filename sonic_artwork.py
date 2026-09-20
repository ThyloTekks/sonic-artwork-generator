"""Sonic Artwork — Audio -> Cover.

Streamlit-Oberflaeche. Die Rechenarbeit liegt im Paket `sonicart`:

    sonicart/analysis.py    Audio einmal analysieren, Ergebnisse cachen
    sonicart/render/        die Bildmodi
    sonicart/artwork.py     Rezept + Pipeline
    sonicart/album.py       Serie statt Einzelbild
    sonicart/cli.py         dasselbe ohne Browser

Start:  streamlit run sonic_artwork.py
CLI:    python -m sonicart --help
"""

from __future__ import annotations

import io
import json
import os
import tempfile

import streamlit as st
from PIL import Image

from sonicart.album import render_album
from sonicart.analysis import Analysis, load
from sonicart.artwork import DEFAULT_PARAMS, Recipe, build
from sonicart.compose import IMAGE_MODES, LAYOUTS, MIX_BLENDS
from sonicart.effects import DEFAULTS as FX_DEFAULTS
from sonicart.export import (PLATFORM_SPECS, SIZES, export_formats, export_svg,
                             export_svg_layers, formats_zip, platform_check,
                             print_report, read_recipe, save_jpeg, save_png,
                             svg_zip)
from sonicart.palette import (HARMONIES, PRESETS, PROFILE_BG, cmap_from_cube,
                              generate_palette, make_cmap, palette_from_key)
from sonicart.render import MODES
from sonicart.riso import INK_SETS, RisoSpec, ink_set
from sonicart.typography import (ANCHORS, CONTRAST_MODES, FONT_PAIRS,
                                 SAFE_AREAS, TYPEFACES, available_faces,
                                 draw_safe_area)
from sonicart.video import SPOTIFY_CANVAS, VIDEO_MODES, loop_length

PREVIEW_PX = 700


# ----------------------------------------------------------------------
# Analyse-Cache
# ----------------------------------------------------------------------
#  cache_resource statt cache_data: Analysis rechnet beim ersten Zugriff nach
#  und merkt sich das im Objekt. cache_data wuerde bei jedem Rerun eine Kopie
#  zurueckgeben und die Nachberechnungen wegwerfen.
@st.cache_resource(show_spinner=False, max_entries=6)
def get_analysis(data: bytes, name: str) -> Analysis:
    return load(io.BytesIO(data), name=name)


@st.cache_resource(show_spinner=False, max_entries=6)
def get_lut(data: bytes):
    p = tempfile.NamedTemporaryFile(suffix=".cube", delete=False)
    p.write(data)
    p.close()
    return cmap_from_cube(p.name)


@st.cache_data(show_spinner=False, max_entries=4)
def _stash(data: bytes, suffix: str) -> str:
    """Upload einmal auf Platte legen; ffmpeg braucht Pfade, keine Puffer."""
    p = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    p.write(data)
    p.close()
    return p.name


@st.cache_data(show_spinner=False, max_entries=8)
def _stash_clut(stops: tuple) -> str:
    """Ein CLUT je Palette — 262144 Eintraege lohnen das Aufheben."""
    from sonicart import footage
    return footage.clut_file(list(stops))


#  Voreinstellung ist ein matter Druck, kein leuchtendes Diagramm: Papier
#  statt Schwarz, Bloom aus, Riso-Schicht an, und als Startmodus das Gitter
#  statt einer weiteren zentrierten Scheibe.
_MATT = Recipe.matte("Zinnober", mode="Gitter")

DEFAULTS = {
    "k_mode": _MATT.mode, "k_mixon": False, "k_mixblend": _MATT.mix_blend,
    "k_preset": "Korrend",
    "k_thick": 1.0, "k_gate": 0.15, "k_gamma": 1.5, "k_rot": 0, "k_inner": 0.15,
    "k_nmels": 110, "k_dthick": True, "k_onset": 0.25, "k_spoke": 0.35,
    "k_sym": 1, "k_symauto": False, "k_turns": 5.0, "k_marks": True,
    "k_nseg": 0, "k_gap": 2.0, "k_labelring": True, "k_source": "mix",
    "k_fxon": True, "k_fxint": 0.5, "k_g": True, "k_b": False, "k_c": False,
    "k_seed": 0, "k_vig": 0.0, "k_post": 0, "k_half": 0, "k_streaks": 0.0,
    "k_depth": 0.0,
    # Druckschicht
    "k_riso_on": True, "k_inkset": "Zinnober", "k_paper": _MATT.riso["paper"],
    "k_cell": 5.0, "k_misreg": 1.0, "k_texture": 0.06, "k_gain": 1.0,
    "k_invert": False, "k_overprint": "auto",
    # Normierung und neue Modi
    "k_whiten": 0.85, "k_tilt": 0.6, "k_stufen": 5, "k_gap_gitter": 0.055,
    "k_spalten": 0, "k_baender": 7, "k_amp": 0.8, "k_glaette": 0.0,
    "k_bandfarbe": False,
    "k_layout": "Zentriert", "k_lscale": 1.0,
    "k_artist": "", "k_title": "", "k_label": "", "k_catalog": "",
    "k_tanchor": "unten links", "k_tsize": 0.052, "k_ttrack": 0.0,
    "k_tupper": False, "k_tcolor": "#ffffff", "k_tshadow": 0.0, "k_tmargin": 0.07,
    "k_safe": "Aus",
    # Schriften und Lesbarkeit
    "k_pair": "Display / Grotesk / Mono",
    "k_face_title": "Bebas Neue", "k_face_artist": "Space Grotesk",
    "k_face_meta": "Space Mono", "k_wtitle": "", "k_wartist": "Medium",
    "k_titleupper": True, "k_ttrack_title": 0.04, "k_linegap": 0.38,
    "k_maxw": 0.86, "k_rule": 0.0,
    "k_contrast": "Farbe umschalten", "k_contrastmin": 4.5,
    "k_size": 3000, "k_transp": False, "k_fmt": "PNG", "k_live": True,
    "k_anchor": "#7b2ff7", "k_harmony": "Monochrom-Ramp", "k_imgmode": "—",
    "n_stops": len(_MATT.stops), "bg_key": _MATT.bg,
}
MIX_KEYS = {m: f"k_w_{i}" for i, m in enumerate(MODES)}


def set_state(**werte):
    """Zustand aendern — ausschliesslich aus einem Rueckruf heraus.

    Rueckrufe (on_click, on_change) laufen zwischen zwei Durchlaeufen: danach
    baut Streamlit das Skript vollstaendig neu auf, und jedes Widget wird
    wieder erzeugt.

    Dasselbe mitten im Aufbau mit st.rerun() zu erzwingen, ist ein Fehler mit
    weitreichenden Folgen: alle Widgets unterhalb der Abbruchstelle werden in
    diesem Durchlauf nicht mehr erzeugt, und Streamlit verwirft ihren Zustand.
    Ein Klick auf 'Preset laden' im Farbe-Pult setzte so stillschweigend
    Schriftgroesse, Rasterweite und alles andere zurueck, was im Skript
    dahinter steht — waehrend die Regler davor ihren Wert behielten.
    """
    st.session_state.update(werte)


# ---------------------------- Rueckrufe ----------------------------
def _cb_preset():
    cp = PRESETS[st.session_state["k_preset"]]
    set_state(n_stops=len(cp), bg_key=PROFILE_BG[st.session_state["k_preset"]],
              **{f"c{i}": c for i, c in enumerate(cp)})


def _cb_inkset():
    from sonicart.riso import ink_set as _ink
    spez = _ink(st.session_state["k_inkset"])
    set_state(n_stops=len(spez.inks), k_paper=spez.paper,
              **{f"c{i}": c for i, c in enumerate(spez.inks)})


def _cb_pair():
    t, a, m = FONT_PAIRS[st.session_state["k_pair"]]
    set_state(k_face_title=t, k_face_artist=a, k_face_meta=m)


def _cb_key_palette(an):
    k = an.key
    neu = palette_from_key(k["tonic"], k["mode"],
                           n=int(st.session_state["n_stops"]),
                           harmony=st.session_state["k_harmony"],
                           template=PRESETS[st.session_state["k_preset"]],
                           hue_spread=0.6 + 0.8 * an.harmonic_complexity)
    set_state(bg_key=neu[0], **{f"c{i}": c for i, c in enumerate(neu)})


def _cb_anchor_palette():
    ss = st.session_state
    tmpl = PRESETS[ss["k_preset"]]
    n = len(tmpl) if ss["k_harmony"] == "Profil-Struktur" else int(ss["n_stops"])
    neu = generate_palette(ss["k_anchor"], n, ss["k_harmony"], template=tmpl)
    set_state(n_stops=len(neu), bg_key=neu[0],
              **{f"c{i}": c for i, c in enumerate(neu)})


def _cb_recipe(rohdaten: bytes, name: str):
    if name.lower().endswith(".png"):
        r = read_recipe(io.BytesIO(rohdaten))
        if r is None:
            st.session_state["_rezept_fehler"] = "In diesem PNG steckt kein Rezept."
            return
        set_state(**recipe_to_state(r))
    else:
        roh = json.loads(rohdaten)
        set_state(**{k: v for k, v in roh.items()
                     if k in DEFAULTS or k == "n_stops"
                     or k.startswith(("c", "k_w_"))})
    st.session_state.pop("_rezept_fehler", None)


def _cb_mix_saat():
    """Beim Einschalten des Mix bekommt der gewaehlte Modus sein Gewicht.

    Vorher stand das mitten im Aufbau und feuerte bei jedem Durchlauf neu, in
    dem das Gewicht 0 war — man konnte die eigene Ebene also gar nicht auf 0
    stellen, sie sprang sofort zurueck.
    """
    ss = st.session_state
    if not ss.get("k_mixon"):
        return
    schluessel = MIX_KEYS[ss["k_mode"]]
    if all(ss.get(k, 0) <= 0 for k in MIX_KEYS.values()) or ss.get(schluessel, 0) <= 0:
        set_state(**{schluessel: 0.8})


def init_state():
    """Zustand herstellen — aus dem Spiegel, sonst aus den Vorgaben.

    Zwei Eigenheiten von Streamlit werden hier aufgefangen:

    1. Der Zustand jedes Widgets, das in einem Durchlauf nicht erzeugt wurde,
       wird verworfen. Wer den Risodruck abschaltet, um zu vergleichen, und
       wieder einschaltet, faende Papierfarbe und Rasterweite sonst auf den
       Vorgaben vor. Der Spiegel ist ein gewoehnlicher Eintrag ohne Widget
       und ueberlebt das.

    2. Ein Widget uebernimmt einen Wert aus dem Zustand nur, wenn dieser im
       *selben* Durchlauf gesetzt wurde, in dem das Widget zum ersten Mal
       entsteht. Steht der Schluessel schon aus einem frueheren Durchlauf da
       — Vorgaben vor dem Datei-Upload, Spiegel beim Wiedereinblenden,
       geladenes Rezept —, zeigt das Widget stattdessen seinen eigenen
       Vorgabewert und schickt diesen beim ersten Klick zurueck in den
       Zustand. So stand nach dem Upload 'Rose gespiegelt' im Auswahlfeld,
       waehrend das Bild im Gitter-Modus gerechnet wurde, und die Farbfelder
       kippten beim ersten Moduswechsel auf Schwarz. Deshalb wird hier jeder
       Wert in jedem Durchlauf neu gesetzt, auch wenn er schon dasteht.
    """
    spiegel = st.session_state.setdefault("_spiegel", {})
    #  Nur was im vorigen Durchlauf ausgeblendet war, wird aus dem Spiegel
    #  zurueckgeholt. Ein sichtbarer Regler traegt dagegen gerade die frische
    #  Eingabe des Nutzers — die duerfte der Spiegel nicht ueberschreiben.
    zuletzt_sichtbar = st.session_state.get("_sichtbar_letzte", ALLE_BEDINGTEN)
    for k, v in DEFAULTS.items():
        if k in ALLE_BEDINGTEN and k not in zuletzt_sichtbar and k in spiegel:
            st.session_state[k] = spiegel[k]
        else:
            st.session_state[k] = st.session_state.get(k, spiegel.get(k, v))
    #  Sechs Farbstufen, auch wenn gerade weniger gezeigt werden: sonst hat
    #  c3 beim Hochschalten keinen Wert und das Feld startet auf Schwarz.
    vorrat = list(_MATT.stops) + ["#8d8a84", "#4b47a8", "#c9c2b2"]
    for i in range(6):
        st.session_state[f"c{i}"] = st.session_state.get(
            f"c{i}", spiegel.get(f"c{i}", vorrat[i]))
    #  Alle Gewichte auf 0: welche Ebene beim Einschalten des Mix hochgezogen
    #  wird, entscheidet die Modusauswahl. Jede feste Vorbelegung wuerde sonst
    #  eine Ebene einblenden, die niemand gewaehlt hat.
    for key in MIX_KEYS.values():
        st.session_state[key] = st.session_state.get(key, spiegel.get(key, 0.0))


#  Regler, die nur unter einer Bedingung erscheinen. Wer ausgeblendet ist,
#  darf nicht gespiegelt werden: Streamlit setzt solche Werte auf die Vorgabe
#  zurueck, und ein blindes Sichern wuerde die Einstellung des Nutzers damit
#  ueberschreiben. Der Schluessel ist der Reglername des Modus (siehe
#  Mode.knobs), der Wert die zugehoerigen Widget-Schluessel.
BEDINGT_MODUS = {
    "gate": ["k_gate"], "inner": ["k_inner"], "n_mels": ["k_nmels"],
    "turns": ["k_turns"], "stufen": ["k_stufen"], "spalten": ["k_spalten"],
    "baender": ["k_baender"], "amp": ["k_amp"], "glaette": ["k_glaette"],
    "onset": ["k_onset"], "spoke_len": ["k_spoke"],
    "n_segments": ["k_nseg"], "gap_deg": ["k_gap"], "gap": ["k_gap_gitter"],
    "whiten_amount": ["k_whiten"], "tilt": ["k_tilt"], "source": ["k_source"],
    "symmetry": ["k_sym", "k_symauto"], "data_thickness": ["k_dthick"],
    "label_ring": ["k_labelring"], "farbe_nach_band": ["k_bandfarbe"],
    "mark_segments": ["k_marks"],
}
BEDINGT_SCHALTER = {
    "k_riso_on": ["k_inkset", "k_paper", "k_cell", "k_misreg", "k_texture",
                  "k_gain", "k_overprint", "k_invert"],
    "k_fxon": ["k_fxint", "k_g", "k_b", "k_c", "k_vig", "k_streaks", "k_depth",
               "k_post", "k_half", "k_seed"],
}
ALLE_BEDINGTEN = {k for v in BEDINGT_MODUS.values() for k in v} | \
                 {k for v in BEDINGT_SCHALTER.values() for k in v}


def sichtbare_regler(knobs) -> set:
    """Welche bedingten Regler standen in diesem Durchlauf tatsaechlich da?"""
    ss = st.session_state
    sichtbar = set()
    for name, keys in BEDINGT_MODUS.items():
        if name in knobs:
            sichtbar.update(keys)
    for schalter, keys in BEDINGT_SCHALTER.items():
        if ss.get(schalter):
            sichtbar.update(keys)
    return sichtbar


def spiegeln(knobs=None):
    """Am Ende des Durchlaufs sichern, was sichtbar war."""
    spiegel = st.session_state.setdefault("_spiegel", {})
    sichtbar = sichtbare_regler(knobs or set())
    for k in list(DEFAULTS) + [f"c{i}" for i in range(6)] + list(MIX_KEYS.values()):
        if k in ALLE_BEDINGTEN and k not in sichtbar:
            continue                    # ausgeblendet -> Wert ist nicht echt
        if k in st.session_state:
            spiegel[k] = st.session_state[k]
    st.session_state["_sichtbar_letzte"] = sichtbar


def state_keys() -> list[str]:
    n = int(st.session_state.get("n_stops", 4))
    return list(DEFAULTS) + list(MIX_KEYS.values()) + [f"c{i}" for i in range(n)]


def stops_from_state() -> list[str]:
    n = int(st.session_state["n_stops"])
    return [st.session_state.get(f"c{i}", "#ffffff") for i in range(n)]


def recipe_from_state(an: Analysis | None, audio_name: str = "") -> Recipe:
    ss = st.session_state
    params = dict(DEFAULT_PARAMS)
    sym = ss["k_sym"]
    if ss["k_symauto"] and an is not None:
        sym = an.meter                      # Taktart -> Zaehligkeit der Rose
    params.update(thickness=ss["k_thick"], gate=ss["k_gate"], gamma=ss["k_gamma"],
                  inner=ss["k_inner"], n_mels=ss["k_nmels"], rotation=ss["k_rot"],
                  data_thickness=ss["k_dthick"], symmetry=int(sym),
                  source=ss["k_source"], turns=ss["k_turns"], onset=ss["k_onset"],
                  spoke_len=ss["k_spoke"], n_segments=int(ss["k_nseg"]),
                  gap_deg=ss["k_gap"], mark_segments=ss["k_marks"],
                  label_ring=ss["k_labelring"],
                  whiten_amount=ss["k_whiten"], tilt=ss["k_tilt"],
                  stufen=int(ss["k_stufen"]), gap=ss["k_gap_gitter"],
                  spalten=int(ss["k_spalten"]), baender=int(ss["k_baender"]),
                  amp=ss["k_amp"], glaette=ss["k_glaette"],
                  farbe_nach_band=ss["k_bandfarbe"])
    fx = None
    if ss["k_fxon"]:
        fx = dict(FX_DEFAULTS, intensity=ss["k_fxint"], grain=ss["k_g"],
                  bloom=ss["k_b"], chroma=ss["k_c"], seed=(ss["k_seed"] or None),
                  vignette=ss["k_vig"], streaks=ss["k_streaks"], depth=ss["k_depth"],
                  posterize_levels=ss["k_post"], halftone=ss["k_half"])
    riso = None
    if ss["k_riso_on"]:
        riso = dict(inks=stops_from_state(), paper=ss["k_paper"],
                    cell=ss["k_cell"], misregister=ss["k_misreg"],
                    texture=ss["k_texture"], gain=ss["k_gain"],
                    invert=ss["k_invert"], overprint=ss["k_overprint"],
                    seed=int(ss["k_seed"]) or 3)
    mix = {m: ss[k] for m, k in MIX_KEYS.items()} if ss["k_mixon"] else {}
    return Recipe(
        mix_blend=ss["k_mixblend"],
        mode=ss["k_mode"], mix=mix, stops=stops_from_state(), bg=ss["bg_key"],
        params=params, effects=fx, riso=riso, layout=ss["k_layout"],
        layout_scale=ss["k_lscale"], transparent=ss["k_transp"],
        size=int(ss["k_size"]), audio_name=audio_name,
        typography=dict(artist=ss["k_artist"], title=ss["k_title"],
                        label=ss["k_label"], catalog=ss["k_catalog"],
                        anchor=ss["k_tanchor"], size=ss["k_tsize"],
                        tracking=ss["k_ttrack"], upper=ss["k_tupper"],
                        color=ss["k_tcolor"], shadow=ss["k_tshadow"],
                        margin=ss["k_tmargin"], font_path=ss.get("font_path"),
                        face_title=ss["k_face_title"],
                        face_artist=ss["k_face_artist"],
                        face_meta=ss["k_face_meta"],
                        weight_title=ss["k_wtitle"],
                        weight_artist=ss["k_wartist"],
                        title_upper=ss["k_titleupper"],
                        title_tracking=ss["k_ttrack_title"],
                        line_gap=ss["k_linegap"], max_width=ss["k_maxw"],
                        rule=ss["k_rule"],
                        contrast_mode=ss["k_contrast"],
                        contrast_min=ss["k_contrastmin"]))


def recipe_to_state(r: Recipe) -> dict:
    """Rezept -> Widget-Zustand (Gegenrichtung zu recipe_from_state).

    Gibt ein dict zurueck, statt selbst zu schreiben: so laesst es sich ueber
    set_state einspielen und ausserdem ohne Streamlit-Kontext testen.
    """
    p = dict(DEFAULT_PARAMS)
    p.update(r.params or {})
    ss = {}
    ss.update({
        "k_mode": r.mode, "k_mixon": bool(r.mix), "bg_key": r.bg,
        "n_stops": len(r.stops), "k_layout": r.layout, "k_lscale": r.layout_scale,
        "k_transp": r.transparent, "k_size": r.size,
        "k_thick": p["thickness"], "k_gate": p["gate"], "k_gamma": p["gamma"],
        "k_inner": p["inner"], "k_nmels": p["n_mels"], "k_rot": p["rotation"],
        "k_dthick": p["data_thickness"], "k_sym": p["symmetry"],
        "k_source": p["source"], "k_turns": p["turns"], "k_onset": p["onset"],
        "k_spoke": p["spoke_len"], "k_nseg": p["n_segments"],
        "k_gap": p["gap_deg"], "k_marks": p["mark_segments"],
        "k_labelring": p["label_ring"], "k_fxon": bool(r.effects),
        "k_whiten": p["whiten_amount"], "k_tilt": p["tilt"],
        "k_stufen": p["stufen"], "k_gap_gitter": p["gap"],
        "k_spalten": p["spalten"], "k_baender": p["baender"],
        "k_amp": p["amp"], "k_glaette": p["glaette"],
        "k_bandfarbe": p["farbe_nach_band"],
        "k_riso_on": bool(r.riso), "k_mixblend": r.mix_blend,
    })
    if r.riso:
        d = RisoSpec.from_dict(r.riso)
        ss.update({"k_paper": d.paper, "k_cell": d.cell,
                   "k_misreg": d.misregister, "k_texture": d.texture,
                   "k_gain": d.gain, "k_invert": d.invert,
                   "k_overprint": d.overprint})
        #  Ein Rezept mit Druckschicht, aber ohne Effekte haette den Seed
        #  sonst verloren — er steuert beide.
        ss.setdefault("k_seed", d.seed if d.seed != 3 else 0)
    for i, c in enumerate(r.stops):
        ss[f"c{i}"] = c
    for m, key in MIX_KEYS.items():
        ss[key] = (r.mix or {}).get(m, 0.0)
    if r.effects:
        fx = dict(FX_DEFAULTS, **r.effects)
        ss.update({"k_fxint": fx["intensity"], "k_g": fx["grain"],
                   "k_b": fx["bloom"], "k_c": fx["chroma"],
                   "k_seed": fx["seed"] or 0, "k_vig": fx["vignette"],
                   "k_streaks": fx["streaks"], "k_depth": fx["depth"],
                   "k_post": fx["posterize_levels"], "k_half": fx["halftone"]})
    t = r.typography or {}
    ss.update({"k_artist": t.get("artist", ""), "k_title": t.get("title", ""),
               "k_label": t.get("label", ""), "k_catalog": t.get("catalog", ""),
               "k_tanchor": t.get("anchor", "unten links"),
               "k_tsize": t.get("size", 0.052), "k_ttrack": t.get("tracking", 0.0),
               "k_tupper": t.get("upper", False),
               "k_tcolor": t.get("color", "#ffffff"),
               "k_tshadow": t.get("shadow", 0.0),
               "k_tmargin": t.get("margin", 0.07),
               "k_face_title": t.get("face_title", "Bebas Neue"),
               "k_face_artist": t.get("face_artist", "Space Grotesk"),
               "k_face_meta": t.get("face_meta", "Space Mono"),
               "k_wtitle": t.get("weight_title", ""),
               "k_wartist": t.get("weight_artist", "Medium"),
               "k_titleupper": t.get("title_upper", True),
               "k_ttrack_title": t.get("title_tracking", 0.04),
               "k_linegap": t.get("line_gap", 0.38),
               "k_maxw": t.get("max_width", 0.86),
               "k_rule": t.get("rule", 0.0),
               "k_contrast": t.get("contrast_mode", "Farbe umschalten"),
               "k_contrastmin": t.get("contrast_min", 4.5)})
    return ss


# ----------------------------------------------------------------------
# Seitenleiste — nur, was die Sitzung eroeffnet und ueberall gilt
# ----------------------------------------------------------------------
def sidebar():
    """Quelle, Rezept, Anzeige. Mehr nicht.

    Vorher standen hier alle 96 Bedienelemente in einer 5200 px langen
    Spalte — man sah nie mehr als ein Zwanzigstel davon und musste zum
    Vergleichen scrollen. Die Regler sitzen jetzt im Steuerpult neben der
    Vorschau, wo man ihre Wirkung sieht.
    """
    ss = st.session_state
    with st.sidebar:
        st.markdown("### Sonic Artwork")
        up = st.file_uploader("Audiodatei",
                              type=["wav", "mp3", "flac", "ogg", "m4a", "aif", "aiff"])
        an = None
        if up is not None:
            try:
                with st.spinner("Analysiere (einmal pro Datei) ..."):
                    an = get_analysis(up.getvalue(), up.name)
                    an.summary()
            except Exception as e:
                st.error(f"Audio konnte nicht gelesen werden: {e}")

        st.divider()
        st.checkbox("Live-Vorschau", key="k_live",
                    help="Aus, wenn das Rechnen bei jedem Regler stoert.")
        st.selectbox("Sperrflaechen", list(SAFE_AREAS), key="k_safe",
                     help="Richtwerte fuer Player-Bedienelemente. "
                          "Nur Vorschau, nie im Export.")

        with st.expander("Rezept"):
            st.caption("Jedes exportierte PNG traegt sein Rezept im Dateikopf — "
                       "ein altes Cover hier hochladen stellt alles wieder her.")
            rfile = st.file_uploader("Rezept (.json) oder Cover (.png)",
                                     type=["json", "png"], key="k_loadfile")
            if rfile is not None:
                st.button("Anwenden", width="stretch", key="b_apply",
                          on_click=_cb_recipe,
                          args=(rfile.getvalue(), rfile.name))
            if ss.get("_rezept_fehler"):
                st.error(ss["_rezept_fehler"])
            cfg = {k: ss[k] for k in state_keys() if k in ss}
            st.download_button("Speichern (.json)", json.dumps(cfg, indent=2),
                               "rezept.json", "application/json",
                               width="stretch")

        with st.expander("Eigene Dateien"):
            lut = st.file_uploader("LUT statt Palette (.cube)", type=["cube"])
            img_up = st.file_uploader("Bild oder Textur", type=["png", "jpg", "jpeg"])
            st.selectbox("Einsatz des Bildes", IMAGE_MODES, key="k_imgmode",
                         help="Masking = Bild NUR innerhalb der Audio-Form. "
                              "Textur-Blend = Textur faerbt die Form.")
            signet_up = st.file_uploader("Signet (PNG)", type=["png"])
            font_up = st.file_uploader("Eigene Schrift (TTF/OTF)",
                                       type=["ttf", "otf"])
            if font_up is not None:
                fp = tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(font_up.name)[1], delete=False)
                fp.write(font_up.getvalue())
                fp.close()
                ss["font_path"] = fp.name
            if ss.get("font_path"):
                st.caption("Eigene Schrift aktiv — sie ueberschreibt alle Rollen.")
                st.button("Verwerfen", width="stretch", key="b_font_weg",
                          on_click=set_state, kwargs={"font_path": None})

        st.caption("Ohne Browser: `python -m sonicart --help`")
    return up, an, lut, img_up, signet_up


# ----------------------------------------------------------------------
# Analyse als eine Zeile statt als sechs grosse Zahlen
# ----------------------------------------------------------------------
def analyse_zeile(an: Analysis):
    s = an.summary()
    chips = [
        ("Tempo", f"{s['tempo_bpm']:.0f} bpm"),
        ("Tonart", f"{s['tonart']} ({s['tonart_sicherheit']:.0%})"),
        ("Takt", s["taktart"]),
        ("Abschnitte", str(s["abschnitte"])),
        ("Harmonik", f"{s['harmonische_komplexitaet']:.2f}"),
        ("Stereo", "ja" if s["stereo"] else "mono"),
        ("Dauer", f"{s['dauer_s']:.0f} s"),
    ]
    html = "".join(
        f'<span style="display:inline-block;margin:0 14px 0 0;font:12px/1.6 '
        f'system-ui,sans-serif"><span style="opacity:.55">{k}</span> '
        f'<strong>{v}</strong></span>' for k, v in chips)
    st.markdown(f'<div style="padding:2px 0 10px">{html}</div>',
                unsafe_allow_html=True)


def _zeige_kontrast(b: dict):
    """Kontrastbefund des Textsatzes anzeigen.

    Die Zahl ist der WCAG-Kontrast zwischen Textfarbe und dem tatsaechlich
    gemessenen Untergrund unter der schwaechsten Zeile — nicht gegen die
    eingestellte Hintergrundfarbe, die unter einem Raster oder einer Form gar
    nicht sichtbar sein muss.
    """
    grund = "#%02x%02x%02x" % tuple(b["grund"])
    zeile = f" · schwaechste Zeile {b['zeile']!r}" if b.get("zeile") else ""
    text = (f"Textkontrast {b['kontrast']:.1f}:1 gegen {grund} "
            f"(Minimum {b['minimum']:.1f}){zeile}")
    if b.get("platte"):
        text += f" · Feld {b['platte']} unterlegt"
    elif b.get("geaendert"):
        text += f" · Farbe auf {b['farbe']} umgeschaltet"
    if b["ok"]:
        st.caption("✓ " + text)
    elif b["modus"] == "Aus":
        st.caption("⚠ " + text + " — Pruefung ist aus")
    elif b["modus"] == "Farbe umschalten":
        st.warning(text + " — keine Farbe reicht hier aus. "
                          "'Feld unterlegen' oder eine andere Stelle waehlen.")
    else:
        st.warning(text + " — auch ein Feld reicht nicht. Andere Stelle waehlen.")


# ----------------------------------------------------------------------
# Steuerpult: Regler dort, wo man ihre Wirkung sieht
# ----------------------------------------------------------------------
def _paare(*regler):
    """Regler paarweise nebeneinander setzen.

    Feste Spalten taugen hier nicht: welche Regler ueberhaupt erscheinen,
    haengt vom Modus ab, und ein weggelassener reisst sonst ein Loch ins
    Raster. Hier werden erst die tatsaechlich vorhandenen gesammelt und dann
    zu zweit gesetzt.
    """
    aktive = [f for f in regler if f]
    for i in range(0, len(aktive), 2):
        spalten = st.columns(2)
        for spalte, fn in zip(spalten, aktive[i:i + 2]):
            with spalte:
                fn()


def _panel_form(an, mode, mix_on, knobs):
    ss = st.session_state
    st.selectbox("Bildmodus", list(MODES), key="k_mode",
                 on_change=_cb_mix_saat)
    st.caption(MODES[mode].hint)
    if MODES[mode].stereo and an is not None and not an.is_stereo:
        st.warning("Mono-Datei: gezeigt wird die Hilbert-Phase als Ersatzachse.")
    if not MODES[mode].time_aware:
        st.caption("Dieser Modus mittelt ueber die Zeit. Fuer den Verlauf "
                   "'Spirale', 'Segmente' oder 'Gitter'.")

    hat = lambda n: n in knobs
    _paare(
        lambda: st.slider("Dicke", 0.3, 3.0, step=0.1, key="k_thick"),
        lambda: st.slider("Gamma (Kontrast)", 0.8, 2.5, step=0.1, key="k_gamma"),
        hat("gate") and (lambda: st.slider("Gate (Rauschen)", 0.0, 0.8,
                                           step=0.05, key="k_gate")),
        lambda: st.slider("Winkel (Grad)", 0, 360, step=5, key="k_rot"),
        # die Regler, die diesen Modus ausmachen
        hat("turns") and (lambda: st.slider("Windungen", 1.0, 14.0, step=0.5,
                                            key="k_turns")),
        hat("stufen") and (lambda: st.slider("Tonwertstufen", 2, 9, key="k_stufen")),
        hat("spalten") and (lambda: st.slider("Spalten (0 = quadratisch)", 0, 16,
                                              key="k_spalten")),
        hat("baender") and (lambda: st.slider(
            "Schichten", 3, 14, key="k_baender",
            help="Ueber etwa zehn verschwimmt es zu Moire.")),
        hat("amp") and (lambda: st.slider("Ausschlag", 0.2, 1.5, step=0.05,
                                          key="k_amp")),
        hat("onset") and (lambda: st.slider("Onset-Akzent", 0.0, 1.0, step=0.05,
                                            key="k_onset")),
        hat("spoke_len") and (lambda: st.slider("Speichen-Laenge", 0.1, 1.0,
                                                step=0.05, key="k_spoke")),
    )

    with st.expander("Feinschliff"):
        _paare(
            hat("inner") and (lambda: st.slider("Innenradius", 0.0, 0.4,
                                                step=0.01, key="k_inner")),
            hat("n_mels") and (lambda: st.slider("Frequenzbaender", 60, 256,
                                                 step=2, key="k_nmels")),
            hat("whiten_amount") and (lambda: st.slider(
                "Bandnormierung", 0.0, 1.0, step=0.05, key="k_whiten",
                help="Hebt leise Frequenzbaender auf denselben Dynamikbereich. "
                     "Ohne das frisst der Bass alles.")),
            hat("tilt") and (lambda: st.slider(
                "Hoehenanhebung", 0.0, 1.0, step=0.05, key="k_tilt",
                help="Die Rose zeigt das zeitgemittelte Spektrum. "
                     "Bandnormierung waere hier falsch — sie wuerde genau "
                     "dieses Mittel einebnen.")),
            hat("source") and (lambda: st.selectbox(
                "Signalquelle", ["mix", "harmonic", "percussive"], key="k_source",
                help="harmonic = Toene ohne Schlagzeug, "
                     "percussive = nur die Transienten")),
            hat("symmetry") and (lambda: st.slider(
                "Zaehligkeit", 1, 8, key="k_sym", disabled=ss["k_symauto"])),
            hat("n_segments") and (lambda: st.slider("Abschnitte (0 = auto)",
                                                     0, 12, key="k_nseg")),
            hat("gap_deg") and (lambda: st.slider("Sektorabstand (Grad)", 0.0,
                                                  8.0, step=0.5, key="k_gap")),
            hat("gap") and (lambda: st.slider("Feldabstand", 0.0, 0.25,
                                              step=0.005, key="k_gap_gitter")),
            hat("glaette") and (lambda: st.slider("Glaettung (0 = auto)", 0.0,
                                                  40.0, step=1.0, key="k_glaette")),
        )
        schalter = [f for f in (
            hat("symmetry") and (lambda: st.checkbox(
                "Zaehligkeit aus der Taktart", key="k_symauto",
                help=(f"Erkannt: {an.meter}/4" if an else
                      "3/4 wird dreizaehlig, 4/4 vierzaehlig"))),
            hat("data_thickness") and (lambda: st.checkbox(
                "Dicke datengetrieben", key="k_dthick")),
            hat("label_ring") and (lambda: st.checkbox("Kennband aussen",
                                                       key="k_labelring")),
            hat("farbe_nach_band") and (lambda: st.checkbox("Farbe nach Band",
                                                            key="k_bandfarbe")),
            hat("mark_segments") and (lambda: st.checkbox(
                "Abschnittsgrenzen markieren", key="k_marks")),
        ) if f]
        for i in range(0, len(schalter), 2):
            for sp, fn in zip(st.columns(2), schalter[i:i + 2]):
                with sp:
                    fn()

    #  Fester Text: ein wechselndes Label gibt dem Aufklapper eine neue
    #  Identitaet, und er faellt bei jeder Aenderung wieder zu.
    with st.expander("Modi mischen"):
        st.checkbox("Modi mischen", key="k_mixon", on_change=_cb_mix_saat)
        if mix_on:
            st.caption("Die Modusauswahl oben zaehlt jetzt ueber ihr Gewicht.")
            st.selectbox("Verfahren", MIX_BLENDS, key="k_mixblend",
                         help="Ueberlagern malt die Ebenen der Reihe nach "
                              "uebereinander. Aufhellen nimmt je Pixel den "
                              "helleren Wert — dabei schluckt eine "
                              "flaechenfuellende Ebene die duenneren.")
            m1, m2 = st.columns(2)
            for i, (m, key) in enumerate(MIX_KEYS.items()):
                (m1 if i % 2 == 0 else m2).slider(m, 0.0, 1.0, step=0.05, key=key)
            aktiv = [m for m, k in MIX_KEYS.items() if ss[k] > 0]
            voll = [m for m in aktiv if MODES[m].full_bleed]
            if voll and len(aktiv) > 1 and ss["k_mixblend"] == "Aufhellen":
                st.warning(f"{voll[0]} fuellt die ganze Flaeche und ueberdeckt "
                           "beim Aufhellen die duenneren Ebenen. "
                           "'Ueberlagern' hilft.")


def _panel_farbe(an):
    ss = st.session_state
    c1, c2 = st.columns([2, 1])
    preset = c1.selectbox("Preset", list(PRESETS), key="k_preset")
    c2.button("Laden", width="stretch", key="b_preset", on_click=_cb_preset)

    st.slider("Farbstufen", 2, 6, key="n_stops")
    n = int(ss["n_stops"])
    #  Immer sechs gleich breite Zellen, davon n belegt. st.columns(n) wuerde
    #  die Felder ueber die volle Breite verteilen: bei zwei Farbstufen stuenden
    #  sie ein Viertel der Panelbreite auseinander und schwebten im Leeren.
    MAX_STUFEN = 6
    cols = st.columns(MAX_STUFEN)
    for i in range(n):
        ss.setdefault(f"c{i}", "#ffffff")
        cols[i].color_picker(f"{i+1}", key=f"c{i}")
    b1, b2 = st.columns([1, MAX_STUFEN - 1])
    b1.color_picker("Grund", key="bg_key")
    b2.caption("Die Farbstufen bilden den Verlauf von dunkel nach hell. "
               "Beim Risodruck sind sie zugleich die Druckfarben.")

    st.divider()
    st.caption("Palette aus dem Stueck ableiten")
    a1, a2 = st.columns(2)
    a1.button("Aus Tonart", width="stretch", disabled=an is None, key="b_key",
              help="Quintenzirkel -> Farbton, Moll dunkler, "
                   "harmonische Dichte -> Farbtonspreizung",
              on_click=_cb_key_palette, args=(an,))
    a2.button("Aus Ankerfarbe", width="stretch", key="b_anchor",
              on_click=_cb_anchor_palette)
    b1, b2 = st.columns(2)
    b1.color_picker("Ankerfarbe", key="k_anchor")
    b2.selectbox("Harmonie", HARMONIES, key="k_harmony")
    st.caption("Profil-Struktur = Helligkeits- und Saettigungsverlauf des "
               "Presets uebernehmen, nur den Farbton tauschen.")


def _panel_druck():
    ss = st.session_state
    st.checkbox("Als Risodruck ausgeben", key="k_riso_on",
                help="Flache Farben, Halbtonraster je Platte, Passerversatz, "
                     "Papier statt Schwarz. Gilt fuer jeden Modus.")
    if not ss["k_riso_on"]:
        st.caption("Aus — das Bild wird als Leuchten auf Farbe ausgegeben.")
        return
    c1, c2 = st.columns([2, 1])
    c1.selectbox("Farbsatz", list(INK_SETS), key="k_inkset")
    c2.button("Laden", width="stretch", key="b_ink", on_click=_cb_inkset)
    st.caption("Die Druckfarben sind die Farbstufen aus 'Farbe' — "
               "hellste zuerst, dunkelste zuletzt.")
    d1, d2 = st.columns(2)
    d1.color_picker("Papier", key="k_paper")
    d2.slider("Rasterweite (px)", 2.0, 14.0, step=0.5, key="k_cell")
    e1, e2 = st.columns(2)
    e1.slider("Passerversatz", 0.0, 5.0, step=0.25, key="k_misreg",
              help="0 = perfekter Passer. Der leichte Versatz ist das, was "
                   "einen Risodruck ausmacht.")
    e2.slider("Papierfaser", 0.0, 0.25, step=0.01, key="k_texture")
    with st.expander("Feinschliff"):
        f1, f2 = st.columns(2)
        f1.slider("Tonwertzunahme", 0.5, 2.0, step=0.05, key="k_gain")
        f2.selectbox("Ueberdruck", ["auto", "multiply", "screen"],
                     key="k_overprint",
                     help="auto waehlt nach Papierhelligkeit. Auf dunklem "
                          "Papier deckt man auf (screen), statt zu lasieren.")
        st.checkbox("Tonwerte tauschen", key="k_invert")
    st.caption("SVG kennt keine Druckschicht — der Vektorexport gibt die "
               "reine Form aus.")


def _panel_text():
    ss = st.session_state
    c1, c2 = st.columns(2)
    c1.text_input("Artist", key="k_artist")
    c2.text_input("Titel", key="k_title")
    c3, c4 = st.columns(2)
    c3.text_input("Label", key="k_label")
    c4.text_input("Katalognr.", key="k_catalog")

    p1, p2 = st.columns([2, 1])
    p1.selectbox("Schriftpaarung", list(FONT_PAIRS), key="k_pair")
    p2.button("Setzen", width="stretch", key="b_pair", on_click=_cb_pair)

    g1, g2 = st.columns(2)
    g1.selectbox("Ankerpunkt", list(ANCHORS), key="k_tanchor")
    g2.slider("Schriftgroesse", 0.02, 0.14, step=0.002, key="k_tsize")

    st.selectbox("Kontrastpruefung", CONTRAST_MODES, key="k_contrast",
                 help="Gemessen wird der tatsaechliche Untergrund unter jeder "
                      "Zeile, nicht die eingestellte Hintergrundfarbe.")

    with st.expander("Schriften"):
        st.caption("Mitgeliefert unter SIL OFL — siehe sonicart/fonts/.")
        st.selectbox("Titel", available_faces(), key="k_face_title")
        tf = TYPEFACES.get(ss["k_face_title"])
        if tf:
            st.caption(tf.note)
            if tf.weights:
                st.select_slider("Schnitt Titel", ["", *tf.weights], key="k_wtitle")
        st.selectbox("Artist", available_faces(), key="k_face_artist")
        tfa = TYPEFACES.get(ss["k_face_artist"])
        if tfa and tfa.weights:
            st.select_slider("Schnitt Artist", ["", *tfa.weights], key="k_wartist")
        st.selectbox("Label und Katalognummer", available_faces(),
                     key="k_face_meta")

    with st.expander("Satz"):
        s1, s2 = st.columns(2)
        with s1:
            st.slider("Blockbreite", 0.3, 1.0, step=0.02, key="k_maxw",
                      help="Zu breite Zeilen werden automatisch verkleinert.")
            st.slider("Laufweite", -0.05, 0.35, step=0.01, key="k_ttrack")
            st.slider("Laufweite Titel", -0.05, 0.35, step=0.01,
                      key="k_ttrack_title")
            st.slider("Zeilenabstand", 0.0, 1.2, step=0.02, key="k_linegap")
        with s2:
            st.slider("Randabstand", 0.02, 0.20, step=0.01, key="k_tmargin")
            st.slider("Haarlinie", 0.0, 3.0, step=0.1, key="k_rule")
            st.slider("Schatten", 0.0, 1.0, step=0.05, key="k_tshadow")
            st.slider("Mindestkontrast", 1.5, 10.0, step=0.5, key="k_contrastmin",
                      help="WCAG: 4.5 fuer Fliesstext, 3.0 fuer grosse Schrift.")
        u1, u2, u3 = st.columns(3)
        u1.checkbox("Versalien", key="k_tupper")
        u2.checkbox("Titel gross", key="k_titleupper")
        u3.color_picker("Farbe", key="k_tcolor")


def _panel_effekte():
    ss = st.session_state
    st.selectbox("Platzierung", list(LAYOUTS), key="k_layout",
                 help="Bricht die immer mittige Scheibe auf.")
    st.slider("Groesse der Form", 0.4, 1.6, step=0.05, key="k_lscale")
    st.divider()
    st.checkbox("Effekte anwenden", key="k_fxon")
    if not ss["k_fxon"]:
        return
    st.slider("Intensitaet", 0.0, 1.5, step=0.05, key="k_fxint")
    f1, f2, f3 = st.columns(3)
    f1.checkbox("Grain", key="k_g")
    f2.checkbox("Bloom", key="k_b")
    f3.checkbox("Aberr.", key="k_c")
    with st.expander("Stilisierung (0 = aus)"):
        g1, g2 = st.columns(2)
        with g1:
            st.slider("Vignette", 0.0, 1.0, step=0.05, key="k_vig")
            st.slider("Licht-Streaks", 0.0, 1.0, step=0.05, key="k_streaks")
            st.slider("Pseudo-Tiefe", 0.0, 1.0, step=0.05, key="k_depth")
        with g2:
            st.slider("Posterize", 0, 12, step=1, key="k_post")
            st.slider("Halftone (px)", 0, 12, step=1, key="k_half")
            st.number_input("Seed (0 = Dateiname)", 0, 999999, step=1, key="k_seed")


def steuerpult(an, mode, mix_on, knobs):
    t = st.tabs(["Form", "Farbe", "Druck", "Text", "Aufbau"])
    with t[0]:
        _panel_form(an, mode, mix_on, knobs)
    with t[1]:
        _panel_farbe(an)
    with t[2]:
        _panel_druck()
    with t[3]:
        _panel_text()
    with t[4]:
        _panel_effekte()


# ----------------------------------------------------------------------
# Export — am Ende des Blickverlaufs, direkt unter der Vorschau
# ----------------------------------------------------------------------
def export_block(an, recipe, kw, cmap):
    """Groesse, Format, Rendern, Laden — in Leserichtung, unter der Vorschau.

    Vorher lagen Groesse und Format unten in der Seitenleiste und der
    Knopf im Hauptbereich: der Abschluss der Arbeit war ueber zwei Orte
    verteilt.
    """
    ss = st.session_state
    c1, c2, c3 = st.columns([3, 2, 3], vertical_alignment="bottom")
    with c1:
        st.select_slider("Groesse (px)", options=[1000, 1500, 2000, 3000, 3543],
                         key="k_size",
                         help="3543 px = 30 cm bei 300 dpi (12-Zoll-Vinyl)")
    with c2:
        if ss["k_transp"]:
            st.selectbox("Format", ["PNG"], disabled=True,
                         help="Transparenz gibt es nur als PNG.")
        else:
            st.selectbox("Format", ["PNG", "JPEG"], key="k_fmt")
        st.checkbox("Transparent", key="k_transp")
    fmt = "PNG" if recipe.transparent else ss["k_fmt"]
    with c3:
        if st.button(f"In {recipe.size} px rendern", type="primary",
                     width="stretch", key="b_render"):
            with st.spinner(f"Rendere {recipe.size} px ..."):
                full = build(an, recipe, recipe.size, **kw)
            ss["full_img"] = full
            ss["full_recipe"] = recipe
            ss["full_bytes"] = (save_jpeg(full) if fmt == "JPEG"
                                else save_png(full, recipe))
        if ss.get("full_bytes"):
            st.download_button(f"{fmt} laden ({len(ss['full_bytes'])/1e6:.1f} MB)",
                               ss["full_bytes"], f"cover.{fmt.lower()}",
                               f"image/{fmt.lower()}", width="stretch")

    with st.expander("Weitere Ausgaben und Pruefungen"):
        if not ss.get("full_bytes"):
            st.caption("Erst in voller Groesse rendern.")
        else:
            data, full, rr = ss["full_bytes"], ss["full_img"], ss["full_recipe"]
            d1, d2 = st.columns(2)
            with d1:
                if st.button("Social-Formate (ZIP)", width="stretch"):
                    with st.spinner("Erzeuge Formate ..."):
                        fmts = export_formats(full, rr.bg, rr.transparent,
                                              rr.layout, rr.layout_scale)
                        ss["zip_bytes"] = formats_zip(fmts, rr)
                if ss.get("zip_bytes"):
                    st.download_button("ZIP laden — 1:1 · 4:5 · 9:16 · 16:9",
                                       ss["zip_bytes"], "social_formate.zip",
                                       "application/zip", width="stretch")
            with d2:
                vec = [m for m in (list(rr.mix) if rr.mix else [rr.mode])
                       if rr.mix.get(m, 1) > 0 and MODES[m].vector]
                if not vec and rr.mix and not any(rr.mix.values()):
                    st.caption("Keine aktive Ebene — alle Mix-Gewichte auf 0.")
                elif not vec:
                    st.caption("Der Modus ist rasterbasiert (kein SVG).")
                elif st.button("SVG erzeugen", width="stretch"):
                    with st.spinner("Vektorisiere ..."):
                        lagen = export_svg_layers(an, rr, 1200, cmap)
                    if len(lagen) == 1:
                        name, d = next(iter(lagen.items()))
                        ss["svg"] = (f"{name}.svg", d, "image/svg+xml")
                    else:
                        ss["svg"] = ("ebenen.zip", svg_zip(lagen, rr),
                                     "application/zip")
                if ss.get("svg"):
                    name, d, mime = ss["svg"]
                    st.download_button(f"{name} ({len(d)/1e6:.1f} MB)", d, name,
                                       mime, width="stretch")
            plat = st.selectbox("Abgabepruefung", list(PLATFORM_SPECS))
            chk = platform_check(full, plat, data)
            (st.success if chk["ok"] else st.warning)(
                "Passt." if chk["ok"] else " · ".join(chk["issues"]))

        rep = print_report(recipe)
        st.caption("Druckpruefung (Naeherung ohne ICC-Profil) — oben die "
                   "Bildschirmfarbe, unten dieselbe nach dem Beschnitt auf "
                   "den Offset-Farbraum.")
        zellen = "".join(
            f'<div style="flex:1;min-width:74px;text-align:center;'
            f'font:11px/1.5 system-ui"><div style="height:30px;'
            f'background:{r["hex"]};border-radius:4px 4px 0 0"></div>'
            f'<div style="height:30px;background:{r["clipped"]};'
            f'border-radius:0 0 4px 4px"></div>{r["hex"]}<br>{r["ratio"]:.2f}x'
            f'{"" if r["printable"] else " ⚠"}</div>' for r in rep["rows"])
        st.markdown(f'<div style="display:flex;gap:6px">{zellen}</div>',
                    unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Reiter
# ----------------------------------------------------------------------
def tab_bild(an, audio_name, cmap, img_pic, sig_path, up):
    ss = st.session_state
    mode, mix_on = ss["k_mode"], ss["k_mixon"]
    if mix_on:
        aktive = [m for m, k in MIX_KEYS.items() if ss[k] > 0] or [mode]
        knobs = set().union(*(MODES[m].knobs() for m in aktive))
    else:
        knobs = MODES[mode].knobs()

    recipe = recipe_from_state(an, audio_name)
    kw = dict(cmap=cmap, image=img_pic, image_mode=ss["k_imgmode"],
              signet=sig_path)

    st.session_state["_knobs"] = knobs
    links, rechts = st.columns([5, 4], gap="large")
    with links:
        if ss["k_live"]:
            try:
                befund = {}
                #  Das vorige Bild bleibt stehen, bis das neue fertig ist.
                #  Sonst ist der Bildplatz waehrend des Rechnens leer, alles
                #  darunter rutscht um die Bildhoehe hoch und wieder zurueck —
                #  das ist das Springen, das die Oberflaeche unruhig macht.
                platz = st.empty()
                if ss.get("_letzte_vorschau") is not None:
                    platz.image(ss["_letzte_vorschau"], width="stretch")
                with st.spinner("Vorschau ..."):
                    img, feat = build(an, recipe, PREVIEW_PX,
                                      return_features=True, report=befund, **kw)
                gezeigt = (draw_safe_area(img, ss["k_safe"])
                           if ss["k_safe"] != "Aus" else img)
                ss["_letzte_vorschau"] = gezeigt
                platz.image(gezeigt, width="stretch")
                unten = st.columns([3, 2])
                unten[0].caption(f"Vorschau {PREVIEW_PX} px · Export "
                                 f"{recipe.size} px")
                if feat:
                    unten[1].caption(f"Grain {feat['roughness']:.2f} · "
                                     f"Bloom {feat['energy']:.2f} · "
                                     f"Aberr. {feat['spread']:.2f}")
                if befund:
                    _zeige_kontrast(befund)
            except Exception as e:
                st.error(f"Render-Fehler: {e}")
        else:
            st.info("Live-Vorschau ist aus — links in der Seitenleiste.")
        st.divider()
        export_block(an, recipe, kw, cmap)

    with rechts:
        steuerpult(an, mode, mix_on, knobs)


def tab_video(an, audio_name, cmap, up):
    ss = st.session_state
    recipe = recipe_from_state(an, audio_name)
    c1, c2, c3 = st.columns(3)
    v_mode = c1.selectbox("Video-Modus", VIDEO_MODES, index=2)
    v_asp = c2.selectbox("Format", list(SIZES), index=2)
    v_fps = c3.select_slider("FPS", [12, 24, 30], value=24)
    c4, c5, c6 = st.columns(3)
    v_dur = c4.number_input("Dauer (s)", 2.0, 300.0, 8.0, 1.0)
    v_turn = c5.slider("Umdrehungen", 0.0, 4.0, 1.0, 0.25)
    v_pulse = c6.slider("Beat-Puls", 0.0, 1.0, 0.4, 0.05,
                        help="Bindet Dicke und Innenradius an den Schlag.")
    c7, c8, c9 = st.columns(3)
    v_loop = c7.checkbox("Loopfaehig", True,
                         help="Laenge auf den Taktanfang, ganze Umdrehungen, "
                              "Ueberblendung am Schleifenpunkt.")
    v_fx = c8.checkbox("Effekte anwenden", True)
    v_audio = c9.checkbox("Originalton einbetten", True)
    v_blend = st.slider("Ueberblendung am Schleifenpunkt", 0.0, 0.5, 0.15, 0.05,
                        disabled=not v_loop)
    if v_loop:
        snapped = loop_length(an, v_dur, (1.0, an.duration))
        ok = SPOTIFY_CANVAS[0] <= snapped <= SPOTIFY_CANVAS[1]
        st.caption(f"Loopfaehige Laenge: **{snapped:.2f} s** (naechster Taktanfang)"
                   + ("" if ok else " — ausserhalb der Spotify-Spanne 3–8 s"))

    if st.button("Video erzeugen", type="primary", width="stretch"):
        from sonicart.video import animate
        bar = st.progress(0.0, "Rendere ...")
        try:
            outp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            info = animate(an, recipe, outp, audio=up, mode=v_mode,
                           duration=v_dur, fps=v_fps, aspect=v_asp,
                           rotate_turns=v_turn, pulse=v_pulse, effects=v_fx,
                           loop_safe=v_loop, loop_blend=v_blend,
                           with_audio=v_audio,
                           progress=lambda p: bar.progress(p, f"Rendere {p:.0%}"))
            bar.empty()
            ss["video"] = open(info["path"], "rb").read()
            msg = (f"{info['duration']:.2f} s · {info['frames']} Frames · "
                   f"{info['turns']:g} Umdrehungen")
            if v_audio and not info["audio"]:
                st.warning(msg + " — Ton konnte nicht gemuxt werden (ffmpeg fehlt).")
            else:
                st.success(msg)
        except ModuleNotFoundError:
            bar.empty()
            st.error("Video braucht: pip install imageio imageio-ffmpeg")
        except Exception as e:
            bar.empty()
            st.error(f"Video-Fehler: {e}")
    if ss.get("video"):
        v1, v2 = st.columns([2, 3])
        with v1:
            st.video(ss["video"])
        v2.download_button("MP4 laden", ss["video"], "visualizer.mp4",
                           "video/mp4", width="stretch")


def tab_footage(up):
    """Fremdmaterial auf die Palette dieses Tracks bringen.

    ffmpeg und das Footage-Modul werden erst hier geladen, wie animate() im
    Video-Reiter. Ein optionales Feature darf die App nicht am Start
    hindern — faellt hier etwas aus, bleiben Bild, Video und Album nutzbar.
    """
    try:
        from sonicart import ffmpeg, footage
    except Exception as e:
        st.error(f"Footage nicht verfuegbar: {e}")
        return

    ss = st.session_state
    stops = stops_from_state()
    st.caption("Beliebigen Clip ueber die Palette dieses Tracks einfaerben. "
               "Der Farbton des Originals faellt weg, Struktur und Helligkeit "
               "bleiben — so passt Fremdmaterial zum Artwork desselben Tracks.")
    vid = st.file_uploader("Videoclip", type=["mp4", "mov", "m4v", "webm"])
    if not vid:
        st.info("Lade einen Clip — erst den Vorschau-Frame, dann rendern.")
        return

    st.image(footage.ramp_strip(stops, w=900, h=28),
             caption="Gradient Map: dunkel -> hell", width="stretch")
    c1, c2, c3 = st.columns(3)
    f_str = c1.slider("Staerke", 0.0, 1.0, 1.0, 0.05,
                      help="0 = Original, 1 = ganz auf die Palette.")
    f_con = c2.slider("Kontrast (vor dem Mapping)", 0.5, 2.0, 1.0, 0.05,
                      help="Verschiebt, welche Helligkeiten wo auf der Rampe "
                           "landen.")
    f_grain = c3.slider("Korn", 0.0, 1.0, 0.0, 0.05)
    c4, c5, c6 = st.columns(3)
    f_fmt = c4.selectbox("Zielformat", footage.FOOTAGE_SIZES)
    f_aud = c5.selectbox("Audio", footage.AUDIO_MODES)

    try:
        pfad = _stash(vid.getvalue(), "." + vid.name.rsplit(".", 1)[-1])
        dauer = ffmpeg.duration(pfad)
    except ModuleNotFoundError:
        st.error("Footage braucht: pip install imageio-ffmpeg")
        return
    t_max = round(max(0.1, (dauer or 5.0) - 0.1), 1)
    f_t = c6.slider("Vorschau-Zeitpunkt (s)", 0.0, t_max, min(1.0, t_max), 0.1)

    groesse = SIZES[f_fmt]
    stand = (vid.name, len(vid.getvalue()), tuple(stops), f_t, f_fmt,
             f_str, f_con, f_grain)
    b1, b2 = st.columns(2)

    if b1.button("Vorschau-Frame", type="primary", width="stretch"):
        try:
            with st.spinner("Frame ..."):
                ss["footage_prev"] = (stand, footage.preview_frame(
                    pfad, _stash_clut(tuple(stops)), t=f_t, size=groesse,
                    contrast=f_con, strength=f_str, grain=f_grain))
        except ModuleNotFoundError:
            st.error("Footage braucht: pip install imageio-ffmpeg")
        except Exception as e:
            ss.pop("footage_prev", None)
            st.error(f"Vorschau-Fehler: {e}")

    if b2.button(f"Video rendern ({groesse[0]}x{groesse[1]})", width="stretch"):
        try:
            with st.spinner("Rendere Video (kann dauern) ..."):
                track = (_stash(up.getvalue(), "." + up.name.rsplit(".", 1)[-1])
                         if f_aud == "Track-Audio" else None)
                outp = tempfile.NamedTemporaryFile(suffix=".mp4",
                                                   delete=False).name
                footage.render_clip(pfad, _stash_clut(tuple(stops)), outp,
                                    size=groesse, contrast=f_con,
                                    strength=f_str, grain=f_grain,
                                    audio_mode=f_aud, track=track)
                ss["footage_mp4"] = open(outp, "rb").read()
            st.success(f"{len(ss['footage_mp4']) / 1e6:.1f} MB")
        except ModuleNotFoundError:
            st.error("Footage braucht: pip install imageio-ffmpeg")
        except Exception as e:
            st.error(f"Render-Fehler: {e}")

    vor = ss.get("footage_prev")
    if vor and vor[0] == stand:
        st.image(vor[1], caption=f"Vorschau bei {f_t:.1f} s — "
                                 f"{groesse[0]}x{groesse[1]}")
    elif vor:
        st.info("Einstellungen geaendert — Vorschau neu erzeugen.")

    if ss.get("footage_mp4"):
        v1, v2 = st.columns([2, 3])
        with v1:
            st.video(ss["footage_mp4"])
        v2.download_button("MP4 laden", ss["footage_mp4"], "footage.mp4",
                           "video/mp4", width="stretch")


def tab_album(cmap):
    st.caption("Ein Ordner mit Titeln wird zur Serie: die Palettenstruktur "
               "bleibt, der Farbton wandert pro Titel innerhalb der Spreizung. "
               "Die Form kommt weiter aus dem Audio, also bleibt jeder Titel eigen.")
    c1, c2 = st.columns(2)
    folder = c1.text_input("Ordner mit Audiodateien", placeholder="/Users/…/release")
    out = c2.text_input("Zielordner", placeholder="/Users/…/release/cover")
    d1, d2, d3 = st.columns(3)
    size = d1.select_slider("Groesse (px)", [1000, 1500, 2000, 3000, 3543],
                            value=3000, key="alb_size")
    spread = d2.slider("Farbspreizung (Grad)", 0, 120, 40, 5,
                       help="0 = alle Titel farbgleich, 120 = stark gefaechert")
    shared = d3.checkbox("Gemeinsame Palette", True)
    e1, e2 = st.columns(2)
    fmts = e1.checkbox("Social-Formate je Titel")
    jpeg = e2.checkbox("JPEG statt PNG")
    if st.button("Serie rendern", type="primary", disabled=not (folder and out),
                 width="stretch"):
        bar = st.progress(0.0, "Start ...")

        def prog(ev):
            kind, i, n, name = ev
            bar.progress(i / (2 * n) + (0.5 if kind == "render" else 0),
                         f"{kind}: {name} ({i}/{n})")
        try:
            rep = render_album(folder, recipe_from_state(None), out, size=size,
                               shared_palette=shared, spread_deg=spread,
                               formats=fmts, jpeg=jpeg, progress=prog)
            bar.empty()
            st.success(f"{len(rep['titel'])} Titel -> {rep['verzeichnis']}")
            f1, f2 = st.columns([3, 2])
            f1.image(os.path.join(out, rep["kontaktbogen"]), caption="Kontaktbogen")
            f2.dataframe([{k: t[k] for k in
                           ("track", "tonart", "tempo_bpm", "taktart")}
                          for t in rep["titel"]], width="stretch")
        except Exception as e:
            bar.empty()
            st.error(f"Album-Fehler: {e}")


# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Sonic Artwork", layout="wide",
                       initial_sidebar_state="expanded")
    init_state()
    up, an, lut, img_up, signet_up = sidebar()

    if not up:
        st.markdown("## Sonic Artwork")
        st.info("Lade links eine Audiodatei — dann erscheinen Analyse, "
                "Vorschau und Export.")
        return
    if an is None:
        return

    analyse_zeile(an)
    try:
        cmap = get_lut(lut.getvalue()) if lut else make_cmap(stops_from_state())
    except Exception as e:
        st.error(f"Palette/LUT ungueltig, nutze Standard: {e}")
        cmap = make_cmap(["#000000", "#ffffff"])

    sig_path = None
    if signet_up:
        p = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        p.write(signet_up.getvalue())
        p.close()
        sig_path = p.name
    img_pic = None
    if img_up is not None and st.session_state["k_imgmode"] != "—":
        try:
            img_pic = Image.open(io.BytesIO(img_up.getvalue())).convert("RGB")
        except Exception as e:
            st.error(f"Bild konnte nicht gelesen werden: {e}")

    t1, t2, t3, t4 = st.tabs(["Bild", "Video", "Footage", "Album"])
    with t1:
        tab_bild(an, up.name, cmap, img_pic, sig_path, up)
    with t2:
        tab_video(an, up.name, cmap, up)
    with t3:
        tab_footage(up)
    with t4:
        tab_album(cmap)
    spiegeln(st.session_state.get("_knobs"))


if __name__ == "__main__":
    main()
