"""Die Streamlit-Oberflaeche wirklich laufen lassen.

Ein reiner Import beweist nichts: der doppelte Widget-Key in der Druckvorschau
ist erst hier aufgefallen, weil er nur bei zwei gleichen Farben auftritt.
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJEKT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest      # noqa: E402


@pytest.fixture(scope="module")
def app_mit_audio(tmp_path_factory):
    """Kopie der App, bei der der Datei-Uploader eine feste WAV liefert."""
    d = tmp_path_factory.mktemp("app")
    wav = d / "t.wav"
    sr = 22050
    teile = []
    for dauer, freqs, rauschen in ((8, (220, 277, 330), 0.02),
                                   (9, (220, 277, 330, 660), 0.22),
                                   (7, (220,), 0.02)):
        t = np.linspace(0, dauer, int(dauer * sr), endpoint=False)
        s = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs) * 0.5
        teile.append(s + np.random.default_rng(0).normal(0, rauschen, len(t)))
    y = np.concatenate(teile)
    sf.write(wav, np.vstack([y, np.roll(y, 150)]).T.astype(np.float32), sr)

    stub = d / "app_stub.py"
    stub.write_text(textwrap.dedent(f"""
        import sys, io
        sys.path.insert(0, {str(PROJEKT)!r})
        import streamlit as st
        DATA = open({str(wav)!r}, "rb").read()

        class FakeUpload(io.BytesIO):
            name = "t.wav"

        st.file_uploader = lambda label, **kw: (
            FakeUpload(DATA) if "Audiodatei" in label else None)

        import sonic_artwork as app
        app.main()
    """))
    return str(stub)


def test_app_ohne_audio_laeuft():
    at = AppTest.from_file(str(PROJEKT / "sonic_artwork.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Audiodatei" in i.value for i in at.info)


def test_app_mit_audio_zeigt_analyse_und_vorschau(app_mit_audio):
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    assert not at.exception, [e for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    #  Die Analyse steht jetzt als eine kompakte Zeile statt als sechs grosse
    #  Kacheln — sechs Metriken fraßen den Platz ueber der Vorschau.
    text = " ".join(m.value for m in at.markdown)
    for feld in ("Tempo", "Tonart", "Takt", "Abschnitte", "Harmonik", "Dauer"):
        assert feld in text, f"{feld} fehlt in der Analysezeile"
    assert "bpm" in text
    # Vorschaubild muss tatsaechlich gezeichnet worden sein
    assert _zaehle(at, "Image") == 1


def test_vollrender_liefert_downloadbutton(app_mit_audio):
    at = AppTest.from_file(app_mit_audio, default_timeout=600)
    at.run()
    knopf = [b for b in at.button if "px rendern" in b.label]
    assert knopf, [b.label for b in at.button]
    knopf[0].click().run()
    assert not at.exception
    assert any("laden" in d.label for d in at.get("download_button"))


def test_alle_modi_sind_waehlbar(app_mit_audio):
    from sonicart.render import MODES
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    assert set(at.selectbox(key="k_mode").options) == set(MODES)


def _zaehle(at, typname):
    def walk(node):
        n = 1 if type(node).__name__ == typname else 0
        kinder = getattr(node, "children", None)
        werte = kinder.values() if isinstance(kinder, dict) else (kinder or [])
        return n + sum(walk(c) for c in werte)
    return walk(at._tree)


def test_standard_ist_matt():
    """Der erste Eindruck ohne einen einzigen Reglerausschlag."""
    import sonic_artwork as app
    d = app.DEFAULTS
    assert d["k_riso_on"] is True
    assert d["k_b"] is False, "Bloom aus"
    assert d["k_c"] is False, "Aberration aus"
    assert d["k_mode"] not in ("Rose gespiegelt", "Rose roh", "Kreis-Wellenform")
    assert d["k_whiten"] > 0 and d["k_tilt"] > 0


def test_beide_richtungen_kennen_dieselben_schluessel():
    """Ein Feld nur in einer Richtung zu pflegen ist der haeufige Fehler.

    recipe_from_state schreibt, recipe_to_state liest zurueck — wer
    einen Regler nur vorwaerts verdrahtet, verliert ihn beim Laden eines
    Rezepts, ohne dass irgendwo ein Fehler auftritt.
    """
    import inspect
    import re

    import sonic_artwork as app
    vor = set(re.findall(r'ss\["(k_[a-z_]+)"\]',
                         inspect.getsource(app.recipe_from_state)))
    zurueck = set(re.findall(r'"(k_[a-z_]+)":',
                             inspect.getsource(app.recipe_to_state)))
    zurueck |= set(re.findall(r'"(k_[a-z_]+)":',
                              inspect.getsource(app.recipe_to_state)))
    # Rein technische Schalter stehen nicht im Rezept.
    ausgenommen = {"k_mixon", "k_imgmode", "k_live", "k_fmt", "k_safe",
                   "k_preset", "k_anchor", "k_harmony", "k_symauto"}
    fehlend = vor - zurueck - ausgenommen
    assert not fehlend, f"wird geschrieben, aber nie zurueckgelesen: {sorted(fehlend)}"


def _knopf(at, key):
    """Ueber den Schluessel suchen, nicht ueber die Beschriftung.

    Seit dem Umbau gibt es zwei Knoepfe mit der Aufschrift 'Laden' (Preset
    und Farbsatz) — eine Suche ueber den Text traefe den falschen.
    """
    return at.button(key=key)


@pytest.mark.parametrize("key", ["b_preset", "b_key", "b_anchor", "b_ink"])
def test_knoepfe_die_farbstufen_setzen_stuerzen_nicht_ab(app_mit_audio, key):
    """StreamlitWidgetAlreadyInstantiatedError auf n_stops.

    Der Farbsatz-Knopf im Druck-Abschnitt steht unterhalb des Reglers
    'Farbstufen' und hat dessen Schluessel geschrieben. Streamlit verbietet
    das, sobald das Widget im selben Aufbau erzeugt wurde — der Knopf stuerzte
    also ab, die gleichartigen weiter oben nicht.
    """
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    _knopf(at, key).click()
    at.run()
    assert not at.exception, [getattr(e, "message", e) for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_analyse_steht_schon_im_ersten_aufbau(app_mit_audio):
    """'Aus Tonart' war gesperrt, weil die Analyse erst nach der Seitenleiste lief."""
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    assert not _knopf(at, "b_key").disabled


def test_mix_zieht_den_gewaehlten_modus_hoch(app_mit_audio):
    """Ohne das war die Modusauswahl beim Mischen wirkungslos.

    Wer Gitter waehlte und 'Modi mischen' einschaltete, sah weiter die Rose:
    der Mix rechnet nur mit den Gewichten, und nur die Rose war vorbelegt.
    """
    import sonic_artwork as app
    for modus in ("Gitter", "Strata", "Spirale", "Rose roh"):
        at = AppTest.from_file(app_mit_audio, default_timeout=300)
        at.run()
        at.selectbox(key="k_mode").set_value(modus).run()
        at.checkbox(key="k_mixon").set_value(True).run()
        assert not at.exception
        aktiv = {m for m, k in app.MIX_KEYS.items()
                 if at.slider(key=k).value > 0}
        assert aktiv == {modus}, f"{modus} -> {aktiv}"


def test_warnung_bei_flaechenfuellender_ebene_im_aufhellen(app_mit_audio):
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    at.checkbox(key="k_mixon").set_value(True).run()
    at.slider(key="k_w_0").set_value(0.7).run()
    at.selectbox(key="k_mixblend").set_value("Aufhellen").run()
    assert any("fuellt die ganze" in w.value for w in at.warning)
    at.selectbox(key="k_mixblend").set_value("Ueberlagern").run()
    assert not any("fuellt die ganze" in w.value for w in at.warning)


def test_alle_modi_im_mix_ohne_absturz(app_mit_audio):
    import sonic_artwork as app
    at = AppTest.from_file(app_mit_audio, default_timeout=600)
    at.run()
    at.checkbox(key="k_mixon").set_value(True).run()
    for m, key in app.MIX_KEYS.items():
        at.slider(key=key).set_value(0.6).run()
        assert not at.exception, f"{m}: {at.exception}"
        assert not at.error, f"{m}: {[e.value for e in at.error]}"


def test_rezept_nach_zustand_braucht_keinen_streamlit_kontext():
    """recipe_to_state gibt ein dict zurueck, statt selbst zu schreiben."""
    import sonic_artwork as app
    from sonicart.artwork import Recipe
    d = app.recipe_to_state(Recipe.matte("Indigo", mode="Strata"))
    assert isinstance(d, dict)
    assert d["k_mode"] == "Strata" and d["k_riso_on"] is True
    assert d["k_mixblend"] == "Ueberlagern"
    assert "k_seed" in d, "Seed steuert Effekte und Druck, darf nicht fehlen"

    from sonicart.artwork import Recipe as R
    nur_druck = app.recipe_to_state(R.matte("Ocker", effects=None))
    assert "k_seed" in nur_druck


def _widgets(view):
    return (len(view.selectbox) + len(view.slider) + len(view.checkbox)
            + len(view.button) + len(view.text_input) + len(view.color_picker)
            + len(view.number_input) + len(view.select_slider))


def test_seitenleiste_bleibt_schlank(app_mit_audio):
    """Vorher standen hier alle 96 Bedienelemente in einer 5200 px langen Spalte.

    Die Seitenleiste traegt jetzt nur noch, was die Sitzung eroeffnet und
    ueberall gilt. Alles Formende sitzt im Steuerpult neben der Vorschau.
    """
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    assert _widgets(at.sidebar) <= 12, _widgets(at.sidebar)
    assert _widgets(at) > 30, "die Regler duerfen nicht verschwunden sein"


def test_steuerpult_hat_die_erwarteten_reiter(app_mit_audio):
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    labels = [t.label for t in at.tabs]
    for name in ("Bild", "Video", "Album", "Form", "Farbe", "Druck", "Text",
                 "Aufbau"):
        assert name in labels, f"Reiter {name} fehlt: {labels}"


def test_exportweg_ist_erreichbar(app_mit_audio):
    """Groesse waehlen, rendern, laden — ohne die Seitenleiste anzufassen."""
    at = AppTest.from_file(app_mit_audio, default_timeout=600)
    at.run()
    at.select_slider(key="k_size").set_value(1000).run()
    at.button(key="b_render").click().run()
    assert not at.exception
    assert any("laden" in d.label for d in at.get("download_button"))


@pytest.mark.parametrize("modus", ["Rose gespiegelt", "Spirale", "Segmente",
                                   "HPSS-Zeit", "Kreis-Wellenform",
                                   "Lissajous (Stereo)", "Gitter", "Strata"])
def test_jeder_modus_zeigt_sein_eigenes_pult(app_mit_audio, modus):
    """Regler werden paarweise gesetzt; ein weggelassener darf kein Loch reissen."""
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    at.selectbox(key="k_mode").set_value(modus).run()
    assert not at.exception, [getattr(e, "message", e) for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.slider(key="k_thick").value is not None


def test_vorschau_und_regler_stehen_nebeneinander(app_mit_audio):
    """Bild links, Steuerpult rechts — beides gleichzeitig sichtbar."""
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    assert _zaehle(at, "Image") == 1
    assert _zaehle(at, "Column") >= 2


# ----------------------------------------------------------------------
# Zustandsverluste — die Fehler hinter dem "fuehlt sich buggy an"
# ----------------------------------------------------------------------
def test_knopf_loescht_keine_spaeteren_einstellungen(app_mit_audio):
    """Der schwerste der gemeldeten Fehler.

    Ein Klick auf 'Preset laden' im Farbe-Pult setzte stillschweigend jede
    Einstellung zurueck, die im Skript dahinter steht: st.rerun() mitten im
    Aufbau laesst alle Widgets darunter ungerendert, und Streamlit verwirft
    deren Zustand. Regler davor behielten ihren Wert — das machte den Fehler
    so schwer greifbar.
    """
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    at.slider(key="k_tsize").set_value(0.11).run()      # Text-Pult, danach
    at.slider(key="k_cell").set_value(9.0).run()        # Druck-Pult, danach
    at.slider(key="k_thick").set_value(2.4).run()       # Form-Pult, davor
    at.button(key="b_preset").click().run()
    assert at.slider(key="k_tsize").value == 0.11, "Schriftgroesse verloren"
    assert at.slider(key="k_cell").value == 9.0, "Rasterweite verloren"
    assert at.slider(key="k_thick").value == 2.4


@pytest.mark.parametrize("knopf", ["b_preset", "b_key", "b_anchor", "b_ink",
                                   "b_pair"])
def test_kein_knopf_loescht_zustand(app_mit_audio, knopf):
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    at.slider(key="k_tsize").set_value(0.11).run()
    at.button(key=knopf).click().run()
    assert not at.exception
    assert at.slider(key="k_tsize").value == 0.11


def _anweisungen(objekt) -> str:
    """Quelltext ohne Kommentare und Doc-Strings.

    Sonst schlaegt eine Suche nach 'st.rerun()' auch dort an, wo genau
    erklaert wird, warum es das nicht mehr gibt.
    """
    import ast
    import inspect
    baum = ast.parse(inspect.getsource(objekt))
    for knoten in ast.walk(baum):
        if (isinstance(knoten, ast.Expr) and isinstance(knoten.value, ast.Constant)
                and isinstance(knoten.value.value, str)):
            knoten.value.value = ""
    return ast.unparse(baum)


def test_keine_rerun_aufrufe_im_aufbau():
    """Strukturell absichern: Zustand nur noch aus Rueckrufen heraus."""
    import sonic_artwork as app
    quelle = _anweisungen(app)
    assert "st.rerun()" not in quelle, \
        "st.rerun() im Aufbau verwirft den Zustand aller Widgets darunter"
    assert "on_click=" in quelle and "on_change=" in quelle


def test_mix_gewicht_laesst_sich_auf_null_setzen(app_mit_audio):
    """Die Saat stand mitten im Aufbau und feuerte bei jedem Durchlauf neu —
    man konnte die eigene Ebene gar nicht abschalten, sie sprang zurueck."""
    import sonic_artwork as app
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    at.selectbox(key="k_mode").set_value("Gitter").run()
    at.checkbox(key="k_mixon").set_value(True).run()
    k = app.MIX_KEYS["Gitter"]
    assert at.slider(key=k).value == 0.8
    at.slider(key=k).set_value(0.0).run()
    assert at.slider(key=k).value == 0.0


def test_moduswechsel_im_mix_behaelt_die_alte_ebene(app_mit_audio):
    import sonic_artwork as app
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    at.checkbox(key="k_mixon").set_value(True).run()
    at.selectbox(key="k_mode").set_value("Spirale").run()
    aktiv = {m for m, k in app.MIX_KEYS.items() if at.slider(key=k).value > 0}
    assert aktiv == {"Gitter", "Spirale"}, aktiv


@pytest.mark.parametrize("schalter,key,wert", [
    ("k_riso_on", "k_cell", 11.0),
    ("k_fxon", "k_vig", 0.6),
])
def test_ausgeblendete_regler_behalten_ihren_wert(app_mit_audio, schalter, key, wert):
    """Wer den Druck abschaltet, um zu vergleichen, und wieder einschaltet,
    fand Papier und Rasterweite auf den Vorgaben vor."""
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    at.slider(key=key).set_value(wert).run()
    at.checkbox(key=schalter).set_value(False).run()
    at.checkbox(key=schalter).set_value(True).run()
    assert at.slider(key=key).value == wert


def test_modus_weg_und_zurueck_behaelt_die_reglerwerte(app_mit_audio):
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    at.selectbox(key="k_mode").set_value("Spirale").run()
    at.slider(key="k_turns").set_value(9.0).run()
    at.selectbox(key="k_mode").set_value("Rose roh").run()
    at.selectbox(key="k_mode").set_value("Spirale").run()
    assert at.slider(key="k_turns").value == 9.0


def test_spiegel_deckt_alle_bedingten_regler_ab():
    """Wer einen bedingten Regler ergaenzt und die Liste vergisst, verliert
    dessen Wert beim Ausblenden — ohne dass irgendwo ein Fehler auftritt."""
    import sonic_artwork as app
    from sonicart.render import MODES
    alle_knobs = set().union(*(m.knobs() for m in MODES.values()))
    #  Regler, die jeder Modus kennt, stehen immer da und brauchen nichts.
    immer = {"thickness", "gamma", "rotation"}
    fehlend = alle_knobs - immer - set(app.BEDINGT_MODUS)
    assert not fehlend, f"nicht im Spiegel beruecksichtigt: {sorted(fehlend)}"
    for keys in app.BEDINGT_MODUS.values():
        for k in keys:
            assert k in app.DEFAULTS, f"{k} fehlt in DEFAULTS"


def test_aufklapper_hat_festes_label():
    """Ein wechselndes Label gibt dem Aufklapper eine neue Identitaet —
    er fiel dann bei jeder Aenderung wieder zu."""
    import sonic_artwork as app
    quelle = _anweisungen(app._panel_form)
    assert "st.expander('Modi mischen')" in quelle


def test_farbfelder_behalten_ihre_breite():
    """st.columns(n) verteilte die Felder ueber die volle Breite: bei zwei
    Farbstufen standen sie weit auseinander und schwebten im Leeren."""
    import sonic_artwork as app
    quelle = _anweisungen(app._panel_farbe)
    assert "st.columns(MAX_STUFEN)" in quelle
    assert "st.columns(n)" not in quelle


def test_zustand_wird_jeden_durchlauf_neu_gesetzt():
    """Streamlit uebernimmt einen Wert nur, wenn er im selben Durchlauf gesetzt
    wurde, in dem das Widget zum ersten Mal entsteht.

    Vor dem Datei-Upload existiert der Hauptbereich nicht. Wurden die Vorgaben
    dort nur einmal gesetzt, zeigte nach dem Upload jedes Widget seinen eigenen
    Vorgabewert — 'Rose gespiegelt' im Auswahlfeld, waehrend das Bild im
    Gitter-Modus gerechnet wurde, alle Regler am Anschlag, die Farbfelder
    schwarz. Beim ersten Klick schickte der Browser diese falschen Werte
    zurueck in den Zustand, und das Bild wurde wirklich schwarz.
    """
    import sonic_artwork as app
    quelle = _anweisungen(app.init_state)
    assert "st.session_state.get(k, spiegel.get(k, v))" in quelle, \
        "Vorgaben muessen in jedem Durchlauf neu gesetzt werden"
    assert "if k not in st.session_state" not in quelle


def test_alle_sechs_farbstufen_haben_einen_wert(app_mit_audio):
    """c3 bis c5 hatten keine Vorgabe — beim Hochschalten der Farbstufen
    starteten die Felder auf Schwarz."""
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    for i in range(6):
        wert = at.session_state[f"c{i}"]
        assert wert and wert != "#000000", f"c{i} = {wert}"


def test_farbstufen_hochschalten_liefert_keine_schwarzen_felder(app_mit_audio):
    at = AppTest.from_file(app_mit_audio, default_timeout=300)
    at.run()
    at.slider(key="n_stops").set_value(6).run()
    felder = [c.value for c in at.color_picker if c.key and c.key.startswith("c")]
    assert len(felder) == 6
    assert "#000000" not in felder, felder


# ----------------------------------------------------------------------
# Footage-Tab
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def app_mit_clip(tmp_path_factory):
    """Kopie der App, bei der Audio- UND Video-Uploader feste Dateien liefern."""
    from sonicart import ffmpeg

    d = tmp_path_factory.mktemp("clipapp")
    wav = d / "t.wav"
    sr = 22050
    t = np.linspace(0, 6, 6 * sr, endpoint=False)
    y = sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330)) / 3 * 0.5
    sf.write(wav, y.astype(np.float32), sr)

    clip = d / "clip.mp4"
    ffmpeg.run(["-y", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc=size=320x240:rate=25:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)])

    stub = d / "app_stub.py"
    stub.write_text(textwrap.dedent(f"""
        import sys, io
        sys.path.insert(0, {str(PROJEKT)!r})
        import streamlit as st
        WAV = open({str(wav)!r}, "rb").read()
        MP4 = open({str(clip)!r}, "rb").read()

        class FakeUpload(io.BytesIO):
            def __init__(self, data, name):
                super().__init__(data)
                self.name = name

        def _upload(label, **kw):
            if "Audiodatei" in label:
                return FakeUpload(WAV, "t.wav")
            if "Videoclip" in label:
                return FakeUpload(MP4, "clip.mp4")
            return None

        st.file_uploader = _upload

        import sonic_artwork as app
        app.main()
    """))
    return str(stub)


def test_footage_tab_laeuft_mit_clip(app_mit_clip):
    at = AppTest.from_file(app_mit_clip, default_timeout=600)
    at.run()
    assert not at.exception, [getattr(e, "message", e) for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert [b for b in at.button if b.label == "Vorschau-Frame"], \
        [b.label for b in at.button]


def test_footage_vorschau_zeichnet_ein_bild(app_mit_clip):
    """Der Knopf muss wirklich einen Frame liefern, nicht bloss existieren."""
    at = AppTest.from_file(app_mit_clip, default_timeout=600)
    at.run()
    vorher = _zaehle(at, "Image")
    [b for b in at.button if b.label == "Vorschau-Frame"][0].click().run()
    assert not at.exception, [getattr(e, "message", e) for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert _zaehle(at, "Image") == vorher + 1


@pytest.fixture(scope="module")
def app_ohne_footage(tmp_path_factory):
    """App, in der sonicart.footage nicht importierbar ist.

    Bildet den Fall nach, der die oeffentliche App einmal komplett lahmgelegt
    hat: ein unvollstaendiger Checkout, bei dem footage.py einen Namen aus
    palette.py zog, den die dortige Fassung noch nicht hatte.
    """
    d = tmp_path_factory.mktemp("nofootage")
    wav = d / "t.wav"
    sr = 22050
    t = np.linspace(0, 6, 6 * sr, endpoint=False)
    sf.write(wav, (np.sin(2 * np.pi * 220 * t) * 0.5).astype(np.float32), sr)

    stub = d / "app_stub.py"
    stub.write_text(textwrap.dedent(f"""
        import sys, io
        sys.path.insert(0, {str(PROJEKT)!r})
        #  Import von sonicart.footage scheitern lassen. Das Attribut am Paket
        #  muss mit weg: ein einmal importiertes Submodul haengt dort und
        #  wuerde den None-Eintrag in sys.modules sonst ueberholen — sonst
        #  haengt das Ergebnis davon ab, welcher Test vorher lief.
        import sonicart
        sonicart.__dict__.pop("footage", None)
        sys.modules["sonicart.footage"] = None
        import streamlit as st
        WAV = open({str(wav)!r}, "rb").read()

        class FakeUpload(io.BytesIO):
            name = "t.wav"

        st.file_uploader = lambda label, **kw: (
            FakeUpload(WAV) if "Audiodatei" in label else None)

        import sonic_artwork as app
        app.main()
    """))
    return str(stub)


def test_kaputtes_footage_legt_die_app_nicht_lahm(app_ohne_footage):
    """Bild, Video und Album muessen weiterlaufen, wenn Footage ausfaellt."""
    at = AppTest.from_file(app_ohne_footage, default_timeout=300)
    at.run()
    assert not at.exception, [getattr(e, "message", e) for e in at.exception]
    #  Das Vorschaubild des Bild-Reiters steht weiterhin
    assert _zaehle(at, "Image") == 1
    #  und der Ausfall wird dort gemeldet, wo er hingehoert
    assert any("Footage nicht verfuegbar" in e.value for e in at.error), \
        [e.value for e in at.error]
