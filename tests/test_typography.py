"""Schriften, Satz und Kontrastpruefung."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sonicart.artwork import Recipe, build
from sonicart.typography import (ANCHORS, CONTRAST_MODES, FONT_PAIRS, TYPEFACES,
                                 TypeSpec, apply_typography,
                                 apply_typography_report, available_faces,
                                 best_ink, check_contrast, contrast_ratio,
                                 layout, line_boxes, load_font,
                                 relative_luminance, sample_background,
                                 worst_contrast)


# ---------------------------------------------------------------- Schriften
def test_mitgelieferte_schriften_sind_da():
    fehlend = [n for n, t in TYPEFACES.items()
               if n != "System" and t.path is None]
    assert not fehlend, f"Schriftdateien fehlen: {fehlend}"
    assert len(available_faces()) >= 5


@pytest.mark.parametrize("name", [n for n in TYPEFACES if n != "System"])
def test_jede_schrift_laedt_und_setzt(name):
    f = load_font(name, 48)
    d = ImageDraw.Draw(Image.new("L", (8, 8)))
    assert d.textlength("KORREND", font=f) > 20


def test_variable_schnitte_unterscheiden_sich():
    """Archivo hat neun Schnitte — sie muessen verschieden breit bauen."""
    d = ImageDraw.Draw(Image.new("L", (8, 8)))
    duenn = d.textlength("KORREND", font=load_font("Archivo", 60, "Thin"))
    fett = d.textlength("KORREND", font=load_font("Archivo", 60, "Black"))
    assert fett > duenn * 1.03, (duenn, fett)


def test_paarungen_verweisen_auf_vorhandene_schriften():
    for name, rollen in FONT_PAIRS.items():
        for face in rollen:
            assert face in TYPEFACES, f"{name}: {face} unbekannt"
            assert TYPEFACES[face].path, f"{name}: {face} fehlt auf der Platte"


def test_eigene_datei_schlaegt_die_auswahl():
    eigen = TYPEFACES["Space Mono"].path
    d = ImageDraw.Draw(Image.new("L", (8, 8)))
    a = d.textlength("MMM", font=load_font("Bebas Neue", 60))
    b = d.textlength("MMM", font=load_font("Bebas Neue", 60, path=eigen))
    assert a != b


def test_unbekannte_schrift_faellt_zurueck():
    assert load_font("gibt es nicht", 40) is not None


def test_bebas_erzwingt_versalien():
    sp = TypeSpec(title="korrend", face_title="Bebas Neue", title_upper=False)
    assert sp.rollen()[0][1] == "KORREND"


# ---------------------------------------------------------------- Kontrast
def test_wcag_werte():
    assert contrast_ratio((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0, abs=0.01)
    assert contrast_ratio((255, 255, 255), (255, 255, 255)) == pytest.approx(1.0)
    assert relative_luminance((255, 255, 255)) == pytest.approx(1.0, abs=1e-6)


def test_schlechtester_teil_statt_median():
    """Der eigentliche Fehler: eine Zeile kreuzt die Kante einer Form.

    Der Median trifft dann die Mehrheitsflaeche und meldet gute Werte,
    waehrend die andere Haelfte der Zeile unlesbar ist.
    """
    img = Image.new("RGB", (200, 40), (10, 10, 10))
    ImageDraw.Draw(img).rectangle([140, 0, 200, 40], fill=(245, 245, 245))
    proben = sample_background(img, (0, 0, 200, 40))
    from sonicart.typography import background_at
    median_kontrast = contrast_ratio((255, 255, 255), background_at(img, (0, 0, 200, 40)))
    assert median_kontrast > 15, "Median sieht nur das Dunkle"
    assert worst_contrast(proben, "#ffffff") < 2.0, "der helle Teil muss auffallen"


def test_perzentil_ignoriert_einzelne_ausreisser():
    """Rasterpunkte duerfen nicht jede Flaeche durchfallen lassen."""
    img = Image.new("RGB", (200, 40), (10, 10, 10))
    ImageDraw.Draw(img).rectangle([0, 0, 3, 3], fill=(255, 255, 255))
    proben = sample_background(img, (0, 0, 200, 40))
    assert worst_contrast(proben, "#ffffff", 5) > 10


def test_best_ink_haelt_den_wunsch_wenn_er_reicht():
    hell = np.full((50, 3), 245, dtype=np.uint8)
    farbe, v = best_ink([hell], ["#111111", "#ffffff"], 4.5)
    assert farbe == "#111111" and v > 4.5


def test_best_ink_bevorzugt_die_palette_vor_weiss():
    dunkel = np.full((50, 3), 20, dtype=np.uint8)
    farbe, _ = best_ink([dunkel], ["#101010", "#c9a0ff", "#ffffff"], 4.5)
    assert farbe == "#c9a0ff", "eine passende Palettenfarbe schlaegt Weiss"


def test_best_ink_beruecksichtigt_alle_zeilen():
    hell = np.full((50, 3), 245, dtype=np.uint8)
    dunkel = np.full((50, 3), 15, dtype=np.uint8)
    farbe, v = best_ink([hell, dunkel], ["#ffffff", "#808080", "#000000"], 4.5)
    assert v < 4.5, "gegen Hell UND Dunkel reicht keine einzelne Farbe"


@pytest.mark.parametrize("grund,wunsch", [((240, 236, 226), "#efe9dd"),
                                          ((18, 18, 24), "#141420")])
def test_farbe_wird_umgeschaltet(grund, wunsch):
    img = Image.new("RGB", (400, 400), grund)
    sp = TypeSpec(artist="THYLO TEKKS", title="Korrend", color=wunsch,
                  contrast_mode="Farbe umschalten")
    _, b = apply_typography_report(img, sp, ["#efe9dd", "#141420"])
    assert b["geaendert"] and b["ok"] and b["kontrast"] > 4.5


def test_modus_aus_meldet_ohne_zu_aendern():
    img = Image.new("RGB", (400, 400), (240, 236, 226))
    sp = TypeSpec(title="Korrend", color="#efe9dd", contrast_mode="Aus")
    _, b = apply_typography_report(img, sp)
    assert not b["ok"] and not b["geaendert"] and b["farbe"] == "#efe9dd"


def test_feld_unterlegen():
    img = Image.new("RGB", (400, 400), (240, 236, 226))
    sp = TypeSpec(title="Korrend", color="#efe9dd",
                  contrast_mode="Feld unterlegen")
    aus, b = apply_typography_report(img, sp, ["#efe9dd", "#141420"])
    assert b.get("platte") and b["ok"]
    # Das Feld muss im Bild sichtbar sein
    assert np.abs(np.asarray(aus).astype(float)
                  - np.asarray(img).astype(float)).mean() > 1.0


def test_befund_nennt_die_schwaechste_zeile():
    """Der Streifen liegt genau auf einer Zeile — die muss gemeldet werden."""
    sp = TypeSpec(artist="THYLO TEKKS", title="Korrend", catalog="PT-004",
                  anchor="unten links", color="#ffffff", contrast_mode="Aus")
    img = Image.new("RGB", (500, 500), (10, 10, 10))
    z, k, start = layout(sp, 500, 500)
    ziel = line_boxes(z, k, start, sp, 500)[-1]          # Katalognummer
    ImageDraw.Draw(img).rectangle(
        [0, int(ziel[1]), 500, int(ziel[3])], fill=(250, 250, 250))
    _, b = apply_typography_report(img, sp)
    assert b["zeile"] == "PT-004", b["zeile"]
    assert not b["ok"]


def test_gut_stehende_zeilen_bleiben_unbeanstandet():
    """Gegenprobe: ohne stoerenden Streifen meldet nichts."""
    sp = TypeSpec(artist="THYLO TEKKS", title="Korrend", catalog="PT-004",
                  anchor="unten links", color="#ffffff", contrast_mode="Aus")
    img = Image.new("RGB", (500, 500), (10, 10, 10))
    _, b = apply_typography_report(img, sp)
    assert b["ok"] and b["kontrast"] > 15


def test_contrast_modi_vollstaendig():
    assert CONTRAST_MODES[0] == "Aus"
    assert "Farbe umschalten" in CONTRAST_MODES
    assert "Feld unterlegen" in CONTRAST_MODES


# ---------------------------------------------------------------- Satz
def test_zeilenkaesten_folgen_dem_satz():
    sp = TypeSpec(artist="A", title="B", label="C", catalog="D")
    z, k, start = layout(sp, 600, 600)
    kaesten = line_boxes(z, k, start, sp, 600)
    assert len(kaesten) == 4
    oben = [b[1] for b in kaesten]
    assert oben == sorted(oben), "Zeilen muessen von oben nach unten laufen"


def test_zu_breiter_block_wird_verkleinert():
    lang = "EIN SEHR SEHR LANGER TITEL DER NIEMALS PASST"
    eng = TypeSpec(title=lang, max_width=0.3)
    weit = TypeSpec(title=lang, max_width=1.0)
    b_eng = layout(eng, 600, 600)[1][2] - layout(eng, 600, 600)[1][0]
    b_weit = layout(weit, 600, 600)[1][2] - layout(weit, 600, 600)[1][0]
    assert b_eng < b_weit
    assert b_eng <= 0.3 * 600 + 1


@pytest.mark.parametrize("anchor", list(ANCHORS))
def test_alle_anker(anchor):
    img = Image.new("RGB", (300, 300), (10, 10, 10))
    sp = TypeSpec(title="Korrend", anchor=anchor)
    aus = apply_typography(img, sp)
    assert np.asarray(aus).std() > 0.5


def test_haarlinie_zeichnet():
    img = Image.new("RGB", (300, 300), (10, 10, 10))
    ohne = np.asarray(apply_typography(img, TypeSpec(title="K", rule=0.0))).astype(float)
    mit = np.asarray(apply_typography(img, TypeSpec(title="K", rule=2.0))).astype(float)
    assert np.abs(ohne - mit).sum() > 0


def test_leerer_satz_aendert_nichts():
    img = Image.new("RGB", (80, 80), (30, 30, 30))
    assert apply_typography(img, TypeSpec()) is img


def test_check_contrast_misst_ohne_zu_zeichnen():
    img = Image.new("RGB", (300, 300), (250, 250, 250))
    b = check_contrast(img, TypeSpec(title="Korrend", color="#ffffff"))
    assert b and not b["ok"] and b["kontrast"] < 2


# ---------------------------------------------------------------- Pipeline
def test_build_liefert_den_befund(an):
    r = Recipe.matte("Indigo", mode="Gitter", size=200, typography=dict(
        artist="THYLO TEKKS", title="Korrend", catalog="PT-004",
        color="#ffffff", contrast_mode="Farbe umschalten"))
    b = {}
    build(an, r, 200, report=b)
    assert b["kontrast"] > 0 and "farbe" in b and "zeile" in b


def test_umgeschaltete_farbe_kommt_aus_der_palette(an):
    r = Recipe.matte("Indigo", mode="Gitter", size=220, typography=dict(
        title="Korrend", color="#e8dcc4", contrast_mode="Farbe umschalten"))
    b = {}
    build(an, r, 220, report=b)
    if b["geaendert"]:
        erlaubt = set(r.stops) | {r.bg, r.riso["paper"], "#ffffff", "#111111"}
        assert b["farbe"] in erlaubt
