"""Die Druckschicht.

Zwei Fehler, die hier festgehalten werden, weil sie beim Bauen aufgetreten
sind: dunkles Papier lief mit Multiply zu Schwarz zu, und die Rose wurde von
der Bandnormierung zu einer flachen Scheibe eingeebnet.
"""

import io

import numpy as np
import pytest
from PIL import Image

from sonicart.artwork import Recipe, build
from sonicart.export import read_recipe, save_png
from sonicart.palette import hex_to_rgb
from sonicart.riso import (DEFAULT_ANGLES, INK_SETS, RisoSpec, apply_riso,
                           density, halftone_mask, ink_set, paper_texture)


def _lum(rgb):
    return np.asarray(rgb, float) @ np.array([0.2126, 0.7152, 0.0722])


def test_dichte_misst_abstand_zum_grund():
    img = Image.new("RGB", (32, 32), hex_to_rgb("#08060d"))
    img.putpixel((5, 5), (255, 255, 255))
    d = density(img, "#08060d")
    assert d[5, 5] == pytest.approx(1.0, abs=0.02)
    assert d[20, 20] == pytest.approx(0.0, abs=0.02)


def test_dichte_auch_auf_hellem_grund():
    img = Image.new("RGB", (32, 32), (240, 240, 240))
    img.putpixel((5, 5), (0, 0, 0))
    d = density(img, "#f0f0f0")
    assert d[5, 5] > 0.9 and d[20, 20] < 0.1


def test_dichte_achtet_auf_alpha():
    img = Image.new("RGBA", (16, 16), (255, 255, 255, 0))
    img.putpixel((8, 8), (255, 255, 255, 255))
    d = density(img, "#000000")
    assert d[8, 8] > 0.9
    assert d[2, 2] == pytest.approx(0.0, abs=1e-6), "Alpha 0 = kein Auftrag"


@pytest.mark.parametrize("v", [0.1, 0.4, 0.8])
def test_raster_deckt_mit_dem_wert(v):
    feld = np.full((120, 120), v)
    deckung = halftone_mask(feld, 6, 15).mean()
    assert 0 < deckung < 1
    mehr = halftone_mask(np.full((120, 120), min(v + 0.15, 1.0)), 6, 15).mean()
    assert mehr > deckung, "groesserer Wert muss groessere Punkte geben"


def test_rasterwinkel_unterscheiden_sich():
    feld = np.full((100, 100), 0.5)
    a = halftone_mask(feld, 6, 15)
    b = halftone_mask(feld, 6, 75)
    assert (a != b).mean() > 0.1, "gleiche Winkel auf zwei Platten geben Moire"
    assert len(set(DEFAULT_ANGLES)) == len(DEFAULT_ANGLES)


@pytest.mark.parametrize("name", list(INK_SETS))
def test_jeder_farbsatz_druckt(an, name):
    r = Recipe.matte(name, mode="Spirale", size=200)
    img = build(an, r, 200)
    a = np.asarray(img).reshape(-1, 3)
    assert img.size == (200, 200)
    # Flache Farben statt Verlauf: wenige verschiedene Werte
    assert len(np.unique(a, axis=0)) < 4000


def test_dunkles_papier_laeuft_nicht_zu(an):
    """War der Fehler: Multiply kann nur abdunkeln, also wurde alles schwarz."""
    spez = ink_set("Nachtdruck")
    assert spez.effective_overprint() == "screen"
    img = build(an, Recipe.matte("Nachtdruck", mode="HPSS-Zeit", size=220), 220)
    hell = _lum(np.asarray(img) / 255.0)
    papier = _lum(np.array(hex_to_rgb(spez.paper)) / 255.0)
    assert hell.max() > papier + 0.25, "die Form muss heller sein als das Papier"


def test_helles_papier_nimmt_multiply():
    assert ink_set("Zinnober").effective_overprint() == "multiply"
    assert RisoSpec(paper="#ffffff", overprint="screen").effective_overprint() == "screen"


def test_passerversatz_wirkt(an):
    basis = Recipe.matte("Ocker", mode="Spirale", size=200)
    ohne = dict(basis.riso, misregister=0.0)
    mit = dict(basis.riso, misregister=4.0)
    a = np.asarray(build(an, Recipe.matte("Ocker", mode="Spirale", size=200,
                                          riso=ohne), 200)).astype(float)
    b = np.asarray(build(an, Recipe.matte("Ocker", mode="Spirale", size=200,
                                          riso=mit), 200)).astype(float)
    assert np.abs(a - b).mean() > 0.5


def test_papierfaser():
    flach = np.ones((40, 40, 3)) * 0.8
    assert paper_texture(flach, 0.0).std() == pytest.approx(0.0, abs=1e-9)
    assert paper_texture(flach, 0.1).std() > 0.001


def test_riso_gilt_fuer_jeden_modus(an):
    from sonicart.render import MODES
    for m in MODES:
        img = build(an, Recipe.matte("Graphit", mode=m, size=150), 150)
        assert img.size == (150, 150)


def test_transparenz_bleibt_erhalten(an):
    r = Recipe.matte("Indigo", mode="Rose roh", size=180)
    r.transparent = True
    img = build(an, r, 180)
    assert img.mode == "RGBA"
    assert (np.asarray(img)[..., 3] == 0).any()


def test_riso_im_rezept_und_im_png(an):
    r = Recipe.matte("Zinnober", mode="Gitter", size=180)
    zurueck = read_recipe(io.BytesIO(save_png(build(an, r, 180), r)))
    assert zurueck.riso == r.riso
    assert zurueck.riso_spec.paper == r.riso_spec.paper


def test_ohne_riso_bleibt_alles_beim_alten(an):
    r = Recipe(mode="Rose roh", size=160)
    assert r.riso is None and r.riso_spec is None
    assert build(an, r, 160).size == (160, 160)
