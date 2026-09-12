import io
import zipfile

import pytest

from sonicart.artwork import Recipe, build
from sonicart.export import (PLATFORM_SPECS, SIZES, export_formats, export_svg,
                             export_svg_layers, formats_zip, platform_check,
                             print_report, save_jpeg, save_png, svg_zip)
from sonicart.render import MODES


@pytest.fixture(scope="module")
def bild(an):
    return build(an, Recipe(mode="Rose roh", size=240), 240)


def test_social_formate_haben_die_richtigen_masse(bild):
    out = export_formats(bild, "#08060d")
    assert set(out) == set(SIZES)
    for name, im in out.items():
        assert im.size == SIZES[name]


def test_zip_enthaelt_alle_formate_und_das_rezept(bild):
    r = Recipe(mode="Rose roh", size=240)
    z = zipfile.ZipFile(io.BytesIO(formats_zip(export_formats(bild, r.bg), r)))
    namen = set(z.namelist())
    assert namen == {f"{k}.png" for k in SIZES} | {"rezept.json"}
    assert Recipe.from_json(z.read("rezept.json")).to_dict() == r.to_dict()


@pytest.mark.parametrize("name", [m for m in MODES if MODES[m].vector])
def test_vektormodi_liefern_svg(an, name):
    svg = export_svg(an, Recipe(mode=name), size_px=300)
    assert svg.startswith(b"<?xml") and b"<svg" in svg
    assert len(svg) < 12_000_000, "Detailgrenze muss die Dateigroesse deckeln"


def test_rastermodus_verweigert_svg(an):
    with pytest.raises(ValueError, match="rasterbasiert"):
        export_svg(an, Recipe(mode="Lissajous (Stereo)"), size_px=200)


def test_svg_ebenen_ueberspringen_rastermodi(an):
    r = Recipe(mix={"Rose roh": 0.8, "Spirale": 0.5, "Lissajous (Stereo)": 0.4})
    lagen = export_svg_layers(an, r, 240)
    assert set(lagen) == {"Rose roh", "Spirale"}
    z = zipfile.ZipFile(io.BytesIO(svg_zip(lagen, r)))
    assert set(z.namelist()) == {"Rose_roh.svg", "Spirale.svg", "rezept.json"}


def test_detailgrenze_wirkt(an):
    """Kleineres Budget muss die Datei messbar schrumpfen."""
    groessen = [len(export_svg(an, Recipe(mode="Spirale"), size_px=300,
                               max_quads=q)) for q in (40_000, 10_000, 3_000)]
    assert groessen == sorted(groessen, reverse=True), groessen
    assert groessen[-1] < groessen[0] / 2


def test_svg_groesse_haengt_nicht_an_der_songlaenge(an, structured):
    """Ohne Budget waechst das SVG linear mit der Spieldauer."""
    kurz = len(export_svg(an, Recipe(mode="Spirale"), size_px=300))
    lang = len(export_svg(structured, Recipe(mode="Spirale"), size_px=300))
    assert 0.5 < lang / kurz < 2.0, (kurz, lang)


def test_plattformpruefung():
    from PIL import Image
    klein = Image.new("RGB", (400, 400))
    assert not platform_check(klein, "Spotify Cover")["ok"]
    gross = Image.new("RGB", (3000, 3000))
    assert platform_check(gross, "Spotify Cover")["ok"]
    breit = Image.new("RGB", (3000, 2000))
    assert "nicht quadratisch" in " ".join(platform_check(breit, "Bandcamp Cover")["issues"])
    riesig = platform_check(gross, "Spotify Cover", data=b"x" * (11 * 1024 * 1024))
    assert any("gross" in i for i in riesig["issues"])


def test_druckbericht_meldet_neonpalette():
    rep = print_report(Recipe(stops=["#08060d", "#3a1d6e", "#7b2ff7", "#c9a0ff"]))
    assert rep["kritisch"] > 0
    grau = print_report(Recipe(stops=["#000000", "#666666", "#ffffff"], bg="#000000"))
    assert grau["kritisch"] == 0


def test_jpeg_ist_kleiner_als_png(bild):
    r = Recipe()
    assert len(save_jpeg(bild)) < len(save_png(bild, r))
