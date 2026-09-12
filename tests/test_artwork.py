import io

import numpy as np
import pytest
from PIL import Image

from sonicart.artwork import DEFAULT_PARAMS, Recipe, build
from sonicart.compose import LAYOUTS, apply_image, compose, mix_modes
from sonicart.export import read_recipe, save_png
from sonicart.palette import PRESETS, make_cmap


def test_rezept_json_hin_und_zurueck():
    r = Recipe(mode="Spirale", layout="Angeschnitten", layout_scale=1.2,
               mix={"Spirale": 0.5}, typography=dict(artist="A", title="B"),
               effects=dict(intensity=0.7, grain=True))
    assert Recipe.from_json(r.to_json()).to_dict() == r.to_dict()


def test_rezept_ergaenzt_fehlende_parameter():
    r = Recipe.from_dict({"mode": "Segmente", "params": {"gamma": 2.0}})
    assert r.params["gamma"] == 2.0
    assert r.params["inner"] == DEFAULT_PARAMS["inner"], "Rest muss aufgefuellt werden"


def test_rezept_ignoriert_unbekannte_felder():
    r = Recipe.from_dict({"mode": "Spirale", "gibt_es_nicht": 1})
    assert r.mode == "Spirale"


def test_rezept_steckt_im_png(an):
    r = Recipe(mode="Segmente", size=200, typography=dict(title="Korrend"))
    img = build(an, r, 200)
    zurueck = read_recipe(io.BytesIO(save_png(img, r)))
    assert zurueck is not None
    assert zurueck.to_dict() == r.to_dict()


def test_png_ohne_rezept_gibt_none(tmp_path):
    p = tmp_path / "leer.png"
    Image.new("RGB", (8, 8)).save(p)
    assert read_recipe(str(p)) is None


def test_seed_folgt_dem_dateinamen():
    r = Recipe(effects=dict(intensity=1.0, seed=0))
    a = r.seeded("track_a.wav")["seed"]
    assert a == Recipe(effects=dict(intensity=1.0, seed=0)).seeded("track_a.wav")["seed"]
    assert a != r.seeded("track_b.wav")["seed"]
    assert Recipe(effects=dict(seed=42)).seeded("x")["seed"] == 42


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_alle_platzierungen(an, layout):
    img = build(an, Recipe(mode="Rose roh", layout=layout, size=200), 200)
    assert img.size == (200, 200)


def test_platzierung_veraendert_das_bild(an):
    a = np.asarray(build(an, Recipe(mode="Rose roh", layout="Zentriert", size=200), 200))
    b = np.asarray(build(an, Recipe(mode="Rose roh", layout="Goldener Schnitt", size=200), 200))
    assert np.abs(a.astype(float) - b).mean() > 3.0


def test_soziale_flaeche_bleibt_leer_bei_transparenz(an):
    img = build(an, Recipe(mode="Rose roh", transparent=True, size=160), 160,
                canvas=(160, 280))
    assert img.mode == "RGBA" and img.size == (160, 280)
    a = np.asarray(img)
    assert (a[..., 3] == 0).any(), "Rand muss durchsichtig bleiben"
    assert (a[a[..., 3] == 0][:, :3] == 0).all(), "kein RGB-Muell unter Alpha 0"


def test_mix_ist_heller_als_die_teile(an):
    cm = make_cmap(PRESETS["Korrend"])
    p = dict(DEFAULT_PARAMS)
    einzeln = np.asarray(mix_modes(an, {"Rose roh": 1.0}, cm, "#08060d", 180, p)).astype(float)
    gemischt = np.asarray(mix_modes(an, {"Rose roh": 1.0, "Kreis-Wellenform": 1.0},
                                    cm, "#08060d", 180, p)).astype(float)
    assert gemischt.mean() >= einzeln.mean() - 1e-6, "Lighten darf nicht abdunkeln"


def test_typografie_landet_im_bild(an):
    ohne = np.asarray(build(an, Recipe(mode="Rose roh", size=300), 300)).astype(float)
    mit = np.asarray(build(an, Recipe(mode="Rose roh", size=300, typography=dict(
        artist="THYLO TEKKS", title="Korrend", anchor="unten links")), 300)).astype(float)
    assert np.abs(ohne - mit).mean() > 0.2


def test_bild_maskierung(an):
    art = build(an, Recipe(mode="Rose roh", size=120), 120)
    tex = Image.new("RGB", (60, 90), (255, 0, 0))
    out = apply_image(art, tex, "Masking", "#000000")
    assert out.size == art.size
    assert np.asarray(out)[..., 0].max() > 100, "Rot der Textur muss durchkommen"


def test_mattes_rezept_ist_kein_leuchten():
    r = Recipe.matte()
    assert r.effects["bloom"] is False, "Bloom ist die Haelfte des Screensaver-Blicks"
    assert r.effects["chroma"] is False
    assert r.riso is not None and r.riso["paper"]
    assert r.params["whiten_amount"] > 0 and r.params["tilt"] > 0
    assert not MODES_ROUND(r.mode), "Standardmodus soll keine zentrierte Scheibe sein"


def MODES_ROUND(name):
    from sonicart.render import MODES
    return name in ("Rose gespiegelt", "Rose roh", "Kreis-Wellenform")


def test_mattes_rezept_laesst_sich_ueberschreiben():
    r = Recipe.matte("Indigo", mode="Strata", size=1234,
                     riso=None, layout="Angeschnitten")
    assert r.mode == "Strata" and r.size == 1234
    assert r.riso is None and r.layout == "Angeschnitten"


def test_druck_kommt_vor_dem_satz(an):
    """Text muss Vollton bleiben, sonst leidet die Lesbarkeit im Raster."""
    typ = dict(artist="THYLO TEKKS", title="Korrend", anchor="unten links",
               color="#ffffff", size=0.09)
    ohne = build(an, Recipe.matte("Graphit", mode="Gitter", size=320), 320)
    mit = build(an, Recipe.matte("Graphit", mode="Gitter", size=320,
                                 typography=typ), 320)
    a, b = np.asarray(ohne).astype(float), np.asarray(mit).astype(float)
    geaendert = np.abs(a - b).sum(axis=2) > 30
    assert geaendert.any(), "Satz muss sichtbar sein"
    # Der Kern der Buchstaben muss eine einzige flache Farbe sein — welche,
    # entscheidet die Kontrastpruefung. Die Kantenglaettung der Glyphen mischt
    # zwangslaeufig mit dem Grund, deshalb ein Anteil und keine Obergrenze
    # fuer die Farbzahl.
    pixel = b[geaendert].astype(np.uint8)
    farben, zahl = np.unique(pixel.reshape(-1, 3), axis=0, return_counts=True)
    voll = zahl.max() / zahl.sum()
    assert voll > 0.25, f"Text wirkt gerastert, haeufigste Farbe nur {voll:.0%}"


def test_mischverfahren_ueberlagern_und_aufhellen(an):
    """Ueberlagern malt der Reihe nach, Aufhellen nimmt den helleren Wert."""
    from sonicart.compose import MIX_BLENDS, mix_modes
    from sonicart.palette import make_cmap
    cm = make_cmap(PRESETS["Korrend"])
    p = dict(DEFAULT_PARAMS, whiten_amount=0.85, tilt=0.6)
    mix = {"Gitter": 0.8, "Rose gespiegelt": 0.9}
    bilder = {b: np.asarray(mix_modes(an, mix, cm, "#08060d", 200, p, blend=b))
              for b in MIX_BLENDS}
    assert set(MIX_BLENDS) == {"Ueberlagern", "Aufhellen"}
    a, b = (bilder[k].astype(float) for k in MIX_BLENDS)
    assert np.abs(a - b).mean() > 0.2, "die Verfahren muessen sich unterscheiden"


def test_ueberlagern_haelt_die_reihenfolge_ein(an):
    """Die spaetere Ebene liegt oben — beim Aufhellen gilt das nicht."""
    from sonicart.compose import mix_modes
    from sonicart.palette import make_cmap
    cm = make_cmap(PRESETS["Korrend"])
    p = dict(DEFAULT_PARAMS, whiten_amount=0.85)
    vorne = np.asarray(mix_modes(an, {"Rose roh": 1.0, "Gitter": 1.0},
                                 cm, "#08060d", 200, p, blend="Ueberlagern"))
    hinten = np.asarray(mix_modes(an, {"Gitter": 1.0, "Rose roh": 1.0},
                                  cm, "#08060d", 200, p, blend="Ueberlagern"))
    assert np.abs(vorne.astype(float) - hinten).mean() > 0.2


def test_aufhellen_bleibt_die_vorgabe_fuer_alte_rezepte():
    """Ein Rezept ohne Angabe darf nicht ploetzlich anders aussehen."""
    assert Recipe().mix_blend == "Aufhellen"
    assert Recipe.from_dict({"mode": "Spirale"}).mix_blend == "Aufhellen"
    assert Recipe.matte().mix_blend == "Ueberlagern"


def test_mischverfahren_im_rezept_ueberlebt_das_png(an):
    import io

    from sonicart.export import read_recipe, save_png
    r = Recipe.matte("Ocker", mode="Gitter", size=180,
                     mix={"Gitter": 0.8, "Strata": 0.5})
    zurueck = read_recipe(io.BytesIO(save_png(build(an, r, 180), r)))
    assert zurueck.mix_blend == "Ueberlagern" and zurueck.mix == r.mix


def test_flaechenfuellende_modi_sind_gekennzeichnet():
    from sonicart.render import MODES
    assert MODES["Gitter"].full_bleed and MODES["Strata"].full_bleed
    assert not MODES["Rose roh"].full_bleed
    assert not MODES["Spirale"].full_bleed
