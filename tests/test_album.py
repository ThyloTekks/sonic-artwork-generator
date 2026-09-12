import json
import os

import numpy as np
import pytest
import soundfile as sf

from sonicart.album import (album_hues, album_palettes, contact_sheet,
                            find_audio, render_album)
from sonicart.analysis import Analysis
from sonicart.artwork import Recipe
from sonicart.palette import hex_to_oklch

SR = 22050
AKKORDE = {
    "01_intro": ((261.63, 329.63, 392.00), 130.81),
    "02_drift": ((293.66, 349.23, 440.00), 146.83),
    "03_seek":  ((220.00, 261.63, 329.63), 110.00),
    "04_outro": ((196.00, 246.94, 293.66), 98.00),
}


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    d = tmp_path_factory.mktemp("release")
    for name, (freqs, bass) in AKKORDE.items():
        t = np.linspace(0, 12, 12 * SR, endpoint=False)
        y = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs) * 0.5
        y = y + np.sin(2 * np.pi * bass * t) * 0.3
        y = y + np.random.default_rng(0).normal(0, 0.03, len(t))
        sf.write(d / f"{name}.wav",
                 np.vstack([y, np.roll(y, 120)]).T.astype(np.float32), SR)
    (d / "liesmich.txt").write_text("keine Audiodatei")
    return str(d)


def test_findet_nur_audiodateien(release):
    f = find_audio(release)
    assert len(f) == 4 and all(x.endswith(".wav") for x in f)
    assert f == sorted(f), "alphabetisch = ueblicherweise die Trackfolge"


def test_leerer_ordner(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_audio(str(tmp_path))


@pytest.fixture(scope="module")
def analysen(release):
    from sonicart.analysis import load
    return [load(p) for p in find_audio(release)]


def test_farbtoene_bleiben_im_rahmen(analysen):
    h = album_hues(analysen, spread_deg=40.0)
    abw = np.rad2deg(np.abs(h["offsets"]))
    assert abw.max() <= 20.001, f"maximal die halbe Spreizung: {abw}"
    assert abw.max() > 1.0, "aber nicht alle gleich"


def test_spreizung_null_macht_alle_gleich(analysen):
    h = album_hues(analysen, spread_deg=0.0)
    assert np.allclose(h["offsets"], 0.0)


def test_palettenstruktur_ist_ueber_das_album_konstant(analysen):
    p = album_palettes(analysen, n=4, template=None, spread_deg=60.0)
    kurven = [[round(hex_to_oklch(c)[0], 2) for c in stops] for stops in p["tracks"]]
    assert all(k == kurven[0] for k in kurven), \
        "Helligkeitskurve muss gleich bleiben, nur der Farbton wandert"
    farbtoene = [round(hex_to_oklch(s[2])[2], 3) for s in p["tracks"]]
    assert len(set(farbtoene)) > 1, "die Titel duerfen nicht farbgleich sein"


def test_kontaktbogen(analysen):
    from PIL import Image
    bilder = [Image.new("RGB", (60, 60), (i * 40, 20, 200)) for i in range(5)]
    sheet = contact_sheet(bilder, [f"t{i}" for i in range(5)], cols=3, cell=80)
    assert sheet.size[0] > 200 and sheet.size[1] > 150
    assert np.asarray(sheet).std() > 5


def test_serie_rendern(release, tmp_path):
    out = str(tmp_path / "cover")
    rep = render_album(release, Recipe(mode="Spirale", size=200), out, size=200)
    assert len(rep["titel"]) == 4
    dateien = set(os.listdir(out))
    assert "00_kontaktbogen.png" in dateien and "album.json" in dateien
    assert sum(f.endswith(".png") for f in dateien) == 5      # 4 Titel + Bogen
    gespeichert = json.loads((tmp_path / "cover" / "album.json").read_text())
    assert gespeichert["titel"][0]["tonart"]
    # Rezept steckt in jedem Titelbild
    from sonicart.export import read_recipe
    erste = sorted(f for f in dateien if f.startswith("01_"))[0]
    r = read_recipe(os.path.join(out, erste))
    assert r is not None and r.mode == "Spirale"


def test_ohne_gemeinsame_palette_bleibt_die_vorgabe(release, tmp_path):
    r = Recipe(mode="Rose roh", stops=["#000000", "#888888", "#ffffff"], size=160)
    rep = render_album(release, r, str(tmp_path / "o"), size=160,
                       shared_palette=False)
    assert all(t["palette"] == list(r.stops) for t in rep["titel"])
