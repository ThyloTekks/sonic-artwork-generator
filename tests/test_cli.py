import json
import os

import pytest

from sonicart.cli import main


def test_inspect(wav_file, capsys):
    assert main(["inspect", wav_file]) == 0
    daten = json.loads(capsys.readouterr().out.split("Abschnitte")[0])
    assert daten["tonart"] and daten["tempo_bpm"] > 0


def test_render_und_rezept_zurueck(wav_file, tmp_path, capsys):
    out = str(tmp_path / "cover.png")
    assert main(["render", wav_file, "-o", out, "--size", "240",
                 "--mode", "Spirale", "--layout", "Goldener Schnitt",
                 "--artist", "THYLO TEKKS", "--title", "Korrend"]) == 0
    assert os.path.getsize(out) > 1000
    capsys.readouterr()
    assert main(["recipe", out]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["mode"] == "Spirale" and r["layout"] == "Goldener Schnitt"
    assert r["typography"]["artist"] == "THYLO TEKKS"


def test_recipe_als_ausgangspunkt(wav_file, tmp_path, capsys):
    rez = tmp_path / "r.json"
    rez.write_text(json.dumps({"mode": "Segmente", "size": 200,
                               "params": {"gamma": 2.1}}))
    out = str(tmp_path / "b.png")
    assert main(["render", wav_file, "-o", out, "--recipe", str(rez)]) == 0
    capsys.readouterr()
    main(["recipe", out])
    r = json.loads(capsys.readouterr().out)
    assert r["mode"] == "Segmente" and r["params"]["gamma"] == 2.1


def test_render_mit_formaten_und_svg(wav_file, tmp_path):
    out = str(tmp_path / "c.png")
    assert main(["render", wav_file, "-o", out, "--size", "240", "--formats",
                 "--svg", "--mode", "Rose roh"]) == 0
    assert (tmp_path / "c_formate.zip").exists()
    assert (tmp_path / "c.svg").exists()


def test_jpeg_ausgabe(wav_file, tmp_path):
    out = str(tmp_path / "c.jpg")
    assert main(["render", wav_file, "-o", out, "--size", "240"]) == 0
    from PIL import Image
    assert Image.open(out).format == "JPEG"


def test_palette_aus_tonart(wav_file, tmp_path, capsys):
    out = str(tmp_path / "k.png")
    assert main(["render", wav_file, "-o", out, "--size", "200",
                 "--key-palette"]) == 0
    capsys.readouterr()
    main(["recipe", out])
    r = json.loads(capsys.readouterr().out)
    from sonicart.palette import PRESETS
    assert r["stops"] != PRESETS["Korrend"], "Palette muss aus der Tonart kommen"


def test_symmetrie_aus_der_taktart(wav_file, tmp_path, capsys):
    out = str(tmp_path / "s.png")
    assert main(["render", wav_file, "-o", out, "--size", "200",
                 "--symmetry", "0"]) == 0
    capsys.readouterr()
    main(["recipe", out])
    r = json.loads(capsys.readouterr().out)
    assert r["params"]["symmetry"] in (2, 3, 4, 5, 6, 7)


def test_recipe_ohne_rezept_meldet_fehler(tmp_path, capsys):
    from PIL import Image
    p = tmp_path / "leer.png"
    Image.new("RGB", (8, 8)).save(p)
    assert main(["recipe", str(p)]) == 1


def test_video(wav_file, tmp_path):
    out = str(tmp_path / "v.mp4")
    assert main(["video", wav_file, "-o", out, "--duration", "3",
                 "--fps", "8", "--video-mode", "Rose roh", "--no-audio"]) == 0
    assert os.path.getsize(out) > 1000
