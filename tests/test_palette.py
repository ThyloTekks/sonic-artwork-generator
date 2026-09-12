import numpy as np
import pytest

from sonicart import palette as P


def test_hex_drei_und_sechsstellig():
    assert P.hex_to_rgb("#000") == (0, 0, 0)
    assert P.hex_to_rgb("#abc") == P.hex_to_rgb("#aabbcc")
    assert P.hex_to_rgb("7b2ff7") == (123, 47, 247)
    with pytest.raises(ValueError):
        P.hex_to_rgb("#12345")


@pytest.mark.parametrize("hx", ["#7b2ff7", "#08060d", "#ffed00", "#ffffff", "#000000"])
def test_oklch_hin_und_zurueck(hx):
    assert P.oklch_to_hex(*P.hex_to_oklch(hx)) == hx


def test_palette_generator_laenge_und_helligkeitsverlauf():
    for mode in P.HARMONIES:
        if mode == "Profil-Struktur":
            continue
        stops = P.generate_palette("#7b2ff7", 5, mode)
        assert len(stops) == 5
        Ls = [P.hex_to_oklch(c)[0] for c in stops]
        assert Ls == sorted(Ls), f"{mode}: Helligkeit muss aufsteigen"


def test_profil_struktur_behaelt_helligkeitskurve():
    tmpl = P.PRESETS["Korrend"]
    out = P.generate_palette("#2fb0f7", len(tmpl), "Profil-Struktur", template=tmpl)
    a = [round(P.hex_to_oklch(c)[0], 2) for c in tmpl]
    b = [round(P.hex_to_oklch(c)[0], 2) for c in out]
    assert a == b, "Nur der Farbton darf wechseln, nicht die Helligkeit"


def test_tonart_zu_farbton_ist_quintenzirkel():
    # Benachbarte Quinten liegen dicht beieinander, Tritonus gegenueber.
    d_gc = abs(P.key_to_hue("G") - P.key_to_hue("C"))
    d_fsc = abs(P.key_to_hue("F#") - P.key_to_hue("C"))
    assert d_gc < d_fsc
    assert P.key_to_hue("A", "moll") != P.key_to_hue("A", "dur")


def test_druckpruefung_erkennt_neon():
    rows = {r["hex"]: r for r in P.print_check(["#7b2ff7", "#666666", "#ffed00"])}
    assert not rows["#7b2ff7"]["printable"], "Neonviolett liegt ausserhalb"
    assert rows["#7b2ff7"]["ratio"] > 1.5
    assert rows["#666666"]["printable"], "Neutrales Grau ist immer druckbar"
    assert rows["#ffed00"]["printable"], "Gelb ist eine Druckprimaerfarbe"
    # Der Beschnitt landet im Farbraum und behaelt den Farbton grob bei
    clipped = P.clip_to_print("#7b2ff7")
    assert P.print_check([clipped])[0]["printable"]


def test_cube_lut(tmp_path):
    p = tmp_path / "t.cube"
    n = 4
    lines = ["LUT_3D_SIZE 4"]
    for b in range(n):
        for g in range(n):
            for r in range(n):
                lines.append(f"{r/(n-1)} {g/(n-1)} {b/(n-1)}")
    p.write_text("\n".join(lines))
    cm = P.cmap_from_cube(str(p))
    assert np.allclose(cm(0.0)[:3], (0, 0, 0), atol=0.02)
    assert np.allclose(cm(1.0)[:3], (1, 1, 1), atol=0.02)
