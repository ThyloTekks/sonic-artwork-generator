import os

import imageio
import numpy as np
import pytest

from sonicart.artwork import Recipe
from sonicart.video import (SPOTIFY_CANVAS, VIDEO_MODES, animate, beat_pulse,
                            loop_length)


def test_loop_laenge_endet_auf_taktanfang(an):
    bars = an.bar_times()
    for ziel in (4.0, 5.5, 7.0):
        L = loop_length(an, ziel, (3.0, 8.0))
        assert np.min(np.abs(bars - L)) < 1e-6, "muss exakt ein Taktanfang sein"
        assert 3.0 <= L <= 8.0


def test_loop_laenge_ohne_beats():
    from sonicart.analysis import Analysis
    still = Analysis(np.zeros(22050 * 5, dtype=np.float32), 22050)
    assert 0.5 <= loop_length(still, 4.0) <= 5.0


def test_beat_puls_vor_dem_ersten_schlag_null(an):
    t0 = an.beat_times[0]
    vor = beat_pulse(an, np.linspace(0, max(t0 - 0.05, 0.01), 5))
    assert np.all(vor == 0.0), "vor dem ersten Schlag darf nichts pulsieren"
    auf = beat_pulse(an, np.array([an.beat_times[2]]))
    assert auf[0] == pytest.approx(1.0, abs=1e-6)
    nach = beat_pulse(an, np.array([an.beat_times[2] + 0.2]))
    assert 0 < nach[0] < 1.0, "danach muss der Puls abfallen"


def _frames(path):
    r = imageio.get_reader(path)
    fr = np.stack([f for f in r]).astype(float)
    r.close()
    return fr


@pytest.mark.parametrize("mode", VIDEO_MODES)
def test_jeder_videomodus_rendert(an, tmp_path, mode):
    out = str(tmp_path / f"{mode[:6]}.mp4")
    info = animate(an, Recipe(size=320), out, mode=mode, duration=3.0, fps=8,
                   with_audio=False, effects=False, loop_safe=True)
    assert os.path.getsize(out) > 1000
    assert info["frames"] == pytest.approx(info["duration"] * 8, abs=1)
    fr = _frames(out)
    assert fr.std() > 1.0, "Video darf nicht standbild-leer sein"


def test_loop_sicherung_schliesst_die_naht(an, tmp_path):
    """Der Sprung am Schleifenpunkt muss unter einem normalen Bildabstand liegen."""
    def naht(**kw):
        out = str(tmp_path / f"n{abs(hash(tuple(sorted(kw.items()))))}.mp4")
        animate(an, Recipe(size=300), out, mode="Live-Kreisspektrum",
                duration=4.0, fps=10, rotate_turns=0.25, with_audio=False,
                effects=False, **kw)
        fr = _frames(out)
        sprung = np.abs(fr[-1] - fr[0]).mean()
        schritt = np.mean([np.abs(fr[i + 1] - fr[i]).mean()
                           for i in range(len(fr) - 1)])
        return sprung / max(schritt, 1e-9)

    assert naht(loop_safe=False) > 2.0, "ungesichert muss die Naht auffallen"
    assert naht(loop_safe=True, loop_blend=0.2) < 1.0, "gesichert darf sie es nicht"


def test_ganze_umdrehungen_bei_loop(an, tmp_path):
    info = animate(an, Recipe(size=220), str(tmp_path / "t.mp4"),
                   mode="Rose roh", duration=4.0, fps=8, rotate_turns=0.3,
                   with_audio=False, effects=False, loop_safe=True)
    assert info["turns"] == int(info["turns"]) and info["turns"] >= 1


def test_beat_puls_veraendert_das_bild(an, tmp_path):
    """Staerkerer Puls muss staerker ausschlagen.

    Gemessen wird der Abstand zum pulsfreien Lauf; absolute Schwellen taugen
    hier nicht, weil die Form nur wenige Prozent der 9:16-Flaeche einnimmt.
    """
    def render(pulse):
        out = str(tmp_path / f"p{pulse}.mp4")
        animate(an, Recipe(size=260), out, mode="Rose roh", duration=3.0, fps=10,
                rotate_turns=0.0, with_audio=False, effects=False, pulse=pulse,
                loop_safe=False)
        return _frames(out)

    ohne = render(0.0)
    abstand = []
    for p in (0.3, 0.6, 0.9):
        mit = render(p)
        n = min(len(ohne), len(mit))
        abstand.append(np.abs(ohne[:n] - mit[:n]).mean())
    assert abstand[0] > 0.02, "ohne Wirkung waere der Regler sinnlos"
    assert abstand == sorted(abstand), f"nicht monoton: {abstand}"


def test_canvas_spanne_wird_gemeldet(an, tmp_path):
    info = animate(an, Recipe(size=200), str(tmp_path / "c.mp4"),
                   mode="Rose roh", duration=5.0, fps=8, with_audio=False,
                   effects=False, loop_safe=True)
    assert info["canvas_ok"] == (SPOTIFY_CANVAS[0] <= info["duration"] <= SPOTIFY_CANVAS[1])
