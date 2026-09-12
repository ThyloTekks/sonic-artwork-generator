"""Gemeinsame Testdaten: synthetisches Audio mit bekannten Eigenschaften.

Echte Musik waere als Fixture unbrauchbar (Groesse, Lizenz, keine bekannte
Wahrheit). Die Signale hier haben eine nachpruefbare Tonart, ein bekanntes
Tempo und eine bekannte Abschnittsstruktur.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonicart.analysis import Analysis          # noqa: E402

SR = 22050
C_MAJOR = (261.63, 329.63, 392.00)              # C E G


def _tone(dur, freqs, noise=0.02, sr=SR):
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return sig * 0.5 + np.random.default_rng(0).normal(0, noise, len(t))


@pytest.fixture(scope="session")
def mono_y():
    """20 s C-Dur mit 120-bpm-Puls."""
    y = _tone(20, C_MAJOR)
    n = len(y)
    click = ((np.arange(n) % (SR // 2)) < 300).astype(float)
    rng = np.random.default_rng(1)
    return (y + rng.normal(0, 1, n) * click * 0.25).astype(np.float32)


@pytest.fixture(scope="session")
def an(mono_y):
    """Analyse eines Stereosignals (rechter Kanal verzoegert)."""
    st = np.vstack([mono_y, np.roll(mono_y, 220)])
    a = Analysis(mono_y, SR, st, "test.wav")
    a.mel(64)                                    # Spektrogramme vorziehen
    return a


@pytest.fixture(scope="session")
def structured():
    """Drei klar getrennte Abschnitte bei 30 s / 70 s."""
    y = np.concatenate([_tone(30, (110, 220), 0.01),
                        _tone(40, (220, 440, 880), 0.25),
                        _tone(25, (110,), 0.02)]).astype(np.float32)
    return Analysis(y, SR, None, "struct.wav")


@pytest.fixture(scope="session")
def wav_file(tmp_path_factory, mono_y):
    p = tmp_path_factory.mktemp("audio") / "test.wav"
    sf.write(p, np.vstack([mono_y, np.roll(mono_y, 220)]).T, SR)
    return str(p)
