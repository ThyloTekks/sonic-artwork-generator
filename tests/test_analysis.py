import numpy as np
import pytest

from sonicart.analysis import Analysis, load, norm01, punch


def test_tonart_wird_erkannt(an):
    k = an.key
    assert k["label"] == "C-dur", f"erwartet C-dur, bekam {k['label']}"
    # Ein blanker Dreiklang ohne Bass ist echt mehrdeutig — die Sicherheit
    # darf das ruhig zeigen, solange die Tonika stimmt.
    assert 0.1 < k["confidence"] <= 1.0


@pytest.mark.parametrize("freqs,bass,erwartet", [
    ((261.63, 329.63, 392.00), 130.81, "C-dur"),
    ((220.00, 261.63, 329.63), 110.00, "A-moll"),
    ((349.23, 440.00, 523.25), 174.61, "F-dur"),
    ((392.00, 493.88, 587.33), 196.00, "G-dur"),
])
def test_tonart_ueber_mehrere_akkorde(freqs, bass, erwartet):
    """Der Basston loest die Mehrdeutigkeit blanker Dreiklaenge auf."""
    sr = 22050
    t = np.linspace(0, 15, 15 * sr, endpoint=False)
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs) * 0.5
    sig = sig + np.sin(2 * np.pi * bass * t) * 0.35
    sig = sig + np.random.default_rng(0).normal(0, 0.02, len(t))
    k = Analysis(sig.astype(np.float32), sr).key
    assert k["label"] == erwartet, f"erwartet {erwartet}, bekam {k['label']}"
    assert k["confidence"] > 0.2


def test_bassbetonung_verbessert_die_bestimmung(an):
    """Gegenprobe: ohne Bassgewicht kippt der Dreiklang auf die Mollparallele."""
    ohne = an.chroma.mean(axis=1)
    ohne = (ohne - ohne.mean()) / (ohne.std() + 1e-9)
    from sonicart.analysis import _KS_MAJOR, _KS_MINOR, PITCH_NAMES
    scores = []
    for i in range(12):
        rot = np.roll(ohne, -i)
        for name, ref in (("dur", _KS_MAJOR), ("moll", _KS_MINOR)):
            r = (ref - ref.mean()) / (ref.std() + 1e-9)
            scores.append((float(np.dot(rot, r)), f"{PITCH_NAMES[i]}-{name}"))
    scores.sort(reverse=True)
    assert scores[0][1] != "C-dur", "ohne Bass waere es nicht eindeutig"
    assert an.key["label"] == "C-dur", "mit Bass schon"


def test_tempo_plausibel(an):
    assert 100 < an.tempo < 145, f"120 bpm erwartet, gemessen {an.tempo}"


def test_stereo_bleibt_erhalten(an):
    assert an.is_stereo
    assert an.stereo_width > 0
    assert an.lissajous.shape[0] == 2


def test_mono_bekommt_ersatzachse(mono_y):
    a = Analysis(mono_y, 22050)
    assert not a.is_stereo
    assert a.stereo_width == 0.0
    lr = a.lissajous
    assert lr.shape[0] == 2
    assert not np.allclose(lr[0], lr[1]), "Hilbert-Phase muss echte 2. Achse liefern"


def test_segmente_treffen_die_uebergaenge(structured):
    t = structured.segment_times()
    assert t[0] == pytest.approx(0, abs=0.5)
    assert any(abs(x - 30) < 4 for x in t), f"Uebergang bei 30 s fehlt: {t}"
    assert any(abs(x - 70) < 4 for x in t), f"Uebergang bei 70 s fehlt: {t}"


def test_keine_winzigen_abschnitte(an, structured):
    for a in (an, structured):
        t = a.segment_times()
        assert np.all(np.diff(t) >= 3.5), f"Abschnitt zu kurz: {np.diff(t)}"


def test_spektrum_ist_zeitblind(an):
    """Dokumentiert die bekannte Grenze der Rose-Modi."""
    y = an.y
    getauscht = np.concatenate([y[len(y) // 2:], y[: len(y) // 2]])
    a = Analysis(y, an.sr).spectrum(64)
    b = Analysis(getauscht, an.sr).spectrum(64)
    assert np.corrcoef(a, b)[0, 1] > 0.99, (
        "Das gemittelte Spektrum ignoriert die Zeit — deshalb gibt es die "
        "zeitbewussten Modi Spirale und Segmente.")


def test_zeitbewusster_modus_sieht_den_unterschied(structured):
    """Gegenprobe: das Spektrogramm unterscheidet, was das Mittel verwischt."""
    y = structured.y
    getauscht = np.concatenate([y[len(y) // 2:], y[: len(y) // 2]])
    a = Analysis(y, structured.sr).mel_norm(64)
    b = Analysis(getauscht, structured.sr).mel_norm(64)
    n = min(a.shape[1], b.shape[1])
    assert np.corrcoef(a[:, :n].ravel(), b[:, :n].ravel())[0, 1] < 0.95


def test_cache_rechnet_nicht_zweimal(an):
    first = an.mel(96, "harmonic")
    assert an.mel(96, "harmonic") is first, "mel muss aus dem Cache kommen"
    assert an._hpss_mag is an._hpss_mag


def test_slice_schneidet_auch_stereo(an):
    s = an.slice(2.0, 8.0)
    assert s.duration == pytest.approx(6.0, abs=0.05)
    assert s.is_stereo and s.y_stereo.shape[1] == len(s.y)
    assert an.slice(0.0, None) is an, "leerer Schnitt darf nicht kopieren"


def test_taktart_ist_ganzzahlig(an, structured):
    for a in (an, structured):
        assert a.meter in (2, 3, 4, 5, 6, 7)


def test_laden_aus_datei(wav_file):
    a = load(wav_file)
    assert a.is_stereo and a.duration == pytest.approx(20, abs=0.1)
    assert a.name.endswith("test.wav")


def test_hilfsfunktionen():
    x = np.array([0.0, 0.5, 1.0])
    assert norm01(x).max() == pytest.approx(1.0)
    assert punch(x, 0.5, 1.0)[0] == 0.0
    assert punch(x, 0.0, 1.0)[2] == pytest.approx(1.0)


def _mit_pegel(art, rms, sr=22050, dauer=12):
    t = np.linspace(0, dauer, int(dauer * sr), endpoint=False)
    if art == "sinus":
        s = np.sin(2 * np.pi * 220 * t)
    elif art == "saege":
        s = 2 * (t * 110 % 1) - 1
    elif art == "rauschen":
        s = np.random.default_rng(0).normal(0, 1, len(t))
    else:
        raise ValueError(art)
    s = s / (np.sqrt((s ** 2).mean()) + 1e-9) * rms
    return Analysis(np.clip(s, -1, 1).astype(np.float32), sr)


def test_lautheit_saettigt_nicht_bei_normalen_pegeln():
    """Frueher war rms * 4 gerechnet — ab -12 dBFS stand alles auf 1,0."""
    werte = [_mit_pegel("sinus", r).features["energy"] for r in (0.05, 0.15, 0.30)]
    assert werte == sorted(werte), werte
    assert all(w < 0.999 for w in werte), f"kein Anschlag erwartet: {werte}"
    assert werte[-1] - werte[0] > 0.3, f"zu wenig Spanne: {werte}"


def test_features_haengen_nicht_am_pegel_wo_sie_es_nicht_sollen():
    """Klangfarbe darf sich nicht aendern, nur weil lauter ausgesteuert wurde."""
    leise = _mit_pegel("saege", 0.05).features
    laut = _mit_pegel("saege", 0.30).features
    for k in ("roughness", "brightness", "spread"):
        assert abs(leise[k] - laut[k]) < 0.05, f"{k} folgt dem Pegel: {leise[k]} vs {laut[k]}"


def test_rauschen_ist_rauer_als_ein_sinus():
    assert (_mit_pegel("rauschen", 0.15).features["roughness"]
            > _mit_pegel("saege", 0.15).features["roughness"]
            > _mit_pegel("sinus", 0.15).features["roughness"])


def test_helligkeit_folgt_dem_spektrum():
    tief = _mit_pegel("sinus", 0.15).features["brightness"]
    reich = _mit_pegel("saege", 0.15).features["brightness"]
    assert tief < 0.2 < reich


def test_alle_features_im_einheitsintervall(an, structured):
    for a in (an, structured):
        for k, v in a.features.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v}"


def test_bandnormierung_gleicht_frequenzbaender_an(an):
    """Ohne sie frisst der Bass den Tonwertumfang."""
    from sonicart.analysis import whiten
    M = an.mel_norm(96)
    drittel = lambda X: [X[:32].mean(), X[32:64].mean(), X[64:].mean()]
    roh = drittel(M)
    gleich = drittel(whiten(M, 0.85))
    assert max(roh) / (min(roh) + 1e-9) > 1.5, "Testsignal muss schief sein"
    assert max(gleich) / (min(gleich) + 1e-9) < 1.2
    assert np.allclose(whiten(M, 0.0), M), "staerke 0 darf nichts tun"


def test_rangnormierung_spreizt_enge_werte():
    from sonicart.analysis import rank01
    eng = np.array([0.50, 0.51, 0.52, 0.53, 0.99])
    r = rank01(eng)
    assert np.allclose(r, [0, 0.25, 0.5, 0.75, 1.0])
    minmax = (eng - eng.min()) / np.ptp(eng)
    assert r[:4].std() > minmax[:4].std() * 3, "genau dafuer ist sie da"
    assert rank01(np.array([7.0])).shape == (1,)


def test_hoehenanhebung_fuellt_die_rose_ohne_sie_einzuebnen(an):
    """Die Rose zeigt das Zeitmittel — Bandnormierung wuerde es loeschen.

    Deshalb hebt sie die Hoehen an, statt bandweise zu normieren: die
    Belegung steigt, die Streuung bleibt.
    """
    ohne = an.spectrum(96, 0.12, 1.3, "mix", tilt=0.0)
    mit = an.spectrum(96, 0.12, 1.3, "mix", tilt=0.6)
    assert (mit > 0.05).mean() > (ohne > 0.05).mean() * 2
    assert mit.std() > ohne.std() * 0.8, "darf nicht flach werden"

    from sonicart.analysis import whiten
    flach = whiten(an.mel_norm(96), 0.85).mean(axis=1)
    assert flach.std() < ohne.std(), "Bandnormierung ebnet das Mittel ein"


def test_taktwerte_nutzen_den_vollen_umfang(an):
    v = an.bar_values()
    assert len(v) >= 4
    assert v.min() == pytest.approx(0.0) and v.max() == pytest.approx(1.0)


def test_taktwerte_ohne_erkannte_beats():
    still = Analysis(np.zeros(22050 * 6, dtype=np.float32) + 1e-5, 22050)
    v = still.bar_values()
    assert len(v) >= 4, "muss auf eine feste Unterteilung zurueckfallen"


def test_bandkurven_form_und_bereich(an):
    K = an.band_curves(7)
    assert K.shape[0] == 7
    assert K.min() >= 0.0 and K.max() <= 1.0
