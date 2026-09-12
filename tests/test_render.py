import inspect

import numpy as np
import pytest

from sonicart.palette import PRESETS, make_cmap
from sonicart.render import MODES, fig_to_pil, render

CMAP = make_cmap(PRESETS["Korrend"])
PARAMS = dict(thickness=1.0, gate=0.2, gamma=1.4, inner=0.15, n_mels=64,
              rotation=30, data_thickness=True, symmetry=3, turns=4.0,
              onset=0.3, spoke_len=0.4, source="mix", n_segments=0,
              gap_deg=2.0, mark_segments=True, label_ring=True)


@pytest.mark.parametrize("name", list(MODES))
def test_jeder_modus_zeichnet(an, name):
    img = fig_to_pil(render(an, name, CMAP, "#08060d", 220, PARAMS), 220)
    assert img.size == (220, 220) and img.mode == "RGB"
    assert np.asarray(img).std() > 1.0, "Bild darf nicht einfarbig sein"


@pytest.mark.parametrize("name", list(MODES))
def test_transparenz(an, name):
    img = fig_to_pil(render(an, name, CMAP, "#08060d", 200, PARAMS,
                            transparent=True), 200, transparent=True)
    a = np.asarray(img)[..., 3]
    assert img.mode == "RGBA"
    assert 0 < (a > 0).mean() < 1.0, "teils deckend, teils durchsichtig erwartet"


@pytest.mark.parametrize("name", list(MODES))
def test_unbekannte_regler_werden_verworfen(an, name):
    """Kein Modus darf an einem Regler scheitern, den er nicht kennt."""
    render(an, name, CMAP, "#08060d", 160, dict(PARAMS, voellig_unbekannt=42))


def test_reglerliste_stammt_aus_der_signatur():
    """Schuetzt gegen Regler im UI, die der Modus stillschweigend ignoriert."""
    for name, m in MODES.items():
        sig = set(inspect.signature(m.fn).parameters)
        bound = set(getattr(m.fn, "keywords", None) or ())
        assert m.params <= sig - bound
        assert "cols" not in m.knobs()


def test_kreis_wellenform_kennt_kein_n_mels():
    """War vorher ein Regler ohne Wirkung."""
    assert "n_mels" not in MODES["Kreis-Wellenform"].params
    assert "data_thickness" not in MODES["Kreis-Wellenform"].params


def test_symmetrie_wirkt(an):
    """Zaehligkeit aus der Taktart muss das Bild sichtbar veraendern."""
    a = np.asarray(fig_to_pil(render(an, "Rose roh", CMAP, "#08060d", 220,
                                     dict(PARAMS, symmetry=1)), 220))
    b = np.asarray(fig_to_pil(render(an, "Rose roh", CMAP, "#08060d", 220,
                                     dict(PARAMS, symmetry=4)), 220))
    assert np.abs(a.astype(float) - b).mean() > 2.0


def test_spirale_unterscheidet_verlaeufe(structured):
    """Der Kernzweck des Modus: eine andere Dramaturgie ergibt ein anderes Bild."""
    from sonicart.analysis import Analysis
    y = structured.y
    rueckwaerts = Analysis(y[::-1].copy(), structured.sr)
    a = np.asarray(fig_to_pil(render(structured, "Spirale", CMAP, "#000000",
                                     260, PARAMS), 260)).astype(float)
    b = np.asarray(fig_to_pil(render(rueckwaerts, "Spirale", CMAP, "#000000",
                                     260, PARAMS), 260)).astype(float)
    assert np.abs(a - b).mean() > 3.0

    r = np.asarray(fig_to_pil(render(structured, "Rose gespiegelt", CMAP,
                                     "#000000", 260, PARAMS), 260)).astype(float)
    rb = np.asarray(fig_to_pil(render(rueckwaerts, "Rose gespiegelt", CMAP,
                                      "#000000", 260, PARAMS), 260)).astype(float)
    assert np.abs(r - rb).mean() < np.abs(a - b).mean(), \
        "die Rose darf den Unterschied schwaecher zeigen als die Spirale"


def test_modus_metadaten_stimmen():
    assert not MODES["Rose gespiegelt"].time_aware
    assert MODES["Spirale"].time_aware and MODES["Segmente"].time_aware
    assert not MODES["Lissajous (Stereo)"].vector
    assert MODES["Lissajous (Stereo)"].stereo


def _radialsymmetrie(img):
    """Wie aehnlich ist das Bild seiner eigenen 90-Grad-Drehung?

    Eine zentrierte Scheibe bleibt sich dabei fast gleich; ein Raster oder
    waagerechte Schichten nicht. Damit laesst sich der Media-Player-Blick
    messen statt nur behaupten.
    """
    a = np.asarray(img.convert("L")).astype(float)
    return float(np.abs(a - np.rot90(a)).mean())


def test_neue_modi_sind_nicht_radial(an):
    scheibe = fig_to_pil(render(an, "Rose roh", CMAP, "#08060d", 300, PARAMS), 300)
    basis = _radialsymmetrie(scheibe)
    for name in ("Gitter", "Strata"):
        img = fig_to_pil(render(an, name, CMAP, "#08060d", 300,
                                dict(PARAMS, whiten_amount=0.85)), 300)
        assert _radialsymmetrie(img) > basis * 2, f"{name} sieht noch rund aus"


def test_gitter_hat_so_viele_felder_wie_takte(an):
    from matplotlib.patches import Rectangle
    fig = render(an, "Gitter", CMAP, "#08060d", 300,
                 dict(PARAMS, whiten_amount=0.85))
    felder = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
    assert len(felder) == len(an.bar_values(96, 0.85, "mix"))
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_gitter_nutzt_mehrere_tonwerte(an):
    """Fiel beim Bauen auf: mit Min-Max landeten fast alle Felder gleich."""
    img = fig_to_pil(render(an, "Gitter", CMAP, "#08060d", 300,
                            dict(PARAMS, whiten_amount=0.85, stufen=5)), 300)
    grau = np.asarray(img.convert("L"))
    haeufig = np.bincount(grau.ravel(), minlength=256)
    belegt = (haeufig > grau.size * 0.01).sum()
    assert belegt >= 3, f"nur {belegt} Tonwerte im Bild"


def test_strata_zeichnet_die_gewuenschte_zahl_schichten(an):
    for n in (5, 9):
        fig = render(an, "Strata", CMAP, "#08060d", 240,
                     dict(PARAMS, baender=n, whiten_amount=0.85))
        assert len(fig.axes[0].collections) == n
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_neue_modi_bleiben_vektorfaehig():
    assert MODES["Gitter"].vector and MODES["Strata"].vector
    assert MODES["Gitter"].time_aware and MODES["Strata"].time_aware
