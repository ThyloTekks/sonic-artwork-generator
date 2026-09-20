"""Footage: Gradient Map, Filtergraph und Export.

Die Wahrheit steckt hier nicht im Python, sondern in dem, was ffmpeg daraus
macht — die Hald-Anordnung und die Reihenfolge der blend-Eingaenge lassen
sich nur pruefen, indem man ffmpeg wirklich laufen laesst.
"""

import subprocess

import numpy as np
import pytest
from PIL import Image

from sonicart import ffmpeg, footage
from sonicart.export import SIZES
from sonicart.palette import PRESETS, oklab_L

STOPS = PRESETS["Korrend"]


# ----------------------------------------------------------------------
# Hilfsmittel
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def clut(tmp_path_factory):
    return footage.clut_file(STOPS, str(tmp_path_factory.mktemp("clut") / "c.png"))


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Vier Sekunden 640x360 mit Ton. 16:9, damit der Crop etwas zu tun hat."""
    p = tmp_path_factory.mktemp("clip") / "clip.mp4"
    ffmpeg.run(["-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=4",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(p)])
    return str(p)


@pytest.fixture(scope="module")
def silent_clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("clip") / "stumm.mp4"
    ffmpeg.run(["-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)])
    return str(p)


@pytest.fixture(scope="module")
def track(tmp_path_factory):
    p = tmp_path_factory.mktemp("audio") / "track.wav"
    ffmpeg.run(["-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=220:duration=6", str(p)])
    return str(p)


def frame_at(path, t=1.0, out=None):
    """Einen Frame aus einem Video als Array."""
    out = out or "/tmp/_f.png"
    ffmpeg.run(["-y", "-loglevel", "error", "-ss", str(t), "-i", path,
                "-frames:v", "1", out])
    return np.asarray(Image.open(out).convert("RGB")).astype(float)


def streams(path):
    return subprocess.run([ffmpeg.exe(), "-i", path],
                          capture_output=True).stderr.decode("utf-8", "replace")


# ----------------------------------------------------------------------
# Rampe und CLUT
# ----------------------------------------------------------------------
def test_rampe_laeuft_von_dunkel_nach_hell():
    r = footage.oklch_ramp(STOPS, 64)
    L = oklab_L(r / 255.0)
    assert L[0] < 0.2 and L[-1] > 0.7
    assert np.all(np.diff(L) > -0.02), "Rampe darf nicht zurueckspringen"


@pytest.mark.parametrize("name", list(PRESETS))
def test_rampe_endet_auf_den_aeusseren_stops(name):
    """Erste und letzte Rampenfarbe sind die Randstops der Palette."""
    from sonicart.palette import hex_to_rgb
    r = footage.oklch_ramp(PRESETS[name], 256)
    for got, want in ((r[0], PRESETS[name][0]), (r[-1], PRESETS[name][-1])):
        assert max(abs(int(a) - b) for a, b in zip(got, hex_to_rgb(want))) <= 2


def test_clut_hat_hald_masse():
    im = footage.hald_clut(STOPS)
    assert im.size == (512, 512) and im.mode == "RGB"


def test_clut_anordnung_passt_zu_ffmpeg(clut, tmp_path):
    """Der Kern: ffmpegs haldclut muss die Tabelle so lesen, wie sie gemeint ist.

    Bei falscher Achsenreihenfolge kaemen vertauschte Farben heraus, nicht
    bloss ungenaue — die Schranke von 6/255 faengt Quantisierung ab, nicht
    ein verdrehtes Layout.
    """
    cols = [(0, 0, 0), (64, 64, 64), (128, 128, 128), (255, 255, 255),
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (200, 120, 40)]
    src = np.zeros((40, len(cols) * 40, 3), np.uint8)
    for i, c in enumerate(cols):
        src[:, i * 40:(i + 1) * 40] = c
    sp = tmp_path / "src.png"
    Image.fromarray(src).save(sp)
    gp = tmp_path / "got.png"
    ffmpeg.run(["-y", "-loglevel", "error", "-i", str(sp), "-i", clut,
                "-filter_complex", "[0:v][1:v]haldclut", "-frames:v", "1", str(gp)])
    got = np.asarray(Image.open(gp).convert("RGB"))

    ramp = footage.oklch_ramp(STOPS)
    for i, c in enumerate(cols):
        exp = footage.map_luma(np.array([oklab_L(np.array(c) / 255.0)]), ramp)[0]
        assert np.abs(exp - got[20, i * 40 + 20]).max() <= 6, f"Farbe {c}"


def test_gleiche_helligkeit_wird_gleich_eingefaerbt(clut, tmp_path):
    """Gradient Map heisst: der Eingangsfarbton spielt keine Rolle mehr."""
    ramp = footage.oklch_ramp(STOPS)
    a = footage.map_luma(np.array([oklab_L(np.array([255, 0, 0]) / 255)]), ramp)[0]
    b = footage.map_luma(np.array([oklab_L(np.array([0, 180, 255]) / 255)]), ramp)[0]
    #  Rot und dieses Blau liegen unterschiedlich hell -> unterschiedliche Farbe
    assert np.abs(a - b).max() > 5
    #  Zwei Farben gleicher Helligkeit dagegen landen auf derselben Stelle
    lum = oklab_L(np.array([255, 0, 0]) / 255)
    same = footage.map_luma(np.array([lum, lum]), ramp)
    assert np.array_equal(same[0], same[1])


def test_ramp_strip_masse():
    im = footage.ramp_strip(STOPS, w=128, h=12)
    assert im.size == (128, 12)


# ----------------------------------------------------------------------
# Filtergraph
# ----------------------------------------------------------------------
def test_graph_ohne_blend_bei_voller_staerke():
    g = footage.graph(1080, 1920)
    assert "haldclut" in g and "blend" not in g and "split" not in g


def test_graph_blendet_eingefaerbt_ueber_original():
    """Erster blend-Eingang ist die obere Ebene, dort muss [g] stehen."""
    g = footage.graph(1080, 1080, strength=0.5)
    assert "[g][a]blend" in g, "sonst wirkt all_opacity auf das Original"
    assert "all_opacity=0.500" in g


def test_graph_schaltet_kontrast_und_korn_nur_bei_bedarf():
    assert "eq=contrast" not in footage.graph(512, 512, contrast=1.0)
    assert "eq=contrast=1.400" in footage.graph(512, 512, contrast=1.4)
    assert "noise" not in footage.graph(512, 512, grain=0.0)
    assert "noise=alls=20" in footage.graph(512, 512, grain=0.5)


@pytest.mark.parametrize("mode,has_track,erwartet", [
    ("Originalton", False, "0:a?"),
    ("Track-Audio", True, "2:a"),
    ("Track-Audio", False, "0:a?"),      # ohne Track auf Originalton zurueck
    ("stumm", False, "-an"),
])
def test_audio_argumente(mode, has_track, erwartet):
    assert erwartet in footage._audio_args(mode, has_track)


# ----------------------------------------------------------------------
# Vorschau
# ----------------------------------------------------------------------
@pytest.mark.parametrize("fmt", footage.FOOTAGE_SIZES)
def test_vorschau_trifft_zielmasse(clip, clut, fmt):
    png = footage.preview_frame(clip, clut, t=1.0, size=SIZES[fmt])
    assert Image.open(__import__("io").BytesIO(png)).size == SIZES[fmt]


def test_vorschau_beschneidet_statt_zu_verzerren(clip, clut):
    """16:9-Quelle auf 9:16 darf nicht gestaucht werden."""
    import io
    png = footage.preview_frame(clip, clut, t=1.0, size=(1080, 1920))
    im = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    assert im.shape[:2] == (1920, 1080)


def test_vorschau_sagt_den_export_voraus(clip, clut, tmp_path):
    """Sonst waere die Vorschau wertlos: sie soll das Rendern ersparen."""
    import io
    kw = dict(size=(1080, 1080), contrast=1.25, strength=0.7)
    prev = np.asarray(Image.open(io.BytesIO(
        footage.preview_frame(clip, clut, t=2.0, **kw))).convert("RGB")).astype(float)
    out = str(tmp_path / "v.mp4")
    footage.render_clip(clip, clut, out, audio_mode="stumm", **kw)
    assert np.abs(prev - frame_at(out, 2.0)).mean() < 6   # nur H.264-Verlust


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------
def test_staerke_null_laesst_das_original_stehen(clip, clut, tmp_path):
    roh = str(tmp_path / "roh.mp4")
    ffmpeg.run(["-y", "-loglevel", "error", "-i", clip, "-filter_complex",
                footage.graph(1080, 1080, strength=1.0).replace(
                    "[b][1:v]haldclut[v]", "[b]null[v]"),
                "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", roh])
    null = str(tmp_path / "null.mp4")
    footage.render_clip(clip, clut, null, size=(1080, 1080), strength=0.0,
                        audio_mode="stumm")
    abstand = np.abs(frame_at(roh, 1.0, "/tmp/_a.png")
                     - frame_at(null, 1.0, "/tmp/_b.png")).mean()
    assert abstand < 3, "strength=0 muss das unveraenderte Bild liefern"


def test_staerke_steigt_monoton(clip, clut, tmp_path):
    abstaende = []
    voll = str(tmp_path / "voll.mp4")
    footage.render_clip(clip, clut, voll, size=(512, 512), strength=1.0,
                        audio_mode="stumm")
    ziel = frame_at(voll, 1.0, "/tmp/_z.png")
    for s in (0.0, 0.5, 1.0):
        p = str(tmp_path / f"s{s}.mp4")
        footage.render_clip(clip, clut, p, size=(512, 512), strength=s,
                            audio_mode="stumm")
        abstaende.append(np.abs(frame_at(p, 1.0, "/tmp/_s.png") - ziel).mean())
    assert abstaende[0] > abstaende[1] > abstaende[2]


def test_export_ist_h264_yuv420p_mit_faststart(clip, clut, tmp_path):
    out = str(tmp_path / "v.mp4")
    footage.render_clip(clip, clut, out, size=(1080, 1080), audio_mode="stumm")
    info = streams(out)
    assert "h264" in info and "yuv420p" in info and "1080x1080" in info
    kopf = open(out, "rb").read(400_000)
    assert kopf.find(b"moov") < kopf.find(b"mdat"), "faststart fehlt"


def test_korn_sprengt_die_dateigroesse_nicht(clip, clut, tmp_path):
    """Ohne Deckel kodiert x264 das Rauschen originalgetreu (>1 GB je Minute)."""
    import os
    out = str(tmp_path / "korn.mp4")
    footage.render_clip(clip, clut, out, size=(1080, 1920), grain=1.0,
                        audio_mode="stumm")
    pro_minute = os.path.getsize(out) / 1e6 * 15          # Clip ist 4 s lang
    assert pro_minute < 200, f"{pro_minute:.0f} MB/min"


@pytest.mark.parametrize("mode,ton_erwartet", [
    ("Originalton", True), ("Track-Audio", True), ("stumm", False)])
def test_audiospur_je_modus(clip, clut, track, tmp_path, mode, ton_erwartet):
    out = str(tmp_path / f"{mode}.mp4")
    footage.render_clip(clip, clut, out, size=(512, 512), audio_mode=mode,
                        track=track if mode == "Track-Audio" else None)
    assert ("Audio:" in streams(out)) is ton_erwartet


def test_tonloser_clip_bricht_originalton_nicht(silent_clip, clut, tmp_path):
    """-map 0:a? muss fehlenden Ton verzeihen, statt ffmpeg abbrechen zu lassen."""
    out = str(tmp_path / "v.mp4")
    footage.render_clip(silent_clip, clut, out, size=(512, 512),
                        audio_mode="Originalton")
    assert "Audio:" not in streams(out)


def test_fehlermeldung_nennt_den_grund(clut, tmp_path):
    with pytest.raises(RuntimeError, match="No such file|Invalid|not found"):
        footage.render_clip("/gibt/es/nicht.mp4", clut,
                            str(tmp_path / "x.mp4"), audio_mode="stumm")
