"""
Sonic Artwork — Audio -> Cover  (Project Tree)
==============================================
Drei Modi, drei Kernregler, editierbare Palette (Presets / Farbwaehler / .cube-LUT).

Start:  streamlit run sonic_artwork.py
Deps:   pip install streamlit librosa soundfile matplotlib numpy pillow
"""

import io
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

# ----------------------------------------------------------------------
# Profile: nur Defaults. Palette ist im UI frei aenderbar.
# ----------------------------------------------------------------------
PRESETS = {
    "Korrend":    ["#08060d", "#3a1d6e", "#7b2ff7", "#c9a0ff"],
    "Type Drift": ["#1a1712", "#5c4a32", "#b08d57", "#e8d8b8"],
    "Seek":       ["#0d0606", "#7a1420", "#e23b2e", "#ffb37a"],
    "Monochrom":  ["#000000", "#666666", "#ffffff"],
}
PROFILE_BG = {"Korrend": "#08060d", "Type Drift": "#1a1712",
              "Seek": "#0d0606", "Monochrom": "#000000"}


# ----------------------------------------------------------------------
# Palette / LUT
# ----------------------------------------------------------------------
def make_cmap(hex_colors):
    return LinearSegmentedColormap.from_list("c", list(hex_colors), N=512)


# ----------------------------------------------------------------------
# Farbtheorie: OKLCH-Palettengenerator
# ----------------------------------------------------------------------
def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb01_to_hex(rgb):
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def _srgb_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin_srgb(c):
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def hex_to_oklch(hx):
    r, g, b = (_srgb_lin(v) for v in _hex_to_rgb01(hx))
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt([l, m, s])
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, float(np.hypot(a, bb)), float(np.arctan2(bb, a))


def oklch_to_hex(L, C, h):
    a, b = C * np.cos(h), C * np.sin(h)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return _rgb01_to_hex((_lin_srgb(r), _lin_srgb(g), _lin_srgb(bl)))


def generate_palette(anchor_hex, n=4, mode="Monochrom-Ramp", template=None):
    """Erzeugt aus EINER Ankerfarbe eine stimmige Palette (dunkel -> hell)."""
    L0, C0, h0 = hex_to_oklch(anchor_hex)
    C0 = max(C0, 0.06)
    deg = np.deg2rad

    if mode == "Profil-Struktur" and template:
        tl = [hex_to_oklch(c) for c in template]
        h_ref = tl[-1][2]
        return [oklch_to_hex(L, C, h0 + (h - h_ref)) for (L, C, h) in tl]

    Ls = np.linspace(0.12, 0.95, n)
    chroma_env = 0.55 + 0.45 * (1 - np.abs(2 * np.linspace(0, 1, n) - 1))  # peak Mitte
    stops = []
    for i, L in enumerate(Ls):
        C = C0 * chroma_env[i]
        if mode == "Monochrom-Ramp":
            h = h0
        elif mode == "Analog":
            h = h0 + deg(35) * (i / (n - 1) - 0.5) * 2
        elif mode == "Komplementaer-Akzent":
            h = h0 if i < n - 1 else h0 + np.pi
            if i == n - 1:
                C = C0
        elif mode == "Triadisch":
            h = [h0, h0 + 2 * np.pi / 3, h0 - 2 * np.pi / 3][i % 3]
        else:
            h = h0
        stops.append(oklch_to_hex(L, C, h))
    return stops


def cmap_from_cube(path, n=9):
    size = None; dim = 3; data = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("#", "TITLE", "DOMAIN_")):
                continue
            if s.startswith("LUT_3D_SIZE"): size = int(s.split()[-1]); dim = 3; continue
            if s.startswith("LUT_1D_SIZE"): size = int(s.split()[-1]); dim = 1; continue
            if s.startswith("LUT_"): continue
            p = s.split()
            if len(p) == 3:
                try: data.append([float(v) for v in p])
                except ValueError: pass
    data = np.array(data)
    if size is None or len(data) == 0:
        raise ValueError("Kein gueltiges .cube LUT")
    if dim == 1:
        ramp = data[np.linspace(0, size - 1, n).round().astype(int)]
    else:
        ks = np.linspace(0, size - 1, n).round().astype(int)
        ramp = data[ks * (1 + size + size * size)]
    return LinearSegmentedColormap.from_list("lut", np.clip(ramp, 0, 1), N=512)


# ----------------------------------------------------------------------
# Analyse-Helfer
# ----------------------------------------------------------------------
def load_audio(f, sr=22050, start=0.0, end=None):
    if hasattr(f, "seek"):
        f.seek(0)                                   # UploadedFile mehrfach lesbar
    y, sr = librosa.load(f, sr=sr, mono=True)
    a = int(start * sr); b = int(end * sr) if end else len(y)
    return y[a:b], sr


def mel_db(y, sr, n_mels):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=sr / 2)
    return librosa.power_to_db(S, ref=1.0)


def norm01(x):
    lo, hi = np.percentile(x, 2), x.max()
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def punch(x, gate, gamma):
    return np.clip((x - gate) / (1 - gate + 1e-9), 0, 1) ** gamma


def _polar_fig(size_px, bg, dpi=100, transparent=False, rotation=0.0):
    inch = size_px / dpi
    fig = plt.figure(figsize=(inch, inch), dpi=dpi)
    face = "none" if transparent else bg
    fig.patch.set_facecolor(face)
    if transparent:
        fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1], projection="polar")
    ax.set_facecolor(face); ax.axis("off")
    ax.set_theta_offset(np.deg2rad(rotation))     # 0 = Osten; dreht die ganze Scheibe
    ax.set_ylim(0, 1.2)
    return fig, ax


def cmap_to_alpha(cmap, knee=0.12):
    """Colormap mit Alpha-Verlauf: leise -> durchsichtig, laut -> deckend."""
    x = np.linspace(0, 1, 256)
    cols = cmap(x)
    cols[:, 3] = np.clip((x - knee) / (1 - knee + 1e-9), 0, 1) ** 0.8
    return ListedColormap(cols)


# ----------------------------------------------------------------------
# Modi.  Gemeinsame Regler:  thickness, gate, gamma, inner
# ----------------------------------------------------------------------
def render_rose(y, sr, cmap, bg, size_px=3000, mirror=True,
                thickness=1.0, gate=0.15, gamma=1.4, inner=0.15,
                n_mels=110, data_thickness=True, transparent=False, rotation=0.0):
    a = norm01(mel_db(y, sr, n_mels)).mean(axis=1)
    a = a / (a.max() + 1e-9)
    a = punch(a, gate, gamma)
    N = len(a)
    fig, ax = _polar_fig(size_px, bg, transparent=transparent, rotation=rotation)
    if mirror:
        base_w = np.pi / N
        w = base_w * thickness * (0.4 + 1.2 * a if data_thickness else 1.0)
        right = np.pi / 2 - np.linspace(0, np.pi, N)
        left = np.pi / 2 + np.linspace(0, np.pi, N)
        ax.bar(right, a, width=w, bottom=inner, color=cmap(a), lw=0)
        ax.bar(left, a, width=w, bottom=inner, color=cmap(a), lw=0)
    else:
        base_w = 2 * np.pi / N
        w = base_w * thickness * (0.4 + 1.2 * a if data_thickness else 1.0)
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        ax.bar(ang, a, width=w, bottom=inner, color=cmap(a), lw=0)
    return fig


def render_hpss_time(y, sr, cmap, bg, size_px=3000,
                     thickness=1.0, gate=0.5, gamma=1.7, inner=0.15,
                     onset=0.25, n_mels=120, transparent=False, rotation=0.0,
                     spoke_len=0.35):
    yh, yp = librosa.effects.hpss(y)
    Sh = punch(norm01(mel_db(yh, sr, n_mels)), gate, gamma)
    Sp = norm01(mel_db(yp, sr, n_mels))
    fig, ax = _polar_fig(size_px, bg, transparent=transparent, rotation=rotation)
    nf, nt = Sh.shape
    th = np.linspace(0, 2 * np.pi, nt + 1); r = np.linspace(inner, 1, nf + 1)
    base_cmap = cmap_to_alpha(cmap) if transparent else cmap
    ax.pcolormesh(th, r, Sh, cmap=base_cmap, shading="flat")
    # Perkussiv-Akzent kraeftiger (niedrigere Schwelle -> mehr sichtbar)
    acc = LinearSegmentedColormap.from_list(
        "a", [(0.95, 0.9, 1, 0), (1.0, 0.85, 1, 1)], N=256)
    ax.pcolormesh(th, r, np.ma.masked_less(Sp, 0.45), cmap=acc, shading="flat")
    if onset > 0:
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        om = oenv.max() + 1e-9
        outer = 1.03
        for f in librosa.onset.onset_detect(onset_envelope=oenv, sr=sr):
            s = oenv[f] / om
            if s < 0.30:                               # nur markante Onsets
                continue
            L = spoke_len * (0.4 + 0.6 * s)            # Laenge ~ Staerke
            ang = 2 * np.pi * f / len(oenv)
            ax.plot([ang, ang], [outer - L, outer], color="#f4ecff",
                    lw=(1.8 + 2.6 * s) * thickness, alpha=onset * (0.55 + 0.45 * s),
                    solid_capstyle="round")
    return fig


def render_wave_ring(y, sr, cmap, bg, size_px=3000, thickness=1.0, gate=0.15,
                     gamma=1.4, inner=0.15, n_mels=110, data_thickness=True,
                     transparent=False, rotation=0.0, cols=1400):
    hop = max(1, len(y) // cols)
    env = np.array([np.abs(y[i * hop:(i + 1) * hop]).max()
                    if y[i * hop:(i + 1) * hop].size else 0.0 for i in range(cols)])
    env = env / (env.max() + 1e-9)
    env = env ** gamma                              # Kontrast der Welle
    theta = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    base = inner + 0.45
    amp = env * 0.24 * thickness
    th = np.append(theta, theta[0])
    ro = np.append(base + amp, base + amp[0]); ri = np.append(base - amp, base - amp[0])
    fig, ax = _polar_fig(size_px, bg, transparent=transparent, rotation=rotation)
    ax.fill_between(th, ri, ro, color=cmap(0.55), alpha=0.5, lw=0)
    for pts in (np.column_stack([th, ro]), np.column_stack([th, ri])):
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        ax.add_collection(LineCollection(segs, colors=cmap(th[:-1] / (2 * np.pi)),
                                         linewidths=2.2 * thickness))
    ax.set_ylim(0, 1.1)
    return fig


MODES = {
    "Rose gespiegelt": lambda **k: render_rose(mirror=True, **k),
    "Rose roh":        lambda **k: render_rose(mirror=False, **k),
    "HPSS-Zeit":       render_hpss_time,
    "Kreis-Wellenform": render_wave_ring,
}
DEFAULT_GATE = {"Rose gespiegelt": 0.15, "Rose roh": 0.15,
                "HPSS-Zeit": 0.5, "Kreis-Wellenform": 0.15}


# ----------------------------------------------------------------------
# Compositing
# ----------------------------------------------------------------------
def fig_to_pil(fig, size_px, transparent=False):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=transparent,
                facecolor="none" if transparent else fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    mode = "RGBA" if transparent else "RGB"
    return Image.open(buf).convert(mode).resize((size_px, size_px), Image.LANCZOS)


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _render_mode_pil(y, sr, mode, cmap, bg, size_px, params, transparent=False):
    p = dict(params); p["transparent"] = transparent
    fig = MODES[mode](y=y, sr=sr, cmap=cmap, bg=bg, size_px=size_px, **p)
    return fig_to_pil(fig, size_px, transparent=transparent)


def _mode_params(mode, base_params):
    p = dict(base_params)
    if mode == "HPSS-Zeit":
        p.pop("data_thickness", None)
        p.setdefault("onset", 0.2)
        p["gate"] = max(p.get("gate", 0.5), 0.4)   # HPSS braucht hoeheres Gate
    else:
        p.pop("onset", None)
        p.setdefault("data_thickness", True)
    return p


def mix_modes(y, sr, weights, cmap, bg, size_px, base_params, transparent=False):
    """Lighten-Komposit mehrerer Modi. weights: {mode: 0..1}."""
    active = [(m, w) for m, w in weights.items() if w > 0]
    if transparent:
        out_rgb = np.zeros((size_px, size_px, 3))
        out_a = np.zeros((size_px, size_px, 1))
        for mode, w in active:
            img = _render_mode_pil(y, sr, mode, cmap, bg, size_px,
                                   _mode_params(mode, base_params), transparent=True)
            arr = np.asarray(img).astype(float) / 255.0
            rgb, a = arr[..., :3], arr[..., 3:] * w
            out_rgb = np.maximum(out_rgb, rgb * a)     # praemultipliziertes Lighten
            out_a = np.maximum(out_a, a)
        rgb = np.where(out_a > 1e-6, out_rgb / np.clip(out_a, 1e-6, 1), 0)
        rgba = np.concatenate([rgb, out_a], axis=2)
        return Image.fromarray(np.clip(rgba * 255, 0, 255).astype("uint8"), "RGBA")

    bg_arr = np.array(_hex_rgb(bg)) / 255.0
    out = None
    for mode, w in active:
        img = _render_mode_pil(y, sr, mode, cmap, bg, size_px,
                               _mode_params(mode, base_params))
        la = np.asarray(img).astype(float) / 255.0
        lw = bg_arr + w * (la - bg_arr)
        out = lw if out is None else np.maximum(out, lw)
    if out is None:
        out = np.tile(bg_arr, (size_px, size_px, 1))
    return Image.fromarray(np.clip(out * 255, 0, 255).astype("uint8"))


def add_signet(img, path, scale=0.16, margin=0.06):
    sig = Image.open(path).convert("RGBA")
    w = int(img.width * scale)
    sig = sig.resize((w, int(w * sig.height / sig.width)), Image.LANCZOS)
    m = int(img.width * margin)
    was_rgba = img.mode == "RGBA"
    base = img.convert("RGBA")
    base.alpha_composite(sig, (img.width - sig.width - m, img.height - sig.height - m))
    return base if was_rgba else base.convert("RGB")


def add_title(img, text, color, font_path=None, scale=0.05, margin=0.07):
    d = ImageDraw.Draw(img); fs = int(img.width * scale)
    try:
        font = ImageFont.truetype(font_path, fs) if font_path else ImageFont.load_default(fs)
    except Exception:
        font = ImageFont.load_default(fs)
    m = int(img.width * margin)
    bb = d.textbbox((0, 0), text, font=font)
    d.text((m, img.height - (bb[3] - bb[1]) - m - bb[1]), text, fill=color, font=font)
    return img


# ----------------------------------------------------------------------
# Audio-reaktive Effekte (alpha-bewusst)
# ----------------------------------------------------------------------
def features(y, sr):
    """Timbre-Features -> 0..1. Skalierung heuristisch, ggf. kalibrieren."""
    flat = librosa.feature.spectral_flatness(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    cen = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    return dict(
        roughness=float(np.clip(flat * 6 + zcr * 2, 0, 1)),    # Verzerrung -> Grain
        energy=float(np.clip(rms * 4, 0, 1)),                  # Lautheit -> Bloom
        brightness=float(np.clip(cen / (sr / 2) * 3, 0, 1)),
        spread=float(np.clip(bw / (sr / 2) * 3, 0, 1)),        # Breite -> Aberration
    )


def _split(img):
    arr = np.asarray(img).astype(float)
    return (arr[..., :3], arr[..., 3:4]) if arr.shape[2] == 4 else (arr, None)


def _merge(rgb, a):
    rgb = np.clip(rgb, 0, 255)
    if a is None:
        return Image.fromarray(rgb.astype("uint8"), "RGB")
    return Image.fromarray(np.concatenate([rgb, np.clip(a, 0, 255)], 2).astype("uint8"), "RGBA")


def add_grain(img, amount, seed=None, cell_ref=2.2):
    if amount <= 0:
        return img
    rgb, a = _split(img)
    H, W = rgb.shape[:2]
    scale = W / 1000.0
    cell = max(1, int(round(cell_ref * scale)))          # Korngroesse ~ konstant relativ
    rng = np.random.default_rng(seed)
    sh, sw = -(-H // cell), -(-W // cell)                # Aufrunden -> deckt Bild ab
    small = rng.standard_normal((sh, sw))
    noise = np.kron(small, np.ones((cell, cell)))[:H, :W, None] * 255 * amount * 0.42
    lum = rgb.mean(2, keepdims=True) / 255.0
    mask = 0.15 + 0.85 * lum
    if a is not None:
        mask = mask * (a / 255.0)                        # kein Grain auf Transparenz
    return _merge(rgb + noise * mask, a)


def add_bloom(img, amount):
    if amount <= 0:
        return img
    rgb, a = _split(img)
    lum = rgb.mean(2)
    thr = np.percentile(lum, 85)
    m = np.clip((lum - thr) / (255 - thr + 1e-9), 0, 1)
    bright = (rgb * m[..., None]).astype("uint8")
    rad = (6 + 22 * amount) * (rgb.shape[1] / 1000.0)      # aufloesungsrelativ
    blur = np.asarray(Image.fromarray(bright).filter(
        ImageFilter.GaussianBlur(radius=rad))).astype(float)
    rgb = 255 - (255 - rgb) * (255 - blur * amount) / 255.0     # Screen
    if a is not None:
        amask = np.asarray(Image.fromarray((m * 255).astype("uint8")).filter(
            ImageFilter.GaussianBlur(radius=rad))).astype(float)[..., None]
        a = np.maximum(a, amask * amount)                        # Halo erweitert Alpha
    return _merge(rgb, a)


def chroma_ab(img, px):
    if px <= 0:
        return img
    rgb, a = _split(img)
    r = np.roll(rgb[:, :, 0], px, 1); b = np.roll(rgb[:, :, 2], -px, 1)
    return _merge(np.stack([r, rgb[:, :, 1], b], 2), a)


def add_vignette(img, amount):
    if amount <= 0:
        return img
    rgb, a = _split(img); H, W = rgb.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    v = 1 - amount * np.clip((r - 0.4) / 0.6, 0, 1) ** 2
    return _merge(rgb * v[..., None], a)


def posterize(img, levels):
    if levels < 2:
        return img
    rgb, a = _split(img)
    q = np.round(rgb / 255 * (levels - 1)) / (levels - 1) * 255
    return _merge(q, a)


def add_halftone(img, cell, bg):
    if cell < 2:
        return img
    rgb, a = _split(img); H, W = rgb.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    dx = (xx % cell) / cell - 0.5; dy = (yy % cell) / cell - 0.5
    d = np.sqrt(dx ** 2 + dy ** 2) / 0.707
    dot = (rgb.mean(2) / 255.0) > d           # heller -> groesserer Punkt
    m = dot[..., None]
    out = np.where(m, rgb, np.array(_hex_rgb(bg)))
    if a is not None:
        a = np.where(m, a, 0)
    return _merge(out, a)


def add_streaks(img, amount):
    if amount <= 0:
        return img
    from scipy.ndimage import uniform_filter1d
    rgb, a = _split(img); W = rgb.shape[1]
    lum = rgb.mean(2)
    thr = np.percentile(lum, 88)
    mask = np.clip((lum - thr) / (255 - thr + 1e-9), 0, 1)
    bright = rgb * mask[..., None]
    k = int(W * 0.16 * amount) + 3
    streak = uniform_filter1d(bright, size=k, axis=1, mode="constant")
    out = 255 - (255 - rgb) * (255 - streak * 1.6) / 255      # Screen
    if a is not None:
        am = uniform_filter1d(mask * 255, size=k, axis=1, mode="constant")[..., None]
        a = np.maximum(a, am * amount)
    return _merge(out, a)


def add_depth(img, amount):
    if amount <= 0:
        return img
    rgb, a = _split(img); H, W = rgb.shape[:2]
    rad = 8 * W / 1000 * amount
    blur = np.asarray(Image.fromarray(np.clip(rgb, 0, 255).astype("uint8"))
                      .filter(ImageFilter.GaussianBlur(radius=rad))).astype(float)
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / np.sqrt((W / 2) ** 2 + (H / 2) ** 2)
    w = (np.clip((r - 0.5) / 0.5, 0, 1) ** 1.5 * amount)[..., None]   # Raender unscharf
    return _merge(rgb * (1 - w) + blur * w, a)


def apply_image(art, image, mode, bg, transparent=False):
    """Masking = Bild NUR innerhalb der Audio-Form; Textur-Blend = Form x Textur."""
    W = art.width
    im = np.asarray(ImageOps.fit(image.convert("RGB"), (W, W),
                                 method=Image.LANCZOS)).astype(float)
    ar = np.asarray(art).astype(float)
    lum = (ar[..., 3] / 255.0 if ar.shape[2] == 4 else ar[..., :3].mean(2) / 255.0)
    lum = np.clip(lum, 0, 1)
    bg_rgb = np.array(_hex_rgb(bg))

    if mode == "Masking":
        if transparent:
            out = np.concatenate([im, (lum * 255)[..., None]], 2)
            return Image.fromarray(np.clip(out, 0, 255).astype("uint8"), "RGBA")
        m = lum[..., None]
        return Image.fromarray(np.clip(im * m + bg_rgb * (1 - m), 0, 255).astype("uint8"), "RGB")

    art_rgb = ar[..., :3]
    blended = art_rgb * (im / 255.0)                    # Multiply -> Textur faerbt Form
    m3 = (lum > 0.03)[..., None]
    out_rgb = np.where(m3, blended, art_rgb)
    if ar.shape[2] == 4:
        out = np.concatenate([out_rgb, ar[..., 3:4]], 2)
        return Image.fromarray(np.clip(out, 0, 255).astype("uint8"), "RGBA")
    return Image.fromarray(np.clip(out_rgb, 0, 255).astype("uint8"), "RGB")


def apply_effects(img, feat, bg="#000000", intensity=1.0, grain=True, bloom=True,
                  chroma=True, seed=None, vignette=0.0, posterize_levels=0,
                  halftone=0, streaks=0.0, depth=0.0):
    scale = img.width / 1000.0
    if depth > 0:
        img = add_depth(img, depth)
    if bloom:
        img = add_bloom(img, feat["energy"] * intensity)
    if streaks > 0:
        img = add_streaks(img, streaks * (0.4 + 0.6 * feat["energy"]))
    if chroma:
        img = chroma_ab(img, int(feat["spread"] * intensity * 8 * scale))
    if posterize_levels and posterize_levels >= 2:
        img = posterize(img, posterize_levels)
    if halftone and halftone >= 2:
        img = add_halftone(img, int(halftone * scale), bg)
    if vignette > 0:
        img = add_vignette(img, vignette)
    if grain:
        img = add_grain(img, feat["roughness"] * intensity, seed=seed)
    return img



def build(audio=None, mode="Rose gespiegelt", cmap=None, bg="#000000", size_px=3000,
          params=None, signet=None, title=None, title_color="#ffffff",
          start=0.0, end=None, mix_weights=None, transparent=False,
          effects=None, return_features=False, y=None, sr=None,
          image=None, image_mode=None):
    params = params or {}
    if y is None:
        y, sr = load_audio(audio, start=start, end=end)
    if mix_weights:
        img = mix_modes(y, sr, mix_weights, cmap, bg, size_px, params,
                        transparent=transparent)
    else:
        p = dict(params); p["transparent"] = transparent
        fig = MODES[mode](y=y, sr=sr, cmap=cmap, bg=bg, size_px=size_px, **p)
        img = fig_to_pil(fig, size_px, transparent=transparent)
    if image is not None and image_mode:          # Bild in die Form (Masking / Textur)
        pic = image if hasattr(image, "size") else Image.open(image)
        img = apply_image(img, pic, image_mode, bg, transparent=transparent)
    feat = None
    if effects:
        feat = features(y, sr)
        img = apply_effects(img, feat, bg=bg, **effects)   # vor Signet/Titel
    if signet: img = add_signet(img, signet)
    if title: img = add_title(img, title, title_color)
    if img.mode == "RGBA":                    # RGB-Muell auf transparenten Pixeln entfernen
        arr = np.asarray(img).copy()
        arr[arr[:, :, 3] == 0, :3] = 0
        img = Image.fromarray(arr, "RGBA")
    return (img, feat) if return_features else img


# ----------------------------------------------------------------------
# Multi-Format-Export (Social)
# ----------------------------------------------------------------------
SIZES = {
    "1x1_Quadrat":     (1080, 1080),
    "4x5_IG-Feed":     (1080, 1350),
    "9x16_Story_Reel_Canvas": (1080, 1920),
    "16x9_YouTube":    (1920, 1080),
}


def export_svg(audio, mode, cmap, bg, params, transparent=False,
               size_px=1000, title=None, title_color="#ffffff"):
    """Einzelner Modus als SVG (Vektor). Ohne Raster-Effekte."""
    y, sr = load_audio(audio)
    p = dict(params); p["transparent"] = transparent
    fig = MODES[mode](y=y, sr=sr, cmap=cmap, bg=bg, size_px=size_px, **p)
    if title:
        fig.text(0.5, 0.05, title, ha="center", color=title_color, fontsize=22)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", transparent=transparent,
                facecolor="none" if transparent else fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    return buf.getvalue()


def export_svg_layers(audio, weights, cmap, bg, base_params,
                      transparent=False, size_px=1000):
    """Aktive Mix-Layer je als eigene SVG (fuer manuelles Stapeln in Affinity)."""
    y, sr = load_audio(audio)
    out = {}
    for mode, w in weights.items():
        if w <= 0:
            continue
        p = _mode_params(mode, base_params); p["transparent"] = transparent
        fig = MODES[mode](y=y, sr=sr, cmap=cmap, bg=bg, size_px=size_px, **p)
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", transparent=transparent,
                    facecolor="none" if transparent else fig.get_facecolor())
        plt.close(fig); buf.seek(0)
        out[mode] = buf.getvalue()
    return out



    """Zentriert das quadratische Artwork auf allen Social-Seitenverhaeltnissen."""
    res = {}
    bg_rgb = _hex_rgb(bg)
    for name, (W, H) in SIZES.items():
        s = int(min(W, H) * margin)
        art = square_img.resize((s, s), Image.LANCZOS)
        pos = ((W - s) // 2, (H - s) // 2)
        if transparent:
            canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canvas.alpha_composite(art.convert("RGBA"), pos)
        else:
            canvas = Image.new("RGB", (W, H), bg_rgb)
            art = art.convert("RGBA")
            canvas.paste(art, pos, art)
        res[name] = canvas
    return res


# ----------------------------------------------------------------------
# Animierter Video-Export (Rose, audio-reaktiv)
# ----------------------------------------------------------------------
def _rose_frame(a, cmap, bg, size, mirror, thickness, inner, rotation,
                data_thickness):
    aa = np.clip(a, 0, 1)
    N = len(aa)
    fig, ax = _polar_fig(size, bg, rotation=rotation)
    dt = (0.4 + 1.2 * aa) if data_thickness else 1.0
    if mirror:
        w = (np.pi / N) * thickness * dt
        right = np.pi / 2 - np.linspace(0, np.pi, N)
        left = np.pi / 2 + np.linspace(0, np.pi, N)
        ax.bar(right, aa, width=w, bottom=inner, color=cmap(aa), lw=0)
        ax.bar(left, aa, width=w, bottom=inner, color=cmap(aa), lw=0)
    else:
        w = (2 * np.pi / N) * thickness * dt
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        ax.bar(ang, aa, width=w, bottom=inner, color=cmap(aa), lw=0)
    return fig_to_pil(fig, size)


def animate_rose(audio, cmap, bg, params, out_path, duration=5.0, fps=24,
                 aspect="9x16_Story_Reel_Canvas", rotate_turns=0.25, react=1.0,
                 mirror=True, smoothing=0.35, with_audio=True):
    """react: 0 = globales Pulsieren (statisches Spektrum),
              1 = jeder Bin folgt seiner Frequenz ueber die Zeit.
       smoothing: 0 = roh/zappelig, ->1 = traeger/weicher.
       with_audio: Originalton unter das Video muxen (ffmpeg)."""
    import imageio
    y, sr = load_audio(audio, end=duration)
    duration = min(duration, len(y) / sr)            # nicht laenger als der Track
    nm = params.get("n_mels", 110)
    S = punch(norm01(mel_db(y, sr, nm)),                 # n_mels x T, global normiert
              params.get("gate", 0.15), params.get("gamma", 1.4))
    a0 = S.mean(axis=1)                                  # statische Referenz
    T = S.shape[1]

    W, H = SIZES[aspect]
    W -= W % 2; H -= H % 2
    side = int(min(W, H) * 0.96)
    bg_rgb = _hex_rgb(bg)
    thick0 = params.get("thickness", 1.0); rot0 = params.get("rotation", 0.0)
    inner = params.get("inner", 0.15); dthick = params.get("data_thickness", True)

    frames = int(duration * fps)
    video_target = out_path
    if with_audio:                                   # zuerst stummes Video, dann muxen
        import tempfile as _tf
        video_target = _tf.NamedTemporaryFile(suffix=".mp4", delete=False).name

    writer = imageio.get_writer(video_target, fps=fps, codec="libx264",
                                quality=8, macro_block_size=None)
    prev = None
    for f in range(frames):
        p = f / max(1, frames - 1)
        c1 = int(p * (T - 1))
        c0 = int((f - 1) / max(1, frames - 1) * (T - 1)) if f > 0 else c1
        live = S[:, min(c0, c1):max(c0, c1) + 1].mean(axis=1)   # Fenster -> Spalte(n)
        a = (1 - react) * a0 + react * live                     # global <-> pro Bin
        if prev is not None:
            a = smoothing * prev + (1 - smoothing) * a          # zeitliche Glaettung
        prev = a
        rot = rot0 + rotate_turns * 360.0 * p                   # Drehen
        art = _rose_frame(a, cmap, bg, side, mirror,
                          thick0, inner, rot, dthick)
        canvas = Image.new("RGB", (W, H), bg_rgb)
        canvas.paste(art, ((W - side) // 2, (H - side) // 2))
        writer.append_data(np.asarray(canvas))
    writer.close()

    if not with_audio:
        return out_path

    # --- Originalton in Videolaenge schneiden und muxen ---
    import subprocess, tempfile as _tf, imageio_ffmpeg, soundfile as _sf
    try:
        if hasattr(audio, "seek"):
            audio.seek(0)
        ay, asr = librosa.load(audio, sr=None, mono=False, duration=duration)
        awav = _tf.NamedTemporaryFile(suffix=".wav", delete=False).name
        _sf.write(awav, ay.T if ay.ndim > 1 else ay, asr)
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([exe, "-y", "-i", video_target, "-i", awav,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", out_path],
                       check=True, capture_output=True)
        return out_path
    except Exception:
        # Fallback: stummes Video ausliefern
        import shutil
        shutil.copy(video_target, out_path)
        return out_path


# ----------------------------------------------------------------------
# Streamlit-UI
# ----------------------------------------------------------------------
def main():
    import streamlit as st, tempfile, hashlib, json, zipfile, io as _io
    ss = st.session_state
    st.set_page_config(page_title="Sonic Artwork", layout="wide")

    @st.cache_data(show_spinner=False)
    def decode(data):
        return load_audio(_io.BytesIO(data))

    defaults = {
        "k_mode": list(MODES.keys())[0], "k_mixon": False, "k_preset": "Korrend",
        "k_w_gesp": 0.7, "k_w_roh": 0.0, "k_w_hpss": 0.8, "k_w_wave": 0.0,
        "k_thick": 1.0, "k_gate": 0.15, "k_gamma": 1.5, "k_rot": 0,
        "k_inner": 0.15, "k_nmels": 110, "k_dthick": True, "k_onset": 0.25,
        "k_spoke": 0.35,
        "k_fxon": False, "k_fxint": 1.0, "k_g": True, "k_b": True, "k_c": True,
        "k_seed": 0, "k_size": 3000, "k_transp": False, "k_fmt": "PNG", "k_live": True,
        "k_anchor": "#7b2ff7", "k_harmony": "Monochrom-Ramp",
        "k_vig": 0.0, "k_post": 0, "k_half": 0, "k_streaks": 0.0, "k_depth": 0.0,
        "k_imgmode": "—",
        "n_stops": len(PRESETS["Korrend"]), "bg_key": PROFILE_BG["Korrend"],
    }
    for k, v in defaults.items():
        ss.setdefault(k, v)
    for i, c in enumerate(PRESETS["Korrend"]):
        ss.setdefault(f"c{i}", c)
    save_keys = list(defaults.keys())

    # ---------------------------- SIDEBAR ----------------------------
    with st.sidebar:
        st.header("Sonic Artwork")
        up = st.file_uploader("Audiodatei", type=["wav", "mp3", "flac", "ogg", "m4a"])

        with st.expander("Presets speichern / laden"):
            pfile = st.file_uploader("Laden (.json)", type=["json"], key="k_loadfile")
            if pfile is not None and st.button("Anwenden"):
                try:
                    data = json.load(pfile)
                    for k, v in data.items():
                        ss[k] = v
                    st.rerun()
                except Exception as e:
                    st.error(f"Preset ungueltig: {e}")
            n_now = int(ss.get("n_stops", 4))
            cfg = {k: ss[k] for k in save_keys}
            cfg["n_stops"] = n_now
            for i in range(n_now):
                cfg[f"c{i}"] = ss.get(f"c{i}", "#ffffff")
            st.download_button("Speichern (.json)", json.dumps(cfg, indent=2),
                               "preset.json", "application/json")

        mode = st.selectbox("Modus", list(MODES.keys()), key="k_mode")
        mix_on = st.checkbox("Modi mischen", key="k_mixon")
        mix_w = {}
        if mix_on:
            mix_w = {
                "Rose gespiegelt":  st.slider("Rose g.", 0.0, 1.0, step=0.05, key="k_w_gesp"),
                "Rose roh":         st.slider("Rose roh", 0.0, 1.0, step=0.05, key="k_w_roh"),
                "HPSS-Zeit":        st.slider("HPSS", 0.0, 1.0, step=0.05, key="k_w_hpss"),
                "Kreis-Wellenform": st.slider("Welle", 0.0, 1.0, step=0.05, key="k_w_wave"),
            }
            st.caption("Rose x HPSS/Welle ist ergiebig; zwei Rosen sind redundant.")

        st.subheader("Palette")
        preset = st.selectbox("Preset", list(PRESETS.keys()), key="k_preset")
        if st.button("Preset laden"):
            cp = PRESETS[preset]; ss["n_stops"] = len(cp)
            for i, c in enumerate(cp):
                ss[f"c{i}"] = c
            ss["bg_key"] = PROFILE_BG[preset]
            st.rerun()

        with st.expander("Palette-Generator (Farbtheorie)"):
            anchor = st.color_picker("Ankerfarbe", key="k_anchor")
            harmony = st.selectbox(
                "Harmonie",
                ["Monochrom-Ramp", "Analog", "Komplementaer-Akzent",
                 "Triadisch", "Profil-Struktur"], key="k_harmony")
            st.caption("Profil-Struktur = Helligkeits-/Saettigungsverlauf des "
                       "gewaehlten Presets uebernehmen, nur Farbton tauschen "
                       "(konsistente Releases).")
            if st.button("Palette generieren"):
                try:
                    if harmony == "Profil-Struktur":
                        tmpl = PRESETS[ss["k_preset"]]
                        newp = generate_palette(anchor, len(tmpl), harmony, template=tmpl)
                    else:
                        newp = generate_palette(anchor, int(ss["n_stops"]), harmony)
                    ss["n_stops"] = len(newp)
                    for i, cc in enumerate(newp):
                        ss[f"c{i}"] = cc
                    ss["bg_key"] = newp[0]
                    st.rerun()
                except Exception as e:
                    st.error(f"Generator-Fehler: {e}")

        n = st.slider("Farbstufen", 2, 6, key="n_stops")
        cols = st.columns(n)
        stops = []
        for i in range(n):
            ss.setdefault(f"c{i}", "#ffffff")
            stops.append(cols[i].color_picker(f"{i+1}", key=f"c{i}"))
        bg = st.color_picker("Hintergrund", key="bg_key")
        lut = st.file_uploader("LUT statt Palette (.cube)", type=["cube"])

        st.subheader("Regler")
        thickness = st.slider("Dicke", 0.3, 3.0, step=0.1, key="k_thick")
        gate = st.slider("Gate (Rauschen)", 0.0, 0.8, step=0.05, key="k_gate")
        gamma = st.slider("Gamma (Kontrast)", 0.8, 2.5, step=0.1, key="k_gamma")
        rotation = st.slider("Winkel drehen (Grad)", 0, 360, step=5, key="k_rot")

        hpss_active = (mode == "HPSS-Zeit") or (mix_on and mix_w.get("HPSS-Zeit", 0) > 0)
        rose_modes = ("Rose gespiegelt", "Rose roh", "Kreis-Wellenform")
        rose_active = (mode in rose_modes) or (mix_on and any(mix_w.get(m, 0) > 0 for m in rose_modes))
        with st.expander("Mehr"):
            inner = st.slider("Innenradius", 0.0, 0.4, step=0.01, key="k_inner")
            n_mels = st.slider("Frequenzaufloesung (Bins)", 60, 256, step=2, key="k_nmels")
            data_thick = (st.checkbox("Dicke datengetrieben (Loudness)", key="k_dthick")
                          if rose_active else ss["k_dthick"])
            onset = (st.slider("Onset-Akzent", 0.0, 1.0, step=0.05, key="k_onset")
                     if hpss_active else ss["k_onset"])
            spoke_len = (st.slider("Speichen-Laenge (HPSS)", 0.1, 1.0, step=0.05, key="k_spoke")
                         if hpss_active else ss["k_spoke"])

        st.subheader("Effekte")
        fx_on = st.checkbox("Effekte anwenden", key="k_fxon")
        fx = None
        if fx_on:
            fx_int = st.slider("Intensitaet", 0.0, 1.5, step=0.05, key="k_fxint")
            f1, f2, f3 = st.columns(3)
            g = f1.checkbox("Grain", key="k_g"); b = f2.checkbox("Bloom", key="k_b")
            c = f3.checkbox("Aberr.", key="k_c")
            seed_in = st.number_input("Seed (0=Datei)", 0, 999999, step=1, key="k_seed")
            st.caption("Stilisierung (0 = aus):")
            vignette = st.slider("Vignette", 0.0, 1.0, step=0.05, key="k_vig")
            streaks = st.slider("Licht-Streaks", 0.0, 1.0, step=0.05, key="k_streaks")
            depth = st.slider("Pseudo-Tiefe (DoF)", 0.0, 1.0, step=0.05, key="k_depth")
            post_lv = st.slider("Posterize (Farbstufen, 0=aus)", 0, 12, step=1, key="k_post")
            half = st.slider("Halftone (Rasterweite px, 0=aus)", 0, 12, step=1, key="k_half")
            fx = dict(intensity=fx_int, grain=g, bloom=b, chroma=c,
                      seed=(seed_in or None), vignette=vignette, streaks=streaks,
                      depth=depth, posterize_levels=post_lv, halftone=half)

        st.subheader("Bild / Textur")
        img_up = st.file_uploader("Bild oder Textur (optional)", type=["png", "jpg", "jpeg"])
        img_mode = st.selectbox("Modus", ["—", "Masking", "Textur-Blend"], key="k_imgmode")
        st.caption("Masking = Bild NUR innerhalb der Audio-Form. "
                   "Textur-Blend = Textur faerbt die Form. "
                   "Fotos wirken besser mit HPSS/gefuellten Formen als mit der Rose.")

        st.subheader("Export-Einstellungen")
        size = st.select_slider("Voll-Groesse (px)", options=[1000, 1500, 2000, 3000], key="k_size")
        transparent = st.checkbox("Transparent (ohne Schwarz, PNG)", key="k_transp")
        fmt = "PNG" if transparent else st.selectbox("Format", ["PNG", "JPEG"], key="k_fmt")
        title = st.text_input("Titel (optional)")
        signet_up = st.file_uploader("Signet (PNG)", type=["png"])
        live = st.checkbox("Live-Vorschau (auto-aktualisiert)", key="k_live")

    # ---------------------------- MAIN ----------------------------
    st.title("Sonic Artwork")
    if not up:
        st.info("Lade links in der Seitenleiste eine Audiodatei — dann erscheinen Vorschau und Export.")
        return

    try:
        y, sr = decode(up.getvalue())
    except Exception as e:
        st.error(f"Audio konnte nicht gelesen werden: {e}")
        return

    try:
        cmap = make_cmap(stops)
        if lut:
            p = tempfile.NamedTemporaryFile(suffix=".cube", delete=False)
            p.write(lut.getvalue()); p.close(); cmap = cmap_from_cube(p.name)
    except Exception as e:
        st.error(f"Palette/LUT ungueltig, nutze Standard: {e}")
        cmap = make_cmap(["#000000", "#ffffff"])

    sig_path = None
    if signet_up:
        p = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        p.write(signet_up.getvalue()); p.close(); sig_path = p.name

    base = dict(thickness=thickness, gate=gate, gamma=gamma,
                inner=inner, n_mels=n_mels, rotation=rotation)
    fx_use = None
    if fx is not None:
        fx_use = dict(fx)
        if fx_use["seed"] is None:
            fx_use["seed"] = int(hashlib.md5(up.name.encode()).hexdigest(), 16) % (2**32)

    img_pic, img_mode_val = None, None
    if img_up is not None and img_mode != "—":
        try:
            img_pic = Image.open(_io.BytesIO(img_up.getvalue())).convert("RGB")
            img_mode_val = img_mode
        except Exception as e:
            st.error(f"Bild konnte nicht gelesen werden: {e}")

    def render(size_px):
        common = dict(cmap=cmap, bg=bg, size_px=size_px, signet=sig_path,
                      title=title or None, title_color=stops[-1],
                      transparent=transparent, effects=fx_use,
                      return_features=True, y=y, sr=sr,
                      image=img_pic, image_mode=img_mode_val)
        if mix_on and any(v > 0 for v in mix_w.values()):
            return build(mode=mode, params=base, mix_weights=mix_w, **common)
        params = dict(base)
        if mode == "HPSS-Zeit":
            params["onset"] = onset; params["gate"] = max(gate, 0.4)
            params["spoke_len"] = spoke_len
        else:
            params["data_thickness"] = data_thick
        return build(mode=mode, params=params, **common)

    col_prev, col_exp = st.columns([3, 2])
    with col_prev:
        if live:
            try:
                with st.spinner("Vorschau ..."):
                    img, feat = render(700)
                st.image(img, caption="Live-Vorschau (700 px) — Export rendert in voller Groesse")
                if feat:
                    st.caption(f"Features -> Grain {feat['roughness']:.2f} · "
                               f"Bloom {feat['energy']:.2f} · Aberration {feat['spread']:.2f}")
            except Exception as e:
                st.error(f"Render-Fehler: {e}")
        else:
            st.info("Live-Vorschau ist aus. Rechts 'In voller Groesse exportieren'.")

    with col_exp:
        st.subheader("Export")
        if st.button("In voller Groesse exportieren", type="primary"):
            try:
                with st.spinner(f"Rendere {size}px ..."):
                    img, _ = render(size)
                    buf = _io.BytesIO()
                    img.save(buf, format=fmt, quality=92 if fmt == "JPEG" else None)
                    buf.seek(0)
                st.download_button(f"{fmt} laden", buf, f"cover.{fmt.lower()}",
                                   f"image/{fmt.lower()}")
            except Exception as e:
                st.error(f"Export-Fehler: {e}")

        if st.button("Social-Formate (ZIP)"):
            try:
                with st.spinner("Erzeuge Formate ..."):
                    img, _ = render(size)
                    fmts = export_formats(img, bg, transparent=transparent)
                    zbuf = _io.BytesIO()
                    with zipfile.ZipFile(zbuf, "w") as z:
                        for name, im in fmts.items():
                            b = _io.BytesIO(); im.save(b, format="PNG")
                            z.writestr(f"{name}.png", b.getvalue())
                    zbuf.seek(0)
                st.caption("Quadrat · 4:5 · 9:16 · 16:9")
                st.download_button("ZIP laden", zbuf, "social_formats.zip", "application/zip")
            except Exception as e:
                st.error(f"Batch-Fehler: {e}")

    with st.expander("Video (Canvas / Reel) — animierte Rose"):
        v_asp = st.selectbox("Format", list(SIZES.keys()), index=2)
        v_dur = st.number_input("Dauer (s)", 2.0, 300.0, 5.0, 1.0,
                                help="Spotify Canvas max. 8 s; Renderzeit steigt linear.")
        v_fps = st.select_slider("FPS", options=[12, 24, 30], value=24)
        v_turn = st.slider("Drehung (Umdrehungen)", 0.0, 1.0, 0.25, 0.05)
        v_react = st.slider("Bin-Bewegung (0=global, 1=pro Frequenz)", 0.0, 1.0, 1.0, 0.05)
        v_smooth = st.slider("Glaettung (gegen Zappeln)", 0.0, 0.9, 0.35, 0.05)
        v_audio = st.checkbox("Originalton einbetten", value=True)
        if st.button("Video erzeugen"):
            try:
                with st.spinner("Rendere Video (kann dauern) ..."):
                    outp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    animate_rose(up, cmap, bg, base, outp, duration=v_dur, fps=v_fps,
                                 aspect=v_asp, rotate_turns=v_turn, react=v_react,
                                 mirror=(mode != "Rose roh"), smoothing=v_smooth,
                                 with_audio=v_audio)
                    vbytes = open(outp, "rb").read()
                st.video(vbytes)
                st.download_button("MP4 laden", vbytes, "visualizer.mp4", "video/mp4")
            except ModuleNotFoundError:
                st.error("Video braucht: pip install imageio imageio-ffmpeg")
            except Exception as e:
                st.error(f"Video-Fehler: {e}")


if __name__ == "__main__":
    main()