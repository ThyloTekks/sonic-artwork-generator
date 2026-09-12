"""Zeichenflaechen und Bildkonvertierung — gemeinsam fuer alle Modi."""

from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image


def polar_fig(size_px, bg, dpi=100, transparent=False, rotation=0.0, rmax=1.2):
    inch = size_px / dpi
    fig = plt.figure(figsize=(inch, inch), dpi=dpi)
    face = "none" if transparent else bg
    fig.patch.set_facecolor(face)
    if transparent:
        fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1], projection="polar")
    ax.set_facecolor(face)
    ax.axis("off")
    ax.set_theta_offset(np.deg2rad(rotation))     # 0 = Osten; dreht die ganze Scheibe
    ax.set_ylim(0, rmax)
    return fig, ax


def cart_fig(size_px, bg, dpi=100, transparent=False, lim=1.1):
    """Kartesische Flaeche — fuer Modi, die keine Scheibe sind (Lissajous)."""
    inch = size_px / dpi
    fig = plt.figure(figsize=(inch, inch), dpi=dpi)
    face = "none" if transparent else bg
    fig.patch.set_facecolor(face)
    if transparent:
        fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(face)
    ax.axis("off")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    return fig, ax


def flat_fig(size_px, bg, dpi=100, transparent=False):
    """Flaeche mit Koordinaten 0..1, Ursprung oben links.

    Fuer Modi, die weder Scheibe noch Goniometer sind — Raster und Schichten
    lesen sich von oben links, wie Satz.
    """
    inch = size_px / dpi
    fig = plt.figure(figsize=(inch, inch), dpi=dpi)
    face = "none" if transparent else bg
    fig.patch.set_facecolor(face)
    if transparent:
        fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(face)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)                       # oben links ist der Anfang
    return fig, ax


def cmap_to_alpha(cmap, knee=0.12):
    """Colormap mit Alpha-Verlauf: leise -> durchsichtig, laut -> deckend."""
    x = np.linspace(0, 1, 256)
    cols = cmap(x)
    cols[:, 3] = np.clip((x - knee) / (1 - knee + 1e-9), 0, 1) ** 0.8
    return ListedColormap(cols)


def fig_to_pil(fig, size_px, transparent=False):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=transparent,
                facecolor="none" if transparent else fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    mode = "RGBA" if transparent else "RGB"
    return Image.open(buf).convert(mode).resize((size_px, size_px), Image.LANCZOS)


def fig_to_svg(fig, transparent=False):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", transparent=transparent,
                facecolor="none" if transparent else fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def resample_time(M: np.ndarray, cols: int) -> np.ndarray:
    """Spektrogramm auf eine feste Spaltenzahl bringen.

    Ohne das haengt die Zeichendauer an der Songlaenge: ein 8-Minuten-Track
    ergibt sonst zehntausende pcolormesh-Vierecke fuer dieselben 3000 Pixel.
    """
    if M.shape[1] <= cols:
        return M
    idx = np.linspace(0, M.shape[1] - 1, cols)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, M.shape[1] - 1)
    w = (idx - lo)[None, :]
    return M[:, lo] * (1 - w) + M[:, hi] * w
