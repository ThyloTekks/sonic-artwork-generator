"""Zugriff auf das ffmpeg-Binary: Pfad, Aufruf, Laufzeit auslesen.

Bewegtbild und Footage brauchen beide ffmpeg. Der Pfad kommt von
imageio-ffmpeg, das eine passende Binary mitbringt; auf Streamlit Cloud
reicht zusaetzlich ein `ffmpeg` in der packages.txt.
"""

from __future__ import annotations

import re
import subprocess

_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def exe() -> str:
    """Pfad zur ffmpeg-Binary. Wirft ModuleNotFoundError ohne imageio-ffmpeg."""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args) -> subprocess.CompletedProcess:
    """ffmpeg aufrufen und bei Fehler das Ende von stderr als Meldung werfen.

    ffmpeg schreibt den eigentlichen Grund in die letzten Zeilen; ein blosser
    Exitcode sagt im UI nichts.
    """
    r = subprocess.run([exe(), *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace").strip()[-700:])
    return r


def duration(path) -> float | None:
    """Laufzeit in Sekunden aus dem Infoblock. None, wenn nicht lesbar.

    `ffmpeg -i` ohne Ausgabedatei endet absichtlich mit Exitcode 1 und legt
    die Angaben nach stderr, deshalb hier nicht ueber run().
    """
    r = subprocess.run([exe(), "-i", str(path)], capture_output=True)
    m = _DURATION.search(r.stderr.decode("utf-8", "replace"))
    if not m:
        return None
    hh, mm, ss = m.groups()
    return int(hh) * 3600 + int(mm) * 60 + float(ss)
