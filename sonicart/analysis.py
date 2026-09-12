"""Audioanalyse: einmal rechnen, oft benutzen.

Vorher lag die Analyse in den Rendermethoden — jeder Reglerausschlag hat
librosa.effects.hpss neu gestartet (~8,5 s bei einem 3-Minuten-Track).
Hier liegt sie in einem Objekt mit cached_property, und die Trennung in
harmonisch/perkussiv passiert im Spektrogramm statt im Zeitsignal: die
Ruecktransformation entfaellt, die Bilder sind dieselben (Korrelation 0,997).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property

import librosa
import numpy as np

DEFAULT_SR = 22050
DEFAULT_HOP = 1024      # 512 waere librosa-Standard; 1024 halbiert die HPSS-Zeit
DEFAULT_NFFT = 2048

# Krumhansl-Schmuckler-Profile fuer die Tonartbestimmung.
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SOURCES = ("mix", "harmonic", "percussive")


def norm01(x):
    """Auf 0..1 skalieren, unteres Perzentil als Nullpunkt (robust gegen Stille)."""
    lo, hi = np.percentile(x, 2), x.max()
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def punch(x, gate, gamma):
    """Gate schneidet Rauschen weg, Gamma spreizt den Rest."""
    return np.clip((x - gate) / (1 - gate + 1e-9), 0, 1) ** gamma


def whiten(M: np.ndarray, staerke: float = 1.0) -> np.ndarray:
    """Jedes Frequenzband auf den eigenen Dynamikbereich normieren.

    Das gemittelte Mel-Spektrum faellt ueber die Frequenz steil ab: der Bass
    frisst den Tonwertumfang, und in den oberen Baendern passiert sichtbar
    nichts. Nach der bandweisen Normierung nutzt jedes Band den vollen Umfang
    — die Rose fuellt dann den ganzen Kreis statt eines schmalen Keils.
    staerke 0 laesst alles wie es war, 1 normiert vollstaendig.
    """
    if staerke <= 0:
        return M
    med = np.median(M, axis=1, keepdims=True)
    spanne = np.percentile(M, 95, axis=1, keepdims=True) - med
    w = (M - med) / (spanne + 1e-9) * 0.5 + 0.5
    return np.clip(M * (1 - staerke) + w * staerke, 0, 1)


def rank01(v: np.ndarray) -> np.ndarray:
    """Rangnormierung: verteilt Werte gleichmaessig ueber 0..1.

    Min-Max reicht nicht, wo Werte dicht beieinanderliegen — Takt- oder
    Spaltenmittel etwa. Nach dem Quantisieren landen die sonst fast alle in
    derselben Stufe und das Bild kippt tonwertlich zusammen.
    """
    v = np.asarray(v, dtype=float).ravel()
    if len(v) < 2:
        return np.zeros_like(v)
    return v.argsort().argsort() / (len(v) - 1)


def _octave_scale(hz, lo, hi) -> float:
    """Frequenz auf 0..1, oktavweise — so wie Gehoer Tonhoehe staffelt."""
    hz = max(float(hz), 1e-6)
    return float(np.clip(np.log2(hz / lo) / np.log2(hi / lo), 0, 1))


@dataclass
class Analysis:
    """Alle Messwerte eines Stuecks. Teures wird beim ersten Zugriff berechnet."""

    y: np.ndarray                       # Mono, das Arbeitssignal
    sr: int = DEFAULT_SR
    y_stereo: np.ndarray | None = None  # (2, n), falls die Quelle stereo war
    name: str = ""
    hop: int = DEFAULT_HOP
    n_fft: int = DEFAULT_NFFT
    _cache: dict = field(default_factory=dict, repr=False)

    # ---------------- Grunddaten ----------------
    @property
    def duration(self) -> float:
        return len(self.y) / self.sr

    @property
    def is_stereo(self) -> bool:
        return self.y_stereo is not None and self.y_stereo.shape[0] == 2

    @property
    def frame_times(self) -> np.ndarray:
        return librosa.frames_to_time(np.arange(self.n_frames),
                                      sr=self.sr, hop_length=self.hop)

    @property
    def n_frames(self) -> int:
        return self.stft_mag.shape[1]

    # ---------------- Spektrogramme ----------------
    @cached_property
    def stft_mag(self) -> np.ndarray:
        """Betragsspektrogramm — Grundlage fuer alles Weitere."""
        return np.abs(librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop))

    @cached_property
    def _hpss_mag(self):
        """Harmonisch/perkussiv getrennt, im Spektrogramm statt im Zeitsignal."""
        return librosa.decompose.hpss(self.stft_mag)

    def _source_mag(self, source: str) -> np.ndarray:
        if source == "mix":
            return self.stft_mag
        if source == "harmonic":
            return self._hpss_mag[0]
        if source == "percussive":
            return self._hpss_mag[1]
        raise ValueError(f"Unbekannte Quelle {source!r}, erlaubt: {SOURCES}")

    def mel(self, n_mels: int = 110, source: str = "mix") -> np.ndarray:
        """Mel-Spektrogramm in dB. source: 'mix' | 'harmonic' | 'percussive'."""
        key = ("mel", n_mels, source)
        if key not in self._cache:
            S = librosa.feature.melspectrogram(
                S=self._source_mag(source) ** 2, sr=self.sr,
                n_mels=n_mels, fmax=self.sr / 2)
            self._cache[key] = librosa.power_to_db(S, ref=1.0)
        return self._cache[key]

    def mel_norm(self, n_mels: int = 110, source: str = "mix",
                 whiten_amount: float = 0.0) -> np.ndarray:
        """Mel-Spektrogramm auf 0..1. whiten_amount normiert bandweise."""
        key = ("meln", n_mels, source, round(float(whiten_amount), 3))
        if key not in self._cache:
            M = norm01(self.mel(n_mels, source))
            self._cache[key] = whiten(M, whiten_amount) if whiten_amount > 0 else M
        return self._cache[key]

    def spectrum(self, n_mels: int = 110, gate: float = 0.15,
                 gamma: float = 1.4, source: str = "mix",
                 tilt: float = 0.0) -> np.ndarray:
        """Ueber die Zeit gemitteltes Spektrum — der Klangfingerabdruck.

        Achtung: hier faellt die Zeitachse weg. Zwei Stuecke mit gleichem
        Frequenzhaushalt, aber verschiedener Dramaturgie, sehen danach gleich
        aus. Fuer zeitabhaengige Bilder mel_norm() oder segments() nehmen.

        tilt hebt die Hoehen an (0 = roh, 1 = kraeftig). Musik faellt zu den
        Hoehen hin natuerlich ab, deshalb draengt sich sonst alles in die
        untersten Baender — bei der Rose in einen schmalen Keil. Bandweise
        Normierung waere hier falsch: sie setzt jedes Bandmittel per
        Konstruktion auf denselben Wert und loescht damit genau das, was
        dieser Modus zeigt.
        """
        a = self.mel_norm(n_mels, source).mean(axis=1)
        if tilt > 0:
            rampe = np.linspace(0.0, 1.0, len(a)) ** 0.7
            a = a * (1.0 + tilt * 3.0 * rampe)
        return punch(a / (a.max() + 1e-9), gate, gamma)

    # ---------------- Rhythmus ----------------
    @cached_property
    def onset_env(self) -> np.ndarray:
        return librosa.onset.onset_strength(S=librosa.power_to_db(
            librosa.feature.melspectrogram(S=self.stft_mag ** 2, sr=self.sr)),
            sr=self.sr, hop_length=self.hop)

    @cached_property
    def onsets(self) -> np.ndarray:
        return librosa.onset.onset_detect(onset_envelope=self.onset_env,
                                          sr=self.sr, hop_length=self.hop)

    @cached_property
    def _beat_track(self):
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=self.onset_env, sr=self.sr,
            hop_length=self.hop, units="frames")
        return float(np.atleast_1d(tempo)[0]), np.asarray(beats)

    @property
    def tempo(self) -> float:
        return self._beat_track[0]

    @property
    def beats(self) -> np.ndarray:
        """Beats als Frame-Indizes (gleiches Raster wie die Spektrogramme)."""
        return self._beat_track[1]

    @cached_property
    def beat_times(self) -> np.ndarray:
        return librosa.frames_to_time(self.beats, sr=self.sr, hop_length=self.hop)

    @cached_property
    def meter(self) -> int:
        """Geschaetzte Taktart (Zaehlzeiten pro Takt).

        Beat-synchrone Onset-Staerke in Gruppen zu m zerlegen; die Taktart mit
        dem deutlichsten wiederkehrenden Akzent gewinnt. Schaetzung, kein Beweis
        — bei gleichmaessig akzentuierter Musik faellt sie auf 4 zurueck.
        """
        b = self.beats
        if len(b) < 8:
            return 4
        strength = self.onset_env[np.clip(b, 0, len(self.onset_env) - 1)]
        strength = (strength - strength.mean()) / (strength.std() + 1e-9)
        best, best_score = 4, -np.inf
        for m in (2, 3, 4, 5, 6, 7):
            usable = (len(strength) // m) * m
            if usable < 2 * m:
                continue
            prof = strength[:usable].reshape(-1, m).mean(axis=0)
            score = (prof.max() - prof.mean()) / (prof.std() + 1e-9)
            score /= np.sqrt(m)          # gegen die Bevorzugung grosser Taktarten
            if score > best_score:
                best, best_score = m, score
        return best

    def bar_values(self, n_mels: int = 96, whiten_amount: float = 0.85,
                   source: str = "mix") -> np.ndarray:
        """Ein Kennwert je Takt — mittlere Energie des Taktes, rangnormiert.

        Faellt auf eine feste Unterteilung zurueck, wenn keine Beats erkannt
        wurden (Ambient, freies Tempo).
        """
        M = self.mel_norm(n_mels, source, whiten_amount)
        nt = M.shape[1]
        takte = self.bar_times()
        if len(takte) < 4:
            grenzen = np.linspace(0, nt - 1, 17).astype(int)
        else:
            grenzen = np.unique(np.clip(
                (takte / max(self.duration, 1e-9) * nt).astype(int), 0, nt - 1))
            if grenzen[-1] < nt - 1:
                grenzen = np.append(grenzen, nt - 1)
        if len(grenzen) < 3:
            grenzen = np.linspace(0, nt - 1, 9).astype(int)
        werte = np.array([M[:, a:max(b, a + 1)].mean()
                          for a, b in zip(grenzen[:-1], grenzen[1:])])
        return rank01(werte)

    def band_curves(self, baender: int = 7, n_mels: int = 96,
                    whiten_amount: float = 0.85, glaette: float = 0.0,
                    source: str = "mix") -> np.ndarray:
        """(baender, Zeit) — je Frequenzband eine geglaettete Verlaufskurve."""
        from scipy.ndimage import gaussian_filter1d
        M = self.mel_norm(n_mels, source, whiten_amount)
        nf, nt = M.shape
        sigma = glaette if glaette > 0 else max(2.0, nt / 60)
        out = np.empty((baender, nt))
        for i in range(baender):
            lo, hi = int(i * nf / baender), int((i + 1) * nf / baender)
            k = gaussian_filter1d(M[lo:max(hi, lo + 1), :].mean(axis=0), sigma)
            out[i] = (k - k.min()) / (np.ptp(k) + 1e-9)
        return out

    def bar_times(self) -> np.ndarray:
        """Taktanfaenge in Sekunden — Basis fuer loopfaehige Videolaengen."""
        bt = self.beat_times
        m = self.meter
        return bt[::m] if len(bt) >= m else bt

    # ---------------- Harmonik ----------------
    @cached_property
    def chroma(self) -> np.ndarray:
        return librosa.feature.chroma_stft(S=self._source_mag("harmonic") ** 2,
                                           sr=self.sr)

    @cached_property
    def bass_chroma(self) -> np.ndarray:
        """Chroma nur aus dem Bassbereich — dort steht meist der Grundton."""
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        low = self._source_mag("harmonic")[freqs < 350] ** 2
        if low.shape[0] < 2 or low.max() < 1e-12:
            return self.chroma
        return librosa.feature.chroma_stft(S=low, sr=self.sr, n_fft=self.n_fft)

    @cached_property
    def key_chroma(self) -> np.ndarray:
        """Chroma fuer die Tonartbestimmung, mit Bassbetonung.

        Ein blanker Dreiklang ist fuer Krumhansl-Schmuckler mehrdeutig: C-E-G
        passt fast gleich gut auf C-dur wie auf e-moll. Erst der Basston loest
        das auf — so hoert man es auch.
        """
        full = self.chroma.mean(axis=1)
        bass = self.bass_chroma.mean(axis=1)
        full = full / (full.max() + 1e-9)
        bass = bass / (bass.max() + 1e-9)
        return full + 0.6 * bass

    @cached_property
    def key(self) -> dict:
        """Tonart nach Krumhansl-Schmuckler: Tonika, Tongeschlecht, Sicherheit."""
        prof = self.key_chroma
        prof = (prof - prof.mean()) / (prof.std() + 1e-9)
        scores = []
        for i in range(12):
            rot = np.roll(prof, -i)
            for name, ref in (("dur", _KS_MAJOR), ("moll", _KS_MINOR)):
                r = (ref - ref.mean()) / (ref.std() + 1e-9)
                scores.append((float(np.dot(rot, r) / 12), PITCH_NAMES[i], name))
        scores.sort(reverse=True)
        top, second = scores[0], scores[1]
        return {"tonic": top[1], "mode": top[2], "score": top[0],
                "confidence": float(np.clip((top[0] - second[0]) * 5, 0, 1)),
                "label": f"{top[1]}-{top[2]}"}

    @cached_property
    def harmonic_complexity(self) -> float:
        """Entropie der Chroma-Verteilung, 0..1.

        Ein Stueck aus drei Akkorden bleibt nahe 0, dichte Harmonik geht Richtung 1.
        Speist die Farbtonspreizung der Palette.
        """
        p = self.chroma.mean(axis=1)
        p = p / (p.sum() + 1e-9)
        ent = -np.sum(p * np.log(p + 1e-12)) / np.log(12)
        return float(np.clip((ent - 0.6) / 0.4, 0, 1))

    # ---------------- Struktur ----------------
    def segments(self, k: int | None = None,
                 min_seconds: float = 4.0) -> np.ndarray:
        """Grenzen der Songabschnitte als Frame-Indizes (inkl. Anfang und Ende).

        Zu dicht beieinander liegende Grenzen werden verworfen: die Clusterung
        setzt bei gleichfoermigem Material gern eine Grenze direkt hinter den
        Anfang, und ein Abschnitt von 50 ms ist weder hoerbar noch zeichenbar.
        """
        if k is None:
            k = int(np.clip(round(self.duration / 20), 3, 10))
        key = ("seg", k, min_seconds)
        if key in self._cache:
            return self._cache[key]
        mel = self.mel(64)
        feat = np.vstack([
            librosa.feature.mfcc(S=mel, n_mfcc=13),
            librosa.util.normalize(self.chroma, axis=0),
        ])
        n = feat.shape[1]
        kk = int(np.clip(k, 2, max(2, n // 4)))
        try:
            bounds = librosa.segment.agglomerative(feat, kk)
        except Exception:
            bounds = np.linspace(0, n - 1, kk + 1).astype(int)
        raw = np.unique(np.concatenate([[0], np.asarray(bounds), [n - 1]]))
        gap = max(1, int(min_seconds * self.sr / self.hop))
        keep = [int(raw[0])]
        for b in raw[1:-1]:
            if b - keep[-1] >= gap and (n - 1) - b >= gap:
                keep.append(int(b))
        keep.append(int(raw[-1]))
        out = np.array(sorted(set(keep)))
        self._cache[key] = out
        return out

    def segment_times(self, k: int | None = None) -> np.ndarray:
        return librosa.frames_to_time(self.segments(k), sr=self.sr,
                                      hop_length=self.hop)

    def segment_of(self, frames: np.ndarray, k: int | None = None) -> np.ndarray:
        """Ordnet jedem Frame seinen Abschnittsindex zu."""
        b = self.segments(k)
        return np.clip(np.searchsorted(b, frames, "right") - 1, 0, len(b) - 2)

    # ---------------- Stereo ----------------
    @cached_property
    def lissajous(self) -> np.ndarray:
        """(2, n) Links/Rechts fuer Goniometer-Darstellung.

        Mono-Quellen ergeben eine Diagonale — dann wird die Hilbert-Phase als
        zweite Achse genommen, damit der Modus trotzdem eine Figur zeichnet.
        """
        if self.is_stereo:
            st = self.y_stereo
            if np.abs(st[0] - st[1]).mean() > 1e-6:
                return st
        from scipy.signal import hilbert
        n = min(len(self.y), self.sr * 60)
        a = hilbert(self.y[:n].astype(np.float64))
        return np.vstack([np.real(a), np.imag(a)])

    @cached_property
    def stereo_width(self) -> float:
        """0 = mono, 1 = maximal breit (Korrelation von L und R)."""
        if not self.is_stereo:
            return 0.0
        l, r = self.y_stereo
        if l.std() < 1e-9 or r.std() < 1e-9:
            return 0.0
        return float(np.clip((1 - np.corrcoef(l, r)[0, 1]) / 2, 0, 1))

    # ---------------- Klangfarbe ----------------
    @cached_property
    def features(self) -> dict:
        """Timbre-Features -> 0..1. Steuern die Effektkette.

        Die Skalierung ist auf den Bereich gelegt, in dem echte Musik sich
        bewegt, nicht auf den theoretisch moeglichen. Die frueheren linearen
        Faktoren (rms * 4, Zentroid / Nyquist * 3) standen bei jedem normal
        ausgesteuerten Master am Anschlag — damit waren die "audio-reaktiven"
        Effekte zwischen zwei Stuecken praktisch gleich. Pegel laufen deshalb
        ueber dBFS, Frequenzen ueber eine Oktavskala.
        """
        S = self.stft_mag
        flat = float(librosa.feature.spectral_flatness(S=S).mean())
        zcr = float(librosa.feature.zero_crossing_rate(self.y).mean())
        cen = float(librosa.feature.spectral_centroid(S=S, sr=self.sr).mean())
        bw = float(librosa.feature.spectral_bandwidth(S=S, sr=self.sr).mean())
        rms = float(librosa.feature.rms(y=self.y).mean())

        db = 20 * np.log10(max(rms, 1e-6))
        return dict(
            # Rauschanteil -> Grain. Weisses Rauschen liegt bei etwa 0,55;
            # Musik zwischen 0,01 und 0,2, deshalb die Wurzelkennlinie.
            roughness=float(np.clip(np.sqrt(flat / 0.30) + zcr * 0.5, 0, 1)),
            # Lautheit -> Bloom. -30 dBFS (leise) bis -6 dBFS (heiss gemastert).
            energy=float(np.clip((db + 30) / 24, 0, 1)),
            # Spektraler Schwerpunkt -> 200 Hz bis 5 kHz, oktavweise.
            brightness=_octave_scale(cen, 200, 5000),
            # Bandbreite -> Aberration. 200 Hz bis 6 kHz, oktavweise.
            spread=_octave_scale(bw, 200, 6000),
        )

    # ---------------- Zusammenfassung ----------------
    def summary(self) -> dict:
        f = self.features
        return {
            "dauer_s": round(self.duration, 1),
            "tempo_bpm": round(self.tempo, 1),
            "tonart": self.key["label"],
            "tonart_sicherheit": round(self.key["confidence"], 2),
            "taktart": f"{self.meter}/4",
            "harmonische_komplexitaet": round(self.harmonic_complexity, 2),
            "abschnitte": int(len(self.segments()) - 1),
            "stereo": self.is_stereo,
            "stereo_breite": round(self.stereo_width, 2),
            **{k: round(v, 3) for k, v in f.items()},
        }

    def slice(self, start: float = 0.0, end: float | None = None) -> "Analysis":
        """Ausschnitt als eigenstaendige Analyse (fuer Start/Ende-Regler)."""
        if start <= 0 and end is None:
            return self
        a = int(start * self.sr)
        b = int(end * self.sr) if end else len(self.y)
        ys = self.y_stereo[:, a:b] if self.y_stereo is not None else None
        return Analysis(self.y[a:b], self.sr, ys, self.name, self.hop, self.n_fft)


def _source_name(f) -> str:
    """Dateiname — egal ob Pfad, Path-Objekt oder hochgeladene Datei."""
    if isinstance(f, (str, os.PathLike)):
        return os.path.basename(os.fspath(f))
    return getattr(f, "name", "") or ""


def load(f, sr: int = DEFAULT_SR, start: float = 0.0, end: float | None = None,
         stereo: bool = True, name: str = "") -> Analysis:
    """Audio laden und als Analysis zurueckgeben. Stereo bleibt erhalten."""
    if hasattr(f, "seek"):
        f.seek(0)                                # UploadedFile mehrfach lesbar
    y, sr = librosa.load(f, sr=sr, mono=not stereo)
    y = np.asarray(y)
    if y.ndim == 1:
        y_st, y_mono = None, y
    else:
        y_st = y if y.shape[0] == 2 else np.vstack([y[0], y[0]])
        y_mono = librosa.to_mono(y)
    a = int(start * sr)
    b = int(end * sr) if end else len(y_mono)
    return Analysis(y_mono[a:b], sr,
                    y_st[:, a:b] if y_st is not None else None,
                    name or _source_name(f))
