# Sonic Artwork

Aus einer Audiodatei ein Cover machen — und zwar eines, das **dieses** Stück
abbildet und nicht bloß irgendeinen Visualizer.

```bash
streamlit run sonic_artwork.py          # Oberfläche
python -m sonicart --help               # dasselbe ohne Browser
```

## Was es macht

Das Stück wird einmal analysiert (Tempo, Tonart, Taktart, Abschnitte,
harmonische Dichte, Stereobild), und diese Messwerte steuern das Bild:

| Messwert | wirkt auf |
|---|---|
| Spektrum über die Zeit | Form der Spirale, Segmente, Gitter, Strata |
| Tonart (Quintenzirkel) | Grundfarbton der Palette; Moll dunkler und matter |
| harmonische Dichte | Farbtonspreizung der Palette |
| Taktart | Zähligkeit der Rose, ein Rasterfeld je Takt |
| Beats und Takte | Videolänge, Drehung, Puls |
| Lautheit, Rauhigkeit, Bandbreite | Bloom, Grain, chromatische Aberration |

### Bildmodi

| Modus | zeitbewusst | vollflächig | Vektor |
|---|---|---|---|
| Rose gespiegelt / roh | nein — mittelt über die Zeit | nein | ja |
| Spirale | ja — innen Anfang, außen Ende | nein | ja |
| Segmente | ja — Sektorbreite = Dauer des Abschnitts | nein | ja |
| HPSS-Zeit | ja | nein | ja |
| Kreis-Wellenform | ja | nein | ja |
| Lissajous (Stereo) | ja — echtes L/R-Goniometer | nein | nein |
| **Gitter** | ja — ein Feld je Takt | ja | ja |
| **Strata** | ja — Aufbau läuft nach rechts | ja | ja |

Die Rose-Modi mitteln das Spektrogramm über die Zeit. Zwei Stücke mit
ähnlichem Frequenzhaushalt sehen danach ähnlich aus, egal wie verschieden
sie aufgebaut sind — das ist so gewollt (Klangfingerabdruck), aber wenn das
Bild den *Verlauf* zeigen soll, sind Spirale, Segmente oder Gitter die
richtige Wahl. `tests/test_analysis.py` hält beides als Gegenprobe fest.

Zwei Normierungen machen die Modi erst brauchbar:

- **Bandnormierung** (`whiten`) für zeitaufgelöste Modi — ohne sie frisst der
  Bass den Tonwertumfang.
- **Rangnormierung** (`rank01`) für Gitter und Barcode — Takt-Mittelwerte
  liegen dicht beieinander, nach Min-Max landen fast alle in derselben
  Quantisierungsstufe und das Raster kippt tonwertlich zusammen.

**Nicht** für die Rose: die zeigt das zeitgemittelte Spektrum, und
Bandnormierung setzt jedes Bandmittel per Konstruktion auf denselben Wert —
sie würde genau das löschen, was der Modus darstellt. Dort hebt stattdessen
`tilt` die Höhen an: Kreisbelegung 9 % → 50 % bei erhaltener Streuung.

## Risodruck statt Leuchten

`sonicart/riso.py` behandelt das fertige Bild als Druck und gilt damit für
jeden Modus, auch für einen Mix:

- auf N flache Farben quantisiert, keine Verläufe
- ein Halbtonraster je Farbe mit eigenem Winkel (15°/75°/45°/0°) — gleiche
  Winkel auf zwei Platten geben Moiré
- Passerversatz je Platte, auflösungsrelativ
- **Multiply auf hellem Papier, Screen auf dunklem.** Multiply kann nur
  abdunkeln; helle Farbe auf dunklem Karton ist Deckfarbe, nicht Lasur.
- Papierfaser statt glatter Fläche

Sechs Farbsätze in `INK_SETS`. Der Druck läuft **nach** der Platzierung (damit
die ganze Fläche Papier wird) und **vor** Signet und Satz (damit Text Vollton
bleibt und lesbar ist). `Recipe.matte()` ist der matte Ausgangspunkt: Papier
statt Schwarz, Bloom und Aberration aus.

## Komposition

Neun Platzierungen statt der immer mittigen Scheibe: zentriert, goldener
Schnitt, angeschnitten, Drittel oben/unten, Raster 2×2 und 3×3 und weitere.

Beim Mischen mehrerer Modi gibt es zwei Verfahren. **Aufhellen** nimmt je
Pixel den helleren Wert — das funktioniert nur bei dünnen Ebenen: Gitter
deckt 55 % der Fläche, eine Rose 3,8 %, und dann bleibt vom Mix praktisch
nur das Gitter. **Überlagern** malt die Ebenen der Reihe nach übereinander,
jeweils nur dort, wo sie wirklich decken.

## Typografie

Fünf mitgelieferte Schriften unter SIL OFL in `sonicart/fonts/`, je nach
Rolle verschieden:

| Schrift | Art | wofür |
|---|---|---|
| Bebas Neue | Display, nur Versalien | große Titel |
| Space Grotesk | Grotesk, 4 Schnitte (variabel) | Artist, Allrounder |
| Archivo | Grotesk, 9 Schnitte (variabel) | Schweizer Anmutung |
| Space Mono / Bold | Monospace | Label, Katalognummer |

Vier fertige Paarungen als ein Klick; `python -m sonicart fonts` listet sie
auf. Eine eigene TTF/OTF überschreibt alle drei Rollen.

### Kontrastprüfung

Gemessen wird der WCAG-Kontrast gegen den **tatsächlichen** Untergrund —
nicht gegen die eingestellte Hintergrundfarbe, die unter einem Raster oder
einer Form gar nicht sichtbar sein muss. Modi: `Aus`, `Farbe umschalten`,
`Feld unterlegen`, `Nur warnen`.

Zwei Feinheiten, die den Unterschied machen:

- **Pro Zeile, nicht pro Block.** Über den ganzen Block gemessen trifft der
  Median die Mehrheitsfläche: der Titel liegt auf hellem Grund, die Zeile
  darunter auf einem dunklen Band — gemeldet wären glatte Werte, während
  eine Zeile unlesbar ist.
- **Ungünstigstes Perzentil, nicht Median.** Kreuzt eine Zeile die Kante
  einer Form, ist der Median weiterhin nutzlos. Bewertet wird das 5.
  Perzentil — nicht das Minimum, weil sonst die Papierlücken *innerhalb* des
  Halbtonrasters jede Fläche durchfallen ließen.

Umgeschaltet wird bevorzugt auf eine Farbe der Palette, nicht auf Weiß. Der
Block behält dabei **eine** Farbe: bewertet wird der schlechteste Fall über
alle Zeilen.

Die Prüfung misst Kontrast, nicht Unruhe. Text über einem stark gerasterten
Grund kann die Schwelle bestehen und trotzdem fusselig wirken — dafür ist
`Feld unterlegen` da.

## Reproduzierbarkeit

Jedes exportierte PNG trägt sein vollständiges Rezept in einem `tEXt`-Chunk.
Ein altes Cover in die App laden stellt die Einstellung wieder her, oder:

```bash
python -m sonicart recipe cover.png     # Rezept ausgeben
```

## Album statt Einzelbild

```bash
python -m sonicart album ./tracks -o ./cover --spread 40
```

Die Palettenstruktur bleibt über das ganze Release konstant, nur der Farbton
wandert pro Titel innerhalb der Spreizung — abgeleitet aus der Tonart. Die
Form kommt weiter aus dem Audio, also bleibt jeder Titel unterscheidbar.
Dazu ein Kontaktbogen und eine `album.json` mit allen Messwerten.

## Video

```bash
python -m sonicart video track.wav -o canvas.mp4 --duration 8 --pulse 0.6
```

Loopfähig heißt drei Dinge: die Länge wird auf einen Taktanfang gezogen, die
Drehung auf ganze Umdrehungen gezwungen, und die letzten Bilder blenden auf
das erste zurück. Gemessen in `tests/test_video.py`: der Sprung am
Schleifenpunkt fällt von über dem Doppelten eines normalen Bildabstands auf
unter einen.

## Footage

Handyclips und Screen-Recordings passen farblich nie zum Artwork desselben
Tracks. Der Reiter *Footage* legt sie auf dieselbe Palette: Aus den
Palettenstops entsteht eine Hald-CLUT der Stufe 8 — eine Gradient Map, die
jeden Eingangsfarbwert durch die Palettenfarbe seiner OKLab-Helligkeit
ersetzt. Der Farbton des Originals fällt weg, Struktur und
Helligkeitsverlauf bleiben. ffmpeg wendet die Tabelle mit `haldclut` an.

Interpoliert wird zwischen den Stops in OKLCH, nicht in sRGB, und die
Rückrechnung senkt bei Bedarf die Chroma (`oklch_to_hex(fit=True)`), statt
Kanäle abzuschneiden — sonst verbiegt sich die Helligkeitskurve genau da,
wo die Palette kräftig wird.

Regler: Stärke (0 = Original, 1 = ganz auf die Palette), Kontrast vor dem
Mapping, Korn, Zielformat (9:16 · 1:1 · 16:9, Center-Crop statt Stauchung)
und Audio (Originalton · Track-Audio · stumm). Weil ein voller Render
Minuten kostet, steht davor ein einzelner gemappter Frame als Standbild;
Vorschau und Export teilen sich denselben Filtergraph, damit das Standbild
nicht lügen kann.

Zwei Dinge stehen als Kommentar im Code, weil beide beim Lesen falsch
wirken: Bei ffmpegs `blend` ist der *erste* Eingang die obere Ebene — steht
dort das Original, ist der Stärkeregler invertiert. Und die Bitrate braucht
einen Deckel, weil x264 das Korn sonst originalgetreu kodiert und eine
Minute damit auf über ein Gigabyte wächst.

## Druck

Die Palettenprüfung meldet Farben außerhalb des Offset-Farbraums und zeigt,
wie sie gedruckt zurückkommen. `#7b2ff7` aus dem Preset *Korrend* liegt bei
Faktor 2,2. Näherung ohne ICC-Profil — ein Warnsignal, kein Proof.

## Oberfläche

Vier Reiter im Hauptbereich: Bild · Video · Footage · Album. Im Bild-Reiter
links die Vorschau, rechts das Steuerpult mit fünf Unterreitern (Form ·
Farbe · Druck · Text · Aufbau), darunter der Export in Leserichtung. Die Seitenleiste
trägt nur, was die Sitzung eröffnet: Audiodatei, Live-Vorschau, Sperrflächen,
Rezept, eigene Dateien.

Drei Streamlit-Eigenheiten sind dabei umschifft, jede mit einem Test
abgesichert:

1. **`st.rerun()` mitten im Aufbau** verwirft den Zustand aller Widgets
   darunter. Ein Klick auf „Preset laden" setzte so stillschweigend
   Schriftgröße und Rasterweite zurück, während Regler davor überlebten.
   Alle Zustandsänderungen laufen deshalb über `on_click`/`on_change`.
2. **Nicht gerenderte Widgets verlieren ihren Zustand.** Risodruck
   abschalten und wieder einschalten setzte Papier und Rasterweite auf die
   Vorgaben. Ein Spiegel hält die Werte — er speichert nur, was sichtbar
   war, und spielt nur zurück, was vorher ausgeblendet war.
3. **Ein Widget übernimmt einen Zustandswert nur, wenn dieser im selben
   Durchlauf gesetzt wurde, in dem das Widget entsteht.** Vor dem Datei-Upload
   existiert der Hauptbereich nicht; danach zeigte jedes Widget seinen eigenen
   Vorgabewert — „Rose gespiegelt" im Auswahlfeld, während das Bild im
   Gitter-Modus gerechnet wurde, und schwarze Farbfelder, die beim ersten
   Klick das Bild wirklich schwarz machten. `init_state()` setzt deshalb jeden
   Wert in jedem Durchlauf neu.

## Aufbau

```
sonicart/
  analysis.py     Audio einmal analysieren, Ergebnisse cachen
  palette.py      OKLCH, Palettengenerator, LUT, Druck-Gamut
  render/         Bildmodi + Zeichenflächen
  riso.py         Druckschicht (Halbton, Passer, Papier)
  effects.py      Post-Effekte (alpha-bewusst, auflösungsrelativ)
  compose.py      Modi mischen, Bild einsetzen, Platzierung
  typography.py   Schriften, Textsatz, Kontrastprüfung
  fonts/          mitgelieferte Schriften (SIL OFL)
  artwork.py      Rezept + Pipeline
  export.py       Social-Formate, SVG, PNG-Metadaten, Abgabeprüfung
  ffmpeg.py       Pfad zur Binary, Aufruf, Laufzeit auslesen
  video.py        Canvas und Reel
  footage.py      Fremdmaterial auf die Palette legen (Hald-CLUT)
  album.py        Serie statt Einzelbild
  cli.py          Kommandozeile
sonic_artwork.py  Streamlit-Oberfläche
```

Die Analyse ist vom Rendern getrennt: `Analysis` rechnet jede teure Größe
höchstens einmal pro Datei. Vorher lief `librosa.effects.hpss` bei jedem
Reglerausschlag neu (~8,5 s bei drei Minuten); jetzt liegt die vollständige
Analyse bei rund 4,8 s, und danach sind die Regler frei.

## Entwicklung

```bash
uv venv --python 3.11 .venv && direnv allow .
uv pip install -r requirements-dev.txt
pytest -q                                # 258 Tests
```

`requirements.txt` hat Ober- und Untergrenzen, `requirements.lock` die exakt
getesteten Versionen.
