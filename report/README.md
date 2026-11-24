# FDM und Balanced Truncation: Bericht und Overleaf

`bericht.pdf` ist der kompilierte deutsche Bericht. Er behandelt die Methoden,
Fehlernormen und -schranken, getrennte Aufbau-/Simulationskosten sowie adaptive
Raum-/Rangwahl und adaptive Zeitintegration.

## In Overleaf verwenden

Am einfachsten `Overleaf_FDM_BT.zip` als neues Overleaf-Projekt hochladen.
Alternativ `overleaf/report.tex` kopieren und `overleaf/Bilder/` hochladen.
Der Ordner muss `Bilder` heißen und direkt neben `report.tex` liegen.
Als Hauptdokument `report.tex` und als Compiler XeLaTeX wählen.
Der Text ist vollständig eingebettet: keine weiteren TeX-Teildateien, kein
Python und kein shell-escape nötig. `Bilder.zip` enthält nur den Bilderordner.

Die zehn eingebundenen Abbildungsgruppen sind PDF-Vektorgrafiken.
PNG-Vorschauen stehen unter `figures/`.

## Reproduzieren

Vom Hauptverzeichnis des Repositories:

```bash
python -m pip install -r report/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python report/experiments.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python report/plots.py
python report/prepare_overleaf.py
cd report
tectonic bericht.tex
```

Die Hauptmessung führt keine referenzgesteuerte Raum-/Rangwahl aus.
`bound_adaptivity.py` wählt ausschließlich anhand von Differenzen zweier
Auflösungen, Sicherheitsfaktor und Hankel-Restsumme. Die Sinusreihenreferenz
wird anschließend zur Fehlerbewertung verwendet.

- `data/runtime.csv`: Medianzeiten und Quartile für Aufbau und Online-Schritte.
- `data/rank_study.csv`: Rang, Feld-/Frequenzfehler und BT-Schranken.
- `data/bounds_adaptive.csv`: erfolgreiche schätzer-/schrankengesteuerte Auswahlen.
- `data/bounds_failures.csv`: vom Budgettest nicht freigegebene Ziele.
- `data/adaptive_time.csv`: feste/adaptive Schritte, Zeiten und Arbeitszähler.
- `data/environment.json`: Softwareversionen, Threadzahl und Referenzparameter.
- `data/manifest.json`: Prüfsummen der Programme.

Die FDM-Raumadaption ist global (automatisch gewählter Gitterabstand), keine
lokale AMR. BT adaptiert den Rang bei festem feinem Ausgangsmodell; ausgeschöpfte
Diskretisierungsbudgets werden gemeldet. Lokale und gekoppelte h/r-Adaption
werden erklärt, aber nicht als gemessene Varianten ausgegeben.

Zeitadaptivität verwendet denselben Euler-Regler mit Schrittverdopplung und
Kontrolle im physikalischen Ausgangsraum für beide Modelle. FLOP-Zahlen sind
arithmetische Strukturmodelle mit dokumentierten Auslassungen.

```bash
PYTHONPATH=report python -m unittest report/test_report.py \
  test_balancing.py test_comparison.py test_gramians.py test_simulation.py
```

`prepare_overleaf.py` erzeugt das eigenständige Overleaf-Dokument aus den
modularen Quelldateien und Messwerttabellen. Änderungen am Text daher in
`bericht.tex`, `theory.tex`, `results.tex`, `adaptivity.tex` oder
`reproduction.tex` vornehmen und danach den Export neu erzeugen.
