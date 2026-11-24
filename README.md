# Model_reduction_balanced_truncation
**Model Reduction via Balanced Truncation**  Implementation of Balanced Truncation for continuous-time linear time-invariant (LTI) systems, using controllability and observability Gramians, Lyapunov equations, and Hankel singular values to construct reduced-order models with stability preservation and theoretical H∞ error bounds.

## Schritt 1: Das volle Wärmeleitungssystem verstehen

```bash
python -m pip install -r requirements.txt
python visualize_system.py
```

`outputs/heat_system_sine.png` zeigt Gitter, räumliche Anregung, Zustand,
Besetzungsmuster der Systemmatrix, Eingang und Ausgang.
`outputs/heat_system_sine.html` kann direkt im Browser geöffnet werden:
Die Flächen lassen sich drehen; Zeitschieber und Abspielknopf zeigen den Verlauf.
Die HTML-Datei enthält alle benötigten Daten und benötigt keine Internetverbindung.
Zustandsfeld, Errormap und Zeitmarkierungen werden gemeinsam neu gezeichnet.
Beim Abspielen wartet jeder Schritt auf die abgeschlossene Darstellung;
bei schnellem Verschieben des Zeitschiebers wird die letzte Auswahl bevorzugt.
Zeit und maximaler Fehler stehen zusätzlich direkt über der Errormap.

Die Errormap läuft synchron zum Zustandsfeld. Sie zeigt den absoluten Unterschied
`|z_h - z_h/2|` zu einer zweiten Rechnung mit halbiertem Gitterabstand
(`d_ref = 2*d+1`, standardmäßig 35 statt 17 innere Punkte pro Richtung).
Verglichen wird an identischen Orten und Zeiten, ohne räumliche Interpolation.
Beide Rechnungen verwenden denselben Zeitschritt. Das ist eine Schätzung der
räumlichen Abweichung; die Referenz ist ebenfalls numerisch und keine exakte Lösung.
Die Rand- und Anfangswerte stimmen exakt überein, daher ist die Errormap dort null.
Eine feste Farbskala macht die Zeitpunkte vergleichbar; die Kurve darunter zeigt
den jeweils größten Fehlerbetrag über die inneren Punkte. Die Darstellung
verwendet absolute Differenzen, damit keine Division durch Null auftritt.

Zusätzlich meldet das Terminal die maximale Änderung bei halbiertem Zeitschritt
auf dem ursprünglichen Raumgitter, als separate Prüfung des zeitlichen Einflusses.
Raumgitter und Zeitschritt sind fest; das Verfahren ist nicht adaptiv.

Die räumliche Diskretisierung stammt unverändert aus `helpers.fdm_2d_poisson(d)`:

- `d`: innere Gitterpunkte pro Raumrichtung; Gitterabstand `h = 1/(d+1)`.
- `n = d**2`: Länge des Zustandsvektors; Randwerte sind fest null.
- `A`: diskreter Laplaceoperator (5-Punkt-Stern), Form `(n, n)`.
- `B`: Werte von `sin(3*pi*x1/2)*sin(4*pi*x2/3)`, Form `(n, 1)`.
- `C1 = h**2 * ones((1,n))`: Quadratur des räumlichen Integrals.
- `C2 = I`: Beobachtung des gesamten diskreten Zustands, direkt als `x` verfügbar.

Simuliert wird `x' = A*x + B*u(t)`, `x(0)=0`, standardmäßig mit
`u(t)=10*sin(2*pi*t)` auf `[0,1]`. Farben zeigen vorzeichenbehaftete Werte,
keine absoluten Temperaturen. Die Zustandsskala bleibt über alle Zeiten fest.
Die dargestellte Anfangsansicht ist ein Zeitpunkt mit großem Zustandsbetrag;
über den Schieber lässt sich auch der Nullzustand bei `t=0` ansehen.

Der implizite Euler in `helpers.py` berechnet
`(I-dt*A)*x[k+1] = x[k] + dt*B*u(t[k+1])` und verwendet eine einzige
Faktorisierung der konstanten linken Matrix. Er speichert alle Zeitschritte;
die Animation zeigt höchstens 81 ausgewählte Zeitpunkte.

Zum Erkunden einer abgeschalteten Quelle und anderer Gitter:

```bash
python visualize_system.py --input pulse
python visualize_system.py --d 8 --steps 500
python -m unittest test_simulation.py
```

Optionaler Browsercheck für macOS mit installiertem Google Chrome:
`python -m pip install websocket-client`, danach
`python check_browser.py outputs/heat_system_sine.html` (oder die Pulsdatei).
Er prüft zusätzlich das tatsächlich gezeichnete Bild sowie Abspielen und Pause
in einer isolierten Headless-Sitzung und speichert einen Screenshot neben der HTML-Datei.

`pulse` setzt `u=10` für `t<0.2*tf` und danach `u=0`: Das Feld klingt anschließend ab.
Die Vorlage `hinf_norm` ist noch offen; `bt_in_action.py`
kann erst nach deren Ergänzung vollständig ausgeführt werden.

## Schritt 2: Gramians für die Balancierung

`helpers.lyapunov_gramians(A, B, C)` berechnet für ein stabiles reelles System
die Steuerbarkeits-Gramian `P` und Beobachtbarkeits-Gramian `Q`:

```python
from helpers import fdm_2d_poisson, lyapunov_gramians

A, B, C, X, Y = fdm_2d_poisson(17)
P, Q = lyapunov_gramians(A, B, C)
```

Es gelten `A@P + P@A.T + B@B.T = 0` und
`A.T@Q + Q@A + C.T@C = 0`. SciPy erwartet auf der rechten Seite
`-B@B.T` beziehungsweise `-C.T@C`. Für die dichten Lyapunov-Löser werden
die kleinen dünnbesetzten Systemmatrizen in dichte Arrays umgewandelt.
Mit `C = np.eye(A.shape[0])` erhält man die zweite Beobachtungsvariante.

`balancing` ruft diesen ersten Schritt auf und verwendet anschließend die
vorgegebenen Faktoren `R = lrcf(P, trunctol)` und `L = lrcf(Q, trunctol)`.
Anschließend berechnet sie die SVD von `L.T @ R`, entfernt numerisch verschwindende
Hankel-Singulärwerte und bildet die beiden Transformationen. Die Rückgabe
`Ab, Bb, Cb, Sigma` enthält das numerisch minimale balancierte System;
die eigentliche Wahl der reduzierten Dimension `r` erfolgt außerhalb der Funktion.

Wir verwenden die Konvention `x_bal = T @ x`, `x ≈ T_tilde @ x_bal`.
Dann lauten die Projektionen `Ab = T @ A @ T_tilde`, `Bb = T @ B` und
`Cb = C @ T_tilde`. Bei rechteckigen Transformationen ist `T_tilde`
keine gewöhnliche Matrixinverse; `T @ T_tilde = I` gilt im behaltenen Raum.

Prüfen: `python -m unittest test_gramians.py test_balancing.py test_simulation.py test_comparison.py`.

## Schritt 3: Grobe FDM und Balanced Truncation vergleichen

```bash
python compare_methods.py
open outputs/comparison_sine.html
```

Die Ansicht zeigt oben die feine FDM-Referenz (17×17 = 289 Zustände),
die grobe FDM (8×8 = 64 Zustände) und das auf 4 Zustände reduzierte BT-Modell.
BT wird für die vollständige Zustandsbeobachtung `C2=I` konstruiert;
`Cb[:, :r] @ x_reduced` rekonstruiert das Feld auf dem feinen Gitter.
Alle Modelle verwenden denselben Eingang, Nullanfangswerte und denselben
impliziten Euler mit derselben Zeitschrittweite.

Darunter stehen die absoluten Errormaps nebeneinander. Beide vergleichen mit
derselben feinen FDM-Referenz, an genau den 8×8 inneren Punkten des groben
Gitters. Feinere Werte werden ohne Interpolation an diesen Punkten ausgewählt.
Damit beschreiben die Zahlen nicht den Fehler zur exakten PDE-Lösung und
erfassen auch keine Maxima zwischen den gemeinsamen Vergleichspunkten.
Die Errormaps verwenden dieselbe, zeitlich feste Farbskala.

Die dritte Karte zeigt `|e_FDM| - |e_BT|`: Positive Werte bedeuten lokal
kleinere BT-Abweichung, negative kleinere FDM-Abweichung. Die Kurven unten
vergleichen die maximalen Fehler auf den gemeinsamen Punkten über der Zeit
(logarithmische Fehlerachse; der exakte Nullwert bei t=0 wird dort ausgelassen).
Alle Felder, Karten, Maximalwerte und Zeitmarkierungen werden synchron aktualisiert.

Zusätzlich entstehen `outputs/comparison_sine.png` als Übersicht,
`outputs/comparison_sine_errors.png` für den gesonderten Errormap-Vergleich
und `outputs/comparison_sine_metrics.json` mit den maximalen Abweichungen
über alle simulierten Zeitschritte und gemeinsamen inneren Gitterpunkte.

```bash
python compare_methods.py --input pulse
python compare_methods.py --r 8
# Gleiche Zustandszahl: 2×2 innere FDM-Punkte und 4 BT-Zustände.
python compare_methods.py --d-coarse 2 --r 4 --output-dir outputs/equal_states
```

Das grobe Gitter muss im feinen enthalten sein: `d_coarse+1` teilt `d_fine+1`.
Die unterschiedlichen Modellgrößen stehen in jeder Ansicht; die Standardwahl
64 gegenüber 4 ist kein Vergleich bei gleichem Rechenaufwand.
Aufbaukosten der Gramians und Balancierung sind nicht als Laufzeiten ausgewiesen.
Beide Verfahren verwenden feste Gitter beziehungsweise feste Modellordnung;
Adaptivität ist in diesem Vergleich nicht implementiert.

Der Browsercheck unterstützt auch `python check_browser.py outputs/comparison_sine.html`
und prüft dabei beide Errormaps und beide Zeitmarkierungen gemeinsam.

## Bericht, Messungen und Overleaf

- [PDF-Bericht](report/bericht.pdf): Theorie, Stärken/Grenzen, Konvergenz,
  Errormaps, Laufzeiten und adaptive Vergleiche mit Arbeits- und Schrittzählern.
- [Overleaf-Paket](report/Overleaf_FDM_BT.zip): eine eigenständige `report.tex`
  und der benötigte Ordner `Bilder/`; keine zusätzlichen TeX-Teildateien nötig.
- [Einzelne report.tex](report/overleaf/report.tex) und [Bilderordner](report/overleaf/Bilder).
- [Anleitung zur Reproduktion](report/README.md).

Die automatische Raum-/Rangwahl wird durch Richardson-Schätzer und die
BT-Restsummenschranke gesteuert. Die unabhängige Sinusreihenreferenz validiert
erst danach. Adaptive Zeitversuche zählen akzeptierte und verworfene Schritte,
lineare Lösungen und Faktorisierungen direkt. Die ausgewiesenen FLOPs sind ein
erläutertes Strukturmodell, keine Hardware-Counter.
