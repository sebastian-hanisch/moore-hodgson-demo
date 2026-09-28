# Moore-Hodgson – möglichst wenige Aufträge verspäten – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-moore-hodgson-demo.streamlit.app/)**

Drittes Stück der **Klassische-Scheduling-Theorie-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch
– Operations Research und Machine Learning": $n$ Aufträge mit Bearbeitungszeit $p_j$ und Fälligkeit $d_j$ auf
**einer** Maschine, Ziel ist die ANZAHL verspäteter Aufträge zu minimieren ($1||\sum U_j$) – anders als $L_{\max}$
im vorigen Stück zählt hier jeder verspätete Auftrag gleich viel, egal wie spät.

**Einordnung in die Linie:** Moore-Hodgson (Moore 1968) ist der erste "zweistufige" Algorithmus dieser Linie –
nicht nur sortieren wie SPT/EDD, sondern sortieren UND gezielt streichen. Aufträge nach EDD einplanen; sobald
einer verspätet würde, wird NICHT dieser Auftrag entfernt, sondern der mit der GRÖSSTEN Bearbeitungszeit unter
allen bisher eingeplanten – ein zweiter, unabhängig interessanter Mechanismus, der genau erklärt, warum "streiche
den Auslöser" die falsche Intuition wäre.
```
SPT (1||ΣCⱼ, Vertauschungsargument)                                              [Stück 1]
EDD (1||Lmax, dasselbe Beweismuster, andere Zielfunktion)                        [Stück 2]
 ├─ Moore-Hodgson (1||ΣUⱼ, EDD + gezieltes Streichen)                            [dieses Stück]
 ├─ WSPT / Smith's Rule (1||ΣwⱼCⱼ, verallgemeinert SPT mit Gewichten)             [Folgestück]
 ├─ Johnson-Regel (F2||Cmax, zweite Maschine)                                     [Folgestück]
 ├─ LPT (Pm||Cmax, parallele Maschinen)                                           [Folgestück]
 └─ Job Shop (Konvergenzpunkt: Reihenfolge UND Maschinenwahl)                     [Folgestück]
```

Ergebnis in Kürze: **Moore-Hodgson trifft auf jeder getesteten Instanz (n = 2 bis 9) exakt das Minimum der
Vollaufzählung.** Bei 20 Aufträgen hält Moore-Hodgson im Mittel **5,2 Aufträge mehr** pünktlich als EDD (die für
dieses Ziel falsche Regel) und **6,5 mehr** als eine zufällige Reihenfolge.
**Der ehrliche Bruch:** auf dem Werkstatt/Logistik-Vehikel (Rüstzeiten zwischen Familien) bleibt Moore-Hodgson bei
Rüstzeit 0 exakt optimal (Konsistenz-Test), aber der Abstand zum echten Minimum wächst mit der Rüstzeit –
bis zu **2,2 zusätzlich verspätete Aufträge** im Mittel bei 60 Minuten je Familienwechsel.

| Frage | Ergebnis (Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds) |
|---|---|
| Standardfall (20 Aufträge) | ✅ Moore-Hodgson hält **5,2** mehr Aufträge pünktlich als EDD, **2,2** mehr als SPT, **6,5** mehr als Zufall |
| **Beweis gegen Vollaufzählung** | ✅ **100 %** Trefferquote bei n = 2 bis 9 |
| **Rechenzeit** | ➖ Vollaufzählung bei n = 9 bereits über 3000 ms, Moore-Hodgson bei rund 0,01 ms |
| **Vehikel Werkstatt/Logistik** | ❌ Rüstzeit 0/5/15/30/60 Minuten: **0/0,4/1,0/2,2/2,2** zusätzlich verspätete Aufträge – ab Rüstzeit > 0 NICHT mehr beweisbar optimal |

## Was die Demo zeigt

1. **Moore-Hodgson in Aktion** (Schritt-Slider): **Aufträge** (Bearbeitungszeit und Fälligkeit) → **Einplanen**
   (Regler "eingeplante Aufträge", pünktlich/verspätet farblich markiert) → **Ergebnis** (vollständige Aufteilung).
2. **Was die Streichregel bringt:** Moore-Hodgson, EDD (falsche Regel hier), SPT, zufällige Reihenfolge,
   Vollaufzählungs-Gegenprobe (n ≤ 9); auf dem Werkstatt/Logistik-Vehikel zusätzlich Moore-Hodgson mit
   Rüstzeiten gegen das echte Optimum mit Rüstzeiten.
3. **📐 Sweep** über Aufträge, Fristen-Anteil und -Streuung.
4. **🔬 Experimente auf Abruf:** Vollaufzählung gegen Moore-Hodgson über n = 2 bis 9 (Beweis-Check); Rechenzeit
   $n!$ gegen $n \log n$; Rüstzeit-Härtetest auf dem Werkstatt/Logistik-Vehikel.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an".

Regler: Aufträge (2–60), **Vehikel** (Neutral/Werkstatt-Logistik – bei Werkstatt zusätzlich Rüstzeit und Anzahl
Familien), Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲, steuert nur die zufällige Vergleichsreihenfolge).

## Die zwei Vehikel (aus den vorigen Stücken übernommen, wortgleich für die ganze Linie)

- **Neutral** (`mh_scenario.py`): $n$ Aufträge mit Bearbeitungszeit $p_j \sim U(1, 100)$ und Fälligkeit
  $d_j \sim U(P \cdot (1-TF-RDD/2),\, P \cdot (1-TF+RDD/2))$ (Potts & Van Wassenhove 1982/1985).
- **Werkstatt/Logistik** (`mh_scenario_logistik.py`): dieselben Aufträge, aber in Familien mit sequenzabhängiger
  Rüstzeit beim Wechsel (wie in Stück 1) – hier die für Moore-Hodgson relevante Erweiterung, weil sie die
  tatsächliche Fertigstellungszeit verschiebt, ohne dass der Algorithmus davon weiß. Rüstzeit 0 kollabiert
  exakt zum neutralen Vehikel (per Test belegt).

## Modell und Verfahren

- **Instanz** (`mh_scenario.py`, `mh_scenario_logistik.py`): wortgleich aus den vorigen Stücken übernommen.
- **Moore-Hodgson** (`mh_algorithm.py`): EDD einplanen, bei Verspätung den größten bisher eingeplanten Auftrag
  streichen, $O(n \log n)$. Dazu die Brute-Force-Vollaufzählung als unabhängige Gegenprobe, und die
  Rüstzeit-Variante für das Werkstatt/Logistik-Vehikel.
- **Auswertung** (`mh_evaluation.py`): Kennzahlen, Sweeps, Optimalitäts- und Timing-Messreihe, Rüstzeit-Härtetest.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "Moore-Hodgson bleibt eine gute Heuristik, auch mit Rüstzeiten"** – **bestätigt, aber nicht
  garantiert**: der Abstand zum echten Optimum wächst mit der Rüstzeit (0,4 zusätzlich verspätete Aufträge bei
  5 Minuten, 2,2 bei 60 Minuten im Mittel über 5 Instanzen bei n = 8) und einzelne Instanzen liegen höher (bis
  zu 3 zusätzlich verspätete Aufträge im Einzelfall, siehe Test) – bei kleinen ganzzahligen Zielwerten wie ΣUⱼ
  wirkt sich jede zusätzliche Verspätung relativ stark aus.
- **Die Vollaufzählung ist die einzige echte Gegenprobe**, praktisch nur bis $n \approx 9$ nutzbar.
- **Synthetische Instanzen:** Bearbeitungszeiten gleichverteilt, Fälligkeiten nach dem TF/RDD-Schema, keine
  Präzedenzen, ein Auftrag = eine Operation.

## Verifikation

- **Beweis gegen unabhängige Vollaufzählung:** für jede getestete Instanzgröße (n = 2 bis 9) und jede der 5
  festen Instanzen trifft Moore-Hodgson exakt das Minimum der Vollaufzählung – 100 % Trefferquote.
- **Rüstzeit-Variante gegen unabhängige Vollaufzählung** und **Konsistenz-Test**: Rüstzeit 0 liefert exakt
  dieselbe Aufteilung wie das neutrale Vehikel.
- **Handrechnung:** zwei kleine, von Hand nachgerechnete Instanzen bestätigen die Streichregel exakt – auch den
  Fall, in dem NICHT der Auftrag gestrichen wird, der die Verspätung ausgelöst hat, sondern ein früher
  eingeplanter mit größerer Bearbeitungszeit.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Standardfall, Optimalitäts-Trefferquote, Rechenzeit,
  Rüstzeit-Härtetest; positive **und** negative Aussagen); alle 5 Presets geprüft; AppTest-Rauchtests
  (Voreinstellung, jedes Preset, jeder Schritt auf beiden Vehikeln, Würfel-Knöpfe, Permalink-Grenzen inkl.
  ungültigem Vehikel, Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente, 🚧 Grenzen, Mathe |
| `mh_algorithm.py` | Moore-Hodgson, Brute-Force-Gegenprobe, Rüstzeit-Variante |
| `mh_scenario.py` | Vehikel Neutral |
| `mh_scenario_logistik.py` | Vehikel Werkstatt/Logistik (Familien, Rüstzeit-Matrix) |
| `mh_constants.py` | Konstanten, Presets |
| `mh_evaluation.py` | Kennzahlen, Sweeps, Optimalitäts- und Timing-Messreihe, Rüstzeit-Härtetest |
| `mh_presets.py`, `mh_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Beweis gegen Vollaufzählung, Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
