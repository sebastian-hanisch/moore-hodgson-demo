"""Vehikel B "Werkstatt/Logistik" der Scheduling-Theorie-Linie: dieselben Aufträge wie Vehikel A (Bearbeitungszeit,
Fälligkeit, Gewicht), aber mit zwei Strukturmerkmalen, die im neutralen Vehikel bewusst fehlen:

1. **Ressourcenkonflikt**: jeder Auftrag gehört zu einer Familie (Werkzeug/Material); wechselt die Maschine
   zwischen zwei Familien, kostet das eine sequenzabhängige Rüstzeit (dieselbe Idee wie die ATCS-Regel in
   `warehouse-transfer-demo`, dort bereits Vepsalainen & Morton 1987 zitiert) - für DIESES Stück (Moore-Hodgson,
   1||ΣUⱼ) die relevante Erweiterung, siehe `mh_evaluation.py`: bleibt die Zahl der pünktlichen Aufträge gleich,
   wenn Rüstzeiten die tatsächliche Fertigstellungszeit verschieben?
2. **Schichten**: der Betrieb hat eine feste Schichtlänge; ein Auftrag, der nicht mehr in die laufende Schicht
   passt, rutscht in die nächste (eine Lücke in der Maschinenverfügbarkeit) - für dieses Stück nicht der
   Schwerpunkt (das war bereits `edd-scheduling-demo`s Experiment), bleibt aber Teil des Vehikels für spätere
   Stücke.

Kein neues Zufallsmodell für Bearbeitungszeit/Fälligkeit/Gewicht - dieselbe Erzeugung wie `spt_scenario.py`,
nur um Familie und Rüstmatrix ergänzt, damit ein Vergleich zwischen den Vehikeln nicht an unterschiedlich
verteilten Grunddaten hängt."""

from dataclasses import dataclass

import numpy as np

import mh_constants as C


@dataclass(frozen=True)
class LogistikInstance:
    n: int
    p: np.ndarray
    d: np.ndarray
    w: np.ndarray
    family: np.ndarray          # Familien-Index je Auftrag
    setup: np.ndarray           # (F, F) Rüstzeit-Matrix, 0 auf der Diagonale
    shift_length: int
    seed: int


def generate(n, seed, n_families=C.DEFAULT_N_FAMILIES, setup_time=C.DEFAULT_SETUP_TIME,
             shift_length=C.DEFAULT_SHIFT_LENGTH, tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD, p_min=C.P_MIN, p_max=C.P_MAX):
    rng = np.random.default_rng(seed)
    p = rng.integers(p_min, p_max + 1, size=n).astype(np.int64)
    total = int(p.sum())
    frac = (1 - tf - rdd / 2) + rdd * rng.random(n)
    d = np.maximum(np.round(total * frac).astype(np.int64), p)
    w = rng.choice(C.WEIGHT_VALUES, size=n, p=C.WEIGHT_PROBS).astype(np.int64)
    family = rng.integers(0, n_families, size=n).astype(np.int64)
    setup = np.full((n_families, n_families), setup_time, dtype=np.int64)
    np.fill_diagonal(setup, 0)
    return LogistikInstance(n, p, d, w, family, setup, shift_length, seed)
