"""Vehikel A "Neutral" der Scheduling-Theorie-Linie (wortgleich aus `spt-scheduling-demo`/`edd-scheduling-demo`
übernommen, dort per Test gegen eingefrorene Werte geprüft): n Aufträge mit Bearbeitungszeit, Fälligkeit und
Gewicht auf EINER Maschine. Bearbeitungszeit UND Fälligkeit sind für dieses Stück (Moore-Hodgson, 1||ΣUⱼ) die
verwendeten Größen; das Gewicht ist für spätere Stücke (WSPT, gewichtete Verspätung) vorbereitet.

Fälligkeiten nach dem TF/RDD-Schema (Tardiness-Faktor, Range der Fälligkeiten): `dⱼ ~ U(P·(1-TF-RDD/2),
P·(1-TF+RDD/2))` mit `P` = Summe aller Bearbeitungszeiten (die Fertigstellungszeit des letzten Auftrags EGAL in
welcher Reihenfolge). Per WebSearch verifiziert (Potts & Van Wassenhove 1982/1985, Standard-Schema der
Scheduling-Literatur für Fristen-Testinstanzen - typische Werte TF ∈ {0.2,...,0.8}, RDD ∈ {0.2,...,1.0})."""

from dataclasses import dataclass

import numpy as np

import mh_constants as C


@dataclass(frozen=True)
class Instance:
    n: int
    p: np.ndarray        # Bearbeitungszeiten
    d: np.ndarray        # Fälligkeiten (für Folgestücke)
    w: np.ndarray        # Gewichte (für Folgestücke)
    seed: int
    tf: float
    rdd: float


def generate(n, seed, tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD, p_min=C.P_MIN, p_max=C.P_MAX):
    rng = np.random.default_rng(seed)
    p = rng.integers(p_min, p_max + 1, size=n).astype(np.int64)
    total = int(p.sum())
    frac = (1 - tf - rdd / 2) + rdd * rng.random(n)
    d = np.maximum(np.round(total * frac).astype(np.int64), p)
    w = rng.choice(C.WEIGHT_VALUES, size=n, p=C.WEIGHT_PROBS).astype(np.int64)
    return Instance(n, p, d, w, seed, tf, rdd)
