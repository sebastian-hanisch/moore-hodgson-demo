"""Moore-Hodgson (Moore 1968) für 1||ΣUⱼ: n Aufträge auf einer Maschine, Ziel ist die ANZAHL verspäteter
Aufträge zu minimieren (jeder verspätete Auftrag zählt 1, egal wie spät - anders als Lmax im vorigen Stück, das
die GRÖSSE der schlimmsten Verspätung misst).

Algorithmus (Hodgsons O(n log n)-Variante aus Moores 1968er-Arbeit): Aufträge nach EDD einplanen; wird ein
Auftrag dabei verspätet (die laufende Fertigstellungszeit überschreitet seine Fälligkeit), wird NICHT dieser
Auftrag verworfen, sondern der mit der GRÖSSTEN Bearbeitungszeit unter allen bisher eingeplanten (der teuerste
Platz im pünktlichen Teil) - er wandert in die Menge der ohnehin verspäteten Aufträge, die verbleibenden rücken
nach. Das maximiert die Zahl der pünktlichen Aufträge, weil jede erzwungene Streichung die laufende Zeit um den
größtmöglichen Betrag verkürzt."""

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    order: np.ndarray          # pünktliche Aufträge (EDD-Reihenfolge) gefolgt von den verspäteten (EDD-Reihenfolge)
    completion: np.ndarray     # Fertigstellungszeiten in dieser Reihenfolge
    late: np.ndarray           # bool je Auftrag (Originalindex): verspätet?
    num_late: int


def _edd_order(d):
    return np.argsort(d, kind="stable")


def moore_hodgson(p, d):
    n = len(p)
    edd = _edd_order(d)
    scheduled = []              # Originalindizes, in EDD-Reihenfolge
    t = 0.0
    late = []
    for j in edd:
        scheduled.append(int(j))
        t += float(p[j])
        if t > float(d[j]) + 1e-9:
            k = max(scheduled, key=lambda idx: p[idx])
            scheduled.remove(k)
            t -= float(p[k])
            late.append(k)
    order = np.array(scheduled + late)
    completion = np.cumsum(np.asarray(p)[order].astype(np.float64))
    late_mask = np.zeros(n, dtype=bool)
    late_mask[late] = True
    return Result(order, completion, late_mask, len(late))


def evaluate_order(p, d, order):
    """Anzahl verspäteter Aufträge für eine BELIEBIGE Reihenfolge (nicht nur die Moore-Hodgson-Aufteilung) -
    für die Brute-Force-Gegenprobe und Vergleichsregeln."""
    order = np.asarray(order)
    completion = np.cumsum(np.asarray(p)[order].astype(np.float64))
    late_in_order = completion > np.asarray(d)[order].astype(np.float64) + 1e-9
    late_mask = np.zeros(len(p), dtype=bool)
    late_mask[order[late_in_order]] = True
    return Result(order, completion, late_mask, int(late_mask.sum()))


def edd_order_result(p, d):
    """EDD ohne die Moore-Hodgson-Streichung - die falsche Regel für DIESES Ziel (Kontrast)."""
    return evaluate_order(p, d, _edd_order(d))


def spt_order_result(p, d):
    return evaluate_order(p, d, np.argsort(p, kind="stable"))


def random_order_result(p, d, n, rng):
    order = np.arange(n)
    rng.shuffle(order)
    return evaluate_order(p, d, order)


def brute_force_optimal(p, d):
    n = len(p)
    best_order, best_num_late = None, n + 1
    for perm in itertools.permutations(range(n)):
        order = np.array(perm)
        num_late = evaluate_order(p, d, order).num_late
        if num_late < best_num_late:
            best_num_late, best_order = num_late, order
    return evaluate_order(p, d, best_order)


# --- Mit Rüstzeiten (Vehikel B: Werkstatt/Logistik) ------------------------------------------------------------


def completion_times_with_setup(p, family, setup, order):
    t = 0.0
    out = np.empty(len(order), dtype=np.float64)
    prev_family = None
    for idx, j in enumerate(order):
        if prev_family is not None:
            t += float(setup[prev_family, family[j]])
        t += float(p[j])
        out[idx] = t
        prev_family = family[j]
    return out


def evaluate_order_with_setup(p, d, family, setup, order):
    order = np.asarray(order)
    completion = completion_times_with_setup(p, family, setup, order)
    late_in_order = completion > np.asarray(d)[order].astype(np.float64) + 1e-9
    late_mask = np.zeros(len(p), dtype=bool)
    late_mask[order[late_in_order]] = True
    return Result(order, completion, late_mask, int(late_mask.sum()))


def brute_force_optimal_with_setup(p, d, family, setup):
    n = len(p)
    best_order, best_num_late = None, n + 1
    for perm in itertools.permutations(range(n)):
        order = np.array(perm)
        num_late = evaluate_order_with_setup(p, d, family, setup, order).num_late
        if num_late < best_num_late:
            best_num_late, best_order = num_late, order
    return evaluate_order_with_setup(p, d, family, setup, best_order)
