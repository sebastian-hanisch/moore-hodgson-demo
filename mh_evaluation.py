"""Auswertung der Moore-Hodgson-Demo: Moore-Hodgson gegen EDD (die richtige Regel für Lmax, aber die falsche für
ΣUⱼ), SPT und zufällige Reihenfolgen, gegen die Brute-Force-Vollaufzählung (nur kleine n), und das
Vehikel-B-Experiment (bleibt die Zahl pünktlicher Aufträge gleich, wenn Rüstzeiten dazukommen)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import mh_algorithm as A
import mh_constants as C
import mh_scenario as S
import mh_scenario_logistik as SL


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    chain_seed: int = 0
    tf: float = C.DEFAULT_TF
    rdd: float = C.DEFAULT_RDD
    vehicle: str = C.DEFAULT_VEHICLE
    setup_time: int = C.DEFAULT_SETUP_TIME
    n_families: int = C.DEFAULT_N_FAMILIES


@lru_cache(maxsize=512)
def instance(n, seed, tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD):
    return S.generate(n, seed, tf=tf, rdd=rdd)


@lru_cache(maxsize=512)
def logistik_instance(n, seed, n_families, setup_time, tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD):
    return SL.generate(n, seed, n_families=n_families, setup_time=setup_time, tf=tf, rdd=rdd)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    mh: object
    edd: object
    spt: object
    random_mean: float
    random_runs: int
    optimal: object

    @property
    def gap_edd(self):
        return self.edd.num_late - self.mh.num_late

    @property
    def gap_spt(self):
        return self.spt.num_late - self.mh.num_late

    @property
    def gap_random(self):
        return self.random_mean - self.mh.num_late

    @property
    def mh_matches_optimum(self):
        return self.optimal is not None and self.mh.num_late == self.optimal.num_late


def analyse(settings, random_draws=20):
    """Wertet Moore-Hodgson auf dem gewählten Vehikel aus - Neutral oder Werkstatt/Logistik (Rüstzeit beim
    Familienwechsel zählt mit). Die Streichregel selbst bleibt in beiden Fällen dieselbe (kennt keine
    Rüstzeiten) - nur die BEWERTUNG der resultierenden Reihenfolge (welche Aufträge dadurch tatsächlich
    verspätet sind, und damit auch die Vollaufzählung) wechselt mit dem Vehikel, damit die Haupt-Kennzahlen
    ehrlich widerspiegeln, was auf dem gewählten Vehikel passiert (statt nur in einer Zusatzbox)."""
    inst = instance(settings.n, settings.seed, settings.tf, settings.rdd)
    p, d = inst.p, inst.d

    if settings.vehicle == "logistik":
        linst = logistik_instance(settings.n, settings.seed, settings.n_families, settings.setup_time, settings.tf, settings.rdd)
        family, setup = linst.family, linst.setup

        def ev(order):
            return A.evaluate_order_with_setup(p, d, family, setup, order)

        optimal = A.brute_force_optimal_with_setup(p, d, family, setup) if settings.n <= C.BRUTE_FORCE_MAX_N else None
    else:
        def ev(order):
            return A.evaluate_order(p, d, order)

        optimal = A.brute_force_optimal(p, d) if settings.n <= C.BRUTE_FORCE_MAX_N else None

    mh = ev(A.moore_hodgson(p, d).order)
    edd = ev(A.edd_order_result(p, d).order)
    spt = ev(A.spt_order_result(p, d).order)
    rng = np.random.default_rng(settings.chain_seed)
    random_counts = [ev(A.random_order_result(p, d, settings.n, rng).order).num_late for _ in range(random_draws)]
    return Analysis(settings, inst, mh, edd, spt, float(np.mean(random_counts)), random_draws, optimal)


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch))
            rows.append({"gap_edd": a.gap_edd, "gap_spt": a.gap_spt, "gap_random": a.gap_random})
    out = {k: _mean(rows, k) for k in rows[0]}
    out["n_runs"] = len(rows)
    return out


SWEEP_VALUES = {"n": (2, 5, 10, 20, 40, 60), "tf": (0.0, 0.2, 0.4, 0.6, 0.8), "rdd": (0.2, 0.4, 0.6, 0.8, 1.0)}
SWEEP_LABELS = {"n": "Aufträge", "tf": "Fristen-Anteil (TF)", "rdd": "Fristen-Streuung (RDD)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def optimality_check(ns=C.BRUTE_FORCE_SWEEP_N, seeds=C.SWEEP_SEEDS):
    rows = []
    for n in ns:
        matches = 0
        for seed in seeds:
            inst = instance(n, seed)
            mh_num_late = A.moore_hodgson(inst.p, inst.d).num_late
            opt_num_late = A.brute_force_optimal(inst.p, inst.d).num_late
            if mh_num_late == opt_num_late:
                matches += 1
        rows.append({"value": n, "match_rate": matches / len(seeds)})
    return rows


def timing_sweep(ns=C.BRUTE_FORCE_SWEEP_N, seed=C.DEFAULT_SEED):
    rows = []
    for n in ns:
        inst = instance(n, seed)
        t0 = time.perf_counter()
        A.brute_force_optimal(inst.p, inst.d)
        t_bf = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(100):
            A.moore_hodgson(inst.p, inst.d)
        t_mh = (time.perf_counter() - t0) / 100
        rows.append({"value": n, "brute_force_seconds": t_bf, "mh_seconds": t_mh})
    return rows


def setup_gap(n=8, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES, setup_time=C.DEFAULT_SETUP_TIME):
    """Vehikel-B-Härtetest: Moore-Hodgson (bestimmt die Aufteilung ohne Rüstzeiten zu kennen) gegen die echte
    Optimallösung MIT Rüstzeiten (Brute-Force, deshalb kleines n)."""
    diffs = []
    for seed in seeds:
        linst = SL.generate(n, seed, n_families=n_families, setup_time=setup_time)
        mh_order = A.moore_hodgson(linst.p, linst.d).order
        mh_num_late = A.evaluate_order_with_setup(linst.p, linst.d, linst.family, linst.setup, mh_order).num_late
        opt_num_late = A.brute_force_optimal_with_setup(linst.p, linst.d, linst.family, linst.setup).num_late
        diffs.append(mh_num_late - opt_num_late)
    return {"diff_mean": float(np.mean(diffs)), "diff_min": float(np.min(diffs)), "diff_max": float(np.max(diffs)), "n_runs": len(diffs)}


def setup_gap_sweep(setup_times=(0, 5, 15, 30, 60), n=8, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES):
    return [{"value": s, **setup_gap(n=n, seeds=seeds, n_families=n_families, setup_time=s)} for s in setup_times]
