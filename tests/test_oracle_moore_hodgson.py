"""Orakel-Test: Moore-Hodgson gegen zwei unabhängige exakte Wege (Teilmengenaufzählung mit Jackson-Zulässigkeit,
Lawler-Moore-DP über die Gesamtzeit) und die Rüstzeit-Variante gegen eigene Auswertung/Permutationsaufzählung."""

import itertools
import random

import numpy as np

import mh_algorithm as A


def _opt_subset(p, d):
    n = len(p)
    idx = sorted(range(n), key=lambda j: (d[j], j))
    best = 0
    for mask in range(1 << n):
        t = 0
        ok = True
        for j in idx:
            if mask >> j & 1:
                t += int(p[j])
                if t > d[j]:
                    ok = False
                    break
        if ok:
            best = max(best, bin(mask).count("1"))
    return n - best


def _lawler_moore(p, d):
    n = len(p)
    idx = sorted(range(n), key=lambda j: (d[j], j))
    total = int(sum(p))
    neg = -10 ** 9
    f = [neg] * (total + 1)
    f[0] = 0
    for j in idx:
        g = f[:]
        for t in range(total + 1):
            if f[t] < 0:
                continue
            nt = t + int(p[j])
            if nt <= d[j] and nt <= total:
                g[nt] = max(g[nt], f[t] + 1)
        f = g
    return n - max(f)


def _late(p, d, order, fam=None, setup=None):
    t = late = 0
    prev = None
    for j in order:
        if fam is not None and prev is not None:
            t += int(setup[prev][fam[j]])
        t += int(p[j])
        prev = fam[j] if fam is not None else None
        late += t > d[j]
    return late


def test_moore_hodgson_is_optimal_against_subset_enumeration_and_dp():
    rng = random.Random(5)
    for _ in range(250):
        n = rng.randint(1, 8)
        hi = rng.choice([2, 4, 20, 100])
        p = np.array([rng.randint(1, hi) for _ in range(n)])
        tot = int(p.sum())
        d = np.array([rng.randint(1, max(1, tot // rng.choice([1, 2, 4]) + 1)) for _ in range(n)])
        res = A.moore_hodgson(p, d)
        opt = _opt_subset(p, d)
        assert opt == _lawler_moore(p, d)
        assert res.num_late == opt
        assert sorted(res.order.tolist()) == list(range(n))
        assert _late(p, d, res.order.tolist()) == res.num_late == int(res.late.sum())
        assert np.allclose(res.completion, np.cumsum(p[res.order]))


def test_setup_variant_evaluation_and_enumeration_match_independent_count():
    rng = random.Random(9)
    for _ in range(60):
        n = rng.randint(1, 5)
        p = np.array([rng.randint(1, 30) for _ in range(n)])
        d = np.array([rng.randint(1, 80) for _ in range(n)])
        fam = np.array([rng.randint(0, 2) for _ in range(n)])
        s = rng.choice([0, 10, 40])
        setup = np.array([[0 if a == b else s for b in range(3)] for a in range(3)])
        order = rng.sample(range(n), n)
        assert A.evaluate_order_with_setup(p, d, fam, setup, order).num_late == _late(p, d, order, fam, setup)
        best = min(_late(p, d, perm, fam, setup) for perm in itertools.permutations(range(n)))
        assert A.brute_force_optimal_with_setup(p, d, fam, setup).num_late == best
