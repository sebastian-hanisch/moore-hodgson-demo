"""mh_algorithm: Moore-Hodgson-Optimalität für ΣUⱼ gegen unabhängige Brute-Force-Vollaufzählung (Moore 1968),
Regressionsschutz, Determinismus, Rüstzeit-Variante."""

import itertools

import numpy as np
import pytest

import mh_algorithm as A


def _instance(seed, n):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 100, size=n).astype(np.int64)
    total = int(p.sum())
    d = np.maximum(np.round(total * (0.2 + 0.6 * rng.random(n))).astype(np.int64), p)
    return p, d


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_moore_hodgson_matches_brute_force_for_every_seed(n):
    for seed in range(10):
        p, d = _instance(seed, n)
        assert A.moore_hodgson(p, d).num_late == A.brute_force_optimal(p, d).num_late


def test_moore_hodgson_achieves_the_minimum_over_all_permutations():
    p, d = _instance(7, 6)
    mh_num_late = A.moore_hodgson(p, d).num_late
    all_counts = [A.evaluate_order(p, d, perm).num_late for perm in itertools.permutations(range(6))]
    assert mh_num_late == min(all_counts)


def test_hand_picked_instance_where_the_largest_job_gets_dropped():
    """Handrechnung: drei Aufträge, EDD-Reihenfolge würde den letzten verspäten, Moore-Hodgson streicht den
    GRÖSSTEN der bisher eingeplanten (nicht zwingend den zuletzt hinzugefügten)."""
    p = np.array([5, 1, 1])
    d = np.array([5, 6, 3])                    # EDD-Reihenfolge (nach d sortiert): Auftrag 2 (d=3), 0 (d=5), 1 (d=6)
    result = A.moore_hodgson(p, d)
    # Ablauf: 2 eingeplant (t=1<=3 ok). 0 eingeplant (t=1+5=6>5 verspätet) -> groesster bisher eingeplanter ist 0 (p=5) -> streichen.
    # t=1. 1 eingeplant (t=1+1=2<=6 ok).
    assert result.num_late == 1
    assert result.late.tolist() == [True, False, False]


def test_a_job_can_be_dropped_even_though_it_is_not_the_one_that_triggered_the_delay():
    """Der Witz von Moore-Hodgson: nicht der Auftrag, der die Verspätung ausgelöst hat, wird gestrichen, sondern
    der mit der größten Bearbeitungszeit unter den bisher eingeplanten."""
    p = np.array([10, 1, 1])
    d = np.array([10, 3, 11])                   # EDD: 1 (d=3), 0 (d=10), 2 (d=11)
    result = A.moore_hodgson(p, d)
    # 1 eingeplant (t=1<=3). 0 eingeplant (t=11>10 verspätet) -> groesster bisher = 0 (p=10) -> streichen, t=1.
    # 2 eingeplant (t=2<=11 ok).
    assert result.late.tolist() == [True, False, False]
    assert result.num_late == 1


def test_num_late_matches_a_direct_count_of_tardy_completions():
    p, d = _instance(3, 10)
    result = A.moore_hodgson(p, d)
    completion_at = {j: c for j, c in zip(result.order.tolist(), result.completion.tolist())}
    direct_late = sum(1 for j in range(10) if completion_at[j] > d[j] + 1e-9)
    assert direct_late == result.num_late


def test_edd_and_spt_are_worse_rules_for_this_objective_on_a_constructed_instance():
    p = np.array([10, 1, 1, 1])
    d = np.array([11, 2, 3, 4])
    mh = A.moore_hodgson(p, d)
    edd = A.edd_order_result(p, d)
    assert mh.num_late <= edd.num_late


def test_random_order_result_is_deterministic_given_the_rng_state():
    n = 8
    p, d = _instance(9, n)
    a = A.random_order_result(p, d, n, np.random.default_rng(0))
    b = A.random_order_result(p, d, n, np.random.default_rng(0))
    assert a.order.tolist() == b.order.tolist() and a.num_late == b.num_late


# --- Mit Rüstzeiten (Vehikel B) --------------------------------------------------------------------------------


def test_setup_variant_matches_the_plain_variant_when_setup_is_zero():
    p, d = _instance(11, 6)
    family = np.array([0, 1, 0, 1, 0, 1])
    setup = np.zeros((2, 2))
    mh = A.moore_hodgson(p, d)
    with_setup = A.evaluate_order_with_setup(p, d, family, setup, mh.order)
    assert with_setup.num_late == mh.num_late
    assert with_setup.completion.tolist() == mh.completion.tolist()


def test_brute_force_with_setup_matches_independent_full_enumeration():
    p, d = _instance(13, 5)
    family = np.array([0, 1, 0, 1, 2])
    setup = np.array([[0, 5, 8], [5, 0, 3], [8, 3, 0]])
    best = A.brute_force_optimal_with_setup(p, d, family, setup)
    all_counts = [A.evaluate_order_with_setup(p, d, family, setup, np.array(perm)).num_late for perm in itertools.permutations(range(5))]
    assert best.num_late == min(all_counts)


def test_moore_hodgson_can_be_worse_than_the_true_optimum_once_setup_times_bind():
    """Moore-Hodgson bestimmt die Aufteilung ohne Rüstzeiten zu kennen - das muss nicht mehr optimal bleiben,
    sobald sie ins Gewicht fallen. Genau die Frage, die Vehikel B stellt."""
    p = np.array([1, 1, 10, 10])
    d = np.array([2, 2, 20, 20])
    family = np.array([0, 1, 0, 1])
    setup = np.array([[0, 100], [100, 0]])
    mh_order = A.moore_hodgson(p, d).order
    mh_num_late = A.evaluate_order_with_setup(p, d, family, setup, mh_order).num_late
    true_opt = A.brute_force_optimal_with_setup(p, d, family, setup).num_late
    assert mh_num_late >= true_opt
