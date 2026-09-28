"""Jede Zahl der App-Texte ist hier über die fünf festen Sweep-Instanzen (je drei Ketten) belegt. Positive UND
negative Aussagen: Moore-Hodgson ist beweisbar optimal auf dem neutralen Vehikel - UND hört auf, beweisbar
optimal zu sein, sobald Rüstzeiten (Vehikel Werkstatt/Logistik) dazukommen. Rechenzeiten nur als Größenordnung
geprüft."""

from functools import lru_cache

import pytest

import mh_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall -------------------------------------------------------------------------------------------------------------------------------


def test_standard_case_numbers():
    std = cfg()
    near(std["gap_edd"], 5.2, 3.0)
    near(std["gap_spt"], 2.2, 2.0)
    near(std["gap_random"], 6.5, 3.0)


def test_moore_hodgson_is_never_worse_than_edd_spt_or_random_on_any_swept_configuration():
    for n in (2, 5, 10, 20, 40, 60):
        row = cfg(n=n)
        assert row["gap_edd"] >= -1e-6 and row["gap_spt"] >= -1e-6 and row["gap_random"] >= -1e-6


# --- Optimalität gegen Brute-Force ---------------------------------------------------------------------------------------------------------------


def test_moore_hodgson_matches_brute_force_on_every_tested_size():
    rows = ev.optimality_check()
    assert all(r["match_rate"] == 1.0 for r in rows)


# --- Timing: n! gegen n log n -----------------------------------------------------------------------------------------------------------------


def test_brute_force_grows_far_faster_than_moore_hodgson():
    rows = ev.timing_sweep()
    small, large = rows[0], rows[-1]
    assert large["brute_force_seconds"] > small["brute_force_seconds"] * 100
    assert large["mh_seconds"] < 0.01


def test_brute_force_becomes_impractical_around_nine_jobs():
    rows = ev.timing_sweep()
    last = rows[-1]
    assert last["value"] == 9
    assert last["brute_force_seconds"] > 0.2


# --- Vehikel B: Rüstzeit-Härtetest ------------------------------------------------------------------------------------------------------------


def test_setup_gap_is_exactly_zero_without_setup_time():
    row = ev.setup_gap(setup_time=0)
    near(row["diff_mean"], 0.0, 1e-6)


@pytest.mark.parametrize("setup_time,diff,tol", [(5, 0.4, 1.0), (15, 1.0, 1.5), (30, 2.2, 2.0), (60, 2.2, 2.0)])
def test_setup_gap_numbers(setup_time, diff, tol):
    row = ev.setup_gap(setup_time=setup_time)
    near(row["diff_mean"], diff, tol)


def test_setup_gap_does_not_shrink_as_the_setup_time_grows():
    values = [ev.setup_gap(setup_time=s)["diff_mean"] for s in (0, 15, 30, 60)]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))


def test_moore_hodgson_on_the_logistik_vehicle_can_be_strictly_worse_than_the_true_optimum():
    """Die zentrale Vehikel-B-Aussage: Moore-Hodgson bleibt hier NICHT beweisbar optimal - ein echter Befund."""
    row = ev.setup_gap(setup_time=60)
    assert row["diff_max"] > 0.0
