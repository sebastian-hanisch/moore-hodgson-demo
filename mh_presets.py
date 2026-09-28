"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Buttons (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import mh_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "chain_seed_input": SettingSpec("cseed", int, 0, 0, C.SEED_MAX),
    "vehicle_radio": SettingSpec("vehicle", str, C.DEFAULT_VEHICLE, None, None),
    "setup_time_slider": SettingSpec("setup", int, C.DEFAULT_SETUP_TIME, C.SETUP_TIME_MIN, C.SETUP_TIME_MAX),
    "n_families_slider": SettingSpec("fam", int, C.DEFAULT_N_FAMILIES, C.N_FAMILIES_MIN, C.N_FAMILIES_MAX),
}
PRESET_KEYS = {"n": "n_slider", "seed": "seed_input", "chain_seed": "chain_seed_input", "vehicle": "vehicle_radio",
               "setup_time": "setup_time_slider", "n_families": "n_families_slider"}
STEPS = {"n_slider": C.N_STEP}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if state_key == "vehicle_radio" and value not in C.VEHICLE_LABELS:
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            lo = SETTING_SPECS[key].lo
            st.session_state[key] = int(lo + round((st.session_state[key] - lo) / step) * step)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


def randomize_chain_seed():
    st.session_state["chain_seed_input"] = random.randint(0, C.SEED_MAX)
