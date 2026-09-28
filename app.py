"""Moore-Hodgson - eine Warteschlange, die möglichst wenige Aufträge verspätet - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Konzepte-Linie "Klassische Scheduling-Theorie": n Aufträge auf einer Maschine, Ziel ist die
ANZAHL verspäteter Aufträge zu minimieren (1||ΣUⱼ) - anders als Lmax im vorigen Stück zählt hier jeder verspätete
Auftrag gleich viel, egal wie spät. Moore-Hodgson (Moore 1968, Hodgsons O(n log n)-Variante) ist dafür beweisbar
optimal: EDD einplanen, und sobald ein Auftrag verspätet würde, den mit der GRÖSSTEN Bearbeitungszeit unter den
bisher eingeplanten streichen - nicht zwingend den, der die Verspätung ausgelöst hat. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import mh_constants as C
from mh_evaluation import Settings, SWEEP_LABELS, analyse, instance, optimality_check, run_config, setup_gap, setup_gap_sweep, sweep, timing_sweep
from mh_presets import KEPT, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_chain_seed, randomize_seed, seed_widget, sync_query_params
from mh_visualization import build_schedule, build_setup_gap, build_sweep, build_timing

st.set_page_config(page_title="Moore-Hodgson – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _optimality():
    return optimality_check()


@st.cache_data(show_spinner=False)
def _timing():
    return timing_sweep()


@st.cache_data(show_spinner=False)
def _setup_gap_sweep(n, n_families):
    return setup_gap_sweep(n=n, n_families=n_families)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("⏱️ Moore-Hodgson – möglichst wenige Aufträge verspäten")
st.markdown(
    r"""
**n Aufträge auf einer Maschine, jeder mit einer Fälligkeit – gesucht ist die Reihenfolge, die die ANZAHL
verspäteter Aufträge minimiert** ($1||\sum U_j$, mit $U_j = 1$, wenn Auftrag $j$ verspätet ist, sonst $0$ - egal
WIE spät). Die Antwort ist **Moore-Hodgson** (Moore 1968): Aufträge nach Fälligkeit einplanen (EDD); sobald einer
dabei verspätet würde, wird NICHT dieser Auftrag gestrichen, sondern der mit der GRÖSSTEN Bearbeitungszeit unter
allen bisher eingeplanten – er wandert in die Menge der ohnehin verspäteten Aufträge. Das maximiert die Zahl der
pünktlichen Aufträge, weil jede erzwungene Streichung die laufende Zeit um den größtmöglichen Betrag verkürzt.
"""
)
st.caption(
    "Drittes Stück der Konzepte-Linie „Klassische Scheduling-Theorie“ - dieselben zwei Vehikel wie in den ersten "
    "beiden Stücken ([spt-scheduling-demo](https://sebastianhanisch-spt-scheduling-demo.streamlit.app/), "
    "[edd-scheduling-demo](https://sebastianhanisch-edd-scheduling-demo.streamlit.app/)): **Neutral** (Aufträge "
    "mit Bearbeitungszeit und Fälligkeit) und **Werkstatt/Logistik** (dieselben Aufträge, aber in Familien mit "
    "Rüstzeit beim Wechsel) - der Umschalter ist in der Seitenleiste."
)

with st.expander("So funktioniert Moore-Hodgson", expanded=True):
    st.markdown(
        r"""
1. **EDD einplanen.** Aufträge aufsteigend nach Fälligkeit $d_j$ nacheinander einplanen, laufende Fertigstellungszeit mitführen.
2. **Bei Verspätung streichen.** Überschreitet die laufende Zeit die Fälligkeit des gerade eingeplanten Auftrags, wird der Auftrag mit der GRÖSSTEN Bearbeitungszeit unter allen bisher eingeplanten aus dem Plan genommen - das kann auch ein früher eingeplanter sein, nicht nur der aktuelle.
3. **Was gemessen wird.** Die Zahl verspäteter Aufträge einer Reihenfolge gegenüber Moore-Hodgson; für kleine $n$ zusätzlich die Vollaufzählung als unabhängige Gegenprobe.
4. **Die Grenze der Annahme.** Moore-Hodgson kennt keine Rüstzeiten. Das Vehikel „Werkstatt/Logistik“ prüft, was passiert, wenn Familienwechsel Zeit kosten.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_jobs = st.slider("Aufträge", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
                        help=f"Anzahl der Aufträge. Bis {C.BRUTE_FORCE_MAX_N} läuft die Vollaufzählung aller n! Reihenfolgen live mit.")
    vehicle = st.radio("Vehikel", list(C.VEHICLE_LABELS), key="vehicle_radio", format_func=lambda k: C.VEHICLE_LABELS[k],
                        help="Neutral: keine Rüstzeiten. Werkstatt/Logistik: dieselben Aufträge, in Familien mit Rüstzeit beim Wechsel.")
    if vehicle == "logistik":
        seed_widget("setup_time_slider")
        setup_time = st.slider("Rüstzeit je Familienwechsel (Minuten)", *bounds("setup_time_slider"), key="setup_time_slider",
                                help="0 Minuten kollabiert exakt zum neutralen Vehikel (siehe Test/Messreihe).")
        st.session_state[KEPT["setup_time_slider"]] = setup_time
        seed_widget("n_families_slider")
        n_families = st.slider("Auftragsfamilien", *bounds("n_families_slider"), key="n_families_slider")
        st.session_state[KEPT["n_families_slider"]] = n_families
    else:
        setup_time = int(st.session_state.get(KEPT["setup_time_slider"], C.DEFAULT_SETUP_TIME))
        n_families = int(st.session_state.get(KEPT["n_families_slider"], C.DEFAULT_N_FAMILIES))
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    chain_seed = st.number_input("Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
                                  help="Steuert nur die zufällige Vergleichsreihenfolge - Moore-Hodgson selbst ist deterministisch.")
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed)

sync_query_params({"n_slider": int(n_jobs), "seed_input": int(seed), "chain_seed_input": int(chain_seed), "vehicle_radio": vehicle,
                    "setup_time_slider": int(setup_time), "n_families_slider": int(n_families)})

settings = Settings(int(n_jobs), int(seed), int(chain_seed), vehicle=vehicle, setup_time=int(setup_time), n_families=int(n_families))
with st.spinner("Rechne..."):
    a = _analysis(settings)
inst = a.inst
p, d = inst.p, inst.d
data_key = settings

# --- Moore-Hodgson in Aktion ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Moore-Hodgson in Aktion")
STEP_LABELS = {1: "1 · Aufträge", 2: "2 · Einplanen", 3: "3 · Ergebnis"}
if "mh_step" not in st.session_state or st.session_state.get("mh_step_owner") != data_key:
    st.session_state["mh_step"] = 1
    st.session_state["mh_step_owner"] = data_key
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="mh_step", format_func=lambda s: STEP_LABELS[s])

if step == 2:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        upto = st.slider("Eingeplante Aufträge", 1, int(n_jobs), value=int(n_jobs), key="mh_upto")
else:
    upto = int(n_jobs)

view_slot = st.empty()
with view_slot.container():
    if step == 1:
        st.markdown(f"**{n_jobs} Aufträge, unsortiert** (Bearbeitungszeit in Minuten)")
        st.bar_chart({"Bearbeitungszeit": p.tolist(), "Fälligkeit": d.tolist()})
    elif step == 2:
        st.markdown(f"**Aufteilung nach {upto} von {n_jobs} Aufträgen** (rot = verspätet, blau = pünktlich)")
        st.plotly_chart(build_schedule(p, a.mh.order, a.mh.completion, a.mh.late, upto=upto), width="stretch", key=f"s2_sched_{upto}")
    else:
        st.markdown("**Vollständige Aufteilung: pünktlich (blau) gefolgt von verspätet (rot)**")
        st.plotly_chart(build_schedule(p, a.mh.order, a.mh.completion, a.mh.late), width="stretch", key="s3_sched")

if step == 1:
    st.caption(f"Bearbeitungszeiten zwischen {int(p.min())} und {int(p.max())}, Fälligkeiten zwischen {int(d.min())} und {int(d.max())} Minuten (Seed {seed}).")
elif step == 2:
    gap_note = " Lücken zwischen Balken sind Rüstzeit bei einem Familienwechsel; sie können einen Auftrag zusätzlich verspäten." if vehicle == "logistik" else ""
    st.caption(f"Rot markierte Aufträge sind bereits endgültig als verspätet erkannt - nicht zwingend der zuletzt hinzugefügte, sondern der mit der größten Bearbeitungszeit.{gap_note}")
else:
    st.caption(f"Moore-Hodgson: {a.mh.num_late} von {n_jobs} Aufträgen verspätet. EDD (falsche Regel hier): {a.edd.num_late} verspätet.")

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Streichregel bringt")
vehicle_note = " Auf dem Werkstatt/Logistik-Vehikel zählt die Rüstzeit beim Familienwechsel mit - Moore-Hodgson kennt sie nicht, alle Zahlen hier berücksichtigen sie trotzdem." if vehicle == "logistik" else ""
st.caption(f"**Abstand:** wie viele Aufträge MEHR bei dieser Reihenfolge verspätet sind, verglichen mit Moore-Hodgson. Moore-Hodgson selbst ist deterministisch - nur die Zufalls-Vergleichsreihenfolge streut.{vehicle_note}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Moore-Hodgson (verspätet)", f"{a.mh.num_late} von {n_jobs}", help="Die Zielgröße: Anzahl verspäteter Aufträge in Moore-Hodgson-Aufteilung, auf dem gewählten Vehikel.")
m2.metric("EDD (falsche Regel hier)", f"+{_fmt_int(a.gap_edd)}", delta_color="off", help="EDD optimiert Lmax, nicht ΣUⱼ - hier zum Vergleich.")
m3.metric(f"Zufällige Reihenfolge (Mittel über {a.random_runs})", f"+{a.gap_random:.1f}", delta_color="off")
if a.optimal is not None:
    m4.metric("Vollaufzählung (Gegenprobe)", "trifft Moore-Hodgson exakt" if a.mh_matches_optimum else "WEICHT AB", delta_color="off",
              help=f"Alle {n_jobs}! Reihenfolgen durchprobiert (auf dem gewählten Vehikel) - unabhängige Bestätigung bzw. Gegenprobe.")
else:
    m4.metric("Vollaufzählung", f"erst ab n ≤ {C.BRUTE_FORCE_MAX_N}", delta_color="off")

if a.optimal is not None and not a.mh_matches_optimum:
    if vehicle == "neutral":
        st.error("⚠️ Moore-Hodgson weicht von der Vollaufzählung ab - das wäre ein Fehler im Beweis oder in der Implementierung, bitte melden.")
    else:
        diff = a.mh.num_late - a.optimal.num_late
        st.warning(f"⚠️ Moore-Hodgson ist hier NICHT mehr optimal: {diff} zusätzlich verspätete Aufträge gegenüber dem echten Optimum MIT Rüstzeiten.")
else:
    tail = " (auch mit Rüstzeiten - bei dieser Instanz trifft Moore-Hodgson trotzdem das Optimum, das ist nicht garantiert)" if vehicle == "logistik" and a.optimal is not None else ""
    st.success(f"✅ Moore-Hodgson hält {_fmt_int(a.gap_edd)} Aufträge mehr pünktlich als EDD und {a.gap_random:.1f} mehr als eine zufällige Reihenfolge im Mittel - bei dieser Zielfunktion beweisbar die beste überhaupt{tail}.")

st.markdown("---")

# --- Sweeps -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt der Vorsprung von der Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
if st.button("Sweep über 5 feste Instanzen berechnen (dauert wenige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {sweep_param}
if sweep_param in st.session_state.get("sweep_done", set()):
    rows_sweep = _sweep(sweep_param, Settings())
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel über 5 feste Instanzen (Seeds 100000–100004) mit je drei Zufalls-Ketten für die Vergleichsreihenfolge.")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Stimmt der Beweis wirklich? Vollaufzählung gegen Moore-Hodgson")
if st.button("Vollaufzählung über n = 2 bis 9 berechnen (dauert etwa 5 Sekunden)", key="opt_start"):
    st.session_state["opt_on"] = True
if st.session_state.get("opt_on"):
    rows_opt = _optimality()
    st.table({"Aufträge": [r["value"] for r in rows_opt], "Trefferquote": [f"{r['match_rate']:.0%}" for r in rows_opt]})
    st.caption("Für jede Instanzgröße 5 feste Instanzen: Moore-Hodgson gegen die Vollaufzählung aller n! Reihenfolgen.")

st.markdown("---")

st.subheader("🔬 Wie teuer ist eine Vollaufzählung wirklich?")
if st.button("Rechenzeit für n = 2 bis 9 messen (dauert etwa 3 Sekunden)", key="timing_start"):
    st.session_state["timing_on"] = True
if st.session_state.get("timing_on"):
    rows_t = _timing()
    st.plotly_chart(build_timing(rows_t), width="stretch", key="timing_chart")
    last = rows_t[-1]
    st.caption(f"Bei {last['value']} Aufträgen braucht die Vollaufzählung bereits {last['brute_force_seconds']*1000:.0f} ms, Moore-Hodgson {last['mh_seconds']*1000:.3f} ms.")

st.markdown("---")

st.subheader("🔬 Werkstatt/Logistik: bleibt die Zahl pünktlicher Aufträge gleich, wenn Rüstzeiten dazukommen?")
if st.button("Rüstzeit von 0 bis 60 Minuten durchfahren (dauert wenige Sekunden)", key="setup_start"):
    st.session_state["setup_on"] = True
if st.session_state.get("setup_on"):
    rows_s = _setup_gap_sweep(min(int(n_jobs), C.BRUTE_FORCE_MAX_N), int(n_families))
    st.plotly_chart(build_setup_gap(rows_s), width="stretch", key="setup_chart")
    st.caption("Moore-Hodgson bestimmt die Aufteilung weiterhin ohne die Rüstzeit beim Familienwechsel zu kennen; verglichen mit der echten Optimallösung MIT Rüstzeiten (Vollaufzählung, deshalb kleine Instanz). Bei Rüstzeit 0 fallen beide exakt zusammen.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Bearbeitungszeit hängt nicht von der Reihenfolge ab** | Sobald Rüstzeiten zwischen Auftragsfamilien dazukommen (Vehikel „Werkstatt/Logistik“), ist Moore-Hodgson nicht mehr beweisbar optimal - der Abstand zum echten Minimum wächst mit der Rüstzeit. | Kein direkter Nachfolger in dieser Linie |
| **Jeder verspätete Auftrag zählt gleich viel** | Zählt nur die GRÖSSE der schlimmsten Verspätung statt ihrer Anzahl, ist EDD (nicht Moore-Hodgson) die richtige Regel. | **EDD** (bereits gebaut) |
| **Alle Aufträge sind gleich wichtig** | Mit unterschiedlichen Gewichten reicht die reine Anzahl nicht mehr - das Ziel wird zu gewichteter Verspätung. | **Gewichtete Verspätung** (Folgestück) |
| **Es gibt nur eine Maschine** | Mit mehreren Maschinen wird aus einer Sortierfrage eine Zuordnungs- UND Reihenfolgefrage. | **Johnson-Regel, LPT, Job Shop** (Folgestücke) |
"""
)
st.caption(
    "Vorige Stücke dieser Linie: [spt-scheduling-demo](https://sebastianhanisch-spt-scheduling-demo.streamlit.app/) "
    "(SPT, 1||ΣCⱼ) und [edd-scheduling-demo](https://sebastianhanisch-edd-scheduling-demo.streamlit.app/) (EDD, 1||Lmax)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem** ($1||\sum U_j$): $n$ Aufträge mit Bearbeitungszeit $p_j$ und Fälligkeit $d_j$ auf einer Maschine.
$U_j = 1$, wenn die Fertigstellungszeit $C_j > d_j$ ist, sonst $0$. Gesucht: die Reihenfolge, die $\sum_j U_j$
minimiert.

**Satz (Moore 1968).** Der folgende Algorithmus minimiert $\sum_j U_j$:
1. Aufträge aufsteigend nach $d_j$ einplanen (EDD), laufende Fertigstellungszeit $t$ mitführen.
2. Überschreitet $t$ nach dem Einplanen von Auftrag $j$ dessen Fälligkeit $d_j$, wird der Auftrag mit der größten
   Bearbeitungszeit unter allen BISHER eingeplanten aus dem Plan entfernt (in die Menge der verspäteten
   Aufträge verschoben) und $t$ entsprechend verringert.
3. Am Ende sind die eingeplanten Aufträge alle pünktlich (in EDD-Reihenfolge), alle übrigen verspätet.

**Warum das optimal ist (Skizze).** Für eine feste Menge pünktlicher Aufträge ist EDD die einzige Reihenfolge,
die alle pünktlich hält, falls das überhaupt möglich ist (Beweis wie bei EDD/Lmax). Die Streichregel maximiert
bei jedem Schritt die verbleibende "Zeitreserve" für die noch nicht eingeplanten Aufträge, indem sie den
teuersten Platz im pünktlichen Teil freigibt - das lässt sich induktiv zu einem Optimalitätsbeweis ausbauen
(Moore 1968; ein moderner, kurzer Beweis: Mor & Mosheiov 2021, *A simple proof of the Moore-Hodgson Algorithm*).

**Kennzahl.** Abstand zu Moore-Hodgson $=$ Anzahl verspäteter Aufträge einer Reihenfolge minus
$\sum U_j(\text{Moore-Hodgson})$. Für $n \le 9$ zusätzlich die Vollaufzählung als unabhängige Gegenprobe.

**Grenzen.** (1) Rüstzeiten verletzen die Voraussetzung "Bearbeitungszeit hängt nicht von der Reihenfolge ab" -
Moore-Hodgson bleibt dann nur eine gute Heuristik (Vehikel B). (2) Ohne Gewichte zählt jeder verspätete Auftrag
gleich viel - gewichtete Verspätung verallgemeinert das.

Implementiert in `mh_algorithm.py` (Moore-Hodgson, Brute-Force-Gegenprobe, Rüstzeit-Variante),
`mh_scenario.py`/`mh_scenario_logistik.py` (die zwei Vehikel, wortgleich aus den vorigen Stücken übernommen),
`mh_evaluation.py` (Kennzahlen, Sweeps, Optimalitäts- und Timing-Messreihe, Rüstzeit-Härtetest).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
