"""Plotly-Abbildungen der Moore-Hodgson-Demo: Gantt-artiges Balkendiagramm (pünktlich/verspätet eingefärbt),
Sweeps, Timing. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

ON_TIME_COLOR = "#4c78a8"
LATE_COLOR = "#e45756"
EDD_COLOR = "#54a24b"
RANDOM_COLOR = "#7f7f7f"
SETUP_COLOR = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_schedule(p, order, completion, late_mask, upto=None):
    """`completion`: die TATSÄCHLICHEN Fertigstellungszeiten (aus `evaluate_order`/`evaluate_order_with_setup`) -
    auf dem Werkstatt/Logistik-Vehikel enthalten sie Lücken durch Rüstzeiten, sichtbar als Leerraum."""
    order = np.asarray(order)
    upto = len(order) if upto is None else upto
    starts = np.asarray(completion) - p[order]
    fig = go.Figure()
    shown_on_time = shown_late = False
    for i in range(upto):
        j = order[i]
        is_late = bool(late_mask[j])
        color = LATE_COLOR if is_late else ON_TIME_COLOR
        name = "verspätet" if is_late else "pünktlich"
        showlegend = not (shown_late if is_late else shown_on_time)
        if is_late:
            shown_late = True
        else:
            shown_on_time = True
        fig.add_trace(go.Bar(x=[float(p[j])], y=["Maschine"], base=[float(starts[i])], orientation="h",
                              marker=dict(color=color, line=dict(width=1, color="white")),
                              name=name, showlegend=showlegend, hovertemplate=f"Auftrag {j}<br>Dauer {p[j]}<extra></extra>"))
    fig.update_xaxes(title_text="Zeit")
    fig.update_yaxes(showticklabels=False)
    return _base(fig, 180)


def build_sweep(rows, param_label, value_key="value", y_keys=(("gap_edd", "MH gegen EDD", EDD_COLOR), ("gap_spt", "MH gegen SPT", LATE_COLOR), ("gap_random", "MH gegen Zufall", RANDOM_COLOR))):
    xs = [r[value_key] for r in rows]
    fig = go.Figure()
    for key, name, color in y_keys:
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", line=dict(color=color, width=2.5), name=name))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Weniger pünktliche Aufträge als Moore-Hodgson")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 360)


def build_timing(rows):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["brute_force_seconds"] * 1000 for r in rows], mode="lines+markers", line=dict(color=LATE_COLOR, width=2.5), name="Brute-Force (O(n!))"))
    fig.add_trace(go.Scatter(x=xs, y=[r["mh_seconds"] * 1000 for r in rows], mode="lines+markers", line=dict(color=ON_TIME_COLOR, width=2.5), name="Moore-Hodgson (O(n log n))"))
    fig.update_xaxes(title_text="Aufträge")
    fig.update_yaxes(title_text="Rechenzeit (ms)", type="log")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 340)


def build_setup_gap(rows):
    xs = [r["value"] for r in rows]
    upper = [r["diff_max"] for r in rows]
    lower = [r["diff_min"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(245,133,24,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["diff_mean"] for r in rows], mode="lines+markers", line=dict(color=SETUP_COLOR, width=2.5), name="Moore-Hodgson über dem echten Minimum (zusätzlich verspätete Aufträge)"))
    fig.update_xaxes(title_text="Rüstzeit je Familienwechsel (Minuten)")
    fig.update_yaxes(title_text="Zusätzlich verspätete Aufträge gegenüber dem Optimum")
    return _base(fig, 340)
