"""Plotly style and two composite figures."""
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

AZURE, NAVY, GREY, RED, AMBER, STEEL = "#019DF1", "#14406B", "#9AA3AD", "#B9382E", "#E0A100", "#5B8DB8"


def style():
    axis = dict(gridcolor="#EEF1F4", linecolor="#D9DEE5", showgrid=True, zeroline=False)
    pio.templates["deck"] = go.layout.Template(layout=dict(
        colorway=[GREY, AZURE, NAVY, RED, AMBER, STEEL], font=dict(family="Helvetica Neue, Helvetica, Arial", size=12, color="#22303F"),
        height=380, legend=dict(orientation="h", y=-0.18), margin=dict(l=50, r=20, t=50, b=50), paper_bgcolor="white", plot_bgcolor="white",
        title=dict(font=dict(size=14), x=0.02), xaxis=axis, yaxis=axis))
    pio.templates.default = "deck"
    pio.renderers.default = "notebook"
    pd.options.plotting.backend = "plotly"


def refits_figure(values, title):
    """One dot per rolling refit of each version."""
    fig = go.Figure()
    for (name, v), color in zip(values.items(), [GREY, AZURE]):
        fig.add_box(y=v, name=name, boxpoints="all", jitter=0.5, pointpos=0, marker=dict(color=color, size=7), line_width=1, fillcolor="rgba(0,0,0,0)")
    return fig.update_layout(title=title, yaxis_title="R² 2015–2025, %", showlegend=False)


def inputs_figure(spread, inputs, title):
    """The spread on top, one row per curve input below."""
    about = ["one week", "two weeks", "one month", "two months", "four months", "eight months"]
    fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=[2.2] + [1] * 6,
                        subplot_titles=["spread, % of price (above zero: backwardation)"]
                        + [f"c({n}): spread against its {n}-day average, about {t}" for n, t in zip(inputs.columns, about)])
    fig.add_scatter(x=spread.index, y=spread, line=dict(color=NAVY, width=1.5), row=1, col=1)
    for k, n in enumerate(inputs.columns):
        fig.add_scatter(x=inputs.index, y=inputs[n], line=dict(color=AZURE, width=1), fill="tozeroy", fillcolor="rgba(1,157,241,0.25)", row=k + 2, col=1)
    fig.update_annotations(font_size=11, x=0, xanchor="left")
    fig.update_yaxes(range=[-5.2, 5.2], tickvals=[-4, 0, 4]).update_yaxes(range=None, tickvals=None, row=1, col=1)
    return fig.update_layout(height=720, showlegend=False, title=title)
