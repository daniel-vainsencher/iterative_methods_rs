from util import error_message
import sys
import os
from operator import itemgetter
try:
    import numpy as np
    import yaml
    import plotly.graph_objects as go
except ModuleNotFoundError as error:
    print(f"ModuleNotFoundError:", error, error_message)
    sys.exit(1)
    
parameters = {}
with open("./visualizations_python/parameters_for_histogram.yaml") as parameters_file:

    parameters = yaml.load(parameters_file, Loader=yaml.CLoader)

num_initial_values = parameters["num_initial_values"]
num_final_values = parameters["num_final_values"]
stream_size = parameters["stream_size"]

with open(parameters["stream_file"]) as pop_file:

    stream = yaml.load_all(pop_file, Loader=yaml.CLoader)
    stream = [value for i, value in stream]
    stream = np.array(stream, dtype=float)

    sigma = float(parameters["sigma"])
    xm = np.min(stream) - 0.2
    xM = np.max(stream) + 0.2
    ym = 0
    yM = 0.3
    bin_size = parameters["bin_size"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=stream[:num_initial_values],
            xbins_size=bin_size,
            histnorm="probability",
            marker_color="#539A99",
            opacity=0.75,
            name=f"Initial Stream Distribution: First {num_initial_values} Samples",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=stream,            
            xbins_size=bin_size,
            histnorm="probability",
            marker_color="#FCA000",
            opacity=0.75,
            name=f"Final Stream Distribution: All {stream_size} Samples",
        )
    )

    fig.layout = go.Layout(
        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False, fixedrange=True),
        hovermode="closest",
        barmode="group",
        bargroupgap=0.1,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#c2d1ef",
        legend=dict(yanchor="top", y=1.00, xanchor="left", x=0.01),
    )

    # To export:
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")
    config = {"staticPlot": True, "displayModeBar": False}
    fig.write_html(
        file="visualizations/reservoir_histograms_initial_final.html", config=config
    )
