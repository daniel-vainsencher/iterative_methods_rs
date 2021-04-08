import numpy as np
import yaml
import plotly.graph_objects as go
import os
from operator import itemgetter


parameters = {}
with open("./visualizations_python/parameters_for_histogram.yaml") as parameters_file:

    parameters = yaml.load(parameters_file, Loader=yaml.CLoader)

num_initial_values = parameters["num_initial_values"]
num_final_values = parameters["num_final_values"]
stream_size = parameters["stream_size"]

with open(parameters["population_file"]) as pop_file:

    population = yaml.load_all(pop_file, Loader=yaml.CLoader)
    population = [value for i, value in population]
    population = np.array(population, dtype=float)

    sigma = float(parameters["sigma"])
    xm = np.min(population) - 0.2
    xM = np.max(population) + 0.2
    ym = 0
    yM = 0.2
    # yM = 1 / (np.sqrt(2 * np.pi) * sigma)
    num_bins = parameters["num_bins"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=population[:num_initial_values],
            nbinsx=num_bins,
            histnorm="probability",
            marker_color="#539A99",
            opacity=0.75,
            name=f"Initial Population Distribution: First {num_initial_values} Samples",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=population,
            nbinsx=num_bins,
            histnorm="probability",
            marker_color="#FCA000",
            opacity=0.75,
            name=f"Final Population Distribution: All {stream_size} Samples",
        )
    )

    fig.layout = go.Layout(
        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False, fixedrange=True),
        # title_text="Drifting Distribution: Reservoir Samples Represent the Stream Distribution",
        # xanchor="center",
        hovermode="closest",
        barmode="group",
        bargroupgap=0.1,
    )

    fig.update_layout(
        title=dict(
            text="The Initial and Final Reservoir Distributions",
            x=0.5,
            y=0.9,
            xanchor="center",
            yanchor="top",
        )
    )

    # To export:
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")
    config = {"displayModeBar": False}
    fig.write_html(
        file="visualizations/reservoir_histograms_initial_final.html", config=config
    )
