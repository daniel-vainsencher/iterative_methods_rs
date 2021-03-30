import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
import plotly.express as px
import os
from operator import itemgetter


def cleanup_test_files(files_to_remove: list):
    """
    The Rust code panics if the files it writes to already exists. This fn
    removes them after they have been used so that the example can be run again.
    It also prevents files used only for the tests from accumulating in the repo.
    """
    for file in files_to_remove:
        os.remove(file)


files_to_remove = []
parameters = {}
with open("./visualizations_python/parameters_for_histogram.yaml") as parameters_file:

    parameters = yaml.load(parameters_file, Loader=yaml.CLoader)

files_to_remove.append(parameters["parameters_file_path"])

with open(parameters["reservoir_samples_file"]) as res_file, open(
    parameters["population_file"]
) as pop_file:

    capacity = int(parameters["capacity"])
    num_res = int(parameters["num_res"])
    print("capacity, num_res:", capacity, num_res)
    reservoirs = yaml.load_all(res_file, Loader=yaml.CLoader)
    population = yaml.load_all(pop_file, Loader=yaml.CLoader)
    population = [value for i, value in population]
    population = np.array(population, dtype=float)
    arr = np.full((capacity * num_res, 2), 0, dtype=float)
    for i, res in enumerate(reservoirs):
        ind = max(map(itemgetter(0), res))
        res = list(map(itemgetter(1), res))
        arr[i * capacity : (i + 1) * capacity, 0] = np.full(capacity, ind)
        arr[i * capacity : (i + 1) * capacity, 1] = np.array(res)

    sigma = float(parameters["sigma"])
    xm = np.min(population) - 0.2
    xM = np.max(population) + 0.2
    ym = 0
    # yM = 1 / (np.sqrt(2 * np.pi) * np.sqrt(2) * sigma)
    yM = 0.4
    num_bins = 20

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=population,
            nbinsx=num_bins,
            histnorm="probability",
            marker_color="#EB89B5",
            opacity=0.75,
            name="Population Distribution",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=population,
            nbinsx=num_bins,
            histnorm="probability",
            marker_color="#EB89B5",
            opacity=0.75,
            name="Population Distribution",
        )
    )

    fig.frames = [
        go.Frame(
            data=[
                go.Histogram(
                    x=arr[i * capacity : (i + 1) * capacity, 1],
                    nbinsx=num_bins,
                    histnorm="probability",
                    marker_color="#330C73",
                    opacity=0.75,
                    name="Reservoir Distribution",
                )
            ],
            layout = go.Layout(annotations= [dict(
            showarrow=False,
            x="1.",
            y=".27",
            text=f"Reservoir Number: {i}",
            font_size=14,
            xanchor="left",
            # xshift=10,
            opacity=1.),])
        )
        for i in range(num_res)
    ]
    fig.layout = go.Layout(
        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
        # title_text="Drifting Distribution: Reservoir Samples Represent the Stream Distribution",
        # xanchor="center",
        hovermode="closest",
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 100},
                            },
                        ],
                    ),
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
            )
        ],
        barmode="group",
        bargroupgap=0.1,
    )

    fig.update_layout(title = dict(text="The Reservoir Distribution Follows The Stream Distribution",
            x = 0.5,
            y = 0.9,
        xanchor="center",
        yanchor="top"))

    # To export:
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")
    fig.write_html("visualizations/reservoir_histogram_animation.html")

files_to_remove.append(parameters["reservoir_samples_file"])
files_to_remove.append(parameters["population_file"])

cleanup_test_files(files_to_remove)
