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

with open("./target/debug/examples/reservoir_means.yaml") as res_file, open(
    "./target/debug/examples/stream_for_histogram.yaml"
) as pop_file:

    # create iterators containing the data from the yaml docs
    reservoir_means = yaml.load_all(res_file, Loader=yaml.CLoader)
    stream = yaml.load_all(pop_file, Loader=yaml.CLoader)
    # For the stream, extract the sample values; forget the enumeration.
    stream = [value for i, value in stream]
    stream = np.array(stream, dtype=float)
    """
    Initialize an array with rows [index, reservoir_mean, stream_slice_mean] where:
        index = enumeration of the reservoirs,
        reservoir_mean = mean of the reservoir
        stream_slice_mean = mean of the slice of the stream from which the reservoir sample was taken
    """
    arr = np.full((parameters["num_res"], 3), 0, dtype=float)
    for i, (ind, res) in enumerate(reservoir_means):
        arr[i, 0] = ind
        arr[i, 1] = res
        arr[i, 2] = np.mean(stream[: ind + 1])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=arr[:, 0], y=arr[:, 1], name="Reservoir Means", mode="lines+markers", marker_color="#539A99"
        )
    )
    fig.add_trace(
        go.Scatter(x=arr[:, 0], y=arr[:, 2], name="Stream Means", mode="lines+markers", marker_color="#FCA000")
    )
    fig.update_layout(
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#c2d1ef",
    )

    # To export a still image:
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")

    config = {"staticPlot": True, "displayModeBar": False}
    fig.write_html(file="visualizations/reservoir_means.html", config=config)
