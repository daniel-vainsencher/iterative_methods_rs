import sys
import os
from operator import itemgetter
try:
    import numpy as np
    import yaml
    import plotly.graph_objects as go
except ModuleNotFoundError as error:
    print(f"ModuleNotFoundError:", error, """\n 
        You do not have the Python modules needed to generate the visualizations for this example.
        You can install the dependencies or run the example without visualizations.

        To install the dependencies use the following steps:

            0) If you don't already have it, install Python3 following the instructions at https://www.python.org/downloads/.
            
            1) Install pip and virtual env according to the instructions here:
            
            https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#:~:text=Installing%20virtualenv&text=venv%20is%20included%20in%20the,Python%20packages%20for%20different%20projects.

            2) Set up a virtual environment that will contain the dependencies:
            `$ virtualenv <name>`
            
            3) Activate the environment:
            `$ source <name>/bin/activate`

            4) Install the requirements using the requirements.txt file:
            `$ pip install -r ./visualizations_python/requirements.txt`

            5) Rerun the examples.

        To run the example without visualizations, add the command line argument 'false':

            `$ ./target/debug/examples/reservoir_histogram_animation false`
        """)
    sys.exit(1)


parameters = {}
with open("./visualizations_python/parameters_for_histogram.yaml") as parameters_file:

    parameters = yaml.load(parameters_file, Loader=yaml.CLoader)

with open(parameters["reservoir_samples_file"]) as res_file, open(
    parameters["stream_file"]
) as pop_file:

    capacity = int(parameters["capacity"])
    num_res = int(parameters["num_res"])
    stream_size = int(parameters["stream_size"])
    print("capacity, num_res:", capacity, num_res)
    reservoirs = yaml.load_all(res_file, Loader=yaml.CLoader)
    stream = yaml.load_all(pop_file, Loader=yaml.CLoader)
    stream = [value for i, value in stream]
    stream = np.array(stream, dtype=float)
    arr = np.full((capacity * num_res, 2), 0, dtype=float)
    for i, res in enumerate(reservoirs):
        ind = max(map(itemgetter(0), res))
        res = list(map(itemgetter(1), res))
        arr[i * capacity : (i + 1) * capacity, 0] = np.full(capacity, ind)
        arr[i * capacity : (i + 1) * capacity, 1] = np.array(res)

    sigma = float(parameters["sigma"])
    xm = np.min(stream) - 0.2
    xM = np.max(stream) + 0.2
    ym = 0
    yM = 0.3
    bin_size = parameters["bin_size"]

    # initialize the reservoir
    current_res = arr[:capacity]

    def make_res(ind):
        """
        This function returns the most recent reservoir given the
        index "ind" of the portion of the stream that has been processed.
        Used in creating the histograms in the frames of the animation.
        """
        if ind > current_res[0, 0]:
            res = arr[arr[:, 0] == ind]
            if res.size > 0:
                return res
        return current_res

    # duration of animation in milliseconds
    total_duration = 4 * 1000
    skip_size = 20
    num_frames = stream_size // skip_size
    frame_duration = total_duration // num_frames
    print("frame duration:", frame_duration)

    fig = go.Figure()
    fig.layout = go.Layout(
        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False, fixedrange=True),
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
                                "frame": {"duration": frame_duration, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
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

    # to adjust position of play/pause buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=-0.1,
                xanchor="right",
                y=1.0,
                yanchor="top",
            ),
        ],
        legend=dict(
            yanchor="top", y=1.00, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0)"
        ),
    )

    fig.update_layout(
        annotations=[
            dict(
                xref="x",
                yref="y",
                showarrow=False,
                x=xM-0.2,
                y=0.22,
                text=f"Percent of Stream Processed: {0}",
                font=dict(color="black"),
                font_size=14,
                xanchor="right",
                opacity=1.0,
            ),
            dict(
                xref="x",
                yref="y",
                showarrow=False,
                x=xM-0.2,
                y=0.17,
                text=f"Reservoir Index: {0}",
                font=dict(color="black"),
                font_size=14,
                xanchor="right",
                opacity=1.0,
            ),
        ]
    ),

    fig.add_trace(
        go.Histogram(
            x=stream[:capacity],
            xbins_size=bin_size,
            histnorm="probability",
            marker_color="#FCA000",
            opacity=0.75,
            name=f"Stream Distribution (Stream Size: {stream_size})",
        )
    )

    fig.add_trace(
        go.Histogram(
            x=current_res[:, 1],
            xbins_size=bin_size,
            histnorm="probability",
            bingroup=2,
            marker_color="#539A99",
            opacity=0.75,
            name=f"Reservoir Distribution (Capacity: {capacity})",
        )
    )

    fig_frames = []
    res_num = 0
    max_index = current_res[0, 0]
    for i in range(capacity - 1, stream_size):
        current_res = make_res(i)        
        if current_res[0, 0] > max_index:
            max_index = current_res[0, 0]
            res_num += 1
        if i % skip_size == 0:
            fig_frames.append(
                go.Frame(
                    data=[
                        go.Histogram(
                            x=stream[:i],                            
                            xbins_size=bin_size,
                            histnorm="probability",                            
                            marker_color="#FCA000",
                            opacity=0.75,
                            name=f"Stream Distribution (Stream Size: {stream_size})",
                        ),
                        go.Histogram(
                            x=current_res[:, 1],
                            xbins_size=bin_size,
                            histnorm="probability",                            
                            marker_color="#539A99",
                            opacity=0.75,
                            name=f"Reservoir Distribution (Capacity: {capacity})",
                        ),
                    ],
                    layout=go.Layout(
                        annotations=[
                            dict(
                                xref="x",
                                yref="y",
                                showarrow=False,
                                x=xM-0.2,
                                y=0.22,
                                text=f"Percent of Stream Processed: {(100*i) // stream_size}",
                                font=dict(color="black"),
                                font_size=14,
                                xanchor="right",                                
                                opacity=1.0,
                            ),
                            dict(
                                xref="x",
                                yref="y",
                                showarrow=False,
                                x=xM-0.2,
                                y=0.17,
                                text=f"Reservoir Index: {res_num}",
                                font=dict(color="black"),
                                font_size=14,
                                xanchor="right",                                
                                opacity=1.0,
                            ),
                        ]
                    ),
                )
            )
    fig.frames = fig_frames

    # To export:
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#c2d1ef")
    config = {"staticPlot": True, "displayModeBar": False}
    fig.write_html(
        file="visualizations/reservoir_histogram_animation.html",
        config=config,
        auto_play=False,
    )
