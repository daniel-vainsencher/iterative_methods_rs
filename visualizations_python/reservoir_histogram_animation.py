import numpy as np
import yaml
import plotly.graph_objects as go
import os
from operator import itemgetter


parameters = {}
with open("./visualizations_python/parameters_for_histogram.yaml") as parameters_file:

    parameters = yaml.load(parameters_file, Loader=yaml.CLoader)

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
    yM = 0.4
    num_bins = parameters["num_bins"]
    bin_size = .1

    fig = go.Figure()
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#c2d1ef")
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
                                "frame": {"duration": 100, "redraw": True},
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
        legend=dict(yanchor="top", y=1.00, xanchor="left", x=0.01, bgcolor='rgba(0,0,0,0)'),
    )

    fig.update_layout(
        annotations=[
            dict(xref = 'x', yref = 'y',
                showarrow=False,
                x=.5,
                y=.35,
                text=f"Reservoir Number: {0}",
                font=dict(color='black'),
                font_size=14,
                xanchor="center",
                # xshift=10,
                opacity=1.0,
            ),
        ]
    ),

    fig.add_trace(
        go.Histogram(
            x=population[:capacity],
            # nbinsx=num_bins,
            xbins_size = bin_size,
            histnorm="probability",
            # bingroup=1,
            marker_color="#EB89B5",
            opacity=0.75,
            name="Stream Distribution",
        )
    )
    
    fig.add_trace(
        go.Histogram(
            x=arr[: capacity, 1],
            # nbinsx=num_bins,
            xbins_size = bin_size,
            histnorm="probability",
            bingroup=2,
            marker_color="#330C73",
            opacity=0.75,
            name="Reservoir Distribution",
        )
    )



    fig.frames = [
        go.Frame(
            data=[
                go.Histogram(
                        x=population[:int(arr[i*capacity,0])],
                        # nbinsx=num_bins,
                        xbins_size = bin_size,
                        histnorm="probability",
                        # bingroup=1,
                        marker_color="#EB89B5",
                        opacity=0.75,
                        name="Stream Distribution",
                    ),
                go.Histogram(
                    x=arr[i * capacity : (i + 1) * capacity, 1],
                    # nbinsx=num_bins,
                    xbins_size = bin_size,
                    histnorm="probability",
                    # bingroup=2,
                    marker_color="#330C73",
                    opacity=0.75,
                    name="Reservoir Distribution",
                ),
            ],

            layout=go.Layout(
                annotations=[
                    dict(xref = 'x', yref = 'y',
                        showarrow=False,
                        x=.5,
                        y=.17,
                        text=f"Reservoir Number: {i}",
                        font=dict(color='black'),
                        font_size=14,
                        xanchor="center",
                        # xshift=10,
                        opacity=1.0,
                    ),
                ]
            ),
        )
        for i in range(num_res)
    ]
    

    # To export:
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")


    config = {
    "staticPlot": True, 
    "displayModeBar": False}
    fig.write_html(
        file="visualizations/reservoir_histogram_animation.html",
        config=config,
        auto_play=False,
    )
