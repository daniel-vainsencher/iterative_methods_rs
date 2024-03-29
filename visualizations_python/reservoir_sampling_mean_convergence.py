import numpy as np
import yaml
import plotly.graph_objects as go
import os
from operator import itemgetter


def cleanup_test_files():
    """
    The Rust code panics if the files it writes to already exists. This fn
    removes them after they have been used so that the example can be run again.
    It also prevents files used only for the tests from accumulating in the repo.
    """
    os.remove("./target/debug/examples/reservoirs.yaml")
    os.remove("./target/debug/examples/population.yaml")


parameters = {}
with open("./visualizations_python/parameters.yaml") as parameters_file:

    parameters = yaml.load(parameters_file, Loader=yaml.CLoader)

with open("./target/debug/examples/reservoirs.yaml") as res_file, open(
    "./target/debug/examples/population.yaml"
) as pop_file:

    # create iterators containing the data from the yaml docs
    reservoirs = yaml.load_all(res_file, Loader=yaml.CLoader)
    population = yaml.load_all(pop_file, Loader=yaml.CLoader)
    # For the population, extract the sample values; forget the enumeration.
    population = [value for i, value in population]
    population = np.array(population, dtype=float)
    # print("population", population)
    # Initialize an array with rows [index, reservoir_mean, population_slice_mean] where:
    # index = enumeration of the reservoirs,
    # reservoir_mean = mean of the reservoir
    # population_slice_mean = mean of the slice of the population from which the reservoir sample was taken
    arr = np.full((parameters["num_res"], 3), 0, dtype=float)
    for i, res in enumerate(reservoirs):
        # print("i, res:", i, res)
        ind = max(map(itemgetter(0), res))
        arr[i, 0] = ind
        arr[i, 1] = np.mean(np.array(res, dtype=float), axis=0)[1]
        arr[i, 2] = np.mean(population[: ind + 1])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=arr[:, 0], y=arr[:, 1], name="Reservoir Means", mode="lines+markers"
        )
    )
    fig.add_trace(
        go.Scatter(x=arr[:, 0], y=arr[:, 2], name="Stream Means", mode="lines+markers")
    )
    fig.update_layout(
        title=f"Reservoir and Stream Means. <br> Stream Size={parameters['stream_size']}, Capacity={parameters['capacity']}, \n Number of Reservoirs={parameters['num_res']}"
    )

    # To export a still image:
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")
    fig.write_image("visualizations/reservoir_and_stream_means.png")

cleanup_test_files()
