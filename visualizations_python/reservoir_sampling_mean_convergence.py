import numpy as np
import pandas as pd
import yaml
from yaml import CLoader
import plotly.graph_objects as go
import plotly.express as px
import os
from operator import itemgetter

parameters = {}
with open("./visualizations_python/parameters.yaml") as parameters_file:
    
    parameters = yaml.load(parameters_file, Loader = CLoader)

with open("./target/debug/examples/reservoirs.yaml") as res_file, open("./target/debug/examples/population.yaml") as pop_file:

    reservoirs = yaml.load_all(res_file, Loader=CLoader)
    population = yaml.load_all(pop_file, Loader = CLoader)
    population = [value for i, value in population]
    population = np.array(population, dtype = float)
    print(population)
    arr = np.full((parameters['num_res'],3),0, dtype = float)
    for i, res in enumerate(reservoirs):
        print("i, res:", i, res)
        ind = max(map(itemgetter(0), res))
        arr[i,0] = ind
        arr[i,1] = np.mean(np.array(res, dtype = float), axis = 0)[1]
        arr[i,2] = np.mean(population[:ind+1]) # the mean of the population from which the current reservoir is drawn
    # print("the array of indices and means:\n", arr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = arr[:,0], y = arr[:,1], name = "Reservoir Means", mode = "lines+markers"))
    fig.add_trace(go.Scatter(x = arr[:,0], y = arr[:,2], name = "Stream Means", mode = "lines+markers"))
    fig.update_layout(title = f"Reservoir and Stream Means. Stream Size={parameters['stream_size']}, Capacity={parameters['capacity']}, Number of Reservoirs={parameters['num_res']}")
    # fig.show()

    # To export a still image:
    if not os.path.exists("visualizations"):
        os.mkdir("visualizations")
    fig.write_image("visualizations/reservoir_and_stream_means.png")


    
    




