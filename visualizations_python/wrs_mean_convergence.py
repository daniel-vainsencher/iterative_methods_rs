import argparse
import numpy as np
import pandas as pd
import yaml
from yaml import CLoader
import plotly.graph_objects as go
import plotly.express as px

parser = argparse.ArgumentParser("Accept population (stream) size, capacity, step.")
parser.add_argument("pop_size_arg", help = "This should be the size of the population or stream.", type = str)
parser.add_argument("capacity_arg", help = "This should be the capacity of the reservoir.", type = str)
parser.add_argument("step_arg", help = "This is the step size.", type = str)
parser.add_argument("num_bins_arg", help = "This should be the number of bins for the histogram.", type = str)
args = parser.parse_args()

num_bins = int(args.num_bins_arg)

capacity = int(args.capacity_arg)
pop_size = int(args.pop_size_arg)
step = int(args.step_arg)
num_res = 1 + (pop_size-capacity) // step

with open("./target/debug/examples/reservoirs.yaml") as res_file, open("./target/debug/examples/population.yaml") as pop_file:

	reservoirs = yaml.load_all(res_file, Loader=CLoader)
	population = yaml.load(pop_file, Loader = CLoader)
	pop = np.array(population, dtype = float)
	arr = np.full((num_res,3),0, dtype = float)
	for i, res in enumerate(reservoirs):
		if res is not None:
			arr[i,0] = i
			arr[i,1] = np.mean(np.array(res, dtype = float))
			arr[i,2] = np.mean(pop[:capacity + i * step]) # the mean of the population from which the current reservoir is drawn
			
	res_df = pd.DataFrame(arr, columns = ["reservoir number", "reservoir mean", "stream mean"])
	
	fig = go.Figure()
	fig.add_trace(go.Scatter(x = res_df["reservoir number"], y = res_df["reservoir mean"], name = "Reservoir Means", mode = "lines+markers"))
	fig.add_trace(go.Scatter(x = res_df["reservoir number"], y = res_df["stream mean"], name = "Stream Means", mode = "lines+markers"))
	fig.update_layout(title = "Reservoir and Stream Means")
	fig.show()


	
	




