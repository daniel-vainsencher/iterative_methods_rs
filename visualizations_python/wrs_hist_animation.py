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
num_res = 1+ (pop_size - capacity) // step

with open("./target/debug/examples/reservoirs.yaml") as res_file, open("./target/debug/examples/population.yaml") as pop_file:

	reservoirs = yaml.load_all(res_file, Loader=CLoader)
	arr = np.full((num_res*num_bins,3),0, dtype = float)
	# print(arr.shape)
	for i, res in enumerate(reservoirs):
		# print(i)
		if res is not None:
			count, division = np.histogram(res, bins = num_bins, density = True, range = (0.0, 1.0))
			r_num = np.full((num_bins,1), i)
			division = division[:num_bins, np.newaxis]
			count = count[:,np.newaxis]
			ar = np.hstack([r_num, division, count])
			# print(ar.shape)
			arr[i*num_bins:(i+1)*num_bins,:] = ar[:,:]
			# print(arr[:50,:])
			
	bar_df = pd.DataFrame(arr, columns = ["res_num", "bin_left_edge", "height"])
	y_max = np.max(arr[:,2]) + 1
	fig2 = px.bar(bar_df, x = "bin_left_edge", y = "height", animation_frame = "res_num", range_y = [0, y_max])
	fig2.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 10
	fig2.show()

	population = yaml.load(pop_file, Loader = CLoader)
	count, division = np.histogram(population, bins = num_bins, density = True, range = (0.0, 1.0))
	fig1 = go.Figure(
		data = go.Bar(x = division[:num_bins], y = count)
		)
	fig1.show()



