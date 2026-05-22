import numpy as np
import sklearn.metrics as sk
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import sys
import tensorflow as tf
import random

from hkml_base import *
from hkml_transfer import *
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras import saving
from matplotlib.ticker import StrMethodFormatter, NullFormatter

import pandas as pd
import warnings

mpl.rcParams.update({
		"font.family": "sans-serif",
		"font.sans-serif": ["DejaVu Sans"],
		"axes.titlesize": 22,
		"axes.labelsize": 18,
		"xtick.labelsize": 16,
		"ytick.labelsize": 16,
		"legend.fontsize": 14,
	})

warnings.filterwarnings('ignore')

# Formats axes for plots in the specified manner
def format_axes(ax, right, lw = 1.8):
	for spine in ax.spines.values():
		spine.set_linewidth(lw)

	ax.tick_params(direction="in", which="major",
					top=True, right=right,
					width=lw, length=7)
	ax.tick_params(which="minor", length=0)

# Darkens a color by the specified pixel amount (op) for plotting
def darken_color(color, op = 5):
	# Recover color in hexadecimal
	color = color[1:]
	c1, c2, c3 = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
	# Darken with ceiling of 0
	def sub_zero(digit, op):
		if (digit - op < 0):
			return 0
		return digit - op
	c1, c2, c3 = sub_zero(c1, op), sub_zero(c2, op), sub_zero(c3, op)
	# Return as two-digit hexadecimal values
	return f"#{c1:02x}{c2:02x}{c3:02x}"

# Extracts predictions from a .keras file given the corresponding metadata
def extract_predictions(model_data, model_type, x_test_data, split_folder, model_name):
	# Grab name, folder
	model_subname = model_data["model_name"]
	model_folder = os.path.join(os.path.join(split_folder, model_name), model_subname)
	# Open appropriate files
	target_scaler = joblib.load(os.path.join(model_folder, f'{model_subname}_t_scaler.kl'))
	model_file = os.path.join(model_folder, f'{model_subname}.keras')
	model = saving.load_model(model_file, custom_objects=None, compile=True, safe_mode=True)
	# If the model type is base, only one model needs to be loaded
	if (model_type == "base"):
		# Load the feature scaler and transform the data
		feature_scaler = joblib.load(os.path.join(model_folder, f'{model_subname}_f_scaler.kl'))
		x_test = feature_scaler.transform(x_test_data)
		# Return the respective targets after inverting the scaler
		return target_scaler.inverse_transform(model.predict(x_test, verbose=0))
	# If the model type is transfer, load two models
	elif (model_type == "transfer"):
		# Find the base model
		base_name = f"{model_data["base_target"]}_100"
		base_folder = os.path.join(os.path.join(split_folder, "base_model"), base_name)
		# Load the feature scaler and transform the data
		feature_scaler = joblib.load(os.path.join(base_folder, f'{base_name}_f_scaler.kl'))
		x_test = feature_scaler.transform(x_test_data)
		# Load the base model, transfer model
		base_model_file = os.path.join(base_folder, f'{base_name}.keras')
		base_model = saving.load_model(base_model_file, custom_objects=None, compile=True, safe_mode=True)
		base_model_partial = Model(inputs=base_model.input, outputs=base_model.layers[1 + int(model_data["extract_layer"])*2].output)
		# Create the transfer predictions
		x_test = np.array(base_model_partial.predict([x_test], verbose=0))
		# Return the respective targets after inverting the scaler
		return target_scaler.inverse_transform(model.predict(x_test, verbose=0))

# Check how many of the top-performing MOFs overlap between a true and predicted set for a single target
def top_overlap(top_threshold, y_true, y_predict, y_names):
	# Match the MOF names to the target lists
	true = {y_names[index] : y_true[index] for index in range(len(y_true))}
	predict = {y_names[index] : y_predict[index] for index in range(len(y_predict))}
	# Sort by magnitude to get top-performers
	true_sorted = dict(sorted(true.items(), key=lambda x: x[1]))
	predict_sorted = dict(sorted(predict.items(), key=lambda x: x[1]))
	# Grab the MOFs names to match between the lists
	true_names = list(true_sorted.keys())
	predict_names = list(predict_sorted.keys())
	# Calculate the index to search
	threshold_index = len(y_predict) - top_threshold
	total = 0
	# Search backwards, finding overlaps
	for entry in range(-1, -top_threshold, -1):
		if (predict_names.index(true_names[entry]) >= threshold_index):
			total += 1
	return total

# Parses the model results contained in a split folder into a Pandas dataframe
def parse_model_results(folder_path, model_names, x_test, y_test, test_names, target_dict={"hk_co2":0, "hk_h2o":1, "hk_n2":2, "hk_nh3":3}):
	# Create a dataframe template
	data_df = 0
	# Create a storage object for the base model parities 
	zipped_parities = []
	# For each model name specified
	for model_name in model_names:
		# Read in the model results file for that model name
		model_data = pd.read_csv(os.path.join(os.path.join(folder_path, model_name), "model_results.csv"))
		# If this is a base model, no need to specify extract layer and base target
		if (model_names[model_name] == "base"):
			model_data["extract_layer"] = [None for x in range(len(model_data))]
			model_data["base_target"] = [None for x in range(len(model_data))]
		# Else, specify extract layer and base target
		elif (model_names[model_name] == "transfer"):
			base_target = model_name.split("_transfer_")[0]
			model_data["base_target"] = [base_target for x in range(len(model_data))]
		# Check for binning, add labeling appropriately
		if ("bin" in model_name):
			model_binning = model_name.split("binning")[1]
			bin_num = model_binning.split("_")[0]
			bin_method = model_binning.split("_")[1].split("bin")[1]
		else:
			bin_num = None
			bin_method = None
		# All submodels in this model name will have the same binning
		model_data["bin_num"] = [bin_num for x in range(len(model_data))]
		model_data["bin_method"] = [bin_method for x in range(len(model_data))]
		# Set up objects to hold model evaluation metrics
		overlaps800 = []
		overlaps400 = []
		overlaps200 = []
		MAE = []
		MAD = []
		div = []
		# Iterate through the loaded model data, extract submodel data
		for index, row in model_data.iterrows():
			# Extract predictions from the submodel
			prediction = extract_predictions(row, model_names[model_name], x_test, folder_path, model_name)[:, 0]
			# Find the target list for the respective target of the model
			y_true = y_test[:, target_dict[row["target"]]]
			# Create metrics from the extracted predictions (MAE, MAD, MAD/MAE)
			MAE_i = sk.mean_absolute_error(y_true, prediction)
			mean_true = np.mean(y_true)
			MAD_i = np.mean(np.abs(y_true - mean_true))
			div_i = MAD_i/MAE_i
			overlaps800.append(top_overlap(800, y_true, prediction, test_names))
			overlaps400.append(top_overlap(400, y_true, prediction, test_names))
			overlaps200.append(top_overlap(200, y_true, prediction, test_names))
			MAE.append(MAE_i)
			MAD.append(MAD_i)
			div.append(div_i)
			# If this is a base model, add to the parity object
			if (row["split"] == 1 and row["base_target"] == None and row["bin_method"] == None):
				parity = [y_true, prediction]
				zipped_parities.append(parity)

		# Add new columns for the model metrics
		model_data[f"{800}_id"] = overlaps800
		model_data[f"{400}_id"] = overlaps400
		model_data[f"{200}_id"] = overlaps200
		model_data[f"MAE"] = MAE
		model_data[f"MAD"] = MAD
		model_data[f"MADMAE"] = div
		# Concatenate this model's data to the total dataframe (making a new dataframe if this is the first iteration)
		if (type(data_df) == int):
			data_df = model_data
		else:
			data_df = pd.concat([data_df, model_data], axis=0, ignore_index=True)
	return data_df, zipped_parities

# Creates model performance curves for a specified set of models, comparing against two metrics with a combined barchart-scatterplot
def plot_model_curves(plot_path, model_results, comparison, comparison_2, target, extract_layers, x_label, y_label, y_label_2, y_lims, y_lims_2, target_dict={"hk_co2":(0, r"CO$_2$"), "hk_h2o":(1, r"H$_2$O"), "hk_n2":(2, r"N$_2$"), "hk_nh3":(3, r"NH$_3$")}, styles=[["#db4200", "#3083ff", "#ffac30", "#4eb500"], ["-", "-", "-.", ":"], ["o", "o", "D", "s"]]):
	# Grab model results from the provided parsed dataframe
	model_results = model_results[model_results["target"] == target]
	# Isolate base target types
	bases = model_results["base_target"].unique().tolist()

	# Set bar width, CHANGE AS NEEDED
	width = 0.8

	# Set categories for plotting
	percents = [100, 50, 30, 10, 5, 3, 1]
	x_cats = np.arange(len(percents))

	# Create the figure with two axes for both metrics
	fig = plt.figure(figsize=(9, 5))
	ax = plt.subplot(111)
	ax2 = ax.twinx()

	# Formate the axes appropriately
	format_axes(ax, right=False)
	format_axes(ax2, right=True)

	# Set the first axis below, add a grid
	ax.set_axisbelow(True)
	ax.yaxis.grid(color='#e6e6e6', linestyle='dashed', linewidth=0.6, zorder=1)
	ax.xaxis.grid(color='#e6e6e6', linestyle='dashed', linewidth=0.6, zorder=1)

	# Create objects to hold the scatterlines and bars
	lines = []
	bars = []

	# Format axes with ticks, limits, and labeling
	ax.set_xticks(x_cats)
	ax.set_xticklabels([f"{p}%" for p in percents])
	ax.set_xlabel(x_label, fontweight="bold")

	ax.set_ylim(y_lims[0], y_lims[1])
	# CHANGE 0.05 AS NEEDED
	ax.set_yticks(np.arange(y_lims[0], y_lims[1]+0.05, 0.05))
	ax.set_ylabel(y_label, fontweight="bold")

	ax2.set_ylim(y_lims_2[0], y_lims_2[1])
	# CHANGE 0.1 AS NEEDED
	ax2.set_yticks(np.arange(y_lims_2[0], y_lims_2[1]+0.1, 0.1))
	ax2.set_ylabel(y_label_2, fontweight="bold")

	# For each base target
	for base_no in range(len(bases)):
		# Check if the base target exists (if not, this is a base model, not a transfer model)
		base = bases[base_no]
		base_check = type(base) != str and np.isnan(base)
		# If this is a base model
		if (base_check):
			# Load results simply
			results_base = model_results[model_results["base_target"].isnull()]
			# Create formatting for the scatterline and barchart accordingly
			line_kwargs = dict(lw=2.4, ms=7, color=darken_color(styles[0][target_dict[target][0]], 20), linestyle=styles[1][0],
							marker=styles[2][0], markeredgecolor="black", markeredgewidth=1.2, zorder=3)

			bar_kwargs = dict(width=width/4, color=styles[0][target_dict[target][0]], zorder=2)

			# Create the y axis data
			x_splits = [float(x)*100 for x in results_base["split"].to_list()]
			y_comparisons = [float(x) for x in results_base[comparison].to_list()]
			y_comparisons_2 = [float(x) for x in results_base[comparison_2].to_list()]

			lines.append(ax2.plot(x_cats, y_comparisons_2, label=f"Base {target_dict[target][1]} {y_label_2}", **line_kwargs)[0])
			bars.append(ax.bar(x_cats - width/2 + width/4*base_no + 0.1, y_comparisons, label=f"Base {target_dict[target][1]} {y_label}", **bar_kwargs))
		# Otherwise, this is a transfer model
		else:
			# Grab the results of the base model
			results_base = model_results[model_results["base_target"] == base]
			# For each extract layer specified, create a new set of scatterlines and barcharts
			for extract_layer in extract_layers:
				# Create formatting for the scatterline and barchart accordingly
				line_kwargs = dict(lw=2.4, ms=7, color=darken_color(styles[0][target_dict[base][0]], 20), linestyle=styles[1][1 + extract_layer], 
							marker=styles[2][1 + extract_layer], markeredgecolor="black", markeredgewidth=1.2, zorder=3)

				bar_kwargs = dict(width=width/4, color=styles[0][target_dict[base][0]], zorder=2)
				# Extract results for this layer
				extract_results = results_base[results_base["extract_layer"] == extract_layer]

				# Create the y axis data
				x_splits = [float(x)*100 for x in extract_results["split"].to_list()]
				y_comparisons = [float(x) for x in extract_results[comparison].to_list()]
				y_comparisons_2 = [float(x) for x in extract_results[comparison_2].to_list()]

				lines.append(ax2.plot(x_cats, y_comparisons_2, label=f"{target_dict[base][1]} → {target_dict[target][1]} {y_label_2}", **line_kwargs)[0])
				bars.append(ax.bar(x_cats - width/2 + width/4*base_no + 0.1, y_comparisons, label=f"{target_dict[base][1]} → {target_dict[target][1]} {y_label}", **bar_kwargs))

	# Create a legend for the plot
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	handles = lines + bars
	labels = [h.get_label() for h in handles]
	leg = ax.legend(handles, labels,
					loc="center left",
					bbox_to_anchor=(1.2, 0.5), 
					frameon=True,
					borderpad=0.6,
					handlelength=2.2,
					labelspacing=1.5)
	leg.get_frame().set_linewidth(1.2)

	plt.savefig(f"{plot_path}/model_curves_{target}_{comparison}.png", dpi=600, bbox_inches="tight")

# Plot comparison of base vs. transfer model from a common adsorbate for a given split
def plot_target_comparison(plot_path, model_results, comparison, comparison_2, base_target, extract_layers, split, y_lims, y_lims_2, target_dict={"hk_h2o":(0, r"H$_2$O"), "hk_co2":(1, r"CO$_2$"), "hk_n2":(2, r"N$_2$"), "hk_nh3":(3, r"NH$_3$")}, styles=["#e97132", "#8ed973", "#ed9f40", "#84d975"], y_label="", y_label_2=""):
	# Grab the transfer learning results
	transfer_results = model_results[model_results["base_target"] == base_target]
	transfer_results = transfer_results[transfer_results["split"] == split]
	# Grab the base model results (base target is null for these)
	base_results = model_results[model_results["base_target"].isnull()]
	base_results = base_results[base_results["split"] == split]
	# Remove the specified base target, as transfer models do not exist from the base target to the base target
	categories = [target for target in target_dict]
	categories.remove(base_target)
	# Create containers for the model results
	comparisons = {"base":[]}
	comparisons_2 = {"base":[]}
	# Iterate through and create the remaining lists based on the extract layers to plot
	for extract_layer in extract_layers:
		comparisons[f"transfer_{extract_layer}"] = []
		comparisons_2[f"transfer_{extract_layer}"] = []
	# Extract results from the dataframe for each model
	for category in categories:
		for extract_layer in extract_layers:
			extract_results = transfer_results[transfer_results["extract_layer"] == extract_layer]
			comparisons[f"transfer_{extract_layer}"].append(extract_results[extract_results["target"] == category].squeeze().to_dict()[comparison])
			comparisons_2[f"transfer_{extract_layer}"].append(extract_results[extract_results["target"] == category].squeeze().to_dict()[comparison_2])
		comparisons["base"].append(base_results[base_results["target"] == category].squeeze().to_dict()[comparison])
		comparisons_2["base"].append(base_results[base_results["target"] == category].squeeze().to_dict()[comparison_2])

	# Creates categories for the chart
	categories = [target_dict[target][1] for target in categories]
	x_cats = np.arange(len(categories))

	# CAN CHANGE HERE, bar width
	width = 0.8

	# Create figure, axes
	fig = plt.figure(figsize=(6, 5))
	ax = plt.subplot(111)
	ax2 = ax.twinx()

	# Format axes
	format_axes(ax, right=False)
	format_axes(ax2, right=True)

	# Set grid for first axis
	ax.set_axisbelow(True)
	ax.yaxis.grid(color='#e6e6e6', linestyle='dashed', linewidth=0.6, zorder=1)
	ax.xaxis.grid(color='#e6e6e6', linestyle='dashed', linewidth=0.6, zorder=1)

	# Format axes with ticks, bins, and labeling
	ax.set_xticks(x_cats)
	ax.set_xticklabels([c for c in categories])

	ax.set_ylim(y_lims[0], y_lims[1])
	# CHANGE 0.1 AS NEEDED
	ax.set_yticks(np.arange(y_lims[0], y_lims[1]+0.1, 0.1))
	ax.set_ylabel(y_label, fontweight="bold")

	ax2.set_ylim(y_lims_2[0], y_lims_2[1])
	# CHANGE 0.4 AS NEEDED
	ax2.set_yticks(np.arange(y_lims_2[0], y_lims_2[1]+0.4, 0.4))
	ax2.set_ylabel(y_label_2, fontweight="bold")

	# Set formatting for scatterlines, bars
	bar_kwargs = dict(width=width/2, zorder=2)

	line_kwargs = dict(lw=2.4, ms=7, linestyle="-",
						marker="o", markeredgecolor="black", markeredgewidth=1.2, zorder=2)

	# Formats a category name for plotting
	def format_category(category):
		if (category.split("_")[0] == "transfer"):
			return "TL"
		else:
			return "Scratch"

	# Create bars and scatterlines accordingly
	count = 0
	handles = []
	for category, comparison_num in comparisons.items():
		handles.append(ax.bar(x_cats - width/2 + width/2*count + 0.2, comparison_num, color=styles[list(comparisons.keys()).index(category)], label=f"{y_label} ({format_category(category)})", **bar_kwargs))
		count += 1
	for category, comparison_num in comparisons_2.items():
		handles.append(ax2.plot(x_cats, comparison_num, color=darken_color(styles[list(comparisons.keys()).index(category)], 20), label=f"{y_label_2} ({format_category(category)})", **line_kwargs)[0])

	# Create a legend for this plot
	box = ax.get_position()
	labels = [h.get_label() for h in handles]
	leg = ax.legend(handles, labels,
					loc="upper left",
					frameon=True,
					borderpad=0.3,
					handlelength=2.2,
					ncol=2,
					labelspacing=0.2)
	leg.get_frame().set_linewidth(1.2)

	plt.savefig(f"{plot_path}/model_comparison_{base_target}_{comparison}.png", dpi=600, bbox_inches="tight")

# Plot a comparison of the average difference in performance across each base adsorbate for a given split, with sample stdev error bars
def plot_average_comparison(plot_path, model_results, comparison, extract_layers, split, y_lims, target_dict={"hk_nh3":(0, r"NH$_3$"), "hk_h2o":(1, r"H$_2$O"), "hk_co2":(2, r"CO$_2$"), "hk_n2":(3, r"N$_2$")}, styles=["#4eb500", "#3083ff", "#db4200", "#ffac30"], y_label=""):
	# Grab model results corresponding to the chosen split
	model_results = model_results[model_results["split"] == split]
	# Grab the base performance of each base target specified (this is the reference for subtracting from transfer models)
	base_performance = {}
	for base_target in target_dict:
		base_results = model_results[model_results["base_target"].isnull()]
		base_results = base_results[base_results["target"] == base_target]
		base_performance[base_target] = list(base_results.to_dict()[comparison].values())[0]
	# Grab categories and corresponding metrics/stdev erros
	categories = [target for target in target_dict]
	comparisons = {}
	errors = {}
	# Create dictionary lists for each comparison/error
	for extract_layer in extract_layers:
		comparisons[f"transfer_{extract_layer}"] = []
		errors[f"transfer_{extract_layer}"] = []
	# Compile lists for plotting
	for category in categories:
		for extract_layer in extract_layers:
			extract_results = model_results[model_results["base_target"] == category]
			extract_results = extract_results[extract_results["extract_layer"] == extract_layer]
			results = {}
			for index, row in extract_results.iterrows():
				results[row["target"]] = row[comparison] - base_performance[row["target"]]
			results = list(results.values())
			comparisons[f"transfer_{extract_layer}"].append(np.mean(results))
			errors[f"transfer_{extract_layer}"].append(np.std(results))

	# DEPRECATED: function for formatting extract layers
	def format_category(category):
		return f"Extract {int(category.split("_")[1])+1}"

	# Create categories based on the labeling
	categories = [target_dict[target][1] for target in categories]
	x_cats = np.arange(len(categories))

	# CAN CHANGE THIS, bar width
	width = 0.6

	# Create plot
	fig = plt.figure(figsize=(6, 5))
	ax = plt.subplot(111)	

	# Format axis, add grid
	format_axes(ax, right=False)
	ax.set_axisbelow(True)
	ax.yaxis.grid(color='#e6e6e6', linestyle='dashed', linewidth=0.6, zorder=1)
	ax.xaxis.grid(color='#e6e6e6', linestyle='dashed', linewidth=0.6, zorder=1)

	# Set axis x and y labeling
	ax.set_xticks(x_cats)
	ax.set_xticklabels([c for c in categories])

	ax.set_ylim(y_lims[0], y_lims[1])
	ax.set_yticks(np.arange(y_lims[0], y_lims[1]+0.05, 0.05))
	ax.set_ylabel(y_label, fontweight="bold")

	# Set up formatting for the scatterline
	line_kwargs = dict(lw=2.4, ms=7, linestyle="-", capsize=5, 
						marker="o", markeredgecolor="black", markeredgewidth=1.2, zorder=3)

	# Make the plot, interpolating the lines between for a continuous scatter
	for category, comparison_num in comparisons.items():
		for n in range(len(x_cats)):
			ax.errorbar(x_cats[n], comparison_num[n], yerr=errors[category][n], color=styles[n], **line_kwargs)
			if (n != 0):
				ax.plot((np.mean([x_cats[n-1], x_cats[n]]), x_cats[n]), (np.mean([comparison_num[n-1], comparison_num[n]]), comparison_num[n]), color=styles[n], lw=2.4, ms=7, linestyle="-")
			if (n != 3):
				ax.plot((np.mean([x_cats[n], x_cats[n+1]]), x_cats[n]), (np.mean([comparison_num[n], comparison_num[n+1]]), comparison_num[n]), color=styles[n], lw=2.4, ms=7, linestyle="-")

	plt.savefig(f"{plot_path}/model_average_transfer_{comparison}.png", dpi=600, bbox_inches="tight")

if __name__ == "__main__":
	# FOR STACKING MODEL RESULTS INTO TABLES IN SI
	# target = "hk_nh3"
	# bin_methods = [None, "quantile", "equal"]
	# model_results_list = [pd.read_csv("parsed_model_results.csv"), pd.read_csv("parsed_model_results.csv"), pd.read_csv("parsed_model_results.csv")]
	# for method in range(len(bin_methods)):
	# 	model_results = model_results_list[method]
	# 	bin_method = bin_methods[method]
	# 	model_results = pd.read_csv("parsed_model_results.csv")
	# 	model_results = model_results[(model_results["base_target"] == target) | (model_results["base_target"].isnull()) & (model_results["target"] == target)]
	# 	model_results = model_results[(model_results["extract_layer"] == 0) | (model_results["extract_layer"].isnull())]
	# 	if bin_method == None:
	# 		model_results = model_results[(model_results["bin_method"].isnull())]
	# 	else:
	# 		model_results = model_results[(model_results["bin_method"] == bin_method)]
	# 	model_results = model_results.drop(columns=["model_name", "loss", "MAD", "MADMAE", "bin_num", "bin_method", "extract_layer"])
	# 	model_results = model_results[["base_target", "target", "split", "MAE", "r2", "800_id", "400_id", "200_id"]]
	# 	model_results['MAE'] = model_results['MAE'].round(3)
	# 	model_results['r2'] = model_results['r2'].round(3)
	# 	model_results = model_results[model_results["split"].isin([0.50, 0.10, 0.05, 0.03, 0.01])]	

	# 	model_results_list[method] = model_results.reset_index()

	# data_list = []
	# for index, row in model_results_list[0].iterrows():
	# 	data_list.append(model_results_list[0].iloc[index, :])
	# 	data_list.append(model_results_list[1].iloc[index, :])
	# 	data_list.append(model_results_list[2].iloc[index, :])
	# model_results = pd.DataFrame(data_list)

	# model_results.to_csv(f"random/{target}.csv", index=False)
	# sys.exit(0)

	# SPECIFY SPLIT TO USE
	split_no = 1
	folder_path = f"other_files/split_{split_no}"

	# SPECIFY MODEL NAMES OT PARSE
	model_names = {
		"base_model" : "base",
		"hk_co2_transfer_model" : "transfer",
		"hk_h2o_transfer_model" : "transfer",
		"hk_n2_transfer_model" : "transfer",
		"hk_nh3_transfer_model" : "transfer",
		"base_model_binning10_binquantile" : "base",
		"hk_co2_transfer_model_binning10_binquantile" : "transfer",
		"hk_h2o_transfer_model_binning10_binquantile" : "transfer",
		"hk_n2_transfer_model_binning10_binquantile" : "transfer",
		"hk_nh3_transfer_model_binning10_binquantile" : "transfer",
		"base_model_binning10_binequal" : "base",
		"hk_co2_transfer_model_binning10_binequal" : "transfer",
		"hk_h2o_transfer_model_binning10_binequal" : "transfer",
		"hk_n2_transfer_model_binning10_binequal" : "transfer",
		"hk_nh3_transfer_model_binning10_binequal" : "transfer",
	}

	# PARSE TEST DATA FOR COMPARISONS
	train_path = "matched_MOFs_train.csv"
	val_path = "matched_MOFs_val.csv"
	test_path = "matched_MOFs_test.csv"
	_, _, x_test_data, _ = recover_data(folder_path, [train_path, val_path, test_path])

	train_path = "matched_targets_train.csv"
	val_path = "matched_targets_val.csv"
	test_path = "matched_targets_test.csv"
	_, _, y_test_data, test_names = recover_data(folder_path, [train_path, val_path, test_path])

	# ASSIGN TARGET INDEXING
	target_dict = {"hk_co2":0, "hk_h2o":1, "hk_n2":2, "hk_nh3":3}

	# CREATE PARSED DATAFRAME, SAVE, AND RECOVER PARITIES
	df, parities = parse_model_results(folder_path, model_names, x_test_data, y_test_data, test_names)
	df.to_csv("parsed_model_results.csv", index=False)
	parities = np.array(parities)
	np.save("parities.npy", parities)

	# PLOT AS NEEDED
	model_results = pd.read_csv("parsed_model_results.csv")
	model_results = model_results[model_results["bin_method"] == "equal"]
	plot_model_curves(".", model_results, "r2", "MAE", "hk_h2o", [0], y_lims=(0.4, 0.9), y_lims_2=(1.1, 2.1), x_label="% Training Data Used", y_label=r"$R^2$", y_label_2="MAE")
	plot_target_comparison(".", model_results, "r2", "MAE", "hk_nh3", [0], 0.03, y_lims=(0.3, 0.9), y_lims_2=(0, 2.4), y_label=r"$R^2$", y_label_2="MAE")
	plot_average_comparison(".", model_results, "r2", [0], 0.03, y_lims=(-0.15, 0.15), y_label='TL $R^2$ - Scratch $R^2$')
	plot_average_comparison(".", model_results, "MAE", [0], 0.03, y_lims=(-0.6, 0.4), y_label='TL MAE - Scratch MAE')

	# PLOT PARITIES
	parities = np.load("parities.npy")
	hk_co2, hk_h2o, hk_n2, hk_nh3 = parities[0], parities[1], parities[2], parities[3]

	# IF BLOCK FOR PLOTTING PARITIES
	if True:
		mpl.rcParams.update({
		    "font.family": "sans-serif",
		    "font.sans-serif": ["Arial"],
		    "axes.edgecolor": "black",
		    "axes.linewidth": 1.6,
		    "axes.grid": True,
		    "grid.color": "#e6e6e6",
		    "grid.linewidth": 0.6,
		    "xtick.direction": "in",
		    "ytick.direction": "in",
		    "xtick.major.pad": 10,
		    "xtick.major.size": 7,
		    "xtick.major.width": 2,
		    "ytick.major.size": 7,
		    "ytick.major.width": 2,
		    "xtick.top": True,
		    "ytick.right": True,
		    "xtick.labelsize": 30,
		    "ytick.labelsize": 30,
		    "axes.titlesize": 30,
		    "axes.titleweight": "bold",
		    "axes.labelsize": 32,
		    "figure.dpi": 300,
		    "savefig.dpi": 600,
		})

		SFS_SIZE   = 22
		SFS_CMAP   = "plasma"
		SFS_SCATTER_ALPHA = None
		SFS_ALPHA_MIN = 0.18
		SFS_ALPHA_MAX = 0.90
		SFS_ALPHA_GAMMA = 0.65

		# Parity line
		PARITY_LS = "--"
		PARITY_LW = 1.6
		PARITY_ALPHA = 0.8

		ticks = [[-13, -10, -7, -4], [-14, -7, 0, 7, 14], [-14, -13, -12, -11], [-12, -6, 0, 6, 12]]
		ticklabels = [["$10^{-13}$", "$10^{-10}$", "$10^{-17}$", "$10^{-4}$"], ["$10^{-14}$", "$10^{-7}$", "$10^{-0}$", "$10^{7}$", "$10^{14}$"], 
					["$10^{-14}$", "$10^{-13}$", "$10^{-12}$", "$10^{-11}$"], ["$10^{-12}$", "$10^{-6}$", "$10^{0}$", "$10^{6}$", "$10^{12}$"]]

		for parity_no in range(len(parities)):
			parity = parities[parity_no]

			fig, ax = plt.subplots(figsize=(8.7, 8.2))

			y_true = parity[0]
			prediction = parity[1]

			mn = np.min(ticks[parity_no])-2*(np.max(ticks[parity_no])-np.min(ticks[parity_no]))/24
			mx = np.max(ticks[parity_no])+2*(np.max(ticks[parity_no])-np.min(ticks[parity_no]))/24

			# Parity line only
			ax.plot([mn, mx], [mn, mx], color="black", lw=PARITY_LW, ls=PARITY_LS, alpha=PARITY_ALPHA, zorder=1)

			x = y_true.reshape(1, -1)
			y = prediction.reshape(1, -1)
			xy = np.vstack([x,y])
			z = gaussian_kde(xy)(xy)
			scaler = MinMaxScaler()
			z = scaler.fit_transform(z.reshape(-1, 1))

			ax.scatter(
			    y_true, prediction,
			    s=SFS_SIZE,
			    c=z,
			    alpha=0.4+z*0.6,
			    edgecolors="none",
			    zorder=2,
			    cmap='plasma'
			)
			plt.axis("square")

			# Colorbar 0..1
			sm = mpl.cm.ScalarMappable(cmap=plt.get_cmap(SFS_CMAP), norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
			sm.set_array([])
			cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
			cbar.ax.tick_params(labelsize=25, direction="in")
			cbar.set_label("Point Density", fontsize=27)

			# Labels and limits*
			ax.set_xlim(mn, mx)
			ax.set_ylim(mn, mx)
			ax.set_xlabel("Widom $K_H$ [mol/kg/Pa]")
			ax.set_ylabel("ML Predicted $K_H$ [mol/kg/Pa]")
			ax.set_xticks(ticks[parity_no])
			ax.set_yticks(ticks[parity_no])
			ax.set_xticklabels(ticklabels[parity_no])
			ax.set_yticklabels(ticklabels[parity_no])

			ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
			ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

			r2 = sk.r2_score(y_true, prediction)
			metric = sk.mean_absolute_error(y_true, prediction)#np.mean(np.abs(y_true - np.mean(y_true)))/sk.mean_absolute_error(y_true, prediction)
			print("RMSE: " + str(sk.root_mean_squared_error(y_true, prediction)))
			# Metrics box
			metrics_txt = (
			    f"R\u00b2 = {r2:.3f}\n"
			    f"MAE = {metric:.3f}"
			)
			ax.text(
			    0.05, 0.93, metrics_txt,
			    transform=ax.transAxes,
			    fontsize=30,
			    fontweight="bold",
			    va="top",
			    ha="left",
			    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8)
			)

			# Legend proxies
			sfs_handle = mpl.lines.Line2D([0], [0], marker='o', linestyle='None',
			                    markerfacecolor=plt.get_cmap(SFS_CMAP)(0.85),
			                    markeredgecolor='none',
			                    markersize=8, label='SFS')

			plt.tight_layout(pad=1.1)

			png1 = f"{list(target_dict.keys())[parity_no]}.png"
			plt.savefig(png1, bbox_inches="tight")
			plt.close(fig)














