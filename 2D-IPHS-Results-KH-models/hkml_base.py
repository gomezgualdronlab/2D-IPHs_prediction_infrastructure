import tensorflow as tf
import numpy as np
import random
import math
import os
import sklearn.metrics as sk
import joblib

from tensorflow.keras import layers, models
from tensorflow.keras import saving
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Load data from provided 3 files in a list
def recover_data(folder_path, files):
    # Load data from each file
    train_data = pd.read_csv(os.path.join(folder_path, files[0]))
    val_data = pd.read_csv(os.path.join(folder_path, files[1]))
    test_data = pd.read_csv(os.path.join(folder_path, files[2]))
    
    # Separate out and store MOF names
    test_names = test_data.iloc[:, 0].to_list()
    
    # Drop name column from dataframes
    train_data = train_data.drop(columns=[train_data.columns[0]]).values
    val_data = val_data.drop(columns=[val_data.columns[0]]).values
    test_data = test_data.drop(columns=[test_data.columns[0]]).values
  	
    return train_data, val_data, test_data, test_names

# Systematically build a sequential model given the provided hyperparameters
def model_builder(input_shape, dropout=0.1, kernel_size=(0, 0), filters=[], neurons=[128, 128, 128], num_cnn_layers=0, num_seq_layers=3, strides=(0, 0), activation=["sigmoid", "sigmoid", "relu"], output_num=1):
	# Initialize list of model layers
	model_layers = []

	# Add all CNN layers
	for layer_no in range(num_cnn_layers):
		model_layers.append(layers.Conv2D(filters[layer_no], kernel_size=kernel_size, strides=strides, padding='same', activation='relu'))
		model_layers.append(layers.MaxPooling2D(padding='same'))
	
	# Flatten CNN output
	if (num_cnn_layers != 0):
		model_layers.append(layers.Flatten())

	# Add all sequential layers
	for layer_no in range(num_seq_layers):
		model_layers.append(layers.Dense(neurons[layer_no], activation=activation[layer_no]))
		model_layers.append(layers.Dropout(dropout))

	# Compile layers into a full model with input and output
	input_layer = layers.Input(shape=input_shape)
	last_layer = input_layer
	for layer in model_layers:
		layer = layer(last_layer)
		last_layer = layer
	layer = layers.Dense(1)(last_layer)
	model = tf.keras.Model(inputs=input_layer, outputs=layer)

	# Return model
	return model

# Create a data split for features, targets using chosen binning method and given target to sort by
def split_bins(features, targets, bin_count, target_number, split, seed=0, bin_method="equal"):
	# Create places to store binned data, split data
	output_features = [[], []]
	output_targets = [[], []]
	bins = {}
	# Iterate over the training/validation data
	for dataset_no in range(len(features)-1):
		# Reset bins, if they exist
		for val in bins:
			bins[val] = []

		# Collect chosen target, target set, and feature set across provided data as tuples
		collected_tuples = []
		feature_set = features[dataset_no]
		target_set = targets[dataset_no]
		for target_no in range(len(target_set)):
			collected_tuples.append((target_set[target_no][target_number], target_set[target_no], feature_set[target_no]))

		# Sort collected tuples by the chosen target
		collected_tuples = sorted(collected_tuples, key=lambda item: item[0])
		# If we are looking at the training dataset, need to set the bin values
		if (dataset_no == 0):
			# If the binning method is equidistant
			if (bin_method == "equal"):
				# Find the per-bin span by dividing sorted data range by the total number of bins
				per_bin_span = (collected_tuples[-1][0] - collected_tuples[0][0])/bin_count
				# Store starting value for each bin
				start_val = collected_tuples[0][0]
				# Create bins by repeatedly adding the span of each bin to the start value
				for bin_no in range(bin_count):
					bins[(start_val, start_val+per_bin_span)] = []
					start_val += per_bin_span
			# If the binning method is quantile
			elif (bin_method == "quantile"):
				# Jump by indices, since quantiles of sorted values correspond to every index: len(sorted)/n_bins 
				per_bin_jump = math.floor(len(collected_tuples)/bin_count)
				start_index = 0
				# Create bins by accessing values from the sorted list at each quantile
				for bin_no in range(bin_count):
					bins[(collected_tuples[start_index][0], collected_tuples[start_index+per_bin_jump][0])] = []
					start_index += per_bin_jump
			# If an invalid binning method was selected, raise an error
			else:
				raise Exception

		# Iterate over the list of sorted values, add each value to the matching bin
		for val in collected_tuples:
			for bin_range in bins:
				if (val[0] >= bin_range[0] and val[0] < bin_range[1]):
					bins[bin_range].append(val)
		# For each bin
		for bin_no in bins:
			# Add the features and targets collected in each bin to lists
			x = []
			y = []
			for item in bins[bin_no]:
				x.append(item[2])
				y.append(item[1])
			# If the given parameters make a valid split, split the collected per-bin data
			if (split != 1.0 and len(x) >= 2):
				_, x_split, _, y_split = train_test_split(x, y, test_size=split, random_state=seed)
			# Otherwise, we don't need to split
			else:
				x_split, y_split = x, y

			# Add split values to the output lists
			output_features[dataset_no] += list(x_split)
			output_targets[dataset_no] += list(y_split)

		# Create a plot of the binned distributions
		"""fig, ax = plt.subplots()
		x = target_set[:, target_number].reshape(1, -1)
		y = np.random.rand(x.shape[1])
		y /= 1000
		y = y.reshape(1, -1)

		print(x.shape)
		print(y.shape)
		xy = np.vstack([x,y])
		z = gaussian_kde(xy)(xy)
		scaler = MinMaxScaler()
		z = scaler.fit_transform(np.log(z.reshape(-1, 1)))
		for bin_no in bins:
			ax.axvline(x=bin_no[0], color='black', linestyle='--', linewidth=2, zorder=3)
			ax.axvline(x=bin_no[1], color='black', linestyle='--', linewidth=2, zorder=3)
		cax = plt.scatter(x, z, c=z, cmap="viridis", s=20)
		plt.show()"""

		# Shuffle the outputted list
		output_features[dataset_no], output_targets[dataset_no] = shuffle(output_features[dataset_no], output_targets[dataset_no], random_state=seed)

	return np.array(output_features[0]), np.array(output_features[1]), np.array(features[2]), np.array(output_targets[0]), np.array(output_targets[1]), np.array(targets[2])

# Creates a density-colored parity plot for model results
def plot_data(y_true, record_prediction, target_no):
	# Estimate point density of data using Gaussian KDE as continuous PDF estimation
	x = y_true.reshape(1, -1)
	y = record_prediction.reshape(1, -1)
	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)
	scaler = MinMaxScaler()
	z = scaler.fit_transform(z.reshape(-1, 1))

	# Set tick marks for resulting plots, CHANGE AS NEEDED
	ticks = [[-4, -7, -10, -13], [14, 7, 1, -7, -14], [-14, -13, -12, -11], [13, 7, 1, -7, -13]]
	fig, ax = plt.subplots(dpi=400)
	
	# Create a parity plot for the best performing ML model, plot with color as the point density 
	plt.axis("square")
	plt.axline(xy1=[0, 0], slope=1, color='k', zorder=1)
	ax.tick_params(axis='x', labelsize=20, direction='in', length=10, pad=5.5, which='major')
	ax.tick_params(axis='y', labelsize=20, direction='in', length=10, pad=5.5, which='major')
	ax.tick_params(axis='x', which='minor', direction='in', left=False)
	ax.tick_params(axis='y', which='minor', direction='in', left=False)
	plt.xlim(min(ticks[target_no]), max(ticks[target_no]))
	plt.ylim(min(ticks[target_no]), max(ticks[target_no]))
	ax.set_xticks(ticks[target_no])
	ax.set_yticks(ticks[target_no])
	cax = plt.scatter(y_true, record_prediction, c=z, cmap="viridis", s=20)
	#cbar = plt.colorbar(cax)

if __name__ == "__main__":
	# Set data split to use
	split_no = 4
	# Set hyperparameters
	num_ml_iters = 3
	learning_rate = 0.0014
	batch_size = 128
	num_epochs = 200
	loss_fn = "mean_squared_error"
	# Set targets for model construction
	targets = ["hk_co2", "hk_h2o", "hk_n2", "hk_nh3"]

	# Parameters for data splitting
	split_data = False
	splits = [0.01] if split_data else [1.0]
	split_seeding = True
	split_seeds = [10300]

	# Set binning parameters
	bin_analysis = False
	bin_count = 10
	bin_method = "equal"

	# Find the folder for data, and create a name for the model notes
	folder_path = f"other_files/split_{split_no}"
	note_file = f"model_notes.txt"

	# Extract features and targets
	train_path = "matched_MOFs_train.csv"
	val_path = "matched_MOFs_val.csv"
	test_path = "matched_MOFs_test.csv"
	x_train_data, x_val_data, x_test_data, x_test_names = recover_data(folder_path, [train_path, val_path, test_path])

	train_path = "matched_targets_train.csv"
	val_path = "matched_targets_val.csv"
	test_path = "matched_targets_test.csv"
	y_train_data, y_val_data, y_test_data, y_test_names = recover_data(folder_path, [train_path, val_path, test_path])

	# Match targets to indices in the data
	target_dict = {"hk_co2":0, "hk_h2o":1, "hk_n2":2, "hk_nh3":3}

	# Create a folder for the new models
	model_path = os.path.join(folder_path, "base_model") if not bin_analysis else os.path.join(folder_path, f"base_model1_binning{bin_count}_bin{bin_method}")
	os.makedirs(model_path, exist_ok=True)

	# Create a list of model results for later csv parsing
	model_results = []

	# Open the note file for writing
	with open(os.path.join(model_path, note_file), "w") as note_file:
		# Add hyperparameter notes, chosen targets, splits for prediction
		note_file.write(f"num epochs: {num_epochs}\n")
		note_file.write(f"targets: {", ".join(targets)}\n")
		note_file.write(f"splits: {", ".join([str(x) for x in splits])}\n")
		# For each target
		for target in targets:
			# For each split
			for split in range(len(splits)):
				# Set appropriate seed for split, write to note file
				split_seed = split_seeds[split] if split_seeding else math.floor(random.random()*100000)
				note_file.write(f"split: {splits[split]*100:.1f}%, seed: {split_seed}\n")

				# If we aren't running a bin_analysis and sk.train_test_split is applicable, split
				# IMPORTANT: target data inputted is NOT SCALED
				if (splits[split] != 1 and not bin_analysis):
					_, x_train_split, _, y_train_split = train_test_split(x_train_data, y_train_data, test_size=splits[split], random_state=split_seed)
					_, x_val_split, _, y_val_split = train_test_split(x_val_data, y_val_data, test_size=splits[split], random_state=split_seed)
					x_test_split, y_test_split = x_test_data, y_test_data
				elif (bin_analysis):
					x_train_split, x_val_split, x_test_split, y_train_split, y_val_split, y_test_split = split_bins(
							(x_train_data, x_val_data, x_test_data), 
							(y_train_data, y_val_data, y_test_data), 
							bin_count, 
							target_dict[target], 
							splits[split],
							seed=split_seed,
							bin_method=bin_method)
				# Don't split anything
				else:
					x_train_split, y_train_split = x_train_data, y_train_data
					x_val_split, y_val_split = x_val_data, y_val_data
					x_test_split, y_test_split = x_test_data, y_test_data

				# Isolate corresponding data for targets
				y_train = y_train_split[:, target_dict[target]].reshape(-1, 1)
				y_val = y_val_split[:, target_dict[target]].reshape(-1, 1)
				y_test = y_test_split[:, target_dict[target]].reshape(-1, 1)

				# Create scalers, scale features and targets
				feature_scaler = StandardScaler()
				x_train = feature_scaler.fit_transform(x_train_split)
				x_val = feature_scaler.transform(x_val_split)
				x_test = feature_scaler.transform(x_test_split)

				target_scaler = StandardScaler()
				Y_train = target_scaler.fit_transform(y_train)
				Y_val = target_scaler.transform(y_val)
				Y_test = target_scaler.transform(y_test)

				# Set model name, create model directories
				model_name = f"{target}_{splits[split]*100:.0f}"
				curr_model_path = os.path.join(model_path, model_name)
				os.makedirs(curr_model_path, exist_ok=True)

				# Save scalers for the model
				joblib.dump(feature_scaler, os.path.join(curr_model_path, f"{model_name}_f_scaler.kl"))
				joblib.dump(target_scaler, os.path.join(curr_model_path, f"{model_name}_t_scaler.kl"))

				# Store record holders for performance, parameters
				record_seed = 0
				record_prediction = []
				record_history = []
				record_loss = np.inf
				# Iterate over the chosen number of ML models for each target
				for i in range(num_ml_iters):
					# Set a random seed
					seed = math.floor(random.random()*100000)
					tf.random.set_seed(seed)
					# Build and compile optimal model
					model = model_builder(input_shape=x_train[0].shape, dropout=0.3, neurons=[128, 128, 128], num_cnn_layers=0, num_seq_layers=3, activation=["sigmoid", "sigmoid", "relu"])
					model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_fn)
					# Fit according to hyperparameters
					history = model.fit(x_train, Y_train, epochs=num_epochs, validation_data=(x_val, Y_val), batch_size=batch_size, verbose=1)

					# Evaluate the model loss on the scaled outputs, make a prediction as needed
					loss = model.evaluate(x_test, Y_test, verbose=0)
					prediction = model.predict(x_test, verbose=0)
					# If the model loss breaks the current record, reset the record and save the new model
					if (loss < record_loss):
						record_loss = loss
						record_seed = seed
						record_prediction = prediction
						record_history = history
						model.save(os.path.join(curr_model_path, f'{model_name}.keras'))
					# Send a status update
					print(f'Model: {model_name}, Loss: {loss}, Record Loss: {record_loss}, Iteration: {i}, Seed: {seed}')

				# Load the best performing model
				model = saving.load_model(os.path.join(curr_model_path, f'{model_name}.keras'), custom_objects=None, compile=True, safe_mode=True)
				# Invert scale on prediction of best model
				record_prediction = target_scaler.inverse_transform(record_prediction)

				# Collect true target data, evaluate MSE and r2 with the log(HK) data
				y_true = y_test
				lossMSE = sk.mean_squared_error(record_prediction, y_true)
				r2 = sk.r2_score(y_true, record_prediction)

				# Create a plot of model results & save
				plot_data(y_true, record_prediction, target_dict[target])
				plt.savefig(os.path.join(model_path, f"{model_name}.png"))

				# Write model notes into note file
				note_file.write(f"model: {model_name}, target: {target}, {loss_fn} loss: {record_loss:.6f}, r2: {r2:.6f}, model seed: {record_seed}, split: {splits[split]}, split seed: {split_seed}\n")

				# Write model training curves into archive file 
				with open(os.path.join(curr_model_path, "training_curve.csv"), "w") as archive_file:
					archive_file.write("epoch,train_loss,val_loss\n")
					for epoch_no in range(len(record_history.history['loss'])):
						archive_file.write(",".join([str(x) for x in (epoch_no, record_history.history['loss'][epoch_no], record_history.history['val_loss'][epoch_no])])+"\n")

				# Add model results to results list
				model_results.append([model_name, target, record_loss, r2, splits[split]])

	# Write model results to file for further processing
	with open(os.path.join(model_path, "model_results.csv"), "w") as results_file:
		results_file.write("model_name,target,loss,r2,split\n")
		for result in model_results:
			results_file.write(",".join([str(x) for x in result])+"\n")


