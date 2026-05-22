import tensorflow as tf
import numpy as np
import random
import math
import os
import sklearn.metrics as sk
import shutil
import joblib

from hkml_base import *
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

# Build a transfer model given extracted layer number from original base model
def build_transfer_model(extraction_layer, input_shape, total_layers=3, params=(0.3, 128, ['sigmoid', 'sigmoid', 'relu'])):
	# Add a model input layer
	input_layer = layers.Input(shape=input_shape)
	last_layer = input_layer
	# Add missing layers, output layer
	for i in range(total_layers-extraction_layer):
		dropout = layers.Dropout(params[0])(last_layer)
		last_layer = layers.Dense(params[1], params[2][i+extraction_layer])(dropout)
	output = layers.Dense(1)(last_layer)
	return tf.keras.Model(inputs=input_layer, outputs=output)

if __name__ == "__main__":
	# Set original target, new targets for model construction
	base_target = "hk_co2"
	targets = ["hk_h2o", "hk_n2", "hk_nh3"]
	
	# Set data split to use
	split_no = 1
	# Set hyperparameters
	num_ml_iters = 3
	learning_rate = 0.0014
	batch_size = 128
	num_epochs = 200
	loss_fn = "mean_squared_error"

	# Set layers to extract outputs from base model
	extraction_layers = [0, 1, 2]
	# Set binning parameters
	bin_analysis = True
	bin_count = 10
	bin_method = "equal2"

	# Parameters for data splitting
	split_data = True
	splits = [1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01] if split_data else [1.0]
	split_seeding = True
	split_seeds = [26432, 55847, 27032, 76455, 3570, 59199, 10300]

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
	model_file = f"{base_target}_transfer_model"
	if (bin_analysis == True):
		model_file += f"_binning{bin_count}_bin{bin_method}"
	model_path = os.path.join(folder_path, model_file)
	os.makedirs(model_path, exist_ok=True)

	# Scale features for partial evaluation in corresponding base model
	base_feature_scaler = joblib.load(os.path.join(folder_path, f'base_model/{base_target}_100/{base_target}_100_f_scaler.kl'))
	x_train_base = base_feature_scaler.transform(x_train_data)
	x_val_base = base_feature_scaler.transform(x_val_data)
	x_test_base = base_feature_scaler.transform(x_test_data)

	# Build the base model to load file into
	base_model = model_builder(input_shape=x_train_base[0].shape, dropout=0.3, neurons=[128, 128, 128], num_cnn_layers=0, num_seq_layers=3, activation=["sigmoid", "sigmoid", "relu"])
	base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_fn)
	base_model = saving.load_model(os.path.join(folder_path, f'base_model/{base_target}_100/{base_target}_100.keras'), custom_objects=None, compile=True, safe_mode=True)

	# Create a list of model results for later csv parsing
	model_results = []

	# Open the note file for writing
	with open(os.path.join(model_path, note_file), "w") as note_file:
		# For each chosen layer to extract outputs from
		for extraction_layer in extraction_layers:
			# Create a new model based on the chosen extraction layer
			base_model_partial = Model(inputs=base_model.input, outputs=base_model.layers[1 + extraction_layer*2].output)
			# Extract neuron outputs from extraction layer given feature data
			x_train = np.array(base_model_partial.predict([x_train_base], verbose=0))
			x_val = np.array(base_model_partial.predict([x_val_base], verbose=0))
			x_test = np.array(base_model_partial.predict([x_test_base], verbose=0))

			# Add hyperparameter notes
			note_file.write(f"extraction_layer: {extraction_layer}\n")
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
						_, x_train_split, _, y_train_split = train_test_split(x_train, y_train_data, test_size=splits[split], random_state=split_seed)
						_, x_val_split, _, y_val_split = train_test_split(x_val, y_val_data, test_size=splits[split], random_state=split_seed)
						x_test_split, y_test_split = x_test, y_test_data
					elif (bin_analysis):
						x_train_split, x_val_split, x_test_split, y_train_split, y_val_split, y_test_split = split_bins(
								(x_train, x_val, x_test), 
								(y_train_data, y_val_data, y_test_data), 
								bin_count, 
								target_dict[target], 
								splits[split],
								seed=split_seed,
								bin_method=bin_method)
					# Don't split anything
					else:
						x_train_split, y_train_split = x_train, y_train_data
						x_val_split, y_val_split = x_val, y_val_data
						x_test_split, y_test_split = x_test, y_test_data

					# Isolate corresponding data for targets
					y_train = y_train_split[:, target_dict[target]].reshape(-1, 1)
					y_val = y_val_split[:, target_dict[target]].reshape(-1, 1)
					y_test = y_test_split[:, target_dict[target]].reshape(-1, 1)

					# Scale targets (IMPORTANT TO DO THIS AFTER SPLITTING, OR WILL OBTAIN INFORMATION OUT-OF-SAMPLE)
					target_scaler = StandardScaler()
					Y_train = target_scaler.fit_transform(y_train)
					Y_val = target_scaler.transform(y_val)
					Y_test = target_scaler.transform(y_test)

					# Set model name, create model directories
					model_name = f"{base_target}->{target}_{splits[split]*100:.0f}_extract{extraction_layer}"
					curr_model_path = os.path.join(model_path, model_name)
					os.makedirs(curr_model_path, exist_ok=True)

					# Save scalers for the model
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
						transfer_model = build_transfer_model(extraction_layer, x_train_split[0].shape)
						transfer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_fn)
						# Fit according to hyperparameters
						history = transfer_model.fit(x_train_split, Y_train, epochs=num_epochs, validation_data=(x_val_split, Y_val), batch_size=batch_size, verbose=1)

						# Evaluate the model loss on the scaled outputs, make a prediction as needed
						loss = transfer_model.evaluate(x_test_split, Y_test, verbose=0)
						prediction = transfer_model.predict(x_test_split, verbose=0)
						# If the model loss breaks the current record, reset the record and save the new model
						if (loss < record_loss):
							record_loss = loss
							record_seed = seed
							record_prediction = prediction
							record_history = history
							transfer_model.save(os.path.join(curr_model_path, f'{model_name}.keras'))
						# Send a status update
						print(f'Model: {model_name}, Loss: {loss}, Record Loss: {record_loss}, Iteration: {i}, Seed: {seed}')

					# Load the best performing model
					transfer_model = saving.load_model(os.path.join(curr_model_path, f'{model_name}.keras'), custom_objects=None, compile=True, safe_mode=True)
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
					note_file.write(f"model: {model_name}, target: {target}, {loss_fn} loss: {record_loss:.6f}, r2: {r2:.6f}, model seed: {record_seed}, split: {splits[split]}, split seed: {split_seed}, extract: {extraction_layer}\n")
	
					# Write model training curves into archive file 
					with open(os.path.join(curr_model_path, "training_curve.csv"), "w") as archive_file:
						archive_file.write("epoch,train_loss,val_loss\n")
						for epoch_no in range(len(record_history.history['loss'])):
							archive_file.write(",".join([str(x) for x in (epoch_no, record_history.history['loss'][epoch_no], record_history.history['val_loss'][epoch_no])])+"\n")

					# Add model results to results list
					model_results.append([model_name, target, record_loss, r2, splits[split], extraction_layer])

	# Write model results to file for further processing
	with open(os.path.join(model_path, "model_results.csv"), "w") as results_file:
		results_file.write("model_name,target,loss,r2,split,extract_layer\n")
		for result in model_results:
			results_file.write(",".join([str(x) for x in result])+"\n")












