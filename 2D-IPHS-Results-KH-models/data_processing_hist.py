import numpy as np
import pandas as pd
import sklearn.metrics as sk
import os
import shutil
import matplotlib.pyplot as plt
import math
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Parses data from input paths to matched features + targets and remaining structures
# NOTE: output_files is a list of 3 files: (matched_MOFs, matched_targets, unmatched_MOFs)
def recover_data(feature_npys, feature_list, target_data_path, output_files, target_list=["hk_co2", "hk_h2o", "hk_n2", "hk_nh3"], input_shape=(74, 20, 3), target_size=4, cell_size_truncate=6):
	# Find all MOF.npys from histogram outputs
	npys = np.load(feature_npys)
	# Read in target Henry's constant files
	target_data = pd.read_csv(target_data_path)

	# Iterate through feature list and identify matched, unmatched MOFs and corresponding targets
	# Each found MOF will be stored as (name, index, targets)
	# Each not found MOF will be stored as (name, index)
	found_data = []
	nfound_data = []
	# Need feature index to match npys with MOF names
	feature_index = 0
	# Open list of MOF names (feature names)
	with open(feature_list, "r") as txt:
		# For each MOF name
		for feature in txt:
			# Process the MOF name
			MOF_name = feature.rstrip()
			# HEURISTIC: Henry's constant dataset only contains MOF names starting with "S"
			# Only consider MOF names that start with S
			if (MOF_name[0] != 'S'):
				nfound_data.append((MOF_name, feature_index))
				feature_index += 1
				continue
			# HEURISTIC: If the last index corresponds to a number, need to filter dimensions from end
			# Example: _1x1x1 is 6 indices long, remove from end to obtain isolated MOF name
			else:
				# Try check will fail if last index is not a number
				try:
					value_test = int(MOF_name[-1])
					MOF_name = MOF_name[:-6]
				except ValueError:
					value_test = -1
			# Check for a set of targets that matches the MOF name
			feature_targets = target_data[target_data["MOF_name"] == MOF_name]
			# If the obtained list is not empty
			if (not feature_targets.empty):
				# Convert the obtained pandas Series to a list
				feature_targets = feature_targets.values.tolist()[0][-target_size:]
				# Check for null targets, throw MOF name into unmatched data
				if (not np.any(np.isnan(feature_targets))):
					# Append to matched MOFs if all checks are passed
					found_data.append((MOF_name, feature_index, feature_targets))
				else:
					nfound_data.append((MOF_name, feature_index))
			# If the target list is empty, no matching targets were found
			else:
				nfound_data.append((MOF_name, feature_index))
			feature_index += 1
			# Print update
			print(f"Processed Count: {feature_index}/{npys.shape[0]} structures, Total Matches: {len(found_data)} structures")

	# Open matched MOFs file
	with open(output_files[0], "w") as feature_file:
		# Add header containing columns for each feature
		header = "MOF_name,"
		for feature_no in range(math.prod(input_shape)):
			header += f"feature_{feature_no},"
		feature_file.write(header[:-1]+'\n')

		# Open matched targets file
		with open(output_files[1], "w") as target_file:
			# Add header containing columns for each target
			header = "MOF_name,"
			for target in target_list:
				header += f"{target},"
			target_file.write(header[:-1]+'\n')

			# For each found entry
			for entry in found_data:
				# Find the corresponding .npy file and store corresponding name
				feature_npy = npys[entry[1]]
				line = f"{entry[0]},"
				# Flatten the histogram in the npy to a vector, write to file
				line += flatten_hist(feature_npy, input_shape)
				feature_file.write(line[:-1]+'\n')

				# Write name, corresponding targets to file
				# NOTE: this is where log transform of HK is taken
				line = f"{entry[0]},"
				for val in entry[2]:
					line += str(np.log(val)) + ','
				target_file.write(line[:-1]+'\n')

	# Open unmatched MOFs file
	with open(output_files[2], "w") as feature_file:
		# Add header containing columns for each feature
		header = "MOF_name,"
		for feature_no in range(math.prod(input_shape)):
			header += f"feature_{feature_no},"
		feature_file.write(header[:-1]+'\n')

		# For each not found entry
		for entry in nfound_data:
			# Flatten the histogram in the npy to a vector, write to file
			feature_npy = npys[entry[1]]
			line = f"{entry[0]},"
			line += flatten_hist(feature_npy, input_shape)
			feature_file.write(line[:-1]+'\n')

	return len(found_data), len(nfound_data)

# Flatten an nparray into a consistently formatted vector
def flatten_hist(hist_npy, input_shape):
	output_str=""
	# Flatten by feature type, density, then distance
	for dim1 in range(input_shape[2]):
		for dim2 in range(input_shape[0]):
			for dim3 in range(input_shape[1]):
				output_str += f"{str(hist_npy[dim2, dim3, dim1])},"
	return output_str

# Given matched features and targets, create random train/validation/test splits
def generate_splits(feature_file, target_file, split_count, dir_path=".", test_split=0.2, val_split=0.2):
	# Open files and read data, store headers for later formatting
	with open(os.path.join(dir_path, feature_file), "r") as feature_f:
		lines = feature_f.readlines()
		feature_header = lines[0]
		features = lines[1:]
	with open(os.path.join(dir_path, target_file), "r") as target_f:
		lines = target_f.readlines()
		target_header = lines[0]
		targets = lines[1:]

	# For each split
	for split in range(split_count):
		# Create a seed
		seed = math.floor(random.random()*100000)

		# Generate a path to the split data
		split_path = f"{dir_path}split_{split+2}/"
		if (os.path.exists(split_path)):
			pass#shutil.rmtree(split_path, ignore_errors=True)
		else:
			os.makedirs(split_path)

		# Create a note file in the new directory with the seed and split sizes
		with open(os.path.join(split_path, "split.txt"), "w") as note_file:
			note_file.write(f"split seed: {seed}\n")
			note_file.write(f"train/test split: {(1-test_split)*100:.1f}%/{test_split*100:.1f}%\n")
			note_file.write(f"train/val split: {(1-val_split)*100:.1f}%/{val_split*100:.1f}%\n")
			note_file.write(f"train/val/test split: {(1-test_split)*(1-val_split)*100:.1f}%/{(1-(1-test_split)*(1-val_split)-test_split)*100:.1f}%/{test_split*100:.1f}%\n")

		# Shuffle features, targets, and split using given numbers
		features, targets = shuffle(features, targets, random_state=seed)
		x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=test_split, random_state=seed)
		x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=seed)

		# Write splits to 3 feature files, 3 target files
		with open(os.path.join(split_path, feature_file[:-4]+"_train.csv"), "w") as feature_f:
			feature_f.write(feature_header)
			for feature in x_train:
				feature_f.write(feature)

		with open(os.path.join(split_path, feature_file[:-4]+"_val.csv"), "w") as feature_f:
			feature_f.write(feature_header)
			for feature in x_val:
				feature_f.write(feature)

		with open(os.path.join(split_path, feature_file[:-4]+"_test.csv"), "w") as feature_f:
			feature_f.write(feature_header)
			for feature in x_test:
				feature_f.write(feature)

		with open(os.path.join(split_path, target_file[:-4]+"_train.csv"), "w") as target_f:
			target_f.write(target_header)
			for target in y_train:
				target_f.write(target)

		with open(os.path.join(split_path, target_file[:-4]+"_val.csv"), "w") as target_f:
			target_f.write(target_header)
			for target in y_val:
				target_f.write(target)

		with open(os.path.join(split_path, target_file[:-4]+"_test.csv"), "w") as target_f:
			target_f.write(target_header)
			for target in y_test:
				target_f.write(target)

if __name__ == "__main__":
	data_path = "other_files/"
	# REQUIRES FULL, ARCHIVED DATAFILES AND IS NOW DEPRECATED
	# recover_output = recover_data(os.path.join(data_path, "MOF_images.npy"), 
	# 										os.path.join(data_path, "MOF_images.txt"), 
	# 										os.path.join(data_path, "HK_50K_ALL_MOLECULES_TEXTURALS.csv"), 
	# 										(os.path.join(data_path, "matched_MOFs.csv"), os.path.join(data_path, "matched_targets.csv"), os.path.join(data_path, "unmatched_MOFs.csv")), 
	# 										target_list=["log(hk_co2)", "log(hk_h2o)", "log(hk_n2)", "log(hk_nh3)"], input_shape=(74, 20, 3), target_size=4, cell_size_truncate=6)
	# Generate split_x/ directories with data splits
	generate_splits("matched_MOFs.csv", "matched_targets.csv", 4, dir_path=data_path, test_split=0.2, val_split=0.2)



















