# hk_ML_models


## Overview
This directory contains the ML models used for regression of Henry's constants on a set of MOFs. In addition to these models, data used for training,  training implementation files, and plotting files are also included. 

Last modified: 2/15/2026
File size: 890 MB

---

## Dependencies
- joblib1.5.0
- matplotlib3.10.3
- numpy2.1.3
- pandas2.2.3
- scikit-learn1.6.1
- scipy1.15.3
- tensorflow2.19.0

---

## Organization
Code implementation, model summaries, and plots are contained within the base directory (hk_ML_models/). All data, model .keras files, and corresponding scalers are contained within the other_files/ directory.

---

## hk_ML_models
hk_ML_models contains the following:
- x.png: various plots originating from output_handling.py
- parities.npy: output npy file containing scratch model parities, originating from output_handling.py
- parsed_model_results.csv: output pandas DataFrame/.csv file containing model names, base targets, transfer targets, data splits, model performance metrics, and other metadata for a **single split_x/ folder** found in other_files/, originating from output_handling.py
- data_processing_hist.py: preliminary data processing file, handles data matching for archived files and data splitting for split_x/ folder generation
- hkml_base.py: creates scratch Henry's constant predictors given a split_x/ directory
- hkml_transfer.py: creates transfer Henry's constant predictors given a split_x/ directory
- output_handling.py: data handling file, parses model results from a **single split_x/ folder** and creates parsed_model_results.csv and plots for presentation purposes.

---

## **data_processing_hist.py**
Handles data recovery from files, histogram flattening, and feature/target file set-up for MOF-Henry's constant combinations

### recover_data
Parses data from input paths to match MOF features and Henry's constant targets, separating remaining structures. MOF features are numerically labeled (feature_0, feature_1, feature_2, etc.), and MOF names are preserved across all files to ensure overlap. Henry's constants are scaled via a log_10 transformation to ensure a semi-normal distribution for model training.

Parameters:
- str feature_npys -> .npy filehandle containing a nparray of all **unflattened** histograms (i.e., 3D histograms)
- str feature_list -> .txt filehandle containing a list of the MOF names corresponding to feature_npys
- str target_data_path -> .csv filehandle containing Henry's constants and matching MOF names
- str list output_files -> contains 3 .csv filehandles for storage of feature MOF histograms, target Henry's constants, and unmatched MOF histograms
- str list target_files -> contains Henry's constants labels for storage
- int tuple input_shape -> holds histogram dimensions for bookkeeping
- int target_size -> number of Henry's constant target types (i.e., CO2, H2O)
- int cell_size_truncate -> last-n MOF name truncation for cell size specification removal

### flatten_hist
Flattens a given histogram (nparray) into a 1D list for ease of storage

Parameters:
- np.array hist_npy -> contains the histogram to be flattened
- int tuple input_shape -> shape of the input histogram

Output:
- str output_str -> 1D list of the histogram, joined with commas for file writing

### generate_splits
Given files for matched feature/target MOFs, creates split_x/ directories containing unique data splits for unbiased model training

Parameters:
- str feature_file -> .csv filehandle containing MOF features (i.e. 1D histograms)
- str target_file ->  .csv filehandle containing Henry's constant targets
- int split_count -> number of split directories to make
- str dir_path -> directory path for split directories
- float test_split -> fraction of **total data** to be used for model testing
- float val_split -> fraction of **total data after test data removal** to be used for model validation

---

## **hkml_base.py**
Handles model building, model training, and model saving for all scratch modeling. Main method handles all model training and hyperparameter specifications

### recover_data
Parses model training data (features or targets) from input paths to recover lists for model training

Parameters:
- string folder_path -> .npy filehandle containing a nparray of all **unflattened** histograms (i.e., 3-D histograms)
- string list files -> 3 .csv filehandles corresponding to training, validation, and testing data for model training

Output:
- float list list -> training data extracted
- float list list -> validation data extracted
- float list list -> testing data extracted
- str list list -> corresponding MOF names for the extracted testing data

### model_builder
Systematically builds a CNN/MLP based on user-defined hyperparameters and using the tensorflow model building interface. Note that an extra input and output layer will **always** be included (input layer has dimensions according to input_shape, output_layer has dimensions according to output_num).

Parameters:
- int tuple input_shape -> dimensions of model inputs (i.e., for 1D histograms, this is 4,440)
- float dropout -> dropout to include between **every** dense neuron layer
- int tuple kernel_size -> CNN: kernel size for convolution
- int list filters -> CNN: filter counts for each convolution layer (must have entry for each layer specified in num_cnn_layers)
- int list neurons -> neuron counts for each MLP layer (must have entry for each layer specified in num_seq_layers)
- int num_cnn_layers -> CNN: number of convolution layers
- int num_seq_layers -> number of MLP layers
- int list strides -> CNN: strides for each convolution layer
- str list activation -> activation function to use for each MLP layer (formatted according to tensorflow guidelines)
- int output_num -> number of outputs at last layer

Output:
- tf.model -> uncompiled model built to required specification

### split_bins
Artificially creates data scarcity by "splitting" given data into bins and sampling to the user-defined extent from each bin.

Parameters:
- float list list list features -> list of feature data (training, validation, testing) to sample (**testing data will be left untouched**)
- float list list list targets -> list of target data (training, validation, testing) to sample (**testing data will be left untouched**)
- int bin_count -> number of bins to create
- int target_number -> column index in targets to base sampling methods on
- float split -> fraction of data to sample
- int seed -> random seed to use while randomly sampling data
- str bin_method -> binning method to use (choose between "equal" for equidistant binning and "quantile" for quantile binning)

Output:
- float list list -> newly sampled training set feature data
- float list list -> newly sampled validation set feature data
- float list list -> newly sampled testing set feature data
- float list list -> newly sampled training set target data
- float list list -> newly sampled validation set target data
- float list list -> newly sampled testing set target data

### plot_data
Creates a rudimentary density-colored parity plot for the given model input/result pair

Parameters:
- float list y_true -> ground truth values for the given **single target**
- float list record_prediction -> ML-predicted values for the given **single target**
- int target_no -> index of target (**used for tick mark indexing inside function**)

Output:
- plt -> implicitly created plt object for saving

---

## **hkml_transfer.py**
Handles model building, model training, and model saving for all transfer modeling. Main method handles all model training and hyperparameter specifications

### build_transfer_model
Creates a new transfer model based on the extracted layer from the scratch (base) model

Parameters:
- int extraction_layer -> extraction layer from base model
- int tuple input_shape -> input shape into transfer model (AKA output count of the extracted layer)
- int total_layers -> total layer count of base model
- tuple params -> hyperparameters for base model training (dropout, batch size, and activation functions)

Output:
- tf.model -> newly created transfer model

---

## **output_handling.py**
Compiles model results for a **single split** directory, and handles plot creation for results presentation. Note that parity plot creation code is included at the bottom of the file for editing convenience, and all plotting functions have defined parameters that must be edited for plot formatting.

### format_axes
Formats an input mpl axis for plot creation

Parameters:
- mpl.ax ax -> axis to format
- bool right -> boolean for whether the axis is on the right or not
- float lw -> line width of the axis

### darken_color
Darkens an input color for distinction between two plotted items of the same color.

Parameters:
- str color -> color in hexadecimal
- int op -> amount to darken by

Output:
- str -> darkened colro in hexadecimal

### extract_predictions
Given model metadata, extracts a model prediction from a given split folder 

Parameters:
- list model_data -> model metadata, including data scaracity, base target, transfer target, extract layer, and binning method
- str model_type -> whether the model is a base (scratch) model, "base", or transfer model, "transfer"
- float list x_test_data -> feature data to evaluate the model on
- str split_folder -> path to the split_x/ folder where the model is housed
- str model_name -> the model's name

Output:
- float list -> corresponding unscaled target data

### top_overlap
Checks the top-n overlap of MOFs between a ground truth and ML-prediction pair (ranked according to a selected Henry's constant)

Parameters:
- int top_threshold -> top threshold to consider (e.g., 200)
- float list y_true -> ground truth target data (indexed identically to y_names)
- float list y_predict -> ML-predicted target data (indexed identically to y_names)
- str list y_names -> MOF names for target data

Output:
- int -> number of overlapping MOFs

### parse_model_results
Given a list of model classifications, parse the corresponding model metadata and performance metrics into a DataFrame for storage and data manipulation

Parameters:
- str folder_path -> path to directory containing all model results (i.e., split_x/)
- str:str dict model_names -> mapping of model name to model type ("base" or "transfer") for iterating through
- float list list x_test -> feature data to evaluate models on (index identically to test_names)
- float list list y_test -> target data to evaluate models on (contains all targets) (index identically to test_names)
- str list test_names -> names of MOFs that models are evaluated on
- str:int dict target_dict -> maps target labels to their indices in the x_test and y_test objects

Output:
- pd.DataFrame -> newly created metadata DataFrame
- float list list -> object containing parities corresponding to the scratch model performances (i.e., 4 for CO2, H2O, N2, NH3)

### plot_model_curves
Creates a plot of model performance curves over data scarcity levels for a specified set of models, comparing against two metrics with a combined barchart-scatterplot

Parameters:
- str plot_path -> path to save plot to
- pd.DataFrame model_results -> DataFrame of parsed model results and metadata
- str comparison -> column label for a performance metric in model_results (becomes left side of plot)
- str comparison_2 -> column label for a performance metric in model_results (becomes right side of plot)
- str target -> target to compare models across
- int list extract_layers -> list of extracted layers to include in plot (MAY NOT WORK)
- str y_label -> y-axis label for left side of plot
- str y_label_2 -> y-axis label for right side of plot
- int list y_lims -> y-axis limits for left side of plot
- int list y_lims_2 -> y-axis limits for left side of plot
- str:tuple dict target_dict -> mapping of target name to target index and LaTeX-formatted target name
- str list list styles -> line/bar styles to use while plotting, including colors

### plot_target_comparison
Creates a plot of transfer model performance from a common base adsorbate for a given data scarcity

Parameters:
- str plot_path -> path to save plot to
- pd.DataFrame model_results -> DataFrame of parsed model results and metadata
- str comparison -> column label for a performance metric in model_results (becomes left side of plot)
- str comparison_2 -> column label for a performance metric in model_results (becomes right side of plot)
- str base_target -> base target to compare models across
- int list extract_layers -> list of extracted layers to include in plot (MAY NOT WORK)
- float split -> model data scarcity to plot
- int list y_lims -> y-axis limits for left side of plot
- int list y_lims_2 -> y-axis limits for left side of plot
- str:tuple dict target_dict -> mapping of target name to target index and LaTeX-formatted target name
- str list list styles -> line/bar styles to use while plotting, including colors
- str y_label -> y-axis label for left side of plot
- str y_label_2 -> y-axis label for right side of plot

### plot_average_comparison
Creates a plot of average differences between transfer and base model performance for each target

Parameters:
- str plot_path -> path to save plot to
- pd.DataFrame model_results -> DataFrame of parsed model results and metadata
- str comparison -> column label for a performance metric in model_results
- int list extract_layers -> list of extracted layers to include in plot (MAY NOT WORK)
- float split -> model data scarcity to plot
- int list y_lims -> y-axis limits
- str:tuple dict target_dict -> mapping of target name to target index and LaTeX-formatted target name
- str list list styles -> line/bar styles to use while plotting, including colors
- str y_label -> y-axis label

---

## other_files
other_files contains the following:
- matched_MOFs.csv: a file containing the flattened histograms for all 39,950 MOFs considered in the analysis
- matched_targets.csv: a file containing the Henry's constant targets for each histogram
- split_x: folders containing data splits and model-related files (all model files are tied to the specific data split contained in this folder)
	- split.txt: file containing the training/validation/testing split metadata for this folder
	- matched_MOFs_train.csv: the training histograms used in this split
	- matched_MOFs_val.csv: the validation histograms used in this split
	- matched_MOFs_test.csv: the testing histograms used in this split
	- matched_targets_train.csv: the training Henry's constants used in this split
	- matched_targets_val.csv: the validation Henry's constants used in this split
	- matched_targets_test.csv: the testing Henry's constants used in this split
- x_model/: folders containing the model named in the title (e.g., "base_model" refers to the scratch models, "hk_co2_transfer_model" refers to all transfer models trained using co2 as the base target)
	- model_notes.txt: a summary of the model training efforts contained in this folder, including the number of training epochs, selected targets, selected data scarcities ("splits: "), model performance metrics, and corresponding seeds for model training/data scarcity selection.
	- hk_{base_target}\_{scarcity_level}.png: a basic parity plot of a corresponding scratch model
	- hk_{base_target}\_{scarcity_level}/: a folder containing the .keras file, feature/target scalers (.kl files), and training curve for a single scratch model
	- hk_{base_target}->hk_{transfer_target}\_{scarcity_level}\_extract{extract_layer}.png: a basic parity plot of a corresponding transfer model
	- hk_{base_target}->hk_{transfer_target}\_{scarcity_level}\_extract{extract_layer}/: a folder containing the .keras file, feature/target scalers (.kl files), and training curve for a single transfer model

___

## Citation

Under revision in peer reviewed journal