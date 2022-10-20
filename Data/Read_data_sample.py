# Read datasets

# Packages for loading and reading json
import json
# Packages for constructing path to datasets location
import os.path
# Function listdir from os to list files in directory
from os import listdir
# Import package regex for deleting part of string matching a pattern
import re
# Packages for opening and reading pickle files
import pickle
### Section of user defined variables ----------------------

# Relative path to datasets
path_to_datasets = "data"
# Define dataset names to be opened
# Possible dataset names:
#                        1. "vyjadreni_politiku.pickle"
#                        2. "stenozaznamy_psp.pickle"
#                        3. "tiskove_konference_vlady.pickle"
# If you don't want some datasets to be loaded just remove them from the list
dataset_names = ["vyjadreni_politiku", "stenozaznamy_psp", "tiskove_konference_vlady"]

# Choose datasets to load. E.g. if you want to load only vyjadreni_politiku and tiskove_konference_vlady. input [0,2]
# Note: Only dataset stenozaznamy-psp is dlownladed, others are not of interest right now
dataset_to_load = [1]

### ---------------------------------------------------------

# Iterating through dataset_names and creating relative path to dataset
dataset_paths = [os.path.join(path_to_datasets, item) for item in dataset_names]


def load_dataset(dataset_path):
    # Opening .pickle file with data from datasets
    # Input:
    #       dataset_path - string with relative path to dataset
    # Output:
    #       python dictionary with data
    # Getting list of part file of dataset
    dataset_parts = listdir(dataset_path)
    # Initializing python dictionary to load parts into
    data = {"results": []}
    # Going through files in dataset location
    for item in dataset_parts:
        with open(os.path.join(dataset_path, item), 'rb') as openfile: # Setting to read mode, no need for writing in this section
            # Reading from .pickle file
            data_part = pickle.load(openfile)
            # Appending data from part to data
            data["results"] += data_part["results"]
    return data

#counter = 0 # Setting counter for acces of indices of dataset_paths
# iterating through each dataset name
for i in dataset_to_load:
    global_Vars = globals() # Getting global variable list to add name of dataset as a variable
    # Calling load_dataset on current dataset path and assigning it to a global variable named after dataset
    # also removing .pickle from dataset name
    global_Vars[re.sub('.pickle', '', dataset_names[i])] = load_dataset(dataset_paths[i])

# Examples of accessing info about data entry
# print(stenozaznamy_psp["results"][1]) # Accessing info about a data entry
# print(stenozaznamy_psp["results"][1]["text"])  # Example of accessing text in data entry
s = 1