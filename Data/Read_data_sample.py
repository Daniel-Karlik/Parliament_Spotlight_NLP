# Read datasets

# Packages for loading and reading json
import json
# Packages for constructing path to datasets location
import os.path
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
dataset_names = ["vyjadreni_politiku.pickle", "tiskove_konference_vlady.pickle", "stenozaznamy_psp.pickle"]

### ---------------------------------------------------------

# Iterating through dataset_names and creating relative path to dataset
dataset_paths = [os.path.join(path_to_datasets, item) for item in dataset_names]


def load_dataset(dataset_path):
    # Opening JSON file with data from datasets
    # Input:
    #       dataset_path - string with relative path to dataset
    # Output:
    #       python dictionary with data
    with open(dataset_path, 'rb') as openfile: # Setting to read mode, no need for writing in this section
        # Reading from json file
        data = pickle.load(openfile)
    return data

counter = 0 # Setting counter for acces of indices of dataset_paths
# iterating through each dataset name
for i in dataset_names:
    global_Vars = globals() # Getting global variable list to add name of dataset as a variable
    # Calling load_dataset on current dataset path and assigning it to a local variable
    # also removing .json from dataset name
    global_Vars[re.sub('.pickle', '', i)] = load_dataset(dataset_paths[counter])
    counter += 1 # Incrementing counter by one so that next dataset path in dataset_paths corresponds to correct dataset

# Uncomment for printing data in json nicely
# stenozaznamy_psp_read = json.dumps(stenozaznamy_psp, indent=4, sort_keys=True, ensure_ascii=False) # Setting json file into readable format
# # Note: ensure_ascii=False is important so that Czech letters are printed correctly

# Examples of accessing info about data entry
# print(stenozaznamy_psp["results"][1]) # Accessing info about a data entry
# print(stenozaznamy_psp["results"][1]["text"])  # Example of accessing text in data entry
