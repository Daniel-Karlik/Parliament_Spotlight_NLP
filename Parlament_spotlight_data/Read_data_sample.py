# Read json sample data
import json
import os.path

# Opening JSON file with sample data
input_file = os.path.join('data', 'sample_stenozaznamy_psp.json')
with open(input_file, 'r') as openfile:
  # Reading from json file
  stenozaznamy_psp_sample = json.load(openfile)

# Uncomment for printing data in json nicely
# stenozaznamy_psp_sample = json.dumps(json_object, indent=4, sort_keys=True, ensure_ascii=False) # Setting json file into readable format
# # Note: ensure_ascii=False is important so that Czech letters are printed correctly

# Examples of accessing info about data entry
# print(stenozaznamy_psp_sample["results"][1]) # Accessing info about a data entry
print(stenozaznamy_psp_sample["results"][1]["text"]) # Example of accessing text in data entry