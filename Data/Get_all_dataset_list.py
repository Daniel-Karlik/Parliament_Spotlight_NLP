# Getting list of all datasets available from Hlídač státu API

import urllib.request as urllib2
import ssl
import json

# Loading packages for constructing path to file
import os.path

# Creating unverified context because it doesn't work otherwise, this is not a good solution as it introduces
# exposure to Man In the Middle attacks
# Note: Need to revise this later
ssl._create_default_https_context = ssl._create_unverified_context

# Standart headers (from example at https://hlidacstatu.docs.apiary.io/#reference/datasety-rozsirene-datove-sady-hlidace-statu/datasety/vsechny-datove-sady)
# Getting authorization token
with open(os.path.join("Access_token", "Token_info.json")) as infile:
    authorization = json.load(infile)
Authorization_token = authorization["Authorization token"]

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'{Authorization_token}'  # Token loaded from json file
}
# Reguesting info about all datasets
request = urllib2.Request('https://api.hlidacstatu.cz/api/v2/datasety', headers=headers) # Reguest Datasets to Hlidac Statu API

response_body = json.load(urllib2.urlopen(request)) # Request returns json type, loading json


### Viewing information about available datasets
# All datasets
# Data_sets_info = json.dumps(response_body, indent=4, sort_keys=True, ensure_ascii=False) # Setting json file into readable format
# print(Data_sets_info) # Printing information about all datasets

# Datasets of interest
dataset_indx = [0, 25, 32]
for i in dataset_indx:
  print(json.dumps(response_body['results'][i], indent=4, sort_keys=True, ensure_ascii=False))
  # Note: ensure_ascii=False is important so that Czech letters are printed correctly


