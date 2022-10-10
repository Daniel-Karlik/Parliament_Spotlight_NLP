# Getting list of all datasets available from Hlídač státu API

import urllib.request as urllib2
import ssl
import json

# Creating unverified context because it doesn't work otherwise, this is not a good solution as it introduces
# exposure to Man In the Middle attacks
# Note: Need to revise this later
ssl._create_default_https_context = ssl._create_unverified_context

# Standart headers (from example at https://hlidacstatu.docs.apiary.io/#reference/datasety-rozsirene-datove-sady-hlidace-statu/datasety/vsechny-datove-sady)
headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Token 30344472ba9c4aeda642e3ea0a276dfe'
}
# Reguesting info about all datasets
request = urllib2.Request('https://www.hlidacstatu.cz/api/v1/Datasets', headers=headers) # Reguest Datasets to Hlidac Statu API

response_body = json.load(urllib2.urlopen(request)) # Request returns json type, loading json
Data_sets_info = json.dumps(response_body, indent=4, sort_keys=True, ensure_ascii=False) # Setting json file into readable format
# Note: ensure_ascii=False is important so that Czech letters are printed correctly

### Viewing information about available datasets
# All datasets
#print(Data_sets_info) # Printing information about all datasets

# Datasets of potential interest
# dataset_indx = [0,1,25,29,31,35,42]
# for i in dataset_indx:
#   print(json.dumps(response_body[i], indent=4, sort_keys=True, ensure_ascii=False))

# Datasets of interest
dataset_indx = [0, 35, 42]
for i in dataset_indx:
  print(json.dumps(response_body[i], indent=4, sort_keys=True, ensure_ascii=False))


