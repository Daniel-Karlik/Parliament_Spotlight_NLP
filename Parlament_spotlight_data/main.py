import urllib.request as urllib2
import certifi
import urllib.error
import urllib.parse
import sys
import html.parser
import ssl
import json

# Creating unverified context because it doesnt work otherwise, this is not a good solution as it introduces
# exposure to Man In the Middle attacks
# Note: Need to revise this later
ssl._create_default_https_context = ssl._create_unverified_context

# Standart headers (from example at https://hlidacstatu.docs.apiary.io/#reference/datasety-rozsirene-datove-sady-hlidace-statu/datasety/vsechny-datove-sady)
headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Token 30344472ba9c4aeda642e3ea0a276dfe'
}
# Reguestinbg
request = urllib2.Request('https://www.hlidacstatu.cz/api/v1/Datasets', headers=headers) # Reguest to hlidacstatu API

response_body = json.load(urllib2.urlopen(request)) # Request returns json type, loading json
Data_set_info = json.dumps(response_body, indent=4, sort_keys=True) # Setting json file into readable format
print(Data_set_info) # Printing


