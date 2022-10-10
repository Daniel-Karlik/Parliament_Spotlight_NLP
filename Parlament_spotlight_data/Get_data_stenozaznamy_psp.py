# Getting data entries in dataset stenozaznamy-psp (stenozáznamy poslanecké sněmovny)

# Loading packages for data scraping
import urllib.request as urllib2
import ssl
import json

# Loading packages for constructing path to file
import os.path

# Loading packages for operation with dates
import datetime
import calendar
from dateutil.relativedelta import relativedelta

# Creating unverified ssl context because it doesn't work otherwise, this is not a good solution as it introduces
# exposure to Man In the Middle attacks
# Note: Need to revise this later
ssl._create_default_https_context = ssl._create_unverified_context

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Token 30344472ba9c4aeda642e3ea0a276dfe'
}

# Name of dataset to download data from
dataset = "stenozaznamy-psp"

### Setting time window of date to get
request = urllib2.Request(
    f'https://www.hlidacstatu.cz/api/v1/DatasetSearch/{dataset}?q=datum%3A[*+TO+{datetime.date.today()}]&page=1&sort=datum&desc=1',
    headers=headers)  # Reguesting first page of results ordered from the latest date
response_body = json.load(urllib2.urlopen(request))  # Request returns json type, loading json
# Getting the latest date from data
date_right = datetime.datetime.strptime(response_body["results"][1]["datum"][:10], "%Y-%m-%d").date()
# Setting right bound of date window to last day of the month of last entry in dataset
date_right = datetime.date(date_right.year, date_right.month, calendar.monthrange(date_right.year, date_right.month)[1])

# Requesting first page of results ordered from the oldest date
request = urllib2.Request(
    f'https://www.hlidacstatu.cz/api/v1/DatasetSearch/{dataset}?q=datum%3A[*+TO+{datetime.date.today()}]&page=1&sort=datum&desc=0',
    headers=headers)
response_body = json.load(urllib2.urlopen(request))  # Request returns json type, loading json
# Getting the oldest date of data entry
year_of_oldest_record = datetime.datetime.strptime(response_body["results"][1]["datum"][:10], "%Y-%m-%d").year
year_of_oldest_record = 2021  # Comment out this line if you want all data from the dataset

time_window_to_get = -1  # Taking window of 1 months
time_window_to_get += 1
# Subtracting number of months defined in time_window_to_get from right bound to receive left bound
date_left = date_right + relativedelta(months=time_window_to_get)
# Setting day of left time bound to first day of the month
date_left = datetime.datetime.strptime(str(date_left)[:7] + "-1", "%Y-%m-%d").date()

# Initializing dictionary for saving downloaded data
stenozaznamy_psp = {"results": []}

# While date of right bound is bigger than year_of_oldest_record from data/set by user
while date_right.year >= year_of_oldest_record:
    # Request to url with data is decided into pages, where maximum number of pages is 200 if number of requested
    # records surpasses 200 pages, they are truncated, this means we cannot simply request full dataset from the url.
    # We circumvent this by sequentially getting data from url for each month.
    page = 0
    while True:
        # Incrementing number of pages
        page += 1
        # Reguesting data from hlidac statu in time window defined by interval <date_left, date_right> and from page defined in page
        request = urllib2.Request(
            f'https://www.hlidacstatu.cz/api/v1/DatasetSearch/{dataset}?q=datum%3A[{date_left}+TO+{date_right}]&page={page}&sort=datum&desc=1',
            headers=headers)
        response_body = json.load(urllib2.urlopen(request))  # Request returns json type, loading json

        # If results in response body is empty: break
        if response_body["results"] == []:
            break
        # Adding results of response_body to stenozaznamy_psp
        stenozaznamy_psp["results"] += response_body["results"]

    # Moving date window back in time, by number of months defined in time_window_to_get-1
    date_right = date_right + relativedelta(
        months=time_window_to_get - 1)  # Moving right bound by number of months defined in time_window_to_get-1
    date_right = datetime.date(date_right.year, date_right.month,
                               calendar.monthrange(date_right.year, date_right.month)[
                                   1])  # Setting day of date_right to last day of the month
    date_left = date_left + relativedelta(
        months=time_window_to_get - 1)  # Moving left bound by number of months defined in time_window_to_get-1

# Writing sample data to json file format
file_name = dataset.replace("-", "_")
output_file_name = os.path.join('data', f'{file_name}.json')
with open(output_file_name, "w") as outfile:
    json.dump(stenozaznamy_psp, outfile)
