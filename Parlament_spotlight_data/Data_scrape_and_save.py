# Functions for scraping data from hlidacstatu.cz and saving scraped data

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

### Section of user defined variables ----------------------

# Define datasets to get data from
dataset1 = "vyjadreni-politiku"  # statements of politicians from Twitter, Facebook, Youtube
dataset2 = "stenozaznamy-psp"  # shorthand records of the Chamber of Deputies (Czech: stenozáznamy poslanecké sněmovny)
dataset3 = "tiskove-konference-vlady"  # records of press conferences of the government of the Czech Republic

# Define starting year for each dataset from which you want to download the records.
# Set equal to [] if you want to get all the data.
# Note: Size of datasets are as follows
#       1. stenozaznamy-psp -> largest dataset
#       2. vyjadreni-politiku -> medium size
#       3. tiskove-konference-vlady -> small dataset

year_of_oldest_record1 = 2022
year_of_oldest_record2 = 2022
year_of_oldest_record3 = 2022

# Define name of relative path for saving data
path1 = "data"
path2 = "data"
path3 = "data"

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Token 30344472ba9c4aeda642e3ea0a276dfe' # insert your token
                                                              # Note: optimally this needs to be loaded from
                                                              #       system variable or from separate .json file.
                                                              # Revise this later
}
### ---------------------------------------------------------

# Creating unverified ssl context because it doesn't work otherwise, this is not a good solution as it introduces
# exposure to Man In the Middle attacks
# Note: Need to revise this later
ssl._create_default_https_context = ssl._create_unverified_context

def data_scraper(dataset, year_of_oldest_record):
    # Request to url with data is divided into pages, where maximum number of pages is 200 if number of requested
    # records surpasses 200 pages, they are truncated, this means we cannot simply request full dataset from the url.
    # We circumvent this by sequentially getting data from url for each day.
    # Note: Getting data for each month was considered, but in case of vyjadreni-politiku dataset, the number of entries
    #       exceed 200 pages.

    ### Getting date of most recent record and year of oldest record
    request = urllib2.Request(
        f'https://www.hlidacstatu.cz/api/v1/DatasetSearch/{dataset}?q=datum%3A[*+TO+{datetime.date.today()}]&page=1&sort=datum&desc=1',
        headers=headers)  # Requesting first page of results ordered from the latest date
    response_body = json.load(urllib2.urlopen(request))  # Request returns json type, loading json
    # Getting the latest date from data and adding +1 day to it
    # this is done because of format of request to url
    # e.g. request for date 12-10-2022 - 13-10-2022 will get data for date 12-10-2022
    date_right = datetime.datetime.strptime(response_body["results"][1]["datum"][:10], "%Y-%m-%d").date() + relativedelta(days=1)
    # Subtracting one day from right bound to receive left bound
    date_left = date_right + relativedelta(days=-1)

    # If statement tests if there is user defined limit to the oldest record if not then get it from the data
    if year_of_oldest_record == []:
        # Requesting first page of results ordered from the oldest date
        request = urllib2.Request(
            f'https://www.hlidacstatu.cz/api/v1/DatasetSearch/{dataset}?q=datum%3A[*+TO+{datetime.date.today()}]&page=1&sort=datum&desc=0',
            headers=headers)
        response_body = json.load(urllib2.urlopen(request))  # Request returns json type, loading json

        # Getting the oldest year of data entry
        year_of_oldest_record = datetime.datetime.strptime(response_body["results"][1]["datum"][:10], "%Y-%m-%d").year

    ### Getting data
    # Initializing dictionary for saving downloaded data
    data = {"results": []}

    # While date of right bound is bigger than year_of_oldest_record from data/set by user download data
    while date_right.year >= year_of_oldest_record:
        # Getting data sequentially for each day and each page
        page = 0
        while True:
            # Incrementing number of pages
            page += 1
            # Requesting data from hlidac statu in time window defined by interval <date_left, date_right} (this means excluding right bound) and from
            # page defined in page
            request = urllib2.Request(
                f'https://www.hlidacstatu.cz/api/v1/DatasetSearch/{dataset}?q=datum%3A[{date_left}+TO+{date_right}'+'}'+f'&page={page}&sort=datum&desc=1',
                headers=headers)
            response_body = json.load(urllib2.urlopen(request))  # Request returns json type, loading json

            # If results in response body is empty: break
            if response_body["results"] == []:
                break
            # Adding results of response_body to data
            data["results"] += response_body["results"]

        # Moving date window back in time by one day
        date_right = date_right + relativedelta(
            days=-1)  # Moving right bound by one day
        date_left = date_left + relativedelta(
            days=-1)  # Moving left bound by one day

    return data


def saving_data(data, file_name, path):
    # Writing sample data to json file format
    file_name = file_name.replace("-", "_")  # Replace - with _ if there are any
                                             # This is set up so you can use dataset string as name of the output file
    output_file_name = os.path.join(path, f'{file_name}.json')  # Adding file name to the path of saving files
    with open(output_file_name, "w") as outfile:
        json.dump(data, outfile)

# Getting data from vyjadreni-politiku
data1 = data_scraper(dataset1, year_of_oldest_record1)
# Getting data from stenozaznamy-psp
data2 = data_scraper(dataset2, year_of_oldest_record2)
# Getting data from tiskove-konference-vlady
data3 = data_scraper(dataset3, year_of_oldest_record3)

# Saving data from vyjadreni-politiku
saving_data(data1, dataset1, path1)
# Saving data from vyjadreni-politiku
saving_data(data2, dataset2, path2)
# Saving data from vyjadreni-politiku
saving_data(data3, dataset2, path3)