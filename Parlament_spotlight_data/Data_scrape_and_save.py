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

# Define time window in months from which you want to get the data
# e.g. Taking window of 1 months (from first day of the month to last day of the month)
time_window_to_get1 = 1
time_window_to_get2 = 1
time_window_to_get3 = 1

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




def data_scraper(dataset, year_of_oldest_record, time_window_to_get):
    # Request to url with data is divided into pages, where maximum number of pages is 200 if number of requested
    # records surpasses 200 pages, they are truncated, this means we cannot simply request full dataset from the url.
    # We circumvent this by sequentially getting data from url for each month.

    ### Setting time window of date to get data
    request = urllib2.Request(
        f'https://www.hlidacstatu.cz/api/v1/DatasetSearch/{dataset}?q=datum%3A[*+TO+{datetime.date.today()}]&page=1&sort=datum&desc=1',
        headers=headers)  # Reguesting first page of results ordered from the latest date
    response_body = json.load(urllib2.urlopen(request))  # Request returns json type, loading json
    # Getting the latest date from data
    date_right = datetime.datetime.strptime(response_body["results"][1]["datum"][:10], "%Y-%m-%d").date()
    # Setting right bound of date window to last day of the month of last entry in dataset
    date_right = datetime.date(date_right.year, date_right.month,
                               calendar.monthrange(date_right.year, date_right.month)[1])

    # If statement tests if there is user defined limit to the oldest record if not then get it from the data
    if year_of_oldest_record == []:
        # Requesting first page of results ordered from the oldest date
        request = urllib2.Request(
            f'https://www.hlidacstatu.cz/api/v1/DatasetSearch/{dataset}?q=datum%3A[*+TO+{datetime.date.today()}]&page=1&sort=datum&desc=0',
            headers=headers)
        response_body = json.load(urllib2.urlopen(request))  # Request returns json type, loading json

        # Getting the oldest date of data entry
        year_of_oldest_record = datetime.datetime.strptime(response_body["results"][1]["datum"][:10], "%Y-%m-%d").year

    time_window_to_get = -time_window_to_get  # Making time_window_to_get negative, so that addition to current date
    # leads to subtractions
    time_window_to_get += 1  # Adding +1 to time_window_to_get so that the subtraction of months leds to correct
    # number of months we want to take into account

    # Subtracting number of months defined in time_window_to_get from right bound to receive left bound
    date_left = date_right + relativedelta(months=time_window_to_get)
    # Setting day of left time bound to first day of the month
    date_left = datetime.datetime.strptime(str(date_left)[:7] + "-1", "%Y-%m-%d").date()


    ### Getting data
    # Initializing dictionary for saving downloaded data
    data = {"results": []}

    # While date of right bound is bigger than year_of_oldest_record from data/set by user
    while date_right.year >= year_of_oldest_record:
        # Getting data sequentially for each month and each page
        page = 0
        while True:
            # Incrementing number of pages
            page += 1
            # Reguesting data from hlidac statu in time window defined by interval <date_left, date_right> and from
            # page defined in page
            request = urllib2.Request(
                f'https://www.hlidacstatu.cz/api/v1/DatasetSearch/{dataset}?q=datum%3A[{date_left}+TO+{date_right}]&page={page}&sort=datum&desc=1',
                headers=headers)
            response_body = json.load(urllib2.urlopen(request))  # Request returns json type, loading json

            # If results in response body is empty: break
            if response_body["results"] == []:
                break
            # Adding results of response_body to vyjadreni_politiku
            data["results"] += response_body["results"]

        # Moving date window back in time, by number of months defined in time_window_to_get-1
        date_right = date_right + relativedelta(
            months=time_window_to_get - 1)  # Moving right bound by number of months defined in time_window_to_get-1
        date_right = datetime.date(date_right.year, date_right.month,
                                   calendar.monthrange(date_right.year, date_right.month)[
                                       1])  # Setting day of date_right to last day of the month
        date_left = date_left + relativedelta(
            months=time_window_to_get - 1)  # Moving left bound by number of months defined in time_window_to_get-1

    return data


def saving_data(data, file_name, path):
    # Writing sample data to json file format
    file_name = file_name.replace("-", "_")  # Replace - with _ if there are any
                                             # This is set up so you can use dataset string as name of the output file
    output_file_name = os.path.join(path, f'{file_name}.json')  # Adding file name to the path of saving files
    with open(output_file_name, "w") as outfile:
        json.dump(data, outfile)

# Getting data from vyjadreni-politiku
data1 = data_scraper(dataset1, year_of_oldest_record1, time_window_to_get1)
# Getting data from stenozaznamy-psp
data2 = data_scraper(dataset2, year_of_oldest_record2, time_window_to_get2)
# Getting data from tiskove-konference-vlady
data3 = data_scraper(dataset3, year_of_oldest_record3, time_window_to_get3)

# Saving data from vyjadreni-politiku
saving_data(data1, dataset1, path1)
# Saving data from vyjadreni-politiku
saving_data(data2, dataset2, path2)
# Saving data from vyjadreni-politiku
saving_data(data2, dataset2, path3)