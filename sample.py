#!/usr/bin/env python
# coding: utf-8
from datetime import datetime, timedelta, timezone
from serpapi import GoogleSearch
import pandas as pd

# Systems Manager - Parameter Storeへのアクセス
API_KEY = "AIzaSyBWB8n9ulVMGHBPC1sELqvGBqRnv12zUWE"
CSE_ID = "kadai3-1688444217771"

# get the API KEY
google_api_key = API_KEY
# get the Search Engine ID
google_cse_id = CSE_ID


def main():
    search = GoogleSearch(
        {"q": "大阪市", "location": "Osaka,Osaka,Japan", "api_key": "your private api key"}
    )
    result = search.get_dict()
    print()


if __name__ == "__main__":
    main()

