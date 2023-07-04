#!/usr/bin/env python
# coding: utf-8
from datetime import datetime, timedelta, timezone

import pandas as pd
import boto3
from googleapiclient.discovery import build

pd.set_option("display.max_colwidth", 1000)

# Systems Manager - Parameter Storeへのアクセス
API_KEY = "AIzaSyBWB8n9ulVMGHBPC1sELqvGBqRnv12zUWE"
CSE_ID = "kadai3-1688444217771"

session = boto3.session.Session(profile_name="xxxxxx")
ssm = session.client("ssm")
params = {}
res = ssm.get_parameters(Names=(API_KEY, CSE_ID), WithDecryption=True)
for p in res["Parameters"]:
    params[p["Name"]] = p["Value"]

# get the API KEY
google_api_key = params[API_KEY]
# get the Search Engine ID
google_cse_id = params[CSE_ID]


def get_search_results(query, start_index):
    # Google Custom Search API
    service = build(
        "customsearch", "v1", cache_discovery=False, developerKey=google_api_key
    )
    # CSEの検索結果を取得
    result = (
        service.cse()
        .list(q=query, cx=google_cse_id, num=10, start=start_index)
        .execute()
    )
    # 検索結果(JSON形式)
    return result


def main():
    query = "いちご農家 HP"

    # ExecDate
    timezone_jst = timezone(timedelta(hours=+9), "JST")
    now = datetime.now(timezone_jst)

    # Google検索 - Custom Search API
    data = get_search_results(query, 1)
    total_results = int(data["searchInformation"]["totalResults"])
    print("total_results", total_results)

    # Google検索結果から任意の項目抽出 & rank付与
    items = data["items"]

    result = []
    num_items = len(items) if len(items) < 10 else 10
    for i in range(num_items):
        title = items[i]["title"] if "title" in items[i] else ""
        link = items[i]["link"] if "link" in items[i] else ""
        snippet = items[i]["snippet"] if "snippet" in items[i] else ""
        result.append("\t".join([str(i + 1), title, link, snippet]))

    # List->DataFrame
    df_search_results = pd.DataFrame(result)[0].str.split("\t", expand=True)
    df_search_results.columns = ["rank", "title", "url", "snippet"]

    # CSV出力
    dt_csv = now.strftime("%Y%m%d%H%M")
    output_csv = f"csv/{query}_{dt_csv}.csv"
    df_search_results.to_csv(
        output_csv, sep=",", index=False, header=True, encoding="utf-8"
    )
    pd.read_csv(output_csv, index_col=0)


if __name__ == "__main__":
    main()

