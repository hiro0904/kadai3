import requests
from bs4 import BeautifulSoup
import urllib
from collections import Counter

# janome のkeep filter
from janome.tokenfilter import POSKeepFilter, LowerCaseFilter, POSStopFilter
from janome.analyzer import Analyzer
from janome.tokenfilter import ExtractAttributeFilter
from janome.charfilter import RegexReplaceCharFilter

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# 描画モジュール
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Webページを取得して解析
def scrap(load_url):
    html = requests.get(load_url)
    soup = BeautifulSoup(html.content, "html.parser")
    # テキストだけ取得
    scrap_text = soup.get_text()
    # print(scrap_text)
    return scrap_text


# 形態素解析して名詞だけ出力
keep_pos = ["名詞"]
analyzer = Analyzer(
    token_filters=[POSKeepFilter(keep_pos), ExtractAttributeFilter("surface")]
)
# print(list(analyzer.analyze(scrap_text)))


def analyze_with_tfidf(texts):
    # TfidfVectorizerを初期化
    vectorizer = TfidfVectorizer()

    # テキストをTF-IDF行列に変換
    tfidf_matrix = vectorizer.fit_transform(texts)

    # 各単語の特徴名（単語） 特徴量を取得
    feature_names = vectorizer.get_feature_names()

    # 結果を表示
    for i, text in enumerate(texts):
        print(f"Text {i+1}:")
        for j, word in enumerate(feature_names):
            tfidf_score = tfidf_matrix[i, j]
            if tfidf_score > 0:
                print(f"  {word}: {tfidf_score}")


# 指定した品詞だけ文からとってくる関数
def get_words(string, keep_pos=None):
    char_filters = [
        # 正規表現で数字のノイズ除去
        RegexReplaceCharFilter("\\d+", ""),
        RegexReplaceCharFilter("-/;:-!0", ""),
        RegexReplaceCharFilter("[a-zA-Z]+", ""),
    ]
    token_filters = [LowerCaseFilter()]
    if keep_pos is None:
        token_filters.append(POSStopFilter(["記号"]))
        token_filters.append(POSStopFilter(["数字"]))
    else:
        # 入力したkeep_posの品詞だけ取って来る
        token_filters.append(POSKeepFilter(keep_pos))
        token_filters.append(ExtractAttributeFilter("surface"))

    tokens = Analyzer(char_filters=char_filters, token_filters=token_filters)
    return " ".join(list(tokens.analyze(string)))


# get_words(scrap("https://news.yahoo.co.jp/categories/sports"), keep_pos=["名詞"])
# 単純頻度重み付け
get_words(scrap("https://news.yahoo.co.jp/categories/sports"), keep_pos=["名詞"])


analyze_with_tfidf(
    [
        get_words(scrap("https://news.yahoo.co.jp/categories/sports"), keep_pos=["名詞"]),
        get_words(scrap("https://news.yahoo.co.jp/categories/it"), keep_pos=["名詞"]),
        get_words(scrap("https://news.yahoo.co.jp/categories/life"), keep_pos=["名詞"]),
    ]
)

# wordクラウドを作成する
# 名詞を取り出してワードクラウドを作ろう
"""
words = get_words(scrap("https://news.yahoo.co.jp/"), keep_pos=["名詞"])
# print(words)
count = Counter(words)
print(count)
"""

