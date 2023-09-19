import requests
from bs4 import BeautifulSoup
import urllib
from collections import Counter

# janome のkeep filter
from janome.tokenfilter import POSKeepFilter, LowerCaseFilter, POSStopFilter
from janome.analyzer import Analyzer
from janome.tokenfilter import ExtractAttributeFilter
from janome.charfilter import RegexReplaceCharFilter

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


def write_text_to_file(text, filename):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"テキストがファイル '{filename}' に正常に書き込まれました。")
    except Exception as e:
        print(f"エラー: ファイル '{filename}' への書き込み中に問題が発生しました。")
        print(str(e))


# get_words(scrap("https://news.yahoo.co.jp/categories/sports"), keep_pos=["名詞"])
# 単純頻度重み付け
def main():
    scrap_Y = get_words(scrap("https://news.yahoo.co.jp"), keep_pos=["名詞"])
    scrap_Y_sports = get_words(
        scrap("https://news.yahoo.co.jp/categories/sports"), keep_pos=["名詞"]
    )
    scrap_Y_it = get_words(
        scrap("https://news.yahoo.co.jp/categories/it"), keep_pos=["名詞"]
    )
    scrap_Y_life = get_words(
        scrap("https://news.yahoo.co.jp/categories/life"), keep_pos=["名詞"]
    )
    write_text_to_file(scrap_Y, "Y")
    write_text_to_file(scrap_Y_sports, "Y-sport")
    write_text_to_file(scrap_Y_it, "Y-it")
    write_text_to_file(scrap_Y_life, "Y-life")
    """
    # wordクラウドを作成する
    # 名詞を取り出してワードクラウドを作ろう
    words = get_words(scrap("https://news.yahoo.co.jp/"), keep_pos=["名詞"])
    # print(words)
    count = Counter(words)
    print(count)
    """


if __name__ == "__main__":
    main()
