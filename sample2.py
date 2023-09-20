import requests
from bs4 import BeautifulSoup
import math
from janome.tokenizer import Tokenizer
import numpy as np


def read_text_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            file_contents = file.read()
        return file_contents
    except Exception as e:
        print(f"エラー: ファイル '{filename}' の読み込み中に問題が発生しました。")
        print(str(e))
        return None


# 文章とクエリをトークン化
def tokenize(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    # トークンを小文字に変換してリストに格納
    tokenized_text = [
        token.surface.lower() for token in tokens if token.surface.strip() != ""
    ]
    return tokenized_text


# 文章と単語集合を渡してtfを返す
def calculate_tf(text, all_word):
    array_tf = []
    tokenized_text = tokenize(text)
    print("分解した文章")
    print(tokenized_text)
    for word in all_word:
        # 全ての集合の一つずつ確認
        instant_var = 0
        for term in tokenized_text:
            if word == term:
                # 一致した回数分増やす
                instant_var += 1
        array_tf.append(instant_var / len(tokenized_text))
    # 単純頻度 → 相対頻度
    print(array_tf)
    return array_tf


def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    # print(f"ベクトルの内積{dot_product}")
    norm_v1 = np.linalg.norm(v1)
    # print(f"ベクトル1のスカラー{norm_v1}")
    norm_v2 = np.linalg.norm(v2)
    # print(f"ベクトル2のスカラー{norm_v2}")
    similarity = dot_product / (norm_v1 * norm_v2)
    # print(f"類似度{similarity}\n")
    return similarity


def main():
    scrap_Y = read_text_file("import-data/Y")
    scrap_Y_it = read_text_file("import-data/Y-it")
    scrap_Y_life = read_text_file("import-data/Y-life")
    scrap_Y_sports = read_text_file("import-data/Y-sport")

    # サンプルの文章とクエリ
    documents = [
        scrap_Y_it,
        scrap_Y_life,
        scrap_Y_sports,
    ]
    query = scrap_Y
    print("クエリ")
    print(query)
    print("ドキュメント")
    for doc in range(len(documents)):
        print(documents[doc])
    # 全てのワードの集合を作る
    all_words = set()
    for doc in documents:
        all_words.update(tokenize(doc))
    print("all wordは")
    print(all_words)

    # tf分解
    tf_query = calculate_tf(query, all_words)
    array_tf_doc = []
    for i in range(len(documents)):
        array_tf_doc.append(calculate_tf(documents[i], all_words))
    # docのtfを集めた集合を作成
    print(array_tf_doc)

    # 文章を比較してidfを作成する。
    print(len(all_words))
    array_idf_doc = []

    for term in range(len(all_words)):
        instant_var = 0
        for i in range(len(array_tf_doc)):
            if array_tf_doc[i][term] != 0:
                instant_var += 1
        if instant_var == 0:
            # 現れない時はidfは無限になるが、tfが0なので
            array_idf_doc.append(0)
        elif instant_var != 0:
            array_idf_doc.append(len(array_tf_doc) / instant_var)
    # 後でlogにしても良いよ
    print(array_idf_doc)
    print(len(array_idf_doc))

    # tf idfを作成 doc0
    print("tf idfを計算")
    array_tf_idf_doc = []
    for doc in range(len(documents)):
        # print(f"{doc}番目の文章")
        np_array_tf = np.array(array_tf_doc[doc])
        np_array_idf = np.array(array_idf_doc)
        # print(np_array_tf * np_array_idf)
        array_tf_idf_doc.append(np_array_tf * np_array_idf)
    print(array_tf_idf_doc)

    # query と  tf_idfの類似度を考える
    np_query_tf = np.array(tf_query)
    for doc in range(len(documents)):
        similarity = cosine_similarity(np_query_tf, array_tf_idf_doc[doc])
        print(f"文章{doc+1}の類似度は {similarity}")


if __name__ == "__main__":
    main()
