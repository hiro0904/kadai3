import requests
from bs4 import BeautifulSoup
import math
from janome.tokenizer import Tokenizer
import numpy as np
import streamlit as st
import plotly.graph_objs as go


def read_text_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            file_contents = file.read()
        return file_contents
    except Exception as e:
        st.warning(f"エラー: ファイル '{filename}' の読み込み中に問題が発生しました。")
        st.write(str(e))
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
    # st.write("分解した文章")
    # st.write(tokenized_text)
    for word in all_word:
        # 全ての集合の一つずつ確認
        instant_var = 0
        for term in tokenized_text:
            if word == term:
                # 一致した回数分増やす
                instant_var += 1
        array_tf.append(instant_var / len(tokenized_text))
    # 単純頻度 → 相対頻度
    # st.write(array_tf)
    return array_tf


# Webページを取得して解析
def scrap(load_url):
    html = requests.get(load_url)
    soup = BeautifulSoup(html.content, "html.parser")
    # テキストだけ取得
    scrap_text = soup.get_text()
    # print(scrap_text)
    return scrap_text


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


def fillin_file():
    # 文字入力形式での読み取り
    string = st.text_input("ここに分析したい文章を記入してください", "これはサンプルです")
    return string


def main():
    st.title("文章関連度算出")
    scrap_python_print = scrap(
        "https://docs.python.org/ja/3.7/tutorial/inputoutput.html"
    )
    scrap_Y_life = read_text_file("Y-life")
    scrap_Y_sports = read_text_file("Y-sport")
    scrap_economy = scrap("https://www.bloomberg.co.jp/")
    scrap_history = scrap(
        "https://ja.wikipedia.org/wiki/%E6%97%A5%E6%9C%AC%E3%81%AE%E6%AD%B4%E5%8F%B2"
    )

    # サンプルの文章とクエリ
    documents = [
        scrap_python_print,
        scrap_Y_life,
        scrap_Y_sports,
        scrap_economy,
        scrap_history,
    ]
    query = fillin_file()

    document_title = [
        "文章1 : Pythonチュートリアル",
        "文章2 : Yahoo lifeカテゴリ",
        "文書3 : Yahoo sportカテゴリ",
        "文章4 : bloomberg 経済誌",
        "文章5 日本の歴史 - wikipedia",
    ]
    st.header("クエリ")
    st.write(query)
    st.header("ドキュメント")
    st.subheader("文章1 : Pythonチュートリアル")
    st.subheader("文章2 : Yahoo lifeカテゴリ ")
    st.subheader("文書3 : Yahoo sportカテゴリ")
    st.subheader("文章4 : bloomberg 経済誌")
    st.subheader("文章5 日本の歴史 - wikipedia")
    st.divider()

    # for doc in range(len(documents)):
    #    st.write(documents[doc])

    # 全てのワードの集合を作る
    all_words = set()
    for doc in documents + [query]:
        all_words.update(tokenize(doc))
    # st.header("all wordは")
    # st.write(all_words)

    # tf分解
    tf_query = calculate_tf(query, all_words)
    array_tf_doc = []
    for i in range(len(documents)):
        array_tf_doc.append(calculate_tf(documents[i], all_words))
    # docのtfを集めた集合を作成
    # st.write(array_tf_doc)

    # 文章を比較してidfを作成する。
    # st.write(len(all_words))
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
    # st.write(array_idf_doc)
    # st.write(len(array_idf_doc))

    # tf idfを作成 doc0
    # st.header("tf idfを計算")
    array_tf_idf_doc = []
    for doc in range(len(documents)):
        # print(f"{doc}番目の文章")
        np_array_tf = np.array(array_tf_doc[doc])
        np_array_idf = np.array(array_idf_doc)
        # print(np_array_tf * np_array_idf)
        array_tf_idf_doc.append(np_array_tf * np_array_idf)
    # st.write(array_tf_idf_doc)
    st.header("文章との類似度")
    # query と  tf_idfの類似度を考える
    np_query_tf = np.array(tf_query)
    similarity_list = []
    for doc in range(len(documents)):
        similarity = cosine_similarity(np_query_tf, array_tf_idf_doc[doc])
        similarity_list.append(similarity)
        st.subheader(document_title[doc])
        st.subheader(f"文章{doc+1}との類似度は {similarity}")
        st.divider()

        # レーダーチャートを作成
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=similarity_list, theta=document_title, fill="toself", name="要素"
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False
    )

    # レーダーチャートを表示
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
