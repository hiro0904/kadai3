import requests
from bs4 import BeautifulSoup
import math
from janome.tokenizer import Tokenizer
import numpy as np

# サンプルの文章とクエリ
documents = [
    "これは最初の文章です。",
    "この文章は2番目の文章です。",
    "そしてこれが3番目の文章です。",
    "ブーリアンモデル、Pythonで配列を結合する方法",
]

query = "これは最初のクエリです。"

# Webページを取得して解析
def scrap(load_url):
    html = requests.get(load_url)
    soup = BeautifulSoup(html.content, "html.parser")
    # テキストだけ取得
    scrap_text = soup.get_text()
    # print(scrap_text)
    return scrap_text


# 文章とクエリをトークン化
def tokenize(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    return [token.surface.lower() for token in tokens]


# Tf-idfを計算
def calculate_tf_idf(word, doc_idx, word_counts):
    if sum(word_counts[word]) == 0:
        return 0  # 単語が出現しない場合はtf-idfをゼロとする

    tf = word_counts[word][doc_idx] / sum(word_counts[word])

    doc_frequency = sum(1 for doc in documents if word_counts[word][doc_idx] > 0)
    idf = math.log(len(documents) / (1 + doc_frequency))

    # 小さな値を分母に追加してゼロ除算を避ける
    epsilon = 1e-6
    idf = idf if idf > epsilon else epsilon

    return tf * idf


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
    all_words = set()
    for doc in documents + [query]:
        all_words.update(tokenize(doc))
        # print(f"doc:  {doc} \n")

    # 出現回数をカウント
    word_counts = {}
    for word in all_words:
        # 文章内に何回出たかのリスト
        word_count_list = []
        for doc in documents:
            word_count = doc.count(word)
            word_count_list.append(word_count)
            # print(f"ワード:{word}  ワードカウント:{word_count}\n")
        word_counts[word] = word_count_list

    # クエリ内の単語の出現回数リスト
    query_counts = []
    for word in all_words:
        word_count = query.count(word)
        query_counts.append(word_count)

    # クエリのTf-idfを計算
    query_tfidf = []
    for word in all_words:
        tfidf_value = calculate_tf_idf(word, -1, word_counts)
        query_tfidf.append(tfidf_value)

    # ドキュメントのTf-idfを計算
    document_tfidfs = []
    for doc_idx in range(len(documents)):
        doc_tfidf = []
        for word in all_words:
            tfidf_value = calculate_tf_idf(word, doc_idx, word_counts)
            doc_tfidf.append(tfidf_value)
        document_tfidfs.append(doc_tfidf)

    # Tf-idfを表示
    print("Query Tf-idf:", query_tfidf)
    print("Document Tf-idfs:")
    for doc_idx, doc_tfidf in enumerate(document_tfidfs):
        print(f"Document {doc_idx + 1}:", doc_tfidf)

    query_vector = np.array(query_tfidf)
    document_vectors = np.array(document_tfidfs)

    # 各ドキュメントとクエリの類似度を計算

    similarities = []
    for doc_vector in document_vectors:
        similarity = cosine_similarity(query_vector, doc_vector)
        similarities.append(similarity)

    print("Cosine Similarities:")
    for doc_idx, similarity in enumerate(similarities):
        print(f"Document {doc_idx + 1}:", similarity)


if __name__ == "__main__":
    main()
