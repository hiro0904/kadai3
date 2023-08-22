import math
from janome.tokenizer import Tokenizer
import numpy as np

# サンプルの文章とクエリ
documents = [
    "これは最初の文章です。",
    "この文章は2番目の文章です。",
    "そしてこれが3番目の文章です。",
    "これは最初の文章ですか？",
]

query = "これは最初のクエリです。"

# 文章とクエリをトークン化
def tokenize(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    # 小文字にして実行
    return [token.surface.lower() for token in tokens]


# ドキュメント全体の単語を集めたリストを作成
all_words = set()
for doc in documents + [query]:
    all_words.update(tokenize(doc))

# 単語の出現回数を数える
# 辞書を初期化
word_counts = {}

for word in all_words:
    # 各単語の出現回数のリスト
    word_count_list = []

    for doc in documents:
        # 文章内で単語が出現する回数をカウントしてリストに追加
        word_count = doc.count(word)
        word_count_list.append(word_count)

    # 辞書に単語をキーとして、出現回数のリストを値として格納
    word_counts[word] = word_count_list


# 文章ごとの単語の出現回数を計算
# クエリ内の単語の出現回数リスト
query_counts = []
# 全ての単語について処理を行う
for word in all_words:
    # クエリ内で単語が出現する回数をカウントしてリストに追加
    word_count = query.count(word)
    query_counts.append(word_count)


# Tf-idfを計算
def tf_idf(word, doc_idx):
    if sum(word_counts[word]) == 0:
        return 0  # 単語が出現しない場合はtf-idfをゼロとする

    tf = word_counts[word][doc_idx] / sum(word_counts[word])

    idf = math.log(
        len(documents)
        / (1 + sum(1 for doc in documents if word_counts[word][doc_idx] > 0))
    )
    return tf * idf


# クエリのTf-idfを計算
query_tfidf = []
for word in all_words:
    tfidf_value = tf_idf(word, -1)  # 単語のTf-idf値を計算
    query_tfidf.append(tfidf_value)  # 計算結果をリストに追加

# ドキュメントのTf-idfを計算
document_tfidfs = []
for doc_idx in range(len(documents)):
    doc_tfidf = []
    for word in all_words:
        tfidf_value = tf_idf(word, doc_idx)  # 単語のTf-idf値を計算
        doc_tfidf.append(tfidf_value)  # 計算結果をリストに追加

    document_tfidfs.append(doc_tfidf)

# Tf-idfを表示
print("Query Tf-idf:", query_tfidf)
print("Document Tf-idfs:")
for doc_idx, doc_tfidf in enumerate(document_tfidfs):
    print(f"Document {doc_idx + 1}:", doc_tfidf)
