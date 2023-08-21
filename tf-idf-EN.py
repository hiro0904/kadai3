import math

# サンプルの文章とクエリ
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

query = "This is the first query."

# 文章とクエリをトークン化
def tokenize(text):
    return text.lower().split()


# ドキュメント全体の単語を集めたリストを作成
all_words = set()
for doc in documents + [query]:
    all_words.update(tokenize(doc))

# 単語の出現回数を数える
word_counts = {word: [doc.count(word) for doc in documents] for word in all_words}

# 文章ごとの単語の出現回数を計算
query_counts = [query.count(word) for word in all_words]

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

