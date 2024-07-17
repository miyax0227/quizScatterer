# -*- coding: utf-8 -*-
import sys

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial import distance

from quizscatterer import qs

# 引数からファイル名を受け取る
filename = sys.argv[1]

# ファイル読込
with open(filename, "r", encoding="utf-8") as f:
    questions = f.read().splitlines()

# 空文字列の行を除く
questions = [q for q in questions if not q == ""]

# 問題文正規化
questionsForVectors = [qs.regulate_question(q) for q in questions]

# ベクター作成
vectors = [qs.get_text_vector(v) for v in questionsForVectors]

# サンプル
# pprint(vectors[41])

# サマリベクタを取得する
# summaryVectors = getSummaryVector(vectors)
# pprint(summaryVectors)

# 単語出現回数dictを取得する
# nounCountDict = getNounCountDict(vectors)

# 距離マトリクス生成
n = len(vectors)
distance_matrix = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        if i == j:
            distance_matrix[i, j] = 0
        elif i > j:
            distance_matrix[i, j] = qs.compute_distance_bw_question_vectors(
                vectors[i], vectors[j]
            )
        else:
            distance_matrix[i, j] = qs.compute_distance_bw_question_vectors(
                vectors[j], vectors[i]
            )

distance_array = distance.squareform(distance_matrix)

# 階層クラスタリング
clustering_result = linkage(distance_array, method="ward")

# テキスト樹形図出力
for i in qs.draw_text_dendrogram(n * 2 - 2, "", clustering_result, questions, n):
    print(i)

# 最遠配置リスト出力
for i in qs.scatter_questions(n * 2 - 2, clustering_result, distance_matrix, n):
    print(str(i) + "." + questions[i])
