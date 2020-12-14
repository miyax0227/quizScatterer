# -*- coding: utf-8 -*-
import os
from pprint import pprint
import sys
import numpy as np
from .classes.qs import *

# 引数からファイル名を受け取る
filename = sys.argv[1]

# ファイル読込
with open(filename) as f:
  questions = f.read().splitlines()

# 空文字列の行を除く
questions = [q for q in questions if not q == ""]

# 問題文正規化
questionsForVectors = [regulateQuestion(q) for q in questions]

# ベクター作成
vectors = [getVector(v) for v in questionsForVectors]

# サンプル
# pprint(vectors[41])

# サマリベクタを取得する
# summaryVectors = getSummaryVector(vectors)
# pprint(summaryVectors)

# 単語出現回数dictを取得する
# nounCountDict = getNounCountDict(vectors)

# 距離マトリクス生成
n = len(vectors)
dMatrix = np.zeros([n,n])
for i in range(n):
  for j in range(n):
    if i == j:
      dMatrix[i,j] = 0
    elif i > j:
      dMatrix[i,j] = getDistance(vectors[i],vectors[j])
      #dMatrix[i,j] = cos_sim(summaryVectors[i],summaryVectors[j])
    else:
      dMatrix[i,j] = getDistance(vectors[j],vectors[i])
      
dArray = distance.squareform(dMatrix)

# 階層クラスタリング
Z = linkage(dArray, method="ward")
#pprint(Z)

# テキスト樹形図出力
for i in getTextDendrogram(n*2-2, "", Z, questions, n):
  print(i)

# 最遠配置リスト出力
for i in scatterQuestion(n*2-2, Z, dMatrix, n):
  print(str(i) + "." + questions[i])
