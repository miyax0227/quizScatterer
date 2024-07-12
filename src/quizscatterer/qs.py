# -*- coding: utf-8 -*-
import os
import re
import math
from pprint import pprint
import MeCab
import numpy as np
import gensim
import itertools
import pandas as pd
import scipy.spatial.distance as distance
from scipy.cluster.hierarchy import dendrogram, linkage

# 実行ファイルパスを取得
execPath = os.path.dirname(__file__)
# 学習済みベクターモデルの読込
wv = gensim.models.Word2Vec.load(execPath + "/gensimModel/word2vec.gensim.model").wv
# MeCab辞書読込
mt = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

# 問題文正規化
def regulateQuestion(q):
  q = q.translate(str.maketrans({'（':'(','）':')'}))
  q = re.sub('\([\u3041-\u309F・]+\)','',q)
  q = re.sub('[?？]','',q)
  return q

# コサイン類似度を得る
def cosSim(v1, v2):
  """ コサイン類似度を得る
  Args: v1(np.Array), v2(np.Array): ベクトル(同次元)
  Returns: float: 類似度(-1～1)
  """
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 問題ベクターから単語対類似度リスト（類似度が高い順）を得る
def getDirectProduct(l1, l2):
  """問題ベクターから単語対類似度リスト（類似度が高い順）を得る
  Args: l1(dict), l2(dict): 問題ベクター
  Returns: list[dic]: 単語対類似度リスト
  """
  directProduct = []
  for v1, v2 in itertools.product(l1, l2):
    directProduct.append({
      'word1': v1['surface'],
      'word2': v2['surface'],
      'cosSim': cosSim(v1['vector'], v2['vector'])
    })
  return sorted(directProduct, key=lambda x:x['cosSim'], reverse=True)

def getWakachigaki(text):
  node = mt.parseToNode(text)
  wakachigaki = []
  while node:
    wakachigaki.append({
      '_surface': node.surface,
      'feature': node.feature
    })
    node = node.next
  return wakachigaki

# 問題文から問題ベクターを得る
def getVector(text):
  """問題文から問題ベクターを得る
  Args: text(str): 問題文
  Returns: dic: 問題ベクター
  """
  node = mt.parseToNode(text)
  nounList = []
  elements = []
  while node:
    fields = node.feature.split(",")
    if fields[0] in ['名詞','動詞','形容詞'] \
      and not (fields[0] == '名詞' and fields[1] in ['代名詞','非自立','数']) \
      and not (fields[0] == '動詞' and fields[1] in ['接尾']) \
      and not (fields[0] == '動詞' and fields[6] in ['する','いう','ある']) \
      and node.surface not in ['年'] \
      and node.surface in wv:
      if node.surface not in elements:
        elements.append(node.surface)
        nounList.append({
          'surface': node.surface,
          'type': fields[0] + "." + fields[1],
          'fields':fields,
          'vector': wv[node.surface],
          'count': 1
        })
      else:
        nounList[min(i for i in range(len(nounList)) if nounList[i]['surface'] == node.surface)]['count'] += 1

    node = node.next
  return nounList

# 問題ベクター情報からTF-IDFに基づくサマリベクタを取得する
def getSummaryVector(questionVectors):
  count=len(questionVectors)
  nounCount = {}
  for qv in questionVectors:
    for v in qv:
      if v['surface'] in nounCount:
        nounCount[v['surface']] += 1
      else:
        nounCount[v['surface']] = 1
  
  returnList = []
  for qv in questionVectors:
    sum = np.zeros([50])
    for v in qv:
      pprint(v['vector'] * v['count'] * math.log(count / nounCount[v['surface']]))
      sum += v['vector'] * v['count'] * math.log(count / nounCount[v['surface']])
    pprint([sum])
    returnList.append(sum)
  
  return returnList

# 単語出現数dictを作成する
def getNounCountDict(questionVectors):
  nounCountDict={}
  for qv in questionVectors:
    for v in qv:
      if v['surface'] in nounCountDict:
        nounCountDict[v['surface']] += 1
      else:
        nounCountDict[v['surface']] = 1
  return nounCountDict

# 問題ベクター間距離関数
def getDistance(l1, l2):
  """問題ベクター間距離関数
  Args: l1(dict), l2(dict): 問題ベクター
  Returns: float: 距離
  """
  threshold = 9
  cosSims = getDirectProduct(l1, l2)
  dist = 0
  for i in range(min(threshold, len(cosSims))):
    dist +=  (1 - cosSims[i]['cosSim']) # * (1 / (i+1) ** 0.5)
  if len(cosSims) < threshold:
    dist += (len(cosSims) - threshold)
  return dist

# テキスト樹形図出力
def getTextDendrogram(num, indent, Z, questions, n):
  """テキスト樹形図出力
  Args: num(float): 枝番号
        indent: 表示する樹形
        Z: クラスタリング結果
        questions(list): 問題文リスト
        n(int): 問題数
  Returns: list: 樹形図（上から順の1行毎リスト）
  """
  if(num < n):
    return [indent + str(int(num)) + "." + questions[int(num)]]
  else:
    branchChars = "①②③④⑤⑥⑦⑧⑨"
    branchRank = int(n*2-num-1)
    if branchRank <= len(branchChars):
      branchChar = branchChars[branchRank-1]
    else:
      branchChar = "┬"
    return getTextDendrogram(Z[int(num-n), 0], indent+branchChar, Z, questions, n) \
         + getTextDendrogram(Z[int(num-n), 1], re.sub("[┬"+branchChars+"]","│",indent).replace("└","　")+"└", Z, questions, n)

# 最遠配置リストを得る
def scatterQuestion(num, Z, dMatrix, n):
  """最遠配置リストを得る
  Args: num(float): 枝番号
        Z: クラスタリング結果
        dMatrix: 距離マトリクス
        n(int): 問題数
  Returns: list: 最遠配置リスト
  """
  if(num < n):
    return [int(num)]
  else:
    v1 = scatterQuestion(Z[int(num-n), 0], Z, dMatrix, n)
    v2 = scatterQuestion(Z[int(num-n), 1], Z, dMatrix, n)
    i1 = 1
    i2 = 1
    d = 1.0 / (2.0 * (len(v1) + 1) * (len(v2) + 1))

    # 2つのリストの間で最も近い要素のインデックスを取得する
    dMatrixv1v2 = dMatrix[np.ix_(v1, v2)]
    # print(dMatrixv1v2)
    minIndex = np.unravel_index(np.argmin(dMatrixv1v2),  dMatrixv1v2.shape)
    minIndexv1 = minIndex[0]
    minIndexv2 = minIndex[1]
    # v1は当該要素が先頭に来るよう要素を移動
    v1 = v1[minIndexv1:] + v1[0:minIndexv1]
    # v2は当該要素が真ん中に来るよう要素を移動
    v2LenHalf = int((len(v2)+1)/2)
    v2 = v2[minIndexv2:] + v2[0:minIndexv2]
    v2 = v2[v2LenHalf:]  + v2[0:v2LenHalf]

    returnList = []
    while i1 <= len(v1) or i2 <= len(v2):
      if (i1 / (len(v1)+1)) > (i2 / (len(v2)+1) + d):
        returnList.append(v2[i2-1])
        i2 += 1
      else:
        returnList.append(v1[i1-1])
        i1 += 1
    return returnList

