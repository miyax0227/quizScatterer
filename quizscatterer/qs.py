# -*- coding: utf-8 -*-
import itertools
import math
import os
import re
from pprint import pprint

import gensim
import MeCab
import numpy as np

# 実行ファイルパスを取得
exec_path = os.path.dirname(__file__)
# 学習済みベクターモデルの読込
word2vec_model = gensim.models.Word2Vec.load(
    exec_path + "/gensimModel/word2vec.gensim.model"
).wv
# MeCab辞書読込
mecab_tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")


def regulate_question(question: str) -> str:
    """問題文を正規化する

    Args:
        question (str): 問題文

    Returns:
        str: 正規化された問題文
    """
    question = question.translate(str.maketrans({"（": "(", "）": ")"}))
    question = re.sub(r"\([\u3041-\u309f・]+\)", "", question)
    question = re.sub(r"[?？]", "", question)
    return question


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """コサイン類似度を得る

    Args:
        v1(np.Array): ベクトル
        v2(np.Array): ベクトル v1と同次元

    Returns: float: 類似度(-1～1)
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def compute_direct_product(
    question_vectors_list_1: list[dict], question_vectors_list_2: list[dict]
) -> list[dict]:
    """問題ベクターから単語対類似度リスト（類似度が高い順）を得る
    Args:
        question_vectors_list_1(list[dict]): 問題ベクター
        question_vectors_list_2(list[dict]): 問題ベクター

    Returns:
        list[dict]: 単語対類似度リスト
    """
    direct_product_list = []
    for question_vector_1, question_vector_2 in itertools.product(
        question_vectors_list_1, question_vectors_list_2
    ):
        direct_product_list.append(
            {
                "word1": question_vector_1["surface"],
                "word2": question_vector_2["surface"],
                "cosSim": compute_cosine_similarity(
                    question_vector_1["vector"], question_vector_2["vector"]
                ),
            }
        )
    return sorted(direct_product_list, key=lambda x: x["cosSim"], reverse=True)


def create_wakachigaki_list(text: str) -> list[dict]:
    """分かち書きリストを作成する

    Args:
        text(str): 問題文

    Returns:
        list[dict]: 分かち書きリスト
    """
    node = mecab_tagger.parseToNode(text)
    wakachigaki_list = []
    while node:
        wakachigaki_list.append({"_surface": node.surface, "feature": node.feature})
        node = node.next
    return wakachigaki_list


def get_text_vector(text: str) -> list[dict]:
    """問題文から問題ベクターを得る

    Args:
        text(str): 問題文

    Returns:
        list[dict]: 問題ベクター
    """
    node = mecab_tagger.parseToNode(text)
    noun_list = []
    elements = []
    while node:
        fields = node.feature.split(",")
        if (
            fields[0] in ["名詞", "動詞", "形容詞"]
            and not (fields[0] == "名詞" and fields[1] in ["代名詞", "非自立", "数"])
            and not (fields[0] == "動詞" and fields[1] in ["接尾"])
            and not (fields[0] == "動詞" and fields[6] in ["する", "いう", "ある"])
            and node.surface not in ["年"]
            and node.surface in word2vec_model
        ):
            if node.surface not in elements:
                elements.append(node.surface)
                noun_list.append(
                    {
                        "surface": node.surface,
                        "type": fields[0] + "." + fields[1],
                        "fields": fields,
                        "vector": word2vec_model[node.surface],
                        "count": 1,
                    }
                )
            else:
                noun_list[
                    min(
                        i
                        for i in range(len(noun_list))
                        if noun_list[i]["surface"] == node.surface
                    )
                ]["count"] += 1

        node = node.next
    return noun_list


def get_summary_vector(question_vectors_list: list[list[dict]]) -> list[np.ndarray]:
    """問題ベクター情報からTF-IDFに基づくサマリベクタを取得する

    Args:
        question_vectors_list(list[list[dict]]): 問題ベクターリスト

    Returns:
        list[np.ndarray]: TF-IDFに基づくサマリベクタリスト
    """
    count = len(question_vectors_list)
    noun_count_dict = {}
    for question_vector in question_vectors_list:
        for vector in question_vector:
            if vector["surface"] in noun_count_dict:
                noun_count_dict[vector["surface"]] += 1
            else:
                noun_count_dict[vector["surface"]] = 1

    return_list = []
    for question_vector in question_vectors_list:
        tf_idf_sum = np.zeros([50])
        for vector in question_vector:
            pprint(
                vector["vector"]
                * vector["count"]
                * math.log(count / noun_count_dict[vector["surface"]])
            )
            tf_idf_sum += (
                vector["vector"]
                * vector["count"]
                * math.log(count / noun_count_dict[vector["surface"]])
            )
        pprint([tf_idf_sum])
        return_list.append(tf_idf_sum)

    return return_list


def compute_noun_count_dict(question_vectors_list: list[list[dict]]) -> dict:
    """単語出現数のdictionaryを作成する

    Args:
        question_vectors_list(list[list[dict]]): 問題ベクターリスト

    Returns:
        dict: 単語出現数のdictionary
    """
    noun_count_dict = {}
    for question_vector in question_vectors_list:
        for vector in question_vector:
            if vector["surface"] in noun_count_dict:
                noun_count_dict[vector["surface"]] += 1
            else:
                noun_count_dict[vector["surface"]] = 1
    return noun_count_dict


# 問題ベクター間距離関数
def compute_distance_bw_question_vectors(
    question_vectors_1: list[dict], question_vectors_2: list[dict]
) -> float:
    """問題ベクター間距離を計算する
    Args:
        question_vectors_1(list[dict]): 問題ベクター1
        question_vectors_2(list[dict]): 問題ベクター2

    Returns:
        float: 距離
    """
    threshold = 9
    cosine_similarity_list = compute_direct_product(
        question_vectors_1, question_vectors_2
    )
    dist = 0
    for i in range(min(threshold, len(cosine_similarity_list))):
        dist += 1 - cosine_similarity_list[i]["cosSim"]  # * (1 / (i+1) ** 0.5)
    if len(cosine_similarity_list) < threshold:
        dist += len(cosine_similarity_list) - threshold
    return dist


# テキスト樹形図出力
def draw_text_dendrogram(
    branch_number: float,
    indent_string: str,
    clustering_result: np.ndarray,
    questions: list[str],
    number_of_questions: int,
) -> list[str]:
    """テキスト樹形図出力

    Args:
        branch_number(float): 枝番号
        indent_string(str): 表示する樹形
        clustering_result(np.ndarray): クラスタリング結果
        questions(list): 問題文リスト
        number_of_questions(int): 問題数

    Returns:
        list[str]: 樹形図（上から順の1行毎リスト）
    """
    if branch_number < number_of_questions:
        return [
            indent_string
            + str(int(branch_number))
            + "."
            + questions[int(branch_number)]
        ]
    else:
        branch_characters = "①②③④⑤⑥⑦⑧⑨"
        branch_rank = int(number_of_questions * 2 - branch_number - 1)
        if branch_rank <= len(branch_characters):
            branch_character = branch_characters[branch_rank - 1]
        else:
            branch_character = "┬"
        return draw_text_dendrogram(
            clustering_result[int(branch_number - number_of_questions), 0],
            indent_string + branch_character,
            clustering_result,
            questions,
            number_of_questions,
        ) + draw_text_dendrogram(
            clustering_result[int(branch_number - number_of_questions), 1],
            re.sub("[┬" + branch_characters + "]", "│", indent_string).replace(
                "└", "　"
            )
            + "└",
            clustering_result,
            questions,
            number_of_questions,
        )


# 最遠配置リストを得る
def scatter_questions(
    branch_number: float,
    clustering_result: np.ndarray,
    distance_matrix: np.ndarray,
    number_of_questions: int,
) -> list[int]:
    """最遠配置リストを得る
    Args:
        branch_number(float): 枝番号
        clustering_result(np.ndarray): クラスタリング結果
        distance_matrix(np.ndarray): 距離マトリクス
        number_of_questions(int): 問題数

    Returns:
        list[int]: 最遠配置リスト
    """
    if branch_number < number_of_questions:
        return [int(branch_number)]
    else:
        v1 = scatter_questions(
            clustering_result[int(branch_number - number_of_questions), 0],
            clustering_result,
            distance_matrix,
            number_of_questions,
        )
        v2 = scatter_questions(
            clustering_result[int(branch_number - number_of_questions), 1],
            clustering_result,
            distance_matrix,
            number_of_questions,
        )
        i1 = 1
        i2 = 1
        d = 1.0 / (2.0 * (len(v1) + 1) * (len(v2) + 1))

        # 2つのリストの間で最も近い要素のインデックスを取得する
        distance_array = distance_matrix[np.ix_(v1, v2)]
        min_index = np.unravel_index(np.argmin(distance_array), distance_array.shape)
        min_index_v1 = int(min_index[0])
        min_index_v2 = int(min_index[1])
        # v1は当該要素が先頭に来るよう要素を移動
        v1 = v1[min_index_v1:] + v1[0:min_index_v1]
        # v2は当該要素が真ん中に来るよう要素を移動
        v2_half_length = int((len(v2) + 1) / 2)
        v2 = v2[min_index_v2:] + v2[0:min_index_v2]
        v2 = v2[v2_half_length:] + v2[0:v2_half_length]

        return_list = []
        while i1 <= len(v1) or i2 <= len(v2):
            if (i1 / (len(v1) + 1)) > (i2 / (len(v2) + 1) + d):
                return_list.append(v2[i2 - 1])
                i2 += 1
            else:
                return_list.append(v1[i1 - 1])
                i1 += 1
        return return_list
