# -*- coding: utf-8 -*-
import itertools
import math
import os
import re
from pprint import pprint
from typing import Any

import gensim
import MeCab
import numpy as np

from quizscatterer.data_structure import SentenceWordsData, WordData

# 実行ファイルパスを取得
exec_path = os.path.dirname(__file__)
# 学習済みベクターモデルの読込
word2vec_model = gensim.models.Word2Vec.load(
    exec_path + "/gensim_model/word2vec.gensim.model"
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


def compute_word_similarity_list(
    sentence_words_data_1: SentenceWordsData, sentence_words_data_2: SentenceWordsData
) -> list[dict]:
    """問題ベクターから単語対類似度リスト（類似度が高い順）を得る
    Args:
        sentence_words_data_1(SentenceWordsData): 問題ベクター1
        sentence_words_data_2(SentenceWordsData): 問題ベクター2

    Returns:
        list[dict]: 単語対類似度リスト
    """
    direct_product_list = []
    for word_1, word_2 in itertools.product(
        sentence_words_data_1.words, sentence_words_data_2.words
    ):
        direct_product_list.append(
            {
                "word1": word_1.surface,
                "word2": word_2.surface,
                "cosSim": compute_cosine_similarity(word_1.vector, word_2.vector),
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


def check_whether_word_is_considered(node: Any) -> tuple[bool, list[str] | None]:
    """ノードを考慮するかどうかを判定する

    Args:
        node(Any): MeCabのNodeオブジェクト

    Returns:
        tuple[bool, list[str] | None]: ノードを考慮するかどうかと，ノードの情報．
            ノードを考慮する場合はTrueとノードの情報，それ以外はFalseとNone
    """
    fields = node.feature.split(",")

    considers = (
        fields[0] in ["名詞", "動詞", "形容詞"]
        and not (fields[0] == "名詞" and fields[1] in ["代名詞", "非自立", "数"])
        and not (fields[0] == "動詞" and fields[1] in ["接尾"])
        and not (fields[0] == "動詞" and fields[6] in ["する", "いう", "ある"])
        and node.surface not in ["年"]
        and node.surface in word2vec_model
    )
    return considers, (fields if considers else None)


def get_sentence_words_data(text: str) -> SentenceWordsData:
    """問題文から問題ベクターを得る

    Args:
        text(str): 問題文

    Returns:
        SentenceWordsData: 問題ベクター
    """
    node = mecab_tagger.parseToNode(text)

    word_data_list: list[WordData] = []
    while node:
        considers, fields = check_whether_word_is_considered(node)
        if considers:
            word_data = WordData(
                surface=node.surface,
                word_type=fields[0] + "." + fields[1],
                vector=word2vec_model[node.surface],
            )
            word_data_list.append(word_data)
        node = node.next

    sentence_words_data = SentenceWordsData.from_word_data_list(word_data_list)
    return sentence_words_data


def get_summary_vector(
    sentence_words_data_list: list[SentenceWordsData],
) -> list[np.ndarray]:
    """問題ベクター情報からTF-IDFに基づくサマリベクタを取得する

    Args:
        question_vectors_list(list[list[dict]]): 問題ベクターリスト

    Returns:
        list[np.ndarray]: TF-IDFに基づくサマリベクタリスト
    """
    number_of_sentences = len(sentence_words_data_list)
    noun_count_dict = compute_noun_count_dict(sentence_words_data_list)

    return_list = []
    for sentence_words_data in sentence_words_data_list:
        tf_idf_sum = np.zeros([50])
        for word, count in sentence_words_data:
            pprint(
                word.vector
                * count
                * math.log(number_of_sentences / noun_count_dict[word.surface])
            )
            tf_idf_sum += (
                word.vector
                * count
                * math.log(number_of_sentences / noun_count_dict[word.surface])
            )
        pprint([tf_idf_sum])
        return_list.append(tf_idf_sum)

    return return_list


def compute_noun_count_dict(sentence_words_data_list: list[SentenceWordsData]) -> dict:
    """単語出現数のdictionaryを作成する

    Args:
        question_vectors_list(list[SentenceWordsData]): 問題ベクターリスト

    Returns:
        dict: 単語出現数のdictionary．キーは単語，値は出現数
    """
    noun_count_dict = {}
    for sentence_words_data in sentence_words_data_list:
        for word in sentence_words_data.words:
            if word.surface in noun_count_dict:
                noun_count_dict[word.surface] += 1
            else:
                noun_count_dict[word.surface] = 1
    return noun_count_dict


# 問題ベクター間距離関数
def compute_distance_bw_question_vectors(
    sentence_words_data_1: SentenceWordsData, sentence_words_data_2: SentenceWordsData
) -> float:
    """問題ベクター間距離を計算する
    Args:
        sentence_words_data_1(SentenceWordsData): 問題ベクター1
        sentence_words_data_2(SentenceWordsData): 問題ベクター2

    Returns:
        float: 距離
    """
    # TODO: しきい値はパラメータ化する
    threshold = 9
    cosine_similarity_list = compute_word_similarity_list(
        sentence_words_data_1, sentence_words_data_2
    )
    distance = 0
    for i in range(min(threshold, len(cosine_similarity_list))):
        distance += 1 - cosine_similarity_list[i]["cosSim"]  # * (1 / (i+1) ** 0.5)
    if len(cosine_similarity_list) < threshold:
        distance += len(cosine_similarity_list) - threshold
    return distance


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
