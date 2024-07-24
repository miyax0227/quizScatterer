from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NounData:
    """名詞のデータを表すクラス

    Attribute:
        surface (str): 表層形
        noun_type (str): 品詞
        vector (np.ndarray): ベクトル表現
    """

    surface: str
    noun_type: str
    vector: np.ndarray


@dataclass(frozen=True)
class SentenceNounData:
    """文にある名詞のデータを表すクラス

    Attribute:
        nouns (list[NounData]): 名詞のリスト
        counts (list[int]): 名詞の出現回数のリスト．
            nouns[i]の出現回数がcounts[i]である．
    """

    nouns: list[NounData]
    counts: list[int]

    def __post_init__(self):
        """データの検証を行うメソッド

        Raises:
            ValueError: nounsとcountsの長さが異なる場合
            ValueError: countsの要素に非正の値が含まれる場合
        """
        if len(self.nouns) != len(self.counts):
            raise ValueError("nouns and counts must have the same length")

        if any(count <= 0 for count in self.counts):
            raise ValueError("counts must be positive")

    @staticmethod
    def from_noun_data_list(nouns_data_list: list[NounData]) -> "SentenceNounData":
        """名詞のリストからSentenceNounDataを作成するメソッド

        Args:
            nouns_list (list[NounData]): 名詞のリスト

        Raises:
            ValueError: nouns_data_listが空の場合

        Returns:
            SentenceNounData: 名詞のリストから作成したSentenceNounData
        """
        if len(nouns_data_list) == 0:
            raise ValueError("nouns_data_list must not be empty")

        nouns = []
        counts = []

        for noun_data in nouns_data_list:
            if noun_data in nouns:
                index = nouns.index(noun_data)
                counts[index] += 1
            else:
                nouns.append(noun_data)
                counts.append(1)
        return SentenceNounData(nouns=nouns, counts=counts)
