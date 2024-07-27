from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WordData:
    """単語データを表すクラス

    Attribute:
        surface (str): 表層形
        word_type (str): 品詞
        vector (np.ndarray): ベクトル表現
    """

    surface: str
    word_type: str
    vector: np.ndarray

    def __eq__(self, __value: object) -> bool:
        """WordDataの等価性を判定する

        Args:
            __value (object): 比較するオブジェクト

        Returns:
            bool: 等価な場合はTrue，それ以外はFalse
        """
        if not isinstance(__value, WordData):
            return False
        return (
            self.surface == __value.surface
            and self.word_type == __value.word_type
            and np.all(self.vector == __value.vector)
        )


@dataclass
class SentenceWordsData:
    """文にある単語のデータを表すクラス

    Attribute:
        words (list[WordData]): 単語のリスト
        counts (list[int]): 単語の出現回数のリスト．
            words[i]の出現回数がcounts[i]である．
    """

    words: list[WordData]
    counts: list[int]

    def __post_init__(self):
        """データの検証を行う

        Raises:
            ValueError: nounsとcountsの長さが異なる場合
            ValueError: countsの要素に非正の値が含まれる場合
        """
        if len(self.words) != len(self.counts):
            raise ValueError("words and counts must have the same length")

        if any(count <= 0 for count in self.counts):
            raise ValueError("counts must be positive")

    @staticmethod
    def from_word_data_list(words_data_list: list[WordData]) -> "SentenceWordsData":
        """単語のリストからSentenceWordsDataを作成する

        Args:
            words_data_list (list[WordData]): 単語のリスト

        Raises:
            ValueError: words_data_listが空の場合

        Returns:
            SentenceWordsData: 単語のリストから作成したSentenceWordsData
        """
        if len(words_data_list) == 0:
            raise ValueError("words_data_list must not be empty")

        words = []
        counts = []

        for word_data in words_data_list:
            if word_data in words:
                index = words.index(word_data)
                counts[index] += 1
            else:
                words.append(word_data)
                counts.append(1)
        return SentenceWordsData(words=words, counts=counts)
