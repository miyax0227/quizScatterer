from collections.abc import Iterator
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

    def __eq__(self, other_value: object) -> bool:
        """WordDataの等価性を判定する

        Args:
            other_value (object): 比較するオブジェクト

        Returns:
            bool: 等価な場合はTrue，それ以外はFalse
        """
        if not isinstance(other_value, WordData):
            return False
        return (
            self.surface == other_value.surface
            and self.word_type == other_value.word_type
            and np.allclose(self.vector, other_value.vector)
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

    def __iter__(self) -> Iterator[tuple[WordData, int]]:
        """SentenceWordsDataのイテレータを返す
        イテレータは，(WordData, int)のタプルを返す

        Returns:
            Iterator: イテレータ
        """
        for word, count in zip(self.words, self.counts):
            yield word, count

    @staticmethod
    def from_word_data_list(words_data_list: list[WordData]) -> "SentenceWordsData":
        """`WordData`のリストから`SentenceWordsData`を作成する

        Args:
            words_data_list (list[WordData]): `WordData`のリスト

        Raises:
            ValueError: `words_data_list`が空の場合

        Returns:
            SentenceWordsData: `WordData`のリストから作成したSentenceWordsData
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
