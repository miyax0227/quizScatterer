import dataclasses

import numpy as np
import pytest

from quizscatterer.data_structure import SentenceWordsData, WordData


class TestWordData:
    @staticmethod
    def test_normal() -> None:
        word_data = WordData(
            surface="猫", word_type="名詞.一般", vector=np.array([1.0, 2.0, 3.0])
        )
        assert word_data.surface == "猫"
        assert word_data.word_type == "名詞.一般"
        np.testing.assert_array_equal(word_data.vector, np.array([1.0, 2.0, 3.0]))

        # 上書きできないことのチェック
        with pytest.raises(dataclasses.FrozenInstanceError):
            word_data.word_type = "名詞.未定義語"

    @staticmethod
    def test_eq_returns_true() -> None:
        word_data1 = WordData(
            surface="猫", word_type="名詞.一般", vector=np.array([1.0, 2.0, 3.0])
        )
        word_data2 = WordData(
            surface="猫", word_type="名詞.一般", vector=np.array([1.0, 2.0, 3.0])
        )
        assert word_data1 == word_data2

    @staticmethod
    @pytest.mark.parametrize(
        "surface, word_type, vector",
        [
            ("犬", "名詞.一般", np.array([1.0, 2.0, 3.0])),  # surfaceが異なる
            ("猫", "名詞.未定義語", np.array([1.0, 2.0, 3.0])),  # word_typeが異なる
            ("猫", "名詞.一般", np.array([1.0, 2.0, 4.0])),  # vectorが異なる
        ],
    )
    def test_eq_returns_false(surface: str, word_type: str, vector: np.ndarray) -> None:
        word_data1 = WordData(
            surface="猫", word_type="名詞.一般", vector=np.array([1.0, 2.0, 3.0])
        )
        word_data2 = WordData(surface=surface, word_type=word_type, vector=vector)
        assert word_data1 != word_data2


class TestSentenceWordsData:
    @staticmethod
    def test_normal() -> None:
        word_data_list = [
            WordData(
                surface="隣", word_type="名詞.一般", vector=np.array([1.0, 2.0, 3.0])
            ),
            WordData(
                surface="客", word_type="名詞.一般", vector=np.array([1.0, 2.0, 4.0])
            ),
            WordData(
                surface="柿", word_type="名詞.一般", vector=np.array([1.0, 3.0, 5.0])
            ),
            WordData(
                surface="喰う", word_type="動詞.自立", vector=np.array([2.0, 3.0, 4.0])
            ),
        ]
        count_list = [1, 2, 1, 1]
        sentence_word_data = SentenceWordsData(words=word_data_list, counts=count_list)
        assert sentence_word_data.words == word_data_list
        assert sentence_word_data.counts == count_list

        # 上書きできないことのチェック

    @staticmethod
    def test_different_length() -> None:
        word_data_list = [
            WordData(
                surface="猫", word_type="名詞.一般", vector=np.array([1.0, 2.0, 3.0])
            )
        ]
        count_list = [1, 2]
        with pytest.raises(ValueError):
            _ = SentenceWordsData(words=word_data_list, counts=count_list)

    @staticmethod
    @pytest.mark.parametrize("count", [-1, 0])
    def test_non_positive_count(count: int) -> None:
        word_data_list = [
            WordData(
                surface="猫", word_type="名詞.一般", vector=np.array([1.0, 2.0, 3.0])
            )
        ]
        count_list = [count]
        with pytest.raises(ValueError):
            _ = SentenceWordsData(words=word_data_list, counts=count_list)

    @staticmethod
    def test_from_word_data_list() -> None:
        word_data_list = [
            WordData(
                surface="隣", word_type="名詞.一般", vector=np.array([1.0, 2.0, 3.0])
            ),
            WordData(
                surface="客", word_type="名詞.一般", vector=np.array([1.0, 2.0, 4.0])
            ),
            WordData(
                surface="柿", word_type="名詞.一般", vector=np.array([1.0, 3.0, 5.0])
            ),
            WordData(
                surface="喰う", word_type="動詞.自立", vector=np.array([2.0, 3.0, 4.0])
            ),
            WordData(
                surface="客", word_type="名詞.一般", vector=np.array([1.0, 2.0, 4.0])
            ),
        ]
        sentence_word_data = SentenceWordsData.from_word_data_list(word_data_list)
        assert isinstance(sentence_word_data, SentenceWordsData)
        # wordsは重複が除かれたリスト
        assert sentence_word_data.words == word_data_list[:-1]
        # 「客」は2回カウントされている．
        assert sentence_word_data.counts == [1, 2, 1, 1]
