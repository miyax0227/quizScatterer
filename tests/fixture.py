import numpy as np
import pytest

from quizscatterer.data_structure import SentenceWordsData, WordData


@pytest.fixture
def prepare_sentence_words_data_list() -> list[SentenceWordsData]:
    def create_vector(index: int) -> np.ndarray:
        vector = np.zeros(50)
        vector[index] = 1.0
        return vector
    sentence_words_data_1 = SentenceWordsData(
        words=[
            WordData(
                surface="隣",
                word_type="名詞.一般",
                vector=create_vector(0),
            ),
            WordData(
                surface="客",
                word_type="名詞.一般",
                vector=create_vector(1),
            ),
            WordData(
                surface="柿",
                word_type="名詞.一般",
                vector=create_vector(2),
            ),
            WordData(
                surface="食う",
                word_type="動詞.自立",
                vector=create_vector(3),
            ),
        ],
        counts=[1, 2, 1, 1],
    )
    sentence_words_data_2 = SentenceWordsData(
        words=[
            WordData(
                surface="柿",
                word_type="名詞.一般",
                vector=create_vector(2),
            ),
            WordData(
                surface="甘く",
                word_type="形容詞.自立",
                vector=create_vector(3),
            ),
            WordData(
                surface="美味しい",
                word_type="形容詞.自立",
                vector=create_vector(4),
            ),
        ],
        counts=[1, 1, 1],
    )
    return [sentence_words_data_1, sentence_words_data_2]
