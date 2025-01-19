import numpy as np
import pytest

from quizscatterer import qs
from quizscatterer.data_structure import SentenceWordsData, WordData


class Node:
    """MeCabのノード用のモッククラス"""

    def __init__(self, surface: str, feature: str):
        self.surface = surface
        self.feature = feature
        self.next = None


class TestRegulateQuestion:
    @staticmethod
    @pytest.mark.parametrize(
        "question, expected",
        [
            ("これはテスト（てすと）ですか", "これはテストですか"),  # 全角括弧
            ("これはテスト(てすと）です", "これはテストです"),  # 半角括弧と全角括弧
            (
                "これはテスト（てすと・てすと）です",
                "これはテストです",
            ),  # 全角括弧と全角中点
            ("これはテストですか？", "これはテストですか"),  # 全角疑問符
            ("これはテストですか?", "これはテストですか"),  # 半角疑問符
            ("これはテストです", "これはテストです"),  # 不変になる場合
        ],
    )
    def test_normal(question: str, expected: str) -> None:
        assert qs.regulate_question(question) == expected


@pytest.mark.parametrize(
    "v1, v2, expected",
    [
        (np.array([1, 0, 0]), np.array([0, 1, 0]), 0.0),
        (np.array([1, 0, 0]), np.array([1, 1, 0]), 1 / np.sqrt(2)),
    ],
)
def test_compute_cosine_similarity(
    v1: np.ndarray, v2: np.ndarray, expected: float
) -> None:
    np.testing.assert_almost_equal(qs.compute_cosine_similarity(v1, v2), expected)


def test_compute_word_similarity_list():
    sentence1 = SentenceWordsData(
        words=[
            WordData(
                surface="テスト", word_type="名詞.一般", vector=np.array([1, 0, 0])
            ),
            WordData(surface="試験", word_type="名詞.一般", vector=np.array([0, 1, 0])),
            WordData(surface="得点", word_type="名詞.一般", vector=np.array([0, 1, 1])),
        ],
        counts=[1, 2, 1],
    )
    sentence2 = SentenceWordsData(
        words=[
            WordData(
                surface="点数", word_type="名詞.一般", vector=np.array([0.5, 1, 1])
            ),
            WordData(
                surface="入試", word_type="名詞.一般", vector=np.array([0, 1, 0.1])
            ),
        ],
        counts=[1, 1],
    )

    result = qs.compute_word_similarity_list(sentence1, sentence2)
    assert isinstance(result, list)
    assert len(result) == len(sentence1.words) * len(sentence2.words)
    for item in result:
        assert isinstance(item, dict)
        assert "cosSim" in item
        assert "word1" in item
        assert "word2" in item
        assert isinstance(item["cosSim"], float)
        assert isinstance(item["word1"], str)
        assert isinstance(item["word2"], str)
        assert item["word1"] in [word.surface for word in sentence1.words]
        assert item["word2"] in [word.surface for word in sentence2.words]
    # 類似度が高い順に並んでいるか
    for i in range(len(result) - 1):
        assert result[i]["cosSim"] >= result[i + 1]["cosSim"]
    # 類似度最大のペアが正しいか
    assert result[0]["word1"] == "試験"
    assert result[0]["word2"] == "入試"
    np.testing.assert_almost_equal(result[0]["cosSim"], 1.0 / np.sqrt(1.01))


def test_create_wakachigaki_list():
    result = qs.create_wakachigaki_list("これからテストを行います")
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, dict)
        assert "_surface" in item
        assert "feature" in item
        assert isinstance(item["_surface"], str)
        assert isinstance(item["feature"], str)
    assert [item["_surface"] for item in result] == [
        "",
        "これから",
        "テスト",
        "を",
        "行い",
        "ます",
        "",
    ]


class TestCheckWhetherWordIsConsidered:
    @staticmethod
    def test_if_not_noun_nor_verb_nor_adjective() -> None:
        node = Node("とても", "副詞,助詞類接続,*,*,*,*,とても,トテモ,トテモ")
        considers, fields = qs.check_whether_word_is_considered(node)
        assert considers is False
        assert fields is None

    @staticmethod
    @pytest.mark.parametrize(
        "surface, fields, expected_considers",
        [
            ("私", "名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ", False),
            ("もの", "名詞,非自立,一般,*,*,*,もの,モノ,モノ", False),
            ("3", "名詞,数,*,*,*,*,*", False),
            ("テスト", "名詞,一般,*,*,*,*,テスト,テスト,テスト", True),
        ],
    )
    def test_if_noun(surface: str, fields: str, expected_considers: bool) -> None:
        node = Node(surface, fields)
        considers, result_fields = qs.check_whether_word_is_considered(node)
        assert considers is expected_considers
        if considers:
            assert result_fields == fields.split(",")
        else:
            assert result_fields is None

    @staticmethod
    @pytest.mark.parametrize(
        "surface, fields, expected_considers",
        [
            ("食べる", "動詞,自立,*,*,五段・ラ行,基本形,食べる,タベル,タベル", True),
            ("れる", "動詞,接尾,*,*,一段,基本形,れる,レル,レル", False),
            ("し", " 動詞,自立,*,*,サ変・スル,連用形,する,シ,シ", False),
            ("いる", " 動詞,非自立,*,*,一段,基本形,いる,イル,イル", False),
            ("いう", "動詞,自立,*,*,五段・ワ行促音便,基本形,いう,イウ,イウ", False),
        ],
    )
    def test_if_verb(surface: str, fields: str, expected_considers: bool) -> None:
        node = Node(surface, fields)
        considers, result_fields = qs.check_whether_word_is_considered(node)
        assert considers is expected_considers
        if considers:
            assert result_fields == fields.split(",")
        else:
            assert result_fields is None

    @staticmethod
    def test_if_nen() -> None:
        node = Node("年", "名詞,接尾,助数詞,*,*,*,年,ネン,ネン")
        considers, fields = qs.check_whether_word_is_considered(node)
        assert considers is False
        assert fields is None


def test_get_sentence_words_data():
    result = qs.get_sentence_words_data("隣の客はよく柿食う客だ")
    assert isinstance(result, SentenceWordsData)
    assert [word.surface for word in result.words] == ["隣", "客", "柿", "食う"]
    assert result.counts == [1, 2, 1, 1]


def test_compute_noun_count_dict(
    prepare_sentence_words_data_list: list[SentenceWordsData],
) -> None:
    result = qs.compute_noun_count_dict(prepare_sentence_words_data_list)
    assert isinstance(result, dict)
    for key, value in result.items():
        assert isinstance(key, str)
        assert isinstance(value, int)
    assert set(result.keys()) == {"隣", "客", "柿", "食う", "甘く", "美味しい"}
    for key, value in result.items():
        if key in ["柿"]:
            assert value == 2
        else:
            assert value == 1


def test_get_summary_vector(
    prepare_sentence_words_data_list: list[SentenceWordsData],
) -> None:
    result = qs.get_summary_vector(prepare_sentence_words_data_list)
    assert isinstance(result, list)
    assert len(result) == len(prepare_sentence_words_data_list)
    for item in result:
        assert isinstance(item, np.ndarray)
        assert item.shape == (50,)
    expected_result_1 = np.zeros(50)
    expected_result_1[0] = 1.0 * 1 * np.log(2 / 1)  # 隣
    expected_result_1[1] = 1.0 * 2 * np.log(2 / 1)  # 客 (文中２回，文間1回)
    expected_result_1[2] = 1.0 * 1 * np.log(2 / 2)  # 柿 (文中1回，文間2回)
    expected_result_1[3] = 1.0 * 1 * np.log(2 / 1)  # 食う

    expected_result_2 = np.zeros(50)
    expected_result_2[2] = 1.0 * 1 * np.log(2 / 2)  # 柿 (文中1回，文間2回)
    expected_result_2[3] = 1.0 * 1 * np.log(2 / 1)  # 甘く
    expected_result_2[4] = 1.0 * 1 * np.log(2 / 1)  # 美味しい
    np.testing.assert_almost_equal(result[0], expected_result_1)
    np.testing.assert_almost_equal(result[1], expected_result_2)
