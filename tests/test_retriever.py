from ragbot.retriever import rrf, tokenize


def test_rrf_prefers_items_present_in_both_rankings():
    scores = rrf([[1, 2, 3], [3, 4, 1]])
    assert scores[1] > scores[2]  # 1 є в обох, 2 лише в одному
    assert scores[3] > scores[4]
    assert set(scores) == {1, 2, 3, 4}


def test_tokenize_lowercases_and_stems_ukrainian():
    assert tokenize("Перехрестя") == tokenize("перехресті") == tokenize("ПЕРЕХРЕСТЯМ")
    assert "50" in tokenize("не більше 50 км/год")
    assert tokenize("рух") == ["рух"]  # короткі слова не чіпаємо
