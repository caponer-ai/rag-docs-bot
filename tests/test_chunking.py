from ragbot.chunking import MAX_CHARS, chunk_markdown, chunk_plain

MD = """# 12. Швидкість руху

### п. 12.4

12.4. У населених пунктах рух дозволяється не більше 50 км/год.

### п. 12.5

12.5. У житлових зонах не більше 20 км/год.

# 33. Дорожні знаки

Абзац перший.

Абзац другий.
"""


def test_clause_becomes_own_chunk_with_heading_path():
    chunks = chunk_markdown(MD, "pdr/12.md")
    clauses = [c for c in chunks if c.clause]
    assert [c.clause for c in clauses] == ["12.4", "12.5"]
    assert clauses[0].heading == "12. Швидкість руху › п. 12.4"
    assert clauses[0].embed_text.startswith("12. Швидкість руху")
    assert "50 км/год" in clauses[0].text


def test_plain_paragraphs_grouped_under_h1():
    chunks = chunk_markdown(MD, "pdr/33.md")
    tail = [c for c in chunks if c.heading.startswith("33.")]
    assert len(tail) == 1
    assert "Абзац перший" in tail[0].text and "Абзац другий" in tail[0].text
    assert tail[0].clause is None


def test_long_paragraph_is_split_by_sentences():
    long = " ".join(f"Речення номер {i} про щось важливе." for i in range(200))
    chunks = chunk_plain(long, "x.txt")
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHARS for c in chunks)
