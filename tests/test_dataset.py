from icd_mock.dataset import load_cases, load_code_descriptions, used_codes


def test_thirty_distinct_cases():
    cases = load_cases()
    assert len(cases) == 30
    assert len({c.id for c in cases}) == 30
    assert len({c.expected_code for c in cases}) == 30


def test_every_code_is_in_the_official_subset():
    descriptions = load_code_descriptions()
    assert used_codes(load_cases()) <= descriptions.keys()


def test_administrative_paragraph_is_never_relevant():
    # Paragraph 0 is the visit or triage line, which the extraction step must drop.
    assert all(0 not in case.relevant for case in load_cases())


def test_interesting_paths_match_the_readme():
    cases = load_cases()
    assert sum(c.is_loop for c in cases) == 6
    assert sum(c.is_retrieval_miss for c in cases) == 3
    assert sum(c.is_wrong_final for c in cases) == 3
