"""
tests/test_phrases.py

Tests for suspicious phrase detection logic.
No model needed — pure string matching.
"""
from __future__ import annotations

import pytest
from main import _find_suspicious_phrases


class TestSuspiciousPhrases:
    def test_english_phrase_detected(self):
        text   = "The machine was in perfect condition before the incident."
        result = _find_suspicious_phrases(text)
        assert len(result) >= 1
        assert any("perfect condition" in r for r in result)

    def test_french_phrase_detected(self):
        text   = "Aucune anomalie n'a été détectée avant l'incident."
        result = _find_suspicious_phrases(text)
        assert len(result) >= 1

    def test_case_insensitive(self):
        text   = "SUDDEN FAILURE WITH NO WARNING reported on site."
        result = _find_suspicious_phrases(text)
        assert len(result) >= 1

    def test_clean_text_no_flags(self):
        text   = "Gradual vibration increase observed over 3 weeks. Temperature rose steadily."
        result = _find_suspicious_phrases(text)
        assert result == []

    def test_multiple_phrases_detected(self):
        text = (
            "Sudden failure with no warning. "
            "All checks passed. "
            "Machine was perfectly maintained."
        )
        result = _find_suspicious_phrases(text)
        assert len(result) >= 2

    def test_returns_list_of_strings(self):
        result = _find_suspicious_phrases("unexpected breakdown occurred.")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_empty_text_returns_empty(self):
        assert _find_suspicious_phrases("") == []