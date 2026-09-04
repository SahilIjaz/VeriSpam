"""Configurable text-preprocessing pipeline.

Every step here is a deterministic, content-only transform (regex
substitution, a fixed stopword list, a rule-based stemmer) — none of it
learns anything from the corpus. That's what makes it safe to run on the
*entire* dataset before the train/test split: the only step that must never
see the test set is TF-IDF vectorization, which lives inside the training
`Pipeline` in `src/modeling.py`, fit only on `X_train`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import (
    EMAIL_PATTERN,
    NUMBER_PATTERN,
    PHONE_PATTERN,
    PUNCTUATION_PATTERN,
    URL_PATTERN,
    WHITESPACE_PATTERN,
)


@dataclass
class PreprocessingConfig:
    lowercase: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = False
    remove_punctuation: bool = True
    remove_numbers: bool = False
    normalize_whitespace: bool = True
    remove_stopwords: bool = False
    normalization: str = "none"  # "none" | "stem" | "lemmatize"

    def as_dict(self) -> dict:
        return {
            "lowercase": self.lowercase,
            "remove_urls": self.remove_urls,
            "remove_emails": self.remove_emails,
            "remove_phone_numbers": self.remove_phone_numbers,
            "remove_punctuation": self.remove_punctuation,
            "remove_numbers": self.remove_numbers,
            "normalize_whitespace": self.normalize_whitespace,
            "remove_stopwords": self.remove_stopwords,
            "normalization": self.normalization,
        }


# --- Rule-based normalization (no NLTK / no network download required) ---

_IRREGULAR_LEMMAS = {
    "is": "be", "are": "be", "was": "be", "were": "be", "am": "be", "been": "be", "being": "be",
    "has": "have", "had": "have", "having": "have",
    "does": "do", "did": "do", "doing": "do", "done": "do",
    "goes": "go", "went": "go", "gone": "go", "going": "go",
}


def _simple_lemmatize_word(word: str) -> str:
    """A lightweight, rule-based lemmatizer (not a full WordNet lemmatizer).

    Handles common inflections (plurals, -ing, -ed) with a short irregular-verb
    map; labeled "simplified lemmatization" in the UI since it trades some
    linguistic precision for zero extra dependencies / no corpus download.
    """
    if word in _IRREGULAR_LEMMAS:
        return _IRREGULAR_LEMMAS[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        return stem + "e" if stem.endswith(("at", "iz", "ol")) else stem
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    return word


class _PorterStemmer:
    """Classic Porter (1980) stemming algorithm, pure Python, no dependencies."""

    def _cons(self, word: str, i: int) -> bool:
        ch = word[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return i == 0 or not self._cons(word, i - 1)
        return True

    def _m(self, word: str, j: int) -> int:
        n, i, count = j, 0, 0
        while True:
            if i > n:
                return count
            if not self._cons(word, i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > n:
                    return count
                if self._cons(word, i):
                    break
                i += 1
            i += 1
            count += 1
            while True:
                if i > n:
                    return count
                if not self._cons(word, i):
                    break
                i += 1
            i += 1

    def _vowel_in_stem(self, word: str, j: int) -> bool:
        return any(not self._cons(word, i) for i in range(j + 1))

    def _double_cons(self, word: str, j: int) -> bool:
        return j > 0 and word[j] == word[j - 1] and self._cons(word, j)

    def _cvc(self, word: str, i: int) -> bool:
        if i < 2 or not self._cons(word, i) or self._cons(word, i - 1) or not self._cons(word, i - 2):
            return False
        return word[i] not in "wxy"

    def stem(self, word: str) -> str:
        word = word.lower()
        if len(word) < 3:
            return word
        k = len(word) - 1

        # Step 1a
        if word.endswith("sses"):
            word, k = word[:-2], k - 2
        elif word.endswith("ies"):
            word, k = word[:-2], k - 2
        elif word.endswith("ss"):
            pass
        elif word.endswith("s"):
            word, k = word[:-1], k - 1

        # Step 1b
        j = k
        if word.endswith("eed"):
            if self._m(word, k - 3) > 0:
                word, k = word[:-1], k - 1
        elif (word.endswith("ed") and self._vowel_in_stem(word, k - 2 - 1)) or (
            word.endswith("ing") and self._vowel_in_stem(word, k - 3 - 1)
        ):
            word = word[:-2] if word.endswith("ed") else word[:-3]
            k = len(word) - 1
            if word.endswith(("at", "bl", "iz")):
                word += "e"
                k += 1
            elif self._double_cons(word, k) and word[-1] not in "lsz":
                word = word[:-1]
                k -= 1
            elif self._m(word, k) == 1 and self._cvc(word, k):
                word += "e"
                k += 1

        # Step 1c
        if word.endswith("y") and self._vowel_in_stem(word, k - 1):
            word = word[:-1] + "i"

        step2_suffixes = {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance", "izer": "ize",
            "abli": "able", "alli": "al", "entli": "ent", "eli": "e", "ousli": "ous",
            "ization": "ize", "ation": "ate", "ator": "ate", "alism": "al", "iveness": "ive",
            "fulness": "ful", "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
        }
        for suffix, repl in step2_suffixes.items():
            if word.endswith(suffix):
                stem_len = len(word) - len(suffix)
                if self._m(word, stem_len - 1) > 0:
                    word = word[:stem_len] + repl
                break

        step3_suffixes = {
            "icate": "ic", "ative": "", "alize": "al", "iciti": "ic", "ical": "ic", "ful": "", "ness": "",
        }
        for suffix, repl in step3_suffixes.items():
            if word.endswith(suffix):
                stem_len = len(word) - len(suffix)
                if self._m(word, stem_len - 1) > 0:
                    word = word[:stem_len] + repl
                break

        step4_suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
            "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ]
        for suffix in step4_suffixes:
            if word.endswith(suffix):
                stem_len = len(word) - len(suffix)
                if suffix == "ion":
                    if stem_len > 0 and word[stem_len - 1] in "st" and self._m(word, stem_len - 1) > 1:
                        word = word[:stem_len]
                elif self._m(word, stem_len - 1) > 1:
                    word = word[:stem_len]
                break
        else:
            if word.endswith("ion"):
                stem_len = len(word) - 3
                if stem_len > 0 and word[stem_len - 1] in "st" and self._m(word, stem_len - 1) > 1:
                    word = word[:stem_len]

        k = len(word) - 1
        if word.endswith("e"):
            a = self._m(word, k - 1)
            if a > 1 or (a == 1 and not self._cvc(word, k - 1)):
                word = word[:-1]
        if word.endswith("l") and self._double_cons(word, len(word) - 1) and self._m(word, len(word) - 2) > 1:
            word = word[:-1]

        return word


_stemmer = _PorterStemmer()
_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)


class TextPreprocessor:
    """Applies a configurable sequence of cleaning steps to a text column."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def clean_text(self, text: str) -> str:
        cfg = self.config
        s = "" if text is None else str(text)

        if cfg.remove_urls:
            s = re.sub(URL_PATTERN, " ", s)
        if cfg.remove_emails:
            s = re.sub(EMAIL_PATTERN, " ", s)
        if cfg.remove_phone_numbers:
            s = re.sub(PHONE_PATTERN, " ", s)
        if cfg.lowercase:
            s = s.lower()
        if cfg.remove_punctuation:
            s = re.sub(PUNCTUATION_PATTERN, " ", s)
        if cfg.remove_numbers:
            s = re.sub(NUMBER_PATTERN, " ", s)

        if cfg.remove_stopwords or cfg.normalization != "none":
            tokens = s.split()
            if cfg.remove_stopwords:
                tokens = [t for t in tokens if t.lower() not in _STOPWORDS]
            if cfg.normalization == "stem":
                tokens = [_stemmer.stem(t) for t in tokens]
            elif cfg.normalization == "lemmatize":
                tokens = [_simple_lemmatize_word(t.lower()) for t in tokens]
            s = " ".join(tokens)

        if cfg.normalize_whitespace:
            s = re.sub(WHITESPACE_PATTERN, " ", s).strip()

        return s

    def transform(self, series: pd.Series) -> pd.Series:
        return series.astype(str).map(self.clean_text)

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z']+", text.lower())
