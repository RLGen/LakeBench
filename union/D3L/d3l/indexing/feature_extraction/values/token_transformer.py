from typing import Iterable, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from d3l.utils.constants import STOPWORDS
from d3l.utils.functions import shingles


class TokenTransformer:
    def __init__(
        self,
        token_pattern: str = r"(?u)\b\w\w+\b",
        max_df: float = 0.5,
        stop_words: Iterable[str] = STOPWORDS,
    ):
        """
        Instantiate a new token-based transformer.
        Parameters
        ----------
        token_pattern : str
            The regex used to identify tokens.
            The default value is scikit-learn's TfidfVectorizer default.
        max_df : float
            Percentage of values the token can appear in before it is ignored.
        stop_words : Iterable[str]
            A collection of stopwords to ignore that defaults to NLTK's English stopwords.
        """

        self._token_pattern = token_pattern
        self._max_df = max_df
        self._stop_words = stop_words

    def transform(self, input_values: Iterable[str]) -> Set[str]:
        """
        Extract the most representative tokens of each value and return the token set.
        Here, the most representative tokens are the ones with the highest TF/IDF scores -
        tokens that best identify each value.
        Parameters
        ----------
        input_values : Iterable[str]
            The collection of values to extract tokens from.

        Returns
        -------
        Set[str]
            A set of representative tokens

        """

        if len(input_values) < 1:
            return set()

        try:
            vectorizer = TfidfVectorizer(
                decode_error="ignore",
                strip_accents="unicode",
                lowercase=True,
                analyzer="word",
                stop_words=self._stop_words,
                token_pattern=self._token_pattern,
                max_df=self._max_df,
                use_idf=True,
            )
            vectorizer.fit_transform(input_values)
        except ValueError:
            return set()

        weight_map = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
        tokenset = set()
        tokenizer = vectorizer.build_tokenizer()
        for value in input_values:
            value = value.lower().replace("\n", " ").strip()
            for shingle in shingles(value):
                tokens = [t for t in tokenizer(shingle)]

                if len(tokens) < 1:
                    continue

                token_weights = [weight_map.get(t, 0.0) for t in tokens]
                max_tok_id = np.argmax(token_weights)
                tokenset.add(tokens[max_tok_id])

        return tokenset
