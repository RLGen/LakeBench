import re
from typing import List, Optional


class QGramTransformer:
    def __init__(self, qgram_size: Optional[int] = None):
        """
        This object performs qgram extraction.
        Parameters
        ----------
        qgram_size : Optional[int]
            The default qgram size.
        """
        self._qgram_size = qgram_size

    @property
    def qgram_size(self) -> Optional[int]:
        return self._qgram_size

    def transform(
        self, input_string: str, qgram_size: Optional[int] = None
    ) -> List[str]:
        """
        Generate a collection of qgrams of configured size from the given string.
        Parameters
        ----------
        input_string : str
            The input string to transform.
        qgram_size : Optional[int]
            The size of each qgram.
            If None, the upper-level size (passed when the object was created) will be used.

        Returns
        -------
        List[str]
            A collection of qgrams of the given string.

        """

        if qgram_size is None and self._qgram_size is None:
            raise ValueError(
                "Expected a qgram_size in this call or at the object level."
            )

        elif qgram_size is None:
            qgram_size = self._qgram_size

        qgrams = []
        for word in re.split(r"\W+", input_string.lower().strip()):
            word = word.strip()
            if len(word) < 1:
                continue
            if len(word) <= qgram_size:
                qgrams.append(word)
                continue
            for i in range((len(word) - qgram_size) + 1):
                qgrams.append(word[i : i + qgram_size])
        return qgrams
