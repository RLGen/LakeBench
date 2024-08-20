import os
import random
import zipfile
from typing import Iterable, Set, Optional, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from d3l.utils.constants import STOPWORDS, GLOVEURL
from d3l.utils.functions import shingles
from urllib.request import urlopen


class GloveTransformer:
    def __init__(
        self,
        token_pattern: str = r"(?u)\b\w\w+\b",
        max_df: float = 0.5,
        stop_words: Iterable[str] = STOPWORDS,
        model_name: str = "glove.42B.300d",
        cache_dir: Optional[str] = None,
    ):
        """
        Instantiate a new embedding-based transformer
        Parameters
        ----------
        token_pattern : str
            The regex used to identify tokens.
            The default value is scikit-learn's TfidfVectorizer default.
        max_df : float
            Percentage of values the token can appear in before it is ignored.
        stop_words : Iterable[str]
            A collection of stopwords to ignore that defaults to NLTK's English stopwords.
        model_name : str
            The embedding model name to download from Stanford's website.
            It does not have to include to *.zip* extension.
            By default, the *Common Crawl 42B* model will be used.
        cache_dir : Optional[str]
            An exising directory path where the model will be stored.
            If not given, the current working directory will be used.
        """

        self._token_pattern = token_pattern
        self._max_df = max_df
        self._stop_words = stop_words
        self._model_name = model_name
        self._cache_dir = (
            cache_dir if cache_dir is not None and os.path.isdir(cache_dir) else None
        )

        self._embedding_model = self.get_embedding_model(
            model_name=model_name,
            overwrite=False
        )
        self._embedding_dimension = self.get_embedding_dimension()

    def __getstate__(self):
        d = self.__dict__
        self_dict = {k: d[k] for k in d if k != "_embedding_model"}
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state
        self._embedding_model = self.get_embedding_model(
            model_name=self._model_name, overwrite=False
        )

    @property
    def cache_dir(self) -> Optional[str]:
        return self._cache_dir

    def _download_glove(self,
                        model_name: str = "glove.42B.300d",
                        chunk_size: int = 2 ** 13):
        """
        Download pre-trained GloVe vectors from Stanford's website
        https://fasttext.cc/docs/en/crawl-vectors.html

        Parameters
        ----------
         model_name : str
            The embedding model name to download from Stanford's website.
            It does not have to include to *.zip* extension.
            By default, the *Common Crawl 42B* model will be used.
        chunk_size : int
            The Fasttext models are commonly large - several GBs.
            The disk writing will therefore be made in chunks.

        Returns
        -------

        """

        url = GLOVEURL + model_name
        print("Downloading %s" % url)
        response = urlopen(url)

        downloaded = 0
        write_file_name = (
            os.path.join(self._cache_dir, model_name)
            if self._cache_dir is not None
            else model_name
        )
        download_file_name = write_file_name + ".part"
        with open(download_file_name, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                downloaded += len(chunk)
                if not chunk:
                    break
                f.write(chunk)
                # print("{} downloaded ...".format(downloaded))

        os.rename(download_file_name, write_file_name)

    def _download_model(self,
                        model_name: str = "glove.42B.300d",
                        if_exists: str = "strict"):
        """
        Download the pre-trained model file.
        Parameters
        ----------
        model_name : str
            The embedding model name to download from Stanford's website.
            It does not have to include to *.zip* extension.
            By default, the *Common Crawl 42B* model will be used.
        if_exists : str
            Supported values:
                - *ignore*: The model will not be downloaded
                - *strict*: This is the default. The model will be downloaded only if it does not exist at the *cache_dir*.
                - *overwrite*: The model will be downloaded even if it already exists at the *cache_dir*.

        Returns
        -------

        """

        base_file_name = "%s.txt" % model_name
        file_name = (
            os.path.join(self._cache_dir, base_file_name)
            if self._cache_dir is not None
            else base_file_name
        )
        gz_file_name = "%s.zip" % model_name
        if os.path.isfile(file_name):
            if if_exists == "ignore":
                return file_name
            elif if_exists == "strict":
                print("File exists. Use --overwrite to download anyway.")
                return file_name
            elif if_exists == "overwrite":
                pass

        absolute_gz_file_name = (
            os.path.join(self._cache_dir, gz_file_name)
            if self._cache_dir is not None
            else gz_file_name
        )
        extract_dir = self._cache_dir if self._cache_dir is not None else "."
        if not os.path.isfile(absolute_gz_file_name):
            self._download_glove(gz_file_name)

        print("Extracting %s" % absolute_gz_file_name)
        with zipfile.ZipFile(absolute_gz_file_name, "r") as f:
            f.extractall(extract_dir)

        """Cleanup"""
        if os.path.isfile(absolute_gz_file_name):
            os.remove(absolute_gz_file_name)

        return file_name

    def get_embedding_model(
        self,
        model_name: str = "glove.42B.300d",
        overwrite: bool = False
    ) -> Dict:
        """
        Download, if not exists, and load the pretrained GloVe embedding model in the working directory.
        Parameters
        ----------
        model_name : str
            The embedding model name to download from Stanford's website.
            It does not have to include to *.zip* extension.
            By default, the *Common Crawl 42B* model will be used.
        overwrite : bool
            If True overwrites the model if exists.
        Returns
        -------

        """
        if_exists = "strict" if not overwrite else "overwrite"

        model_file = self._download_model(model_name=model_name, if_exists=if_exists)
        embedding_model = {}
        print("Loading embeddings. This may take a few minutes ...")
        with open(model_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embedding_model[word] = vector

        return embedding_model

    def get_embedding_dimension(self) -> int:
        """
        Retrieve the embedding dimensions of the underlying model.
        Returns
        -------
        int
            The dimensions of each embedding
        """
        return len(self._embedding_model.get(random.choice(list(self._embedding_model.keys()))))

    def get_vector(self, word: str) -> np.ndarray:
        """
        Retrieve the embedding of the given word.
        If the word is out of vocabulary a zero vector is returned.
        Parameters
        ----------
        word : str
            The word to retrieve the vector for.

        Returns
        -------
        np.ndarray
            A vector of float numbers.
        """
        vector = self._embedding_model.get(str(word).strip().lower(),
                                           np.random.randn(self._embedding_dimension))
        return vector

    def get_tokens(self, input_values: Iterable[str]) -> Set[str]:
        """
        Extract the most representative tokens of each value and return the token set.
        Here, the most representative tokens are the ones with the lowest TF/IDF scores -
        tokens that describe what the values are about.
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

        weight_map = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        tokenset = set()
        tokenizer = vectorizer.build_tokenizer()
        for value in input_values:
            value = value.lower().replace("\n", " ").strip()
            for shingle in shingles(value):
                tokens = [t for t in tokenizer(shingle)]

                if len(tokens) < 1:
                    continue

                token_weights = [weight_map.get(t, 0.0) for t in tokens]
                min_tok_id = np.argmin(token_weights)
                tokenset.add(tokens[min_tok_id])

        return tokenset

    def transform(self, input_values: Iterable[str]) -> np.ndarray:
        """
         Extract the embeddings of the most representative tokens of each value and return their **mean** embedding.
         Here, the most representative tokens are the ones with the lowest TF/IDF scores -
         tokens that describe what the values are about.
         Given that the underlying embedding model is a n-gram based one,
         the number of out-of-vocabulary tokens should be relatively small or zero.
         Parameters
         ----------
        input_values : Iterable[str]
             The collection of values to extract tokens from.

         Returns
         -------
         np.ndarray
             A Numpy vector representing the mean of all token embeddings.
        """

        embeddings = [self.get_vector(token) for token in self.get_tokens(input_values)]
        if len(embeddings) == 0:
            return np.empty(0)
        return np.mean(np.array(embeddings), axis=0)
