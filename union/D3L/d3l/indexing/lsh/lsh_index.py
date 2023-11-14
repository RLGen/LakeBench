"""
Copyright (C) 2021 Alex Bogatu.
This file is part of the D3L Data Discovery Framework.
Notes
-----
This module exposes the LSH indexing functionality.
Based on https://github.com/ekzhu/datasketch
and the LSH Forest paper <http://ilpubs.stanford.edu:8090/678/1/2005-14.pdf>.
"""

import struct
import pdb
from collections import Counter, defaultdict
from typing import Any, ByteString, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import quad as integrate

from d3l.indexing.hashing.hash_generators import (
    BaseHashGenerator,
    MinHashHashGenerator,
    RandomProjectionsHashGenerator,
)

MIN_LSH_PARAMS = (10, 3)


class LSHIndex:
    def __init__(
        self,
        hash_size: int,
        similarity_threshold: float,
        dimension: Optional[int] = None,
        fp_fn_weights: Tuple[float, float] = (0.5, 0.5),
        seed: int = 12345,
    ):
        """
        The base LSH index class.

        Parameters
        ----------
        hash_size : int
            The expected size of the input hashcodes.
        similarity_threshold : float
            Must be in [0, 1].
            Represents the minimum similarity score between two sets to be considered similar.
            The similarity type is given by the type of hash used to generate the index inputs.
            E.g.,   *MinHash* hash function corresponds to Jaccard similarity,
                    *RandomProjections* hash functions corresponds to Cosine similarity.
        dimension : Optional[int]
            The number of dimensions expected for each input to be indexed.
            If this is not None the LSH index will be RandomProjections-based.
            Otherwise, the LSH index will be MinHash-based.
        fp_fn_weights : Tuple[float, float]
            A pair of values between 0 and 1 denoting a preference for high precision or high recall.
            If the fp weight is higher then indexing precision is preferred. Otherwise, recall is preferred.
            Their sum has to be 1.
        seed : int
            The random seed for the underlying hash generator.
        """

        self._hash_size = hash_size
        self._similarity_threshold = similarity_threshold
        self._dimension = dimension
        self._fp_fn_weights = fp_fn_weights
        self._seed = seed

        if not all(
            [
                self._similarity_threshold >= 0,
                self._similarity_threshold <= 1,
                self._fp_fn_weights[0] >= 0,
                self._fp_fn_weights[0] <= 1,
                self._fp_fn_weights[1] >= 0,
                self._fp_fn_weights[1] <= 1,
                sum(self._fp_fn_weights) == 1.0,
            ]
        ):
            raise ValueError(
                "The similarity threshold and the weights have to be float values between 0 and 1. "
                "The weights also have to sum to 1."
            )

        if self._dimension is None:
            self._hash_generator = MinHashHashGenerator(
                hash_size=self._hash_size, seed=self._seed
            )
        else:
            self._hash_generator = RandomProjectionsHashGenerator(
                hash_size=self._hash_size, seed=self._seed, dimension=self._dimension
            )

        """
        LSH-specific parameters:
            b: the number of hashtables used internally
            r: the sixe of the key of each entry of each hashtable
        """
        self._b, self._r = self._lsh_error_minimization()
        self._hashtables = [defaultdict(set) for _ in range(self._b)]
        self._hashranges = [(i * self._r, (i + 1) * self._r) for i in range(self._b)]
        self._keys = defaultdict(list)

    @property
    def hash_generator(self) -> BaseHashGenerator:
        return self._hash_generator

    @property
    def hash_size(self) -> int:
        return self._hash_size

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    @property
    def fp_fn_weights(self):
        return self._fp_fn_weights

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def keys(self) -> Dict:
        return self._keys

    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    @property
    def lsh_parameters(self) -> Tuple[float, float]:
        return self._b, self._r

    @property
    def hashtables(self) -> List[defaultdict]:
        return self._hashtables

    @staticmethod
    def lsh_false_positive_probability(threshold: float, b: int, r: int) -> float:
        """
        Computes the probability of false positive occurrence in LSH.
        Parameters
        ----------
        threshold : float
            The minimum similarity threshold.
        b : int
            The number of bands used with LSH.
            b * r = the size of the underlying hash.
        r : int
            The number of rows in each band.
            b * r = the size of the underlying hash.
        Returns
        -------
        float
            The probability of false positive occurrence
        """

        def _probability(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, err = integrate(_probability, 0.0, threshold)
        return a

    @staticmethod
    def lsh_false_negative_probability(threshold: float, b: int, r: int) -> float:
        """
        Computes the probability of false negative occurrence in LSH.
        Parameters
        ----------
        threshold : float
            The minimum similarity threshold.
        b : int
            The number of bands used with LSH.
            b * r = the size of the underlying hash.
        r : int
            The number of rows in each band.
            b * r = the size of the underlying hash.
        Returns
        -------
        float
            The probability of false negative occurrence
        """

        def _probability(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, err = integrate(_probability, threshold, 1.0)
        return a

    def _get_lsh_probabilities(self, b: int, r: int) -> Tuple[float, float]:
        """
        Compute the false positive and false negative probabilities given the LSH parameters.

        Parameters
        ----------
        b : int
            The number of hashtables the inner index uses.
        r : int
            The keys size used with each hashtable.

        Returns
        -------
        Tuple[float, float]
            The (false positive, false negative) probability tuple.

        """

        false_positive = LSHIndex.lsh_false_positive_probability(
            self._similarity_threshold, b, r
        )
        false_negative = LSHIndex.lsh_false_negative_probability(
            self._similarity_threshold, b, r
        )
        return false_positive, false_negative

    def _lsh_error_minimization(self) -> Tuple[int, int]:
        """
        Determine the values for LSH parameters b and r that minimize
        the probability of false negatives (guarantees high recall).
        Note that, in theory, it is possible to have a likelihood of false negatives of 1,
        but this will virtually make all items neighbours.

        Returns
        -------
        Tuple[int, int]
            A pair of (b, r) tuples that minimizes false negative probability.
        """

        min_b, min_r = MIN_LSH_PARAMS
        min_error = float("inf")
        opt = (0, 0)

        for b in range(min_b, int(self._hash_size / min_r) + 1):
            max_r = int(self._hash_size / b)
            for r in range(min_r, max_r + 1):
                fp, fn = self._get_lsh_probabilities(b, r)
                error = (fp * self._fp_fn_weights[0]) + (fn * self._fp_fn_weights[1])
                if (
                    error <= min_error
                    and (1 / b) ** (1 / r) >= self._similarity_threshold
                ):
                    min_error = error
                    opt = (b, r)
        return opt

    def _get_lsh_keys(self, input_hash: Iterable[int]) -> List[ByteString]:
        """
        Transform a given hashcode into a collection of index keys.

        Parameters
        ----------
        input_hash : List[int]
            The hashcode to transform

        Returns
        -------
        ByteString
            A collection of LSH index keys represented as a bytestring.
        """

        hash_chunks = [
            bytes(np.array(input_hash[start:end]).byteswap().data)
            for start, end in self._hashranges
        ]
        return hash_chunks

    def _get_hash(self, input_id: str) -> Optional[np.ndarray]:
        """
        Reconstruct the hash back from the index keys.
        This should be used only if the index keys originate from hash hashcode.

        Parameters
        ----------
        input_id : str
            The item identifier that is already indexed.

        Returns
        -------
        Optional[np.ndarray]
            The item hashcode as a Numpy array or None if the item has not been indexed.

        """

        hashcode = self._keys.get(input_id, None)
        if hashcode is None:
            return None

        hashcode = b"".join(hashcode)
        original_hash = []
        for counter in range(0, len(hashcode), 8):
            chunk = struct.unpack("=Q", hashcode[counter: counter + 8])[0]
            original_hash.append(np.array(chunk).byteswap())

        return np.array(original_hash, dtype=np.uint64)

    def get_similarity_score(
        self,
        left_element: Union[Iterable[Any], str],
        right_element: Union[Iterable[Any], str],
    ) -> float:
        """
        Estimate the similarity between the two sets.

        Parameters
        ----------
        left_element : Union[Iterable[Any], str]
            The id of an already indexed element or a hashcode.
        right_element : Union[Iterable[Any], str]
            The id of an already indexed element or a hashcode.

        Returns
        -------
        np.float16
            The estimated similarity score.

        """

        if isinstance(left_element, str):
            left_hashcode = self._get_hash(left_element)
        else:
            left_hashcode = left_element

        if isinstance(right_element, str):
            right_hashcode = self._get_hash(right_element)
        else:
            right_hashcode = right_element

        if left_hashcode is None or right_hashcode is None:
            return 0.0

        max_size = min([left_hashcode.size, right_hashcode.size])
        return np.float16(
            np.count_nonzero(left_hashcode[:max_size] == right_hashcode[:max_size])
        ) / np.float16(max_size)

    def add(self, input_id: str, input_set: Iterable) -> bool:
        """
        Add a new item to the index.

        Parameters
        ----------
        input_id : str
            The id that will identify the input item as a string.
            Only the ids are stored in the index buckets.
        input_set : Iterable
            Since this is a set-based index, the *input* has to be an iterable.
            It will be chunked into multiple keys that will be added to the index.

        Returns
        -------
        bool
            True if the item has been successfully added, False otherwise.

        """

        if input_id in self._keys:
            raise ValueError("Input identifier already used: {}".format(input_id))

        input_hash = self._hash_generator.hash(input_set, hashvalues=None)

        if len(input_hash) != self._hash_size:
            raise ValueError(
                "The resulting input hash has inconsistent length. Expected {} but got {}".format(
                    self._hash_size, len(input_hash)
                )
            )

        hash_chunks = self._get_lsh_keys(input_hash)
        self._keys[input_id] = hash_chunks
        for hash_entry, hash_table in zip(hash_chunks, self._hashtables):
            hash_table[hash_entry].add(input_id)
        return True

    def union(self, lsh, id):
        '''
        把另一lshensemble的值合并过来

        Args:
            lshE 待加的lsh索引
        '''
        for i in range(len(lsh._hashtables)):
            for j in lsh._hashtables[i].keys():
                #if id>0:
                    #pdb.set_trace()
                self._hashtables[i][j].update(lsh._hashtables[i][j])
 

    def query(
        self,
        query_id: Optional[str] = None,
        query: Optional[Iterable] = None,
        k: Optional[int] = None,
        with_scores: bool = False,
    ) -> Union[List[Any], List[Tuple[Any, float]]]:
        """
        Search for the nearest neighbours of the given query.

        Parameters
        ----------
        query_id : Optional[str]
            The the id of the query_engine.
            If defined then *query* is ignored.
            If None then it is assumed that the item has not been indexed and *query_hash* must be defined.
        query: Optional[Iterable]
            Since this is a set-based index, the *query* has to be an iterable.
            If None then it is assumed that the item has been indexed and *query_id* must be defined.
            If *query_id* is defined then this is ignored.
        k : int
            The number of neighbours to return.
        with_scores : bool
            Whether or not to return the estimated similarity scores associated with each result.
        Returns
        -------
        Union[List[Any], List[Tuple[Any, float]]]:
            A list of the nearest neighbours ids with or without associated similarity scores,
             depending on the values of *with_scores*.
        """

        query_hash = None
        if query_id is None:
            query_hash = self._hash_generator.hash(query, hashvalues=None)
            hash_chunks = self._get_lsh_keys(query_hash)
        else:
            hash_chunks = self._keys.get(query_id, None)
            if hash_chunks is None:
                raise ValueError(
                    "query_id must be an existing identifier in the index. Item with id {} not found!".format(
                        query_id
                    )
                )

        neighbours = [
            n
            for hash_entry, hash_table in zip(hash_chunks, self._hashtables)
            for n in hash_table.get(hash_entry, [])
        ]

        neighbour_counter = Counter(neighbours)
        neighbours = [w for w, _ in neighbour_counter.most_common(k) if w != query_id]
        if with_scores:
            similarity_scores = [
                self.get_similarity_score(query_hash, n)
                if query_id is None
                else self.get_similarity_score(query_id, n)
                for n in neighbours
            ]
            return list(zip(neighbours, similarity_scores))
        return neighbours
