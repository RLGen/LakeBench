from typing import Iterable, Union

import numpy as np


class DistributionTransformer:
    def __init__(self, num_bins: int = 300, use_density: bool = True):
        """
        Instantiate a new distribution transformer that extracts distribution information from input values.
        Parameters
        ----------
        num_bins : int
            Defines the dimension of the resulting distribution representation and the number of equal-width bins.
        use_density : bool
            If True the distribution representation defines a probability density function rather than just a count histogram.
        """

        self._num_bins = num_bins
        self._use_density = use_density

    @property
    def num_bins(self) -> int:
        return self._num_bins

    @property
    def use_density(self) -> bool:
        return self._use_density

    def transform(self, input_values: Iterable[Union[int, float]]) -> np.ndarray:
        """
        Generate a probability distribution representation for the given inputs.
        Parameters
        ----------
        input_values : Iterable[Union[int, float]]
            A collection of numbers seen as a representative sample of the probability distribution.

        Returns
        -------
        np.ndarray
            A vectorised representation of the distribution.

        """

        input_array = np.array(input_values)
        # print(f"Original input array: {input_array}")

        if len(input_array) < 1:
            return np.empty(self._num_bins)

        input_array = input_array[~np.isnan(input_array)]
        # print(f"Input array after removing NaN: {input_array}")
        hist, bin_edges = np.histogram(
            input_array, bins=self._num_bins, density=self._use_density
        )
        return hist
