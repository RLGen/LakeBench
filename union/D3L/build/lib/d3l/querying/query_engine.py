import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Iterable, Tuple, Optional, Dict, Union, List

from d3l.indexing.similarity_indexes import (
    SimilarityIndex,
    NameIndex,
)


class QueryEngine:
    def __init__(self, *query_backends: SimilarityIndex):
        """
        Create a new querying engine to perform data discovery in datalakes.
        Parameters
        ----------
        query_backends : SimilarityIndex
            A variable number of similarity indexes.
        """

        self.query_backends = query_backends

    @staticmethod
    def group_results_by_table(
        target_id: str,
        results: Iterable[Tuple[str, Iterable[float]]],
        table_groups: Optional[Dict] = None,
    ) -> Dict:
        """
        Groups column-based results by table.
        For a given query column, at most one candidate column is considered for each candidate table.
        This candidate column is the one with the highest sum of similarity scores.

        Parameters
        ----------
        target_id : str
            Typically the target column name used to get the results.
        results : Iterable[Tuple[str, Iterable[float]]]
            One or more pairs of column names (including the table names) and backend similarity scores.
        table_groups: Optional[Dict]
            Iteratively created table groups.
            If None, a new dict is created and populated with the current results.

        Returns
        -------
        Dict
            A mapping of table names to similarity scores.
        """

        if table_groups is None:
            table_groups = defaultdict(list)
        candidate_scores = {}
        for result_item, result_scores in results:
            name_components = result_item.split(".")
            table_name, column_name = (
                ".".join(name_components[:-1]),
                name_components[-1:][0],
            )

            candidate_column, existing_scores = candidate_scores.get(
                table_name, (None, None)
            )
            if existing_scores is None or sum(existing_scores) < sum(result_scores):
                candidate_scores[table_name] = (column_name, result_scores)

        for table_name, (candidate_column, result_scores) in candidate_scores.items():
            table_groups[table_name].append(
                ((target_id, candidate_column), result_scores)
            )
        return table_groups

    @staticmethod
    def get_cdf_scores(
        score_distributions: List[np.ndarray], scores: np.ndarray
    ) -> np.ndarray:
        """
        Parameters
        ----------
        score_distributions : List[np.ndarray]
            The samples of all existing scores for each of the LSH backend.
            Each of these has to be sorted in order to extract the
            Empirical Cumulative Distribution Function (ECDF) value.
        scores : np.ndarray
            An array of current scores for which to extract the ECDF values.

        Returns
        -------
        np.ndarray
            A vector of scores of size (1xc).

        """

        def ecdf(samples, values):
            return [
                np.searchsorted(samples[:, j], value, side="right") / len(samples)
                for j, value in enumerate(values)
            ]

        ecdf_weights = []
        for i in range(len(scores)):
            ecdfs = ecdf(score_distributions[i], scores[i])
            ecdf_weights.append(ecdfs)
        ecdf_weights = np.array(ecdf_weights)
        return np.average(scores, axis=0, weights=ecdf_weights)

    def column_query(
        self,
        column: pd.Series,
        aggregator: Optional[callable] = None,
        k: Optional[int] = None,
    ) -> Iterable[Tuple[str, Iterable[float]]]:
        """
        Perform column-level top-k nearest neighbour search over the configured LSH backends.
        Parameters
        ----------
        column : pd.Series
            The column query as a Pandas Series.
            The series name will give the name queries.
            The series values will give the value queries.
        aggregator: Optional[callable] = None
            An aggregating function used to merge the results of all configured backends.
            If None then all scores are returned.
        k : Optional[int]
            Only the top-k neighbours will be retrieved from each backend.
            Then, these results are aggregated using the aggregator function and the results re-ranked to retrieve
            the top-k aggregated neighbours.
            If this is None all results are retrieved.

        Returns
        -------
        Iterable[Tuple[str, Iterable[float]]]
            A collection of (column id, aggregated score values) pairs.
            The scores are the values returned by the backends or one aggregated value if an aggregator is passed.
        """

        results = defaultdict(lambda: [0.0] * len(self.query_backends))
        query_name = str(column.name)
        query_value = column.values.tolist()

        for i, backend in enumerate(self.query_backends):
            if isinstance(backend, NameIndex):
                query_results = backend.query(query=query_name, k=k)
            else:
                query_results = backend.query(query=query_value, k=k)

            for rid, score in query_results:
                results[rid][i] = score

        if aggregator is None:
            # If not aggregation is used results are sorted by the mean of the scores.
            # Reverse sorting because the scores are similarities.
            results = sorted(
                results.items(),
                key=lambda items: sum(items[1]) / len(self.query_backends),
                reverse=True,
            )
        else:
            results = {rid: [aggregator(scores)] for rid, scores in results.items()}
            # Reverse sorting because the scores are similarities.
            results = sorted(
                results.items(), key=lambda items: items[1][0], reverse=True
            )

        if k is None:
            return results
        return results[:k]

    def table_query(
        self,
        table: pd.DataFrame,
        aggregator: Optional[callable] = None,
        k: Optional[int] = None,
        verbose: bool = False,
    ) -> Union[Iterable[Tuple], Tuple[Iterable[Tuple], Iterable[Tuple]]]:
        """
        Perform table-level top-k nearest neighbour search over the configured LSH backends.
        Note that this functions assumes that the table name is part of the canonical indexed item ids.
        In other words, it considers the first part of the item id separated by a dot to be the table name.
        Parameters
        ----------
        table : pd.DataFrame
            The table query as a Pandas DataFrame.
            Each column will be the subject of a column-based query.
        aggregator: callable
            An aggregating function used to merge the results of all configured backends at table-level.
        k : Optional[int]
            Only the top-k neighbours will be retrieved from each backend.
            Then, these results are aggregated using the aggregator function and the results re-ranked to retrieve
            the top-k aggregated neighbours.
            If this is None all results are retrieved.
        verbose: bool
            Whether or not to also return the detailed scores for each similar column to some query column.

        Returns
        -------
         Union[Iterable[Tuple], Tuple[Iterable[Tuple], Iterable[Tuple]]]
            Pairs of the form (candidate table name, aggregated similarity score).
            If verbosity is required, also return pairs with column-level similarity details.
        """

        extended_table_results = None
        score_distributions = {}
        for column in table.columns:
            """Column scores are not aggregated when performing table queries."""
            column_results = self.column_query(
                column=table[column], aggregator=None, k=None
            )

            score_distributions[column] = np.sort(
                np.array([scores for _, scores in column_results]), axis=0
            )
            extended_table_results = self.group_results_by_table(
                target_id=column,
                results=column_results,
                table_groups=extended_table_results,
            )

        table_results = {}
        for candidate in extended_table_results.keys():
            candidate_scores = np.array(
                [details[1] for details in extended_table_results[candidate]]
            )
            distributions = [
                score_distributions[details[0][0]]
                for details in extended_table_results[candidate]
            ]
            weighted_scores = self.get_cdf_scores(distributions, candidate_scores)
            if aggregator is None:
                table_results[candidate] = weighted_scores.tolist()
            else:
                table_results[candidate] = aggregator(weighted_scores.tolist())

        # Reverse sorting because the scores are similarities.
        table_results = sorted(table_results.items(), key=lambda pair: pair[1], reverse=True)

        if k is not None:
            table_results = table_results[:k]

        if verbose:
            extended_table_results = [(cand, extended_table_results[cand])
                                      for cand, _ in table_results]
            return table_results, extended_table_results
        return table_results
