"""
Robust Rank Aggregation (RRA) for combining biomarker rankings from multiple
methods or folds into a consensus ranking.

Usage:
    from aggregate_rankings import aggregate_rankings

    # rankings: list of lists/arrays, each being a ranked list of feature names
    #   (higher-ranked features appear earlier)
    consensus = aggregate_rankings([ranking1, ranking2, ranking3])
    print(consensus)  # DataFrame with 'name' index and 'p-value' column
"""

import pandas as pd
import numpy as np


def aggregate_rankings(rankings: list) -> pd.DataFrame:
    """Aggregate multiple ranked lists into a consensus ranking using RRA.

    Uses the Robust Rank Aggregation method (Kolde et al., 2012) via the
    R package 'RobustRankAggreg'. Requires rpy2 and the R package installed.

    Install R package (if not already):
        R -e 'install.packages("RobustRankAggreg")'

    Args:
        rankings: A list of ranked lists. Each list contains feature names
            (strings) ordered from most important to least important.
            Lists can have different lengths and different elements.

    Returns:
        A DataFrame indexed by feature name, with a 'p-value' column.
        Lower p-values indicate features that are consistently highly ranked
        across the input lists. Sorted by p-value ascending.

    Example:
        >>> from aggregate_rankings import aggregate_rankings
        >>> r1 = ['mRNA@TP53', 'mRNA@KRAS', 'mRNA@EGFR', 'mRNA@BRAF']
        >>> r2 = ['mRNA@KRAS', 'mRNA@TP53', 'mRNA@PTEN', 'mRNA@EGFR']
        >>> r3 = ['mRNA@TP53', 'mRNA@PTEN', 'mRNA@KRAS', 'mRNA@BRAF']
        >>> result = aggregate_rankings([r1, r2, r3])
        >>> print(result.head())
    """
    from rpy2.robjects import ListVector
    from rpy2.robjects.vectors import StrVector
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    RRA = importr('RobustRankAggreg')

    # Convert to string lists
    str_rankings = []
    for ranking in rankings:
        str_rankings.append([str(elem) for elem in ranking])

    # Total number of unique elements
    unique_items = set(item for lst in str_rankings for item in lst)
    N = len(unique_items)

    r_lists = ListVector({
        f"rank{i+1}": StrVector(lst)
        for i, lst in enumerate(str_rankings)
    })

    r_res = RRA.aggregateRanks(glist=r_lists, N=N, method='RRA')
    consensus_df = pandas2ri.rpy2py(r_res)
    consensus_df.columns = ['name', 'p-value']
    consensus_df = consensus_df.sort_values('p-value', ascending=True).reset_index(drop=True)
    consensus_df.index = consensus_df['name']
    consensus_df = consensus_df.drop(columns=['name'])

    return consensus_df


def aggregate_rankings_from_ft_scores(
    ft_scores: list,
    ascending: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper: convert feature score DataFrames to rankings, then run RRA.

    Args:
        ft_scores: List of single-column DataFrames with feature importance scores.
            Each DataFrame should have feature names as index and scores as values.
        ascending: If False (default), higher scores = more important (ranked first).
            Set to True if lower scores = more important.

    Returns:
        Consensus DataFrame with 'p-value' column, same as aggregate_rankings().
    """
    rankings = []
    for ft in ft_scores:
        if isinstance(ft, pd.DataFrame):
            s = ft.iloc[:, 0]
        else:
            s = ft
        ranked = s.sort_values(ascending=ascending)
        rankings.append(ranked.index.tolist())
    return aggregate_rankings(rankings)
