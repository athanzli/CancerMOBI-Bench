"""
Robust Rank Aggregation (RRA) for combining biomarker rankings from multiple
methods or folds into a consensus ranking.

Usage:
    from aggregate_rankings import aggregate_rankings, aggregate_rankings_from_gene_scores

    # Option A: from pre-sorted ranked gene lists (strings, most important first)
    consensus = aggregate_rankings([ranking1, ranking2, ranking3])

    # Option B: from gene-level score DataFrames (index = gene names, column = scores)
    #   NOTE: these should be gene-level scores, NOT raw method output (MOD@molecule).
    #   Use convert_ft_score_to_gene_level() to convert raw method output first.
    consensus = aggregate_rankings_from_gene_scores([gene_scores1, gene_scores2])

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


def aggregate_rankings_from_gene_scores(
    gene_scores: list,
    ascending: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper: convert gene-level score DataFrames to ranked lists, then run RRA.

    Important: the input should be **gene-level** score DataFrames (index = gene names,
    e.g., 'TP53', 'KRAS'), NOT raw method output (which uses 'MOD@molecule' format,
    e.g., 'mRNA@TP53', 'DNAm@cg00000029'). If you have raw method output, first convert
    it to gene-level using `convert_ft_score_to_gene_level()` from `benchmark_pipeline.py`.

    Args:
        gene_scores: List of single-column DataFrames with gene-level importance scores.
            Each DataFrame should have gene names as index and scores as values.
        ascending: If False (default), higher scores = more important (ranked first).
            Set to True if lower scores = more important.

    Returns:
        Consensus DataFrame with 'p-value' column, same as aggregate_rankings().
    """
    rankings = []
    for gs in gene_scores:
        if isinstance(gs, pd.DataFrame):
            s = gs.iloc[:, 0]
        else:
            s = gs
        ranked = s.sort_values(ascending=ascending)
        rankings.append(ranked.index.tolist())
    return aggregate_rankings(rankings)
