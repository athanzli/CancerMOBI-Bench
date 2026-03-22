"""
Unified interface for running any of the 20 benchmarked biomarker identification
methods on arbitrary multi-omics data.

Usage:
    from run_method import run_method, run_method_rra
    ft_score = run_method('GAUDI', X_train=X_trn)
    ft_score = run_method('DIABLO', X_train=X_trn, y_train=y_trn, X_test=X_tst, y_test=y_tst)
    ft_score = run_method('DeePathNet', X_train=X_trn, y_train=y_trn,
                          X_val=X_val, y_val=y_val, X_test=X_tst, y_test=y_tst, device='cuda:0')

    # Run multiple methods and aggregate with RRA
    ft_score = run_method_rra(['GAUDI', 'DIABLO', 'DeepKEGG'],
                              X_train=X_trn, y_train=y_trn,
                              X_val=X_val, y_val=y_val,
                              X_test=X_tst, y_test=y_tst, device='cuda:0')
"""

import sys
import os
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_REPO_ROOT, 'code')

# Keep repo root on sys.path so sibling modules (e.g., aggregate_rankings,
# benchmark_pipeline) stay importable and the root-level utils.py is found.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import models from code/ directory. This requires:
# 1. code/ in sys.path (for code/utils.py and selected_models/)
# 2. cwd = code/ (models read relative paths like '../data/')
# After import, we restore state and save the code/utils module separately
# so it doesn't conflict with the root utils.py used by benchmark_pipeline.
_prev_cwd = os.getcwd()
sys.path.insert(0, _CODE_DIR)
os.chdir(_CODE_DIR)

from models import (
    run_gaudi, run_diablo, run_asmplsda, run_stabl, run_gdf,
    run_deepathnet, run_deepkegg, run_moglam, run_customics,
    run_tmonet, run_genius, run_pathformer, run_mofa, run_mcia,
    run_mogonet, run_more, run_moaglsa, run_pnet, run_gnnsubnet,
    run_dpm,
)

# Save code/utils as _code_utils and remove from sys.modules so that
# benchmark_pipeline can import the root-level utils.py as 'utils'.
_code_utils = sys.modules.get('utils')
if _code_utils and hasattr(_code_utils, '__file__') and 'code' in _code_utils.__file__:
    sys.modules.pop('utils', None)

if sys.path[0] == _CODE_DIR:
    sys.path.pop(0)
os.chdir(_prev_cwd)

# Method categories by input pattern
_UNSUPERVISED = ['GAUDI', 'MCIA']  # train data only, no labels
_TRAIN_TEST = ['DIABLO', 'asmPLSDA', 'Stabl', 'GDF']  # train + test
_TRAIN_VAL_TEST = [
    'DeePathNet', 'DeepKEGG', 'MOGLAM', 'CustOmics', 'TMONet',
    'GENIUS', 'Pathformer', 'MOGONET', 'MORE', 'MoAGLSA',
    'PNet', 'GNNSubNet',
]
_TRAIN_ONLY_LABELED = ['MOFA', 'DPM']  # train data + labels (no test needed for ft_score)

# Methods that need a GPU device argument
_NEEDS_DEVICE = [
    'DeePathNet', 'DeepKEGG', 'MOGLAM', 'CustOmics', 'TMONet',
    'GENIUS', 'Pathformer', 'MOGONET', 'MORE', 'MoAGLSA',
    'PNet', 'GNNSubNet',
]

# Modality constraints per method
_EXACTLY_3 = ['GAUDI', 'MCIA', 'asmPLSDA', 'DeepKEGG', 'DPM']
_THREE_OR_FOUR = ['GENIUS']
_TWO_OR_THREE = ['GDF']

ALL_METHODS = _UNSUPERVISED + _TRAIN_TEST + _TRAIN_VAL_TEST + _TRAIN_ONLY_LABELED


def run_method(
    method_name: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame = None,
    X_val: pd.DataFrame = None,
    y_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    y_test: pd.DataFrame = None,
    device: str = 'cuda:0',
) -> pd.DataFrame:
    """Run a benchmarked biomarker identification method and return feature importance scores.

    Args:
        method_name: Name of the method. One of:
            Unsupervised (no labels needed): GAUDI, MCIA
            Train+Test: DIABLO, asmPLSDA, Stabl, GDF
            Train+Val+Test (deep learning): DeePathNet, DeepKEGG, MOGLAM,
                CustOmics, TMONet, GENIUS, Pathformer, MOGONET, MORE, MoAGLSA,
                PNet, GNNSubNet
            Train-only: MOFA, DPM
        X_train: Training feature matrix. Columns must follow the 'MOD@molecule'
            naming convention, e.g., 'mRNA@TP53', 'DNAm@cg00000029'.
        y_train: Training labels. A DataFrame with a single 'label' column.
            Not needed for unsupervised methods (GAUDI, MCIA).
        X_val: Validation feature matrix. Required for deep learning methods.
            If not provided for DL methods, 20% of training data is used.
        y_val: Validation labels. Required for deep learning methods.
        X_test: Test feature matrix. Required for train+test and DL methods.
            If not provided for DL methods, validation set is reused.
        y_test: Test labels. Required for train+test and DL methods.
        device: PyTorch device string for DL methods. Default 'cuda:0'.

    Returns:
        ft_score: A single-column DataFrame with feature importance scores.
            Index follows the 'MOD@molecule' naming convention.
            Column name is 'score'. Higher values indicate higher importance.
    """
    if method_name not in ALL_METHODS:
        raise ValueError(
            f"Unknown method '{method_name}'. Available methods: {ALL_METHODS}"
        )

    # Validate inputs based on method category
    if method_name in _UNSUPERVISED:
        pass  # only X_train needed
    elif method_name in _TRAIN_TEST:
        if y_train is None:
            raise ValueError(f"{method_name} requires y_train.")
        if X_test is None or y_test is None:
            raise ValueError(f"{method_name} requires X_test and y_test.")
    elif method_name in _TRAIN_VAL_TEST:
        if y_train is None:
            raise ValueError(f"{method_name} requires y_train.")
        # Auto-split if val/test not provided
        if X_val is None or y_val is None:
            if X_test is not None and y_test is not None:
                # Use test as both val and test
                X_val, y_val = X_test.copy(), y_test.copy()
                print(f"[run_method] No validation set provided for {method_name}; "
                      f"reusing test set as validation.")
            else:
                # Split training data 80/20 for train/val, use val as test too
                n = len(X_train)
                idx = np.random.permutation(n)
                split = int(0.8 * n)
                train_idx = X_train.index[idx[:split]]
                val_idx = X_train.index[idx[split:]]
                X_val = X_train.loc[val_idx]
                y_val = y_train.loc[val_idx]
                X_train = X_train.loc[train_idx]
                y_train = y_train.loc[train_idx]
                X_test, y_test = X_val.copy(), y_val.copy()
                print(f"[run_method] No val/test sets provided for {method_name}; "
                      f"split training data 80/20.")
        if X_test is None or y_test is None:
            X_test, y_test = X_val.copy(), y_val.copy()
            print(f"[run_method] No test set provided for {method_name}; "
                  f"reusing validation set as test.")
    elif method_name in _TRAIN_ONLY_LABELED:
        if y_train is None and method_name != 'MOFA':
            raise ValueError(f"{method_name} requires y_train.")

    # Validate number of input modalities
    from utils import mod_mol_dict
    mmdic = mod_mol_dict(X_train.columns)
    n_mods = len(mmdic['mods_uni'])
    mods_str = ', '.join(mmdic['mods_uni'])

    if method_name in _EXACTLY_3:
        assert n_mods == 3, (
            f"{method_name} requires exactly 3 omics types, but got {n_mods}: [{mods_str}]. "
            f"Please provide a tri-omics combination (e.g., ['mRNA', 'DNAm', 'miRNA'])."
        )
    elif method_name in _THREE_OR_FOUR:
        assert n_mods in [3, 4], (
            f"{method_name} requires 3 or 4 omics types, but got {n_mods}: [{mods_str}]."
        )
    elif method_name in _TWO_OR_THREE:
        assert n_mods in [2, 3], (
            f"{method_name} requires 2 or 3 omics types, but got {n_mods}: [{mods_str}]."
        )
    else:
        assert n_mods >= 2, (
            f"{method_name} requires at least 2 omics types, but got {n_mods}: [{mods_str}]."
        )

    # Ensure cwd is code/ and code/ is in sys.path for model implementations
    # that use relative paths and lazy imports (e.g., from selected_models.X import ...)
    prev_cwd = os.getcwd()
    prev_utils = sys.modules.get('utils')
    os.chdir(_CODE_DIR)
    sys.path.insert(0, _CODE_DIR)
    if _code_utils is not None:
        sys.modules['utils'] = _code_utils

    # Dispatch
    ft_score = _dispatch(method_name, X_train, y_train, X_val, y_val,
                         X_test, y_test, device)

    # Restore previous state
    if sys.path[0] == _CODE_DIR:
        sys.path.pop(0)
    if prev_utils is not None:
        sys.modules['utils'] = prev_utils
    elif 'utils' in sys.modules and _code_utils is not None:
        sys.modules.pop('utils', None)
    os.chdir(prev_cwd)

    # Normalize output to single-column DataFrame with 'score' column
    ft_score = _normalize_output(ft_score, method_name)

    return ft_score


def _dispatch(method_name, X_train, y_train, X_val, y_val, X_test, y_test, device):
    """Call the underlying method function with the correct arguments."""

    if method_name == 'GAUDI':
        return run_gaudi(data=X_train)

    elif method_name == 'MCIA':
        return run_mcia(data=X_train)

    elif method_name == 'MOFA':
        ft, ft_score = run_mofa(data=X_train)
        return ft_score

    elif method_name == 'DPM':
        return run_dpm(data=X_train, label=y_train)

    elif method_name == 'DIABLO':
        ft_score, ft_score_rank, perf = run_diablo(
            data_trn=X_train, label_trn=y_train,
            data_tst=X_test, label_tst=y_test)
        # Convert ranks to scores (higher = more important) so downstream
        # aggregation (max pooling, descending sort in RRA) works correctly.
        max_rank = ft_score_rank['rank'].max()
        return pd.DataFrame({'score': max_rank + 1 - ft_score_rank['rank']},
                            index=ft_score_rank.index)

    elif method_name == 'asmPLSDA':
        ft_score, ft_score_rank, perf = run_asmplsda(
            data_trn=X_train, label_trn=y_train,
            data_tst=X_test, label_tst=y_test)
        max_rank = ft_score_rank['score'].max()
        return pd.DataFrame({'score': max_rank + 1 - ft_score_rank['score']},
                            index=ft_score_rank.index)

    elif method_name == 'Stabl':
        # NOTE: Stabl has anomalous arg order (data_trn, data_tst, label_trn, label_tst)
        ft_score, perf = run_stabl(
            data_trn=X_train, data_tst=X_test,
            label_trn=y_train, label_tst=y_test)
        return ft_score

    elif method_name == 'GDF':
        ft_score, perf = run_gdf(
            data_trn=X_train, label_trn=y_train,
            data_tst=X_test, label_tst=y_test)
        return ft_score

    elif method_name == 'DeePathNet':
        ft_score, perf = run_deepathnet(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score

    elif method_name == 'DeepKEGG':
        ft_score, perf = run_deepkegg(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score

    elif method_name == 'MOGLAM':
        ft_score, perf = run_moglam(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score

    elif method_name == 'CustOmics':
        ft_score, perf = run_customics(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score

    elif method_name == 'TMONet':
        ft_score, perf = run_tmonet(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score

    elif method_name == 'GENIUS':
        ft_score, perf = run_genius(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score

    elif method_name == 'Pathformer':
        ft_score, perf = run_pathformer(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score

    elif method_name == 'MOGONET':
        ft_score, perf = run_mogonet(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score

    elif method_name == 'MORE':
        ft_score, perf = run_more(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score

    elif method_name == 'MoAGLSA':
        ft_score, perf = run_moaglsa(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score

    elif method_name == 'PNet':
        ft_score, perf = run_pnet(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score

    elif method_name == 'GNNSubNet':
        ft_score, perf = run_gnnsubnet(
            data_trn=X_train, label_trn=y_train,
            data_val=X_val, label_val=y_val,
            data_tst=X_test, label_tst=y_test,
            device=device)
        return ft_score


def run_method_rra(
    method_names: list,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame = None,
    X_val: pd.DataFrame = None,
    y_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    y_test: pd.DataFrame = None,
    device: str = 'cuda:0',
) -> pd.DataFrame:
    """Run multiple methods sequentially and aggregate their rankings via RRA.

    Runs each method in `method_names` using `run_method()`, converts each
    result to gene-level scores (mapping CpGs and miRNAs to genes), then
    combines the gene-level rankings using Robust Rank Aggregation (RRA)
    to produce a consensus gene-level importance score.

    Args:
        method_names: List of method names to run (e.g., ['GAUDI', 'DIABLO', 'DeepKEGG']).
        X_train, y_train, X_val, y_val, X_test, y_test, device:
            Same as `run_method()`. All arguments are passed to each method;
            methods that don't need certain arguments (e.g., unsupervised methods
            ignore y_train) will handle them accordingly.

    Returns:
        ft_score: A single-column DataFrame with 'score' column, indexed by
            gene name. Scores are -log10(p-value) from RRA, so higher values
            indicate genes more consistently ranked highly across methods.
    """
    from aggregate_rankings import aggregate_rankings_from_gene_scores
    from benchmark_pipeline import convert_ft_score_to_gene_level

    # Method output mode for convert_ft_score_to_gene_level:
    #   mode 0: molecule-level (most methods - CpG/miRNA/gene names with MOD@ prefix)
    #   mode 1: gene-level with MOD@ prefix (GDF converts CpGs/miRNAs to genes internally)
    #   mode 2: gene-level without prefix (DPM strips MOD@ prefix internally)
    _METHOD_MODE = {'GDF': 1, 'DeePathNet': 1, 'DPM': 2}

    ft_scores = []
    for method_name in method_names:
        print(f"\n[run_method_rra] Running {method_name}...")
        ft = run_method(
            method_name, X_train=X_train.copy(), y_train=y_train.copy(),
            X_val=X_val.copy() if X_val is not None else None,
            y_val=y_val.copy() if y_val is not None else None,
            X_test=X_test.copy() if X_test is not None else None,
            y_test=y_test.copy() if y_test is not None else None,
            device=device,
        )
        print(f"[run_method_rra] {method_name} done. Output shape: {ft.shape}")
        # Convert to gene-level so all methods use the same feature space
        mode = _METHOD_MODE.get(method_name, 0)
        ft_gene = convert_ft_score_to_gene_level(ft, mode=mode)
        print(f"[run_method_rra] {method_name} gene-level shape: {ft_gene.shape}")
        ft_scores.append(ft_gene)

    print(f"\n[run_method_rra] Aggregating {len(ft_scores)} gene-level rankings via RRA...")
    consensus = aggregate_rankings_from_gene_scores(ft_scores)

    # Convert p-values to scores: -log10(p-value), higher = more important
    scores = -np.log10(consensus['p-value'].clip(lower=1e-300))
    ft_score = pd.DataFrame({'score': scores}, index=consensus.index)
    ft_score.index.name = None

    print(f"[run_method_rra] Done. Consensus shape: {ft_score.shape}")
    return ft_score


def _normalize_output(ft_score, method_name):
    """Normalize method output to a single-column DataFrame with 'score' column."""
    if ft_score is None:
        raise RuntimeError(f"{method_name} returned None for feature scores.")

    if isinstance(ft_score, pd.DataFrame):
        if ft_score.shape[1] == 1:
            ft_score.columns = ['score']
        elif 'score' in ft_score.columns:
            ft_score = ft_score[['score']]
        else:
            # For methods that return per-class scores, take the mean across columns
            ft_score = pd.DataFrame(
                ft_score.mean(axis=1), columns=['score']
            )
    elif isinstance(ft_score, pd.Series):
        ft_score = ft_score.to_frame(name='score')
    else:
        raise RuntimeError(
            f"{method_name} returned unexpected type: {type(ft_score)}"
        )

    return ft_score
