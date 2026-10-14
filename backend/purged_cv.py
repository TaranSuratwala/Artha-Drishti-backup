"""
purged_cv.py — Combinatorial Purged Cross-Validation (CPCV) and Anchored Walk-Forward Validation.

TIMING CONVENTION:
Features use data <= close of day t -> entry at OPEN t+1 -> exit at CLOSE t+k or earlier on barrier touch.
Barriers use high/low of t+1…t+k; same-day double touch -> stop first; open through a barrier -> fill at open.
Adjusted prices; point-in-time universe incl. delisted; reindex to master trading calendar (k = market days).
"""

import numpy as np
import pandas as pd
from itertools import combinations
import math
from typing import Dict, List, Optional, Tuple, Generator, Union
import logging

logger = logging.getLogger(__name__)


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV) for GBDT Models:
    Implements de Prado (2018) Advances in Financial Machine Learning, Chapter 12.

    Mathematical Formulation:
      1. Partition trading calendar into N groups: G_0, G_1, ..., G_{N-1}.
      2. Choose k groups as the test set: S = C(N, k) combinations.
      3. For each combination s:
         - Test Set: Samples whose t0 falls in the chosen k groups.
         - Purging: Remove any training sample i whose holding span [t0_i, t1_i] overlaps
           the test span [T_start, T_test_exit]:
               Overlap: t0_train <= T_test_exit AND t1_train >= T_test_start.
         - Embargo: Remove any training sample i starting within embargo_tdays trading days
           after the test span exit:
               Embargo: T_test_exit < t0_train <= T_embargo_end.
      4. Number of Historical Backtest Paths:
         phi = C(N-1, k-1)
         Each path covers the entire historical timeline continuously without gaps or overlap.
    """

    def __init__(
        self,
        n_groups: int = 6,
        k_test_groups: int = 2,
        embargo_tdays: int = 21,
        calendar: Optional[pd.DatetimeIndex] = None
    ):
        if n_groups <= k_test_groups:
            raise ValueError(f"n_groups ({n_groups}) must be > k_test_groups ({k_test_groups})")
        if embargo_tdays < 0:
            raise ValueError("embargo_tdays cannot be negative")

        self.n_groups = n_groups
        self.k_test_groups = k_test_groups
        self.embargo_tdays = embargo_tdays
        self.calendar = calendar
        self.num_splits = math.comb(n_groups, k_test_groups)
        self.num_paths = math.comb(n_groups - 1, k_test_groups - 1)

    def get_num_splits(self) -> int:
        """Returns total number of train/test combinations C(N, k)."""
        return self.num_splits

    def get_num_paths(self) -> int:
        """Returns total number of reconstructed chronological backtest paths C(N-1, k-1)."""
        return self.num_paths

    def split(
        self,
        df: pd.DataFrame,
        t0_col: str = 't0',
        t1_col: str = 't1'
    ) -> Generator[Tuple[int, np.ndarray, np.ndarray, List[int]], None, None]:
        """
        Generates purged and embargoed train/test index splits.
        Yields: (split_id, train_indices, test_indices, test_group_ids)
        """
        t0_vals = pd.to_datetime(df[t0_col])
        t1_vals = pd.to_datetime(df[t1_col])

        # Master trading calendar
        if self.calendar is not None:
            cal_idx = pd.DatetimeIndex(self.calendar)
            unique_dates = cal_idx[(cal_idx >= t0_vals.min()) & (cal_idx <= t0_vals.max())].values
        else:
            unique_dates = np.sort(t0_vals.unique())

        if len(unique_dates) < self.n_groups:
            raise ValueError(f"Insufficient unique dates ({len(unique_dates)}) for {self.n_groups} groups")

        # Partition dates into n_groups
        groups = np.array_split(unique_dates, self.n_groups)

        # Map each date to group index
        date_to_group = {}
        for g_idx, g_dates in enumerate(groups):
            for d in g_dates:
                date_to_group[d] = g_idx

        # Assign sample groups based on t0
        sample_groups = t0_vals.map(date_to_group).values

        # Calendar lookup for embargo calculation
        date_series = pd.Series(np.arange(len(unique_dates)), index=unique_dates)

        test_combinations = list(combinations(range(self.n_groups), self.k_test_groups))

        for split_id, test_groups in enumerate(test_combinations):
            # Test mask: samples whose t0 belongs to any of the test groups
            test_mask = np.isin(sample_groups, test_groups)
            test_idx = np.where(test_mask)[0]

            if len(test_idx) == 0:
                continue

            # Identify test blocks and compute purge + embargo boundaries
            purge_mask = np.zeros(len(df), dtype=bool)
            embargo_mask = np.zeros(len(df), dtype=bool)

            for g_id in test_groups:
                g_dates = groups[g_id]
                g_start = g_dates[0]
                g_end = g_dates[-1]

                # Samples in this specific test block
                block_test_mask = (sample_groups == g_id)
                if not block_test_mask.any():
                    continue

                # Max exit date for samples in this test group
                block_max_exit = t1_vals.iloc[block_test_mask].max()
                if pd.isna(block_max_exit) or block_max_exit < g_end:
                    block_max_exit = g_end

                # Purge condition: training sample span [t0, t1] overlaps test block [g_start, block_max_exit]
                # Overlap: t0 <= block_max_exit AND t1 >= g_start
                block_purge = (t0_vals <= block_max_exit) & (t1_vals >= g_start)
                purge_mask |= block_purge.values

                # Embargo condition: sample starts within embargo_tdays trading days after block_max_exit
                if block_max_exit in date_series.index:
                    exit_loc = date_series.loc[block_max_exit]
                    if isinstance(exit_loc, pd.Series):
                        exit_loc = exit_loc.iloc[-1]
                    embargo_loc = min(exit_loc + self.embargo_tdays, len(unique_dates) - 1)
                    embargo_end_date = unique_dates[embargo_loc]
                else:
                    # Fallback approximation: 1.5 * calendar days
                    embargo_end_date = block_max_exit + np.timedelta64(int(self.embargo_tdays * 1.5), 'D')

                block_embargo = (t0_vals > block_max_exit) & (t0_vals <= embargo_end_date)
                embargo_mask |= block_embargo.values

            # Clean training set: neither test, nor purged, nor embargoed
            train_mask = (~test_mask) & (~purge_mask) & (~embargo_mask)
            train_idx = np.where(train_mask)[0]

            yield split_id, train_idx, test_idx, list(test_groups)

    def build_backtest_paths(
        self,
        split_predictions: Dict[int, pd.DataFrame],
        df: pd.DataFrame,
        t0_col: str = 't0'
    ) -> List[pd.DataFrame]:
        """
        Assembles exactly C(N-1, k-1) chronological out-of-sample backtest paths.
        split_predictions: Dict mapping split_id -> DataFrame of test predictions (with original df indices).
        Returns: List of DataFrames, each containing continuous historical predictions across all N groups.
        """
        t0_vals = pd.to_datetime(df[t0_col])
        if self.calendar is not None:
            cal_idx = pd.DatetimeIndex(self.calendar)
            unique_dates = cal_idx[(cal_idx >= t0_vals.min()) & (cal_idx <= t0_vals.max())].values
        else:
            unique_dates = np.sort(t0_vals.unique())

        groups = np.array_split(unique_dates, self.n_groups)
        date_to_group = {}
        for g_idx, g_dates in enumerate(groups):
            for d in g_dates:
                date_to_group[d] = g_idx

        sample_groups = t0_vals.map(date_to_group)

        test_combinations = list(combinations(range(self.n_groups), self.k_test_groups))

        # Map each group g to the splits in which g was tested
        group_splits: Dict[int, List[int]] = {g: [] for g in range(self.n_groups)}
        for s_idx, test_groups in enumerate(test_combinations):
            for g in test_groups:
                group_splits[g].append(s_idx)

        # For every group, exactly num_paths splits tested it
        for g in range(self.n_groups):
            if len(group_splits[g]) != self.num_paths:
                raise RuntimeError(f"Group {g} has {len(group_splits[g])} splits; expected {self.num_paths}")

        paths: List[pd.DataFrame] = []
        for p in range(self.num_paths):
            path_pieces = []
            for g in range(self.n_groups):
                chosen_split = group_splits[g][p]
                if chosen_split not in split_predictions:
                    raise KeyError(f"Missing predictions for split {chosen_split}")

                split_df = split_predictions[chosen_split]
                # Filter to samples in group g
                group_mask = (sample_groups.loc[split_df.index] == g)
                group_preds = split_df.loc[group_mask]
                path_pieces.append(group_preds)

            path_df = pd.concat(path_pieces).sort_index()
            paths.append(path_df)

        return paths

    @staticmethod
    def assert_no_leakage(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        t0_col: str = 't0',
        t1_col: str = 't1',
        embargo_tdays: int = 21,
        calendar: Optional[pd.DatetimeIndex] = None
    ):
        """
        Strict leakage assertion verifying:
          1. Zero temporal overlap: No training sample [t0, t1] overlaps any test sample [t0, t1].
          2. Embargo compliance: No training sample starts within embargo_tdays after any test block exit.
        """
        if len(train_df) == 0 or len(test_df) == 0:
            return

        tr_t0 = pd.to_datetime(train_df[t0_col]).values
        tr_t1 = pd.to_datetime(train_df[t1_col]).values
        te_t0 = pd.to_datetime(test_df[t0_col]).values
        te_t1 = pd.to_datetime(test_df[t1_col]).values

        # 1. Overlap Check: tr_t0 <= te_t1 AND tr_t1 >= te_t0
        min_te_t0 = te_t0.min()
        max_te_t1 = te_t1.max()

        # Potential overlapping candidates
        cand_mask = (tr_t0 <= max_te_t1) & (tr_t1 >= min_te_t0)
        if cand_mask.any():
            cand_t0 = tr_t0[cand_mask]
            cand_t1 = tr_t1[cand_mask]

            overlap = (cand_t0[:, None] <= te_t1[None, :]) & (cand_t1[:, None] >= te_t0[None, :])
            if overlap.any():
                viol_tr_idx, viol_te_idx = np.where(overlap)
                v_tr = viol_tr_idx[0]
                v_te = viol_te_idx[0]
                raise AssertionError(
                    f"CPCV Overlap Leakage Detected!\n"
                    f"Train sample span: [{cand_t0[v_tr]}, {cand_t1[v_tr]}]\n"
                    f"Test sample span:  [{te_t0[v_te]}, {te_t1[v_te]}]"
                )

        # 2. Embargo Check: Train sample starts within embargo after test exit
        # For all test blocks:
        if calendar is not None:
            cal_idx = pd.DatetimeIndex(calendar)
            exit_loc = cal_idx.get_indexer([max_te_t1])[0]
            if exit_loc >= 0:
                embargo_loc = min(exit_loc + embargo_tdays, len(cal_idx) - 1)
                embargo_end = cal_idx[embargo_loc].to_datetime64()
            else:
                embargo_end = max_te_t1 + np.timedelta64(int(embargo_tdays * 1.5), 'D')
        else:
            embargo_end = max_te_t1 + np.timedelta64(int(embargo_tdays * 1.5), 'D')

        embargo_viol = (tr_t0 > max_te_t1) & (tr_t0 <= embargo_end)
        if embargo_viol.any():
            viol_t0 = tr_t0[embargo_viol][0]
            raise AssertionError(
                f"CPCV Embargo Leakage Detected!\n"
                f"Test block exit: {max_te_t1}, Embargo end: {embargo_end}\n"
                f"Violating train sample starts at: {viol_t0}"
            )


class AnchoredWalkForwardCV:
    """
    Anchored Expanding-Window Walk-Forward Cross-Validation for Neural Networks:
    Ensures sequential, strictly chronological out-of-sample evaluation without future-to-past contamination.

    Structure:
      Fold m:
        Train:   [0, T_train_m]  (anchored at inception, expanding window)
        Embargo: [T_train_m + 1, T_train_m + embargo_tdays] (purged)
        Test:    [T_train_m + embargo_tdays + 1, T_test_m]
    """

    def __init__(
        self,
        min_train_tdays: int = 252,    # ~1 year minimum training window
        test_tdays: int = 63,          # ~1 quarter evaluation window
        embargo_tdays: int = 21,       # mandatory embargo gap >= 21 trading days
        step_tdays: Optional[int] = None,
        calendar: Optional[pd.DatetimeIndex] = None
    ):
        if min_train_tdays <= 0 or test_tdays <= 0:
            raise ValueError("min_train_tdays and test_tdays must be positive")
        if embargo_tdays < 0:
            raise ValueError("embargo_tdays cannot be negative")

        self.min_train_tdays = min_train_tdays
        self.test_tdays = test_tdays
        self.embargo_tdays = embargo_tdays
        self.step_tdays = step_tdays or test_tdays
        self.calendar = calendar

    def split(
        self,
        df: pd.DataFrame,
        t0_col: str = 't0',
        t1_col: str = 't1'
    ) -> Generator[Tuple[int, np.ndarray, np.ndarray, Dict[str, pd.Timestamp]], None, None]:
        """
        Generates expanding-window train/test splits.
        Yields: (fold_id, train_indices, test_indices, metadata_dict)
        """
        t0_vals = pd.to_datetime(df[t0_col])
        t1_vals = pd.to_datetime(df[t1_col])

        if self.calendar is not None:
            cal_idx = pd.DatetimeIndex(self.calendar)
            unique_dates = cal_idx[(cal_idx >= t0_vals.min()) & (cal_idx <= t0_vals.max())].values
        else:
            unique_dates = np.sort(t0_vals.unique())

        total_dates = len(unique_dates)
        required_span = self.min_train_tdays + self.embargo_tdays + self.test_tdays
        if total_dates < required_span:
            raise ValueError(f"Insufficient dates ({total_dates}) for anchored walk-forward span ({required_span})")

        fold_id = 0
        train_end_idx = self.min_train_tdays - 1

        while True:
            test_start_idx = train_end_idx + self.embargo_tdays + 1
            test_end_idx = test_start_idx + self.test_tdays - 1

            if test_end_idx >= total_dates:
                break

            train_start_date = unique_dates[0]
            train_end_date = unique_dates[train_end_idx]
            test_start_date = unique_dates[test_start_idx]
            test_end_date = unique_dates[test_end_idx]

            # Train mask: t0 in [train_start_date, train_end_date] and t1 <= train_end_date (purged exit)
            train_mask = (t0_vals >= train_start_date) & (t0_vals <= train_end_date) & (t1_vals <= train_end_date)
            # Test mask: t0 in [test_start_date, test_end_date]
            test_mask = (t0_vals >= test_start_date) & (t0_vals <= test_end_date)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            meta = {
                'fold': fold_id,
                'train_start': pd.Timestamp(train_start_date),
                'train_end': pd.Timestamp(train_end_date),
                'test_start': pd.Timestamp(test_start_date),
                'test_end': pd.Timestamp(test_end_date),
            }

            yield fold_id, train_idx, test_idx, meta

            fold_id += 1
            train_end_idx += self.step_tdays

    @staticmethod
    def assert_no_leakage(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        t0_col: str = 't0',
        t1_col: str = 't1',
        embargo_tdays: int = 21
    ):
        """Validates zero lookahead in anchored walk-forward fold."""
        CombinatorialPurgedCV.assert_no_leakage(
            train_df, test_df, t0_col=t0_col, t1_col=t1_col, embargo_tdays=embargo_tdays
        )
