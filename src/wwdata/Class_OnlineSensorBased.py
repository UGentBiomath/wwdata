"""
Class_OnlineSensorBased provides functionalities for data handling of data obtained with online sensors in the field of (waste)water treatment.
Copyright (C) 2025 Chaim De Mulder, Saba Daneshgar, BIOMATH, Ghent University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.
"""

from __future__ import annotations



import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import datetime as dt
import warnings as wn
import random as rn
import inspect

from .Class_HydroData import HydroData
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union, Tuple, Literal, Any, Hashable, Iterable, List, Callable
import logging
import types
import inspect
logger = logging.getLogger(__name__)


class OnlineSensorBased(HydroData):
    """
    Adds utilities for online/continuous sensor datasets on top of HydroData.

    Parameters
    ----------
    data : pd.DataFrame or array-like
        Time-indexed sensor data (rows = timestamps, cols = signals).
    timedata_column : str, default 'index'
        Use 'index' when the index holds the time axis. Otherwise provide a column
        name containing time values; it will not be set as index automatically here.
    data_type : str, default 'WWTP'
        A short label for the dataset type (e.g., 'WWTP').
    experiment_tag : str, default 'No tag given'
        A free-form tag identifying the dataset/experiment.
    time_unit : Optional[str], default None
        Unit of the time axis if non-datetime (e.g., 'min', 'hr', 'd').
    """

    def __init__(
        self,
        data: pd.DataFrame,
        timedata_column: str = "index",
        data_type: str = "WWTP",
        experiment_tag: str = "No tag given",
        time_unit: Optional[str] = None,
    ) -> None:
        super().__init__(
            data=data,
            timedata_column=timedata_column,
            data_type=data_type,
            experiment_tag=experiment_tag,
            time_unit=time_unit,
        )

        # DataFrame to store imputed/filled values; same index/columns as data.
        self.filled: pd.DataFrame = pd.DataFrame(index=self.data.index, dtype=float)

        # Metadata for filled values: mirrors meta_valid but for filling provenance.
        # Initialize from meta_valid (if present) to keep shape; otherwise empty aligned frame.
        base_meta = getattr(self, "meta_valid", None)
        if base_meta is not None and not base_meta.empty:
            self.meta_filled: pd.DataFrame = base_meta.reindex(self.data.index)
        else:
            self.meta_filled = pd.DataFrame(index=self.data.index)

        # Per-signal imputation error summary (%); initialize with NaN.
        self.filling_error: pd.DataFrame = pd.DataFrame(
            data=np.nan,
            index=pd.Index(self.data.columns, name="signal"),
            columns=["imputation error [%]"],
        )

        # Internal flags for one-time warnings
        self._filling_warning_issued: bool = False
        self._rain_warning_issued: bool = False



    def drop_index_duplicates(
        self,
        *,
        keep: Literal["first", "last", False] = "first",
        sort_index: bool = True,
        print_number: bool = True
    ) -> None:
        """
        Drop duplicate index rows in `data` and apply the SAME positional keep-mask
        to `meta_valid`, `meta_filled`, and `filled`.

        Parameters
        ----------
        keep : {'first','last',False}, default 'first'
            Which duplicate to keep. `False` drops *all* occurrences of duplicated labels.
        sort_index : bool, default True
            Sort all frames by index after de-duplication.

        Returns
        -------
        int
            Number of duplicate rows removed from `data`.
        """
        if self.data is None or self.data.empty:
            return None

        # Count duplicates for reporting
        if keep is False:
            counts = self.data.index.value_counts()
            n_dupes = int(counts[counts > 1].sum())
        else:
            n_total = int(len(self.data))
            n_unique = int(self.data.index.nunique())
            n_dupes = n_total - n_unique

        # Build ONE keep mask from data's index and use it positionally everywhere
        orig_index = self.data.index
        keep_mask = ~orig_index.duplicated(keep=keep) if keep in ("first", "last") else ~orig_index.duplicated(keep=False)
        keep_mask_pos = np.asarray(keep_mask, dtype=bool)

        # Apply to main data (positional keep preserves current order)
        self.data = self.data.iloc[keep_mask_pos]

        # Apply the SAME positional mask to aux frames (avoid reindex on duplicate labels)
        for attr in ("meta_valid", "meta_filled", "filled"):
            frame = getattr(self, attr, None)
            if isinstance(frame, pd.DataFrame):
                if len(frame) == len(orig_index):
                    setattr(self, attr, frame.iloc[keep_mask_pos])
                else:
                    # Fallback if lengths drifted: best effort align to data index
                    # (may still have duplicate labels, but avoids positional mismatch)
                    setattr(self, attr, frame.reindex(self.data.index))
            else:
                setattr(self, attr, pd.DataFrame(index=self.data.index))

        # Optional consistent sorting for all frames
        if sort_index:
            self.data = self.data.sort_index()
            self.meta_valid = self.meta_valid.sort_index()
            self.meta_filled = self.meta_filled.sort_index()
            self.filled = self.filled.sort_index()

        # Housekeeping
        self._update_time()

        # Friendly warning for object dtype index
        if len(self.data.index) >= 2 and self.data.index.dtype == object:
            wn.warn(
                "Index has object dtype; ordering may be unexpected. Consider converting "
                "to datetime or numeric and calling .sort_index().",
                RuntimeWarning,
                stacklevel=2,
            )
        if print_number:
            print(f"{n_dupes} rows have been dropped as duplicates.")
        return None
    

    def calc_total_proportional(
        self,
        Q_tot: str,
        Q: Sequence[str],
        conc: Sequence[str],
        *,
        new_name: str = "new",
        unit: str = "mg/l",
        filled: bool = False,
    ) -> None:
        """
        Compute a total (flow-weighted) concentration:
            new = sum_i( Q_i * conc_i ) / Q_tot

        Parameters
        ----------
        Q_tot : str
            Column name containing the total flow (denominator).
        Q : sequence of str
            Column names for the contributing flows (numerator terms).
        conc : sequence of str
            Column names for the corresponding concentrations (same order as Q).
        new_name : str, default 'new'
            Name of the output column to create.
        unit : str, default 'mg/l'
            Unit string to assign to the new column (if self.units is dict-like).
        filled : bool, default False
            If True, compute using self.filled; otherwise use self.data.

        Returns
        -------
        None
            Adds `new_name` to the chosen frame.
        """
        # Choose the working frame
        df = self.filled if filled else self.data
        if df is None or df.empty:
            raise ValueError("No data available to compute proportional total.")

        # Basic validations
        if Q_tot not in df.columns:
            raise KeyError(f"Total flow column '{Q_tot}' not found.")
        if not Q or not conc:
            raise ValueError("Q and conc must be non-empty sequences of column names.")
        if len(Q) != len(conc):
            raise ValueError(f"Q and conc must have the same length (got {len(Q)} vs {len(conc)}).")

        missing_q = [c for c in Q if c not in df.columns]
        missing_c = [c for c in conc if c not in df.columns]
        if missing_q or missing_c:
            missing = ", ".join(missing_q + missing_c)
            raise KeyError(f"Missing columns: {missing}")

        # Coerce numeric; keep index alignment
        Q_tot_s = pd.to_numeric(df[Q_tot], errors="coerce")

        # Build numerator: sum_i(Q_i * conc_i)
        numerator = pd.Series(0.0, index=df.index)
        for Qi, Ci in zip(Q, conc):
            Qi_s = pd.to_numeric(df[Qi], errors="coerce")
            Ci_s = pd.to_numeric(df[Ci], errors="coerce")
            numerator = numerator.add(Qi_s * Ci_s, fill_value=0.0)

        # Safe division: avoid inf on zero denominators
        with np.errstate(divide="ignore", invalid="ignore"):
            result = numerator.divide(Q_tot_s)
        result = result.replace([np.inf, -np.inf], np.nan)

        # Assign
        df[new_name] = result.astype(float)

        # Write back (if we operated on self.filled it's already a view)
        if not filled:
            self.data = df
            # keep columns cache if you maintain one
            if hasattr(self, "columns"):
                try:
                    self.columns = np.array(self.data.columns)
                except Exception:
                    pass

        # Update units if dict-like
        try:
            if hasattr(self, "units") and isinstance(self.units, dict):
                self.units[new_name] = unit
            elif hasattr(self, "units") and isinstance(self.units, pd.DataFrame):
                # Optional: if you store units as a 1-row DataFrame with columns = signals
                # we try to set it; otherwise warn.
                if new_name in self.units.columns:
                    self.units.loc[:, new_name] = unit
                else:
                    # attempt to add the column if shape allows
                    try:
                        self.units[new_name] = unit
                    except Exception:
                        wn.warn(
                            "Could not set unit on units DataFrame; please update manually.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
            else:
                # No units structure; ignore silently
                pass
        except Exception:
            wn.warn(
                "Something went wrong while updating units; verify self.units.",
                RuntimeWarning,
                stacklevel=2,
            )

        return None


    def calc_daily_average(
        self,
        column_name: str,
        arange: Optional[Tuple[object, object]] = None,
        *,
        plot: bool = False,
    ) -> Optional[Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]]:
        """
        Calculate daily averages (and std) for a column within a given range.

        - If index is DatetimeIndex: resamples by calendar day ('D').
        - If index is numeric (int/float): groups days by integer bins [n, n+1).

        Parameters
        ----------
        column_name : str
            Column to aggregate.
        arange : (start, end), optional
            Slice bounds for selecting the data to aggregate.
            Must be compatible with the index type.
            If None, use the full available range.
        plot : bool, default False
            If True, returns (fig, ax) with error bars (mean ± std).

        Returns
        -------
        None or (fig, ax)
            Updates self.daily_average[column_name] with a DataFrame containing:
            ['day', 'mean', 'std'].

            If plot=True, returns the matplotlib figure and axes.
        """
        # Ensure output dict exists
        if not hasattr(self, "daily_average") or not isinstance(getattr(self, "daily_average"), dict):
            self.daily_average = {}

        # Validate column
        if column_name not in self.data.columns:
            raise KeyError(f"Column '{column_name}' not found in data.")

        # Slice range
        try:
            if arange is None:
                series = self.data[column_name].copy()
            else:
                series = self.data.loc[arange[0]:arange[1], column_name].copy()
        except TypeError as e:
            raise TypeError(
                f"Slicing not possible for index type {type(self.data.index[0])} "
                f"with arange element types {type(arange[0])}, {type(arange[1])}. "
                "Ensure arange values are compatible with the index."
            ) from e

        if series.empty:
            wn.warn("Selected range is empty; no daily averages computed.", RuntimeWarning, stacklevel=2)
            self.daily_average[column_name] = pd.DataFrame(columns=["day", "mean", "std"])
            return None

        # Numeric coercion and drop NaNs
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            wn.warn("No numeric samples in selected range; no daily averages computed.", RuntimeWarning, stacklevel=2)
            self.daily_average[column_name] = pd.DataFrame(columns=["day", "mean", "std"])
            return None

        # DatetimeIndex path
        if isinstance(series.index, pd.DatetimeIndex):
            daily_mean = series.resample("D").mean()
            daily_std = series.resample("D").std()
            to_return = pd.DataFrame({
                "day": daily_mean.index,
                "mean": daily_mean.values,
                "std": daily_std.values,
            })

        # Numeric index path
        elif np.issubdtype(series.index.dtype, np.number):
            days = np.floor(series.index.values).astype(int)
            df_tmp = pd.DataFrame({"day": days, "val": series.values})
            grouped = df_tmp.groupby("day")
            to_return = pd.DataFrame({
                "day": grouped["day"].first().index,
                "mean": grouped["val"].mean().values,
                "std": grouped["val"].std().values,
            })
        else:
            raise TypeError(
                "Unsupported index type. Use a DatetimeIndex or numeric index for daily averaging."
            )

        # Store
        self.daily_average[column_name] = to_return

        # Plot if requested
        if plot:
            fig, ax = plt.subplots(figsize=(16, 6))
            if isinstance(series.index, pd.DatetimeIndex):
                ax.errorbar(pd.to_datetime(to_return["day"]), to_return["mean"], yerr=to_return["std"], fmt="o")
                ax.set_xlabel("Time")
            else:
                ax.errorbar(to_return["day"], to_return["mean"], yerr=to_return["std"], fmt="o")
                ax.set_xlabel("Day (index units)")
            ax.set_ylabel(column_name)
            ax.tick_params(labelsize=12)
            fig.tight_layout()
            return fig, ax

        return None

#     ###############################################################################
#     ##                        FILLING HELP FUNCTIONS                             ##
#     ###############################################################################
    
    def _reset_meta_filled(self, data_name: Optional[str] = None) -> None:
        """
        Reset the `meta_filled` DataFrame.

        Parameters
        ----------
        data_name : str, optional
            If provided, only reset the given column in `meta_filled` to match
            `meta_valid`. If None, reset the entire DataFrame.

        Notes
        -----
        - Ensures index alignment with `self.data.index`.
        - If `data_name` does not exist in `meta_valid`, a warning is issued.
        """
        if data_name is None:
            # Full reset
            if hasattr(self, "meta_valid") and isinstance(self.meta_valid, pd.DataFrame):
                self.meta_filled = self.meta_valid.copy().reindex(self.data.index)
            else:
                self.meta_filled = pd.DataFrame(index=self.data.index)
        else:
            # Column-specific reset
            if hasattr(self, "meta_valid") and data_name in self.meta_valid.columns:
                # ensure column exists in meta_filled too
                if not hasattr(self, "meta_filled") or not isinstance(self.meta_filled, pd.DataFrame):
                    self.meta_filled = pd.DataFrame(index=self.data.index)
                if data_name not in self.meta_filled.columns:
                    self.meta_filled[data_name] = None
                self.meta_filled[data_name] = self.meta_valid[data_name].copy()
            else:
                wn.warn(
                    f"Column '{data_name}' not found in meta_valid; nothing was reset.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    
    def add_to_filled(self, column_names: Union[str, Sequence[str]]) -> None:
        """
        Add one or more columns into `self.filled`, seeding them with the *validated*
        (i.e., 'original' in meta_valid) values from `self.data`. All other rows
        become NaN (to be filled later by imputation methods).

        Parameters
        ----------
        column_names : str or sequence of str
            Column(s) to add/seed into `self.filled`.

        Notes
        -----
        - If `meta_valid` is missing or does not contain a column tag, the entire
        column is copied (with a warning).
        - `self.filled` is created/reindexed to match `self.data.index`.
        """
        self._plot = "filled"

        # Normalize input to a list
        if isinstance(column_names, str):
            names = [column_names]
        else:
            names = list(column_names)

        # Ensure `self.filled` exists and is aligned to the main index
        if not hasattr(self, "filled") or not isinstance(self.filled, pd.DataFrame):
            self.filled = pd.DataFrame(index=self.data.index)
        else:
            self.filled = self.filled.reindex(self.data.index)

        # Ensure meta_valid exists (used to filter for 'original')
        meta = getattr(self, "meta_valid", None)
        has_meta = isinstance(meta, pd.DataFrame) and not meta.empty

        for col in names:
            if col not in self.data.columns:
                raise KeyError(f"Column '{col}' not found in data.")

            series = self.data[col].copy()

            if has_meta and (col in meta.columns):
                # Keep only validated/original values; others become NaN
                mask_original = (meta[col].reindex(self.data.index) == "original")
                seeded = series.where(mask_original, np.nan)
            else:
                wn.warn(
                    f"meta_valid missing or no tags for '{col}'; copying entire column.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                seeded = series

            # Assign into filled; reindex again to be safe
            self.filled[col] = pd.to_numeric(seeded, errors="coerce")
            self.filled = self.filled.reindex(self.data.index)

        return None
                
    
    def _add_to_meta(self, to_fill: str) -> None:
        """
        Ensure `to_fill` exists and is aligned across:
        - self.meta_valid  (tags for filtering)
        - self.meta_filled (tags for filling provenance)
        - self.filled      (values to be filled)

        Behavior
        --------
        - Creates frames if missing; aligns all to `self.data.index`.
        - Makes indices unique in aux frames (keeps 'first') to avoid reindex errors.
        - Ensures `meta_valid[to_fill]` exists and has no NaNs/placeholder tags (defaults to 'original').
        - Ensures `meta_filled[to_fill]` exists; if missing, copies from `meta_valid[to_fill]`;
        otherwise fills missing tags from `meta_valid` and defaults remaining to 'original'.
        - Ensures `filled[to_fill]` exists and is seeded with the *validated/original* values
        from `self.data[to_fill]`. Non-validated rows are set to NaN (to be filled later).

        Returns
        -------
        None
        """
        # --- basic checks ---
        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")
        idx = self.data.index

        # --- ensure/align meta_valid ---
        if not hasattr(self, "meta_valid") or not isinstance(self.meta_valid, pd.DataFrame):
            self.meta_valid = pd.DataFrame(index=idx)
        else:
            # guard against duplicate index labels before reindex
            if not self.meta_valid.index.is_unique:
                self.meta_valid = self.meta_valid.loc[~self.meta_valid.index.duplicated(keep="first")]
            self.meta_valid = self.meta_valid.reindex(idx)

        # guarantee column & clean tags
        if to_fill not in self.meta_valid.columns:
            self.meta_valid[to_fill] = "original"
        else:
            self.meta_valid[to_fill] = (
                self.meta_valid[to_fill]
                .astype(object)
                .fillna("original")
                .replace({"!!": "original"})
            )

        # --- ensure/align meta_filled ---
        if not hasattr(self, "meta_filled") or not isinstance(self.meta_filled, pd.DataFrame):
            self.meta_filled = pd.DataFrame(index=idx)
        else:
            if not self.meta_filled.index.is_unique:
                self.meta_filled = self.meta_filled.loc[~self.meta_filled.index.duplicated(keep="first")]
            self.meta_filled = self.meta_filled.reindex(idx)

        if to_fill not in self.meta_filled.columns:
            # start from validated tags
            self.meta_filled[to_fill] = self.meta_valid[to_fill].copy()
        else:
            # re-use validated tags to fill holes, then default remaining to 'original'
            self.meta_filled[to_fill] = (
                self.meta_filled[to_fill]
                .astype(object)
                .where(self.meta_filled[to_fill].notna(), self.meta_valid[to_fill])
                .fillna("original")
                .replace({"!!": "original"})
            )

        # --- ensure/align filled values ---
        if not hasattr(self, "filled") or not isinstance(self.filled, pd.DataFrame):
            self.filled = pd.DataFrame(index=idx)
        else:
            if not self.filled.index.is_unique:
                self.filled = self.filled.loc[~self.filled.index.duplicated(keep="first")]
            self.filled = self.filled.reindex(idx)

        if to_fill not in self.filled.columns:
            # seed with validated/original values only; others NaN
            mask_original = (self.meta_valid[to_fill] == "original")
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce").where(mask_original)
        else:
            # ensure numeric dtype & alignment (don’t overwrite existing user-filled values)
            self.filled[to_fill] = pd.to_numeric(self.filled[to_fill], errors="coerce")

        # final tag normalization (paranoia)
        self.meta_valid[to_fill] = self.meta_valid[to_fill].replace({"!!": "original"}).fillna("original")
        self.meta_filled[to_fill] = self.meta_filled[to_fill].replace({"!!": "original"}).fillna("original")

        return None
    

    def _warning(
        self,
        message,                 # str or Warning instance
        category,                # Warning subclass
        filename: str,
        lineno: int,
        file=None,
        line=None,
    ) -> None:
        """
        Optional custom formatter for warnings.showwarning.
        Produces: "<filename>:<lineno>: <CategoryName>: <message>"
        """
        cat_name = category.__name__ if hasattr(category, "__name__") else str(category)
        msg_text = str(message)
        out = f"{filename}:{lineno}: {cat_name}: {msg_text}"
        # Write to the provided file-like or fallback to stderr
        stream = file if file is not None else wn._showwarnmsg_impl.__self__ if hasattr(wn._showwarnmsg_impl, "__self__") else None
        if stream and hasattr(stream, "write"):
            try:
                stream.write(out + "\n")
            except Exception:
                print(out)  # final fallback
        else:
            print(out)


    def _use_custom_warning_format(self, enable: bool = True) -> None:
        """
        Enable/disable the custom warning formatting for the current process.
        This modifies the global warnings.showwarning hook.
        """
        if enable:
            wn.showwarning = types.MethodType(self._warning, self)  # bind to instance method
        else:
            # Restore default behavior
            wn.showwarning = wn._showwarning_orig if hasattr(wn, "_showwarning_orig") else warnings._showwarning  # type: ignore[attr-defined]


    def _filling_warning(self, *, use_custom_format: bool = False, stacklevel: int = 2) -> None:
        """
        One-time notice for filling workflows. Shown only on first call.
        """
        if getattr(self, "_filling_warning_issued", False):
            return

        if use_custom_format:
            try:
                # Save original once
                if not hasattr(wn, "_showwarning_orig"):
                    wn._showwarning_orig = wn.showwarning
                self._use_custom_warning_format(True)
            except Exception:
                pass  # fall back silently

        wn.warn(
            "When using filling functions, start with small gaps and progressively "
            "move to larger gaps. This improves algorithm reliability. "
            "This notice is shown only once.",
            UserWarning,
            stacklevel=stacklevel,
        )

        # Restore default formatting if we temporarily changed it
        if use_custom_format:
            try:
                self._use_custom_warning_format(False)
            except Exception:
                pass

        self._filling_warning_issued = True


    def _rain_warning(self, *, use_custom_format: bool = False, stacklevel: int = 2) -> None:
        """
        Reminder for operations affecting rain/high-event data. Shown EVERY time.
        """
        if use_custom_format:
            try:
                if not hasattr(wn, "_showwarning_orig"):
                    wn._showwarning_orig = wn.showwarning
                self._use_custom_warning_format(True)
            except Exception:
                pass

        wn.warn(
            "Data points obtained during rain/high events will be replaced. "
            "Ensure the chosen method is appropriate for gaps during rain events.",
            UserWarning,
            stacklevel=stacklevel,
        )

        if use_custom_format:
            try:
                self._use_custom_warning_format(False)
            except Exception:
                pass
    def _check_rain(
        self,
        arange: Optional[Tuple[object, object]] = None,
    ) -> bool:
        """
        Check if the selected range overlaps rain/high events and warn if so.

        Parameters
        ----------
        arange : (start, end), optional
            Index slice for the check. If None, checks the entire available range.

        Returns
        -------
        bool
            True if a rain/high event was detected in the range (and a warning issued),
            False otherwise.
        """
        if getattr(self, "data_type", None) != "WWTP":
            return False

        # Require a highs DataFrame with a 'highs' column
        highs = getattr(self, "highs", None)
        if not isinstance(highs, pd.DataFrame) or "highs" not in highs.columns or highs.empty:
            return False

        if arange is None:
            sub = highs["highs"]
        else:
            try:
                sub = highs.loc[arange[0]:arange[1], "highs"]
            except Exception:
                # If slicing fails, bail out quietly
                return False

        if sub.sum() > 0:
            self._rain_warning()
            return True
        return False
    
    def _check_daily_profile(
        self,
        column_name: Optional[str] = None,
        *,
        return_df: bool = False,
    ) -> Union[bool, pd.DataFrame]:
        """
        Validate that a daily profile exists (optionally for a specific column).

        Parameters
        ----------
        column_name : str, optional
            If provided, check that `self.daily_profile[column_name]` exists.
        return_df : bool, default False
            If True, return the profile DataFrame instead of just a boolean.

        Returns
        -------
        bool or pd.DataFrame
            - If return_df=False (default): True if the profile exists, False otherwise.
            - If return_df=True: the requested profile DataFrame is returned.

        Raises
        ------
        AttributeError
            If `self.daily_profile` has not been created yet.
        TypeError
            If `self.daily_profile` is not a dict-like.
        KeyError
            If `column_name` is provided but not present in the dictionary.
        """
        if not hasattr(self, "daily_profile"):
            raise AttributeError(
                "self.daily_profile doesn't exist yet. Run calc_daily_profile(...) first."
            )
        if not isinstance(self.daily_profile, dict):
            raise TypeError(
                "self.daily_profile should be a dict. Recompute via calc_daily_profile(...)."
            )

        if column_name is None:
            if not self.daily_profile:
                return False if not return_df else pd.DataFrame()
            if return_df:
                return pd.concat(self.daily_profile, names=["column", "time_of_day"])
            return True

        if column_name not in self.daily_profile:
            return False if not return_df else pd.DataFrame()

        return self.daily_profile[column_name] if return_df else True
    
    def _align_aux_frames(self, keep: str = "first") -> None:
        """Make aux frames unique-indexed and aligned to self.data.index."""
        for attr in ("meta_valid", "meta_filled", "filled"):
            frame = getattr(self, attr, None)
            if isinstance(frame, pd.DataFrame):
                if not frame.index.is_unique:
                    frame = frame.loc[~frame.index.duplicated(keep=keep)]
                setattr(self, attr, frame.reindex(self.data.index))
            else:
                setattr(self, attr, pd.DataFrame(index=self.data.index))
    
   

    def _reset_meta_filled_column(self, to_fill: str) -> None:
        """
        Used when clear=True for a fill method.
        Resets meta_filled[to_fill] back to meta_valid[to_fill] and re-seeds
        self.filled[to_fill] with originals (NaN for filtered).
        """
        # ensure columns exist
        if to_fill not in self.meta_valid.columns:
            self.add_to_meta_valid([to_fill])  # sets to 'original'
        self.meta_filled[to_fill] = self.meta_valid[to_fill].copy()
        # seed 'filled' with originals; NaN where filtered
        ser = self.data[to_fill].copy()
        ser[self.meta_filled[to_fill] == "filtered"] = np.nan
        self.filled[to_fill] = ser.reindex(self.index())


    def _get_fill_targets(
        self,
        to_fill: str,
        arange: tuple | None = None,
        only_checked: bool = True,
    ) -> pd.Index:
        """
        Return the index labels that are eligible to be filled NOW.
        By design, we ONLY return rows currently tagged 'filtered' in meta_filled.
        (We never overwrite rows previously filled by another method.)
        """
        # make sure meta_filled exists/aligns
        self.meta_filled = self.meta_filled.reindex(self.index())
        if to_fill not in self.meta_filled.columns:
            # initialize from meta_valid; do NOT mark any 'filled_*'
            if to_fill not in self.meta_valid.columns:
                self.add_to_meta_valid([to_fill])
            self.meta_filled[to_fill] = self.meta_valid[to_fill].copy()

        mv = self.meta_filled[to_fill]

        # base mask: only currently 'filtered'
        mask = (mv == "filtered")

        # arange restriction if provided
        if arange is not None:
            try:
                idx_range = self.data.loc[arange[0]:arange[1]].index
            except Exception:
                raise TypeError(
                    "Invalid `arange` bounds for index slicing; use datetime-like or matching index type."
                )
            mask = mask & mv.index.isin(idx_range)

        # only_checked controls nothing new here—by policy we only fill filtered.
        # If you ever want a mode that overwrites 'original' too (not recommended),
        # you'd branch here.

        targets = mv.index[mask]
        return targets
            
#     ###############################################################################
#     ##                          FILLING FUNCTIONS                                ##
#     ###############################################################################
    def fill_missing_interpolation(
        self,
        to_fill: str,
        range_: int,
        arange: Optional[Tuple[object, object]] = None,
        *,
        method: str = "time",           # 'time' for DatetimeIndex; 'index'/'linear' for numeric index
        limit_direction: str = "both",
        plot: bool = False,
        clear: bool = False,
        **kwargs,
    ) -> None:
        """
        Fill short filtered runs (<= range_) in `to_fill` by interpolation.
        - Keeps original values for non-filtered points in self.filled[to_fill]
        - Fills only eligible filtered runs inside `arange`
        - Updates self.meta_filled[to_fill] to 'filled_interpol' only for points actually filled
        - Leaves longer filtered runs as 'filtered' (and NaN in self.filled)

        This method only touches the specified column; it does not reshape/reindex other frames.
        """
        self._plot = "filled"
        self._filling_warning()

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")
        if range_ <= 0:
            raise ValueError("`range_` must be a positive integer.")

        # Start from a clean, aligned state for this column
        if clear:
            self._reset_meta_filled(to_fill)
        
        self._add_to_meta(to_fill)  # seeds self.filled[to_fill] with original values where validated

        # Optional warning (no side effects)
        if arange is not None:
            self._check_rain(arange)

        idx_all = self.data.index

        # Window to operate in
        if arange is None:
            win_idx = idx_all
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` bounds must match the index dtype.") from e
            if len(win_idx) == 0:
                if plot:
                    self.plot_analysed(to_fill)
                return

        
        tags_valid = (
            self.meta_filled[to_fill]
            .reindex(idx_all)
            .fillna("original")
            .replace({"!!": "original"})
        )
        filtered_all = (tags_valid == "filtered")
        filtered_win = filtered_all.reindex(win_idx, fill_value=False)

        if not filtered_win.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        # Compute run-lengths on the boolean filtered mask in the window
        # (True runs indicate consecutive filtered points)
        change = filtered_win.ne(filtered_win.shift(1)).cumsum()
        run_lengths = change.map(change.value_counts())
        eligible = filtered_win & (run_lengths <= range_)

        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        # Work on a copy of the filled series (aligned to the global index)
        s = pd.to_numeric(self.filled[to_fill], errors="coerce").copy()

        # Inside the window, force eligible indices to NaN so interpolate can fill them
        s_win = s.loc[win_idx].copy()
        s_win.loc[eligible] = np.nan

        # Interpolate *within the window*.
        # limit=range_ ensures only gaps of that length are filled.
        interp_kwargs = dict(method=method, limit=range_, limit_direction=limit_direction)
        interp_kwargs.update(kwargs)
        try:
            s_win_interp = s_win.interpolate(**interp_kwargs, limit_area="inside")
        except TypeError:
            # for older pandas without limit_area
            s_win_interp = s_win.interpolate(**interp_kwargs)

        # Determine which eligible points actually got filled
        newly_filled_idx = eligible.index[eligible & s_win_interp.notna()]

        # Write the window back and assign the updated series to the column
        s.loc[win_idx] = s_win_interp
        self.filled[to_fill] = s

        # Initialize / normalize meta_filled for this column and flip tags only for newly filled
        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(idx_all)
            .fillna("original")
            .replace({"!!": "original"})
        )
        
        if len(newly_filled_idx) > 0:
            # if self.meta_filled.loc[newly_filled_idx, to_fill] == "filtered":
            self.meta_filled.loc[newly_filled_idx, to_fill] = "filled_interpol" 
        # (Indices still filtered remain 'filtered'; non-filtered remain 'original')

        if plot:
            self.plot_analysed(to_fill)
    

    def fill_missing_kalman(
        self,
        to_fill: str,
        arange: Optional[Tuple[object, object]] = None,
        *,
        model: Literal["local_level", "local_linear_trend"] = "local_level",
        seasonal_periods: Optional[int] = None,   # e.g. 96 for 15-min diurnal, 24 for hourly
        max_gap: Optional[int] = None,            # only fill filtered runs with length <= max_gap
        plot: bool = False,
        clear: bool = False,
        fit_kwargs: Optional[dict] = None,        # e.g. {"disp": False, "maxiter": 200}
    ) -> None:
        """
        Fill filtered values in `to_fill` using a state-space Kalman smoother.

        - Preserves original values.
        - Fills only points currently tagged 'filtered' (optionally only short runs).
        - Updates `self.filled[to_fill]` with smoothed estimates at eligible points.
        - Sets `self.meta_filled[to_fill] = 'filled_kalman'` only where a value was produced.

        Parameters
        ----------
        to_fill : str
            Column name to fill.
        arange : (start, end) slice bounds in index dtype; None = full index.
        model : {'local_level','local_linear_trend'}
            Structural model to use.
        seasonal_periods : int, optional
            Add a deterministic seasonal component with this period (in samples).
        max_gap : int, optional
            Only fill filtered runs with length <= `max_gap`. If None, no cap by length.
        plot : bool
            If True, calls `plot_analysed(to_fill)` after filling.
        clear : bool
            If True, reset meta_filled for this column before filling.
        fit_kwargs : dict, optional
            Extra keyword args passed to `results = model.fit(...)`.
        """
        # --- Safety & setup ---
        import warnings as wn
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            # older signature with lineno() – ignore
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")

        if clear:
            self._reset_meta_filled(to_fill)

        # Seed aux frames/columns: originals → self.filled[to_fill], filtered → NaN
        self._add_to_meta(to_fill)

        # Select window
        if arange is None:
            win_idx = self.data.index
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` bounds must match index dtype.") from e
            if len(win_idx) == 0:
                if plot:
                    self.plot_analysed(to_fill)
                return

        
        tags_valid = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )

        filtered_all = (tags_valid == "filtered")
        filtered_win = filtered_all.reindex(win_idx, fill_value=False)
        if not filtered_win.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        # Optional short-gap eligibility
        eligible = filtered_win.copy()
        if max_gap is not None and max_gap > 0:
            # run-length encoding on filtered_win
            change = filtered_win.ne(filtered_win.shift(1)).cumsum()
            run_lengths = change.map(change.value_counts())
            eligible = filtered_win & (run_lengths <= max_gap)

        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        # Endog for the model: start from filled (originals present, filtered as NaN)
        y_full = pd.to_numeric(self.filled[to_fill], errors="coerce")
        y = y_full.loc[win_idx]

        # Ensure monotonic index for time-mode Kalman
        if isinstance(y.index, pd.DatetimeIndex) and not y.index.is_monotonic_increasing:
            y = y.sort_index()
            eligible = eligible.reindex(y.index, fill_value=False)
            win_idx = y.index  # keep aligned

        # --- Build & fit the structural model ---
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except Exception as e:
            raise ImportError(
                "statsmodels is required for Kalman filling. Install via `pip install statsmodels`."
            ) from e

        level = None
        if model == "local_level":
            level = "llevel"
        elif model == "local_linear_trend":
            level = "ltrend"
        else:
            raise ValueError("`model` must be 'local_level' or 'local_linear_trend'.")

        seasonal = seasonal_periods if (seasonal_periods and seasonal_periods > 1) else None

        ss_mod = UnobservedComponents(
            endog=y,
            level=level,
            seasonal=seasonal,
            # You could expose more options (stochastic_level, irregular, etc.)
        )

        # Fit (let statsmodels handle missing values in endog)
        fit_kwargs = dict() if fit_kwargs is None else dict(fit_kwargs)
        fit_kwargs.setdefault("disp", False)

        try:
            res = ss_mod.fit(**fit_kwargs)
        except Exception as err:
            # Fall back to a simpler model if fitting fails
            wn.warn(f"Kalman fit failed with `{model}`; retrying with local level. Error: {err}")
            ss_mod = UnobservedComponents(endog=y, level="llevel", seasonal=seasonal)
            res = ss_mod.fit(**fit_kwargs)

        # Get in-sample smoothed estimates for the window (includes missing positions)
        try:
            pred = res.get_prediction()
            y_hat = pred.predicted_mean
        except Exception:
            # fallback
            y_hat = res.fittedvalues

        y_hat = pd.to_numeric(y_hat, errors="coerce").reindex(win_idx)

        # Only assign to indices that are eligible AND got a finite estimate
        to_update = eligible.index[eligible & y_hat.notna()]
        if len(to_update) > 0:
            self.filled.loc[to_update, to_fill] = y_hat.loc[to_update]
            # Tag just those indices as filled by Kalman
            self.meta_filled[to_fill] = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            
            # if self.meta_filled.loc[to_update, to_fill] == "filtered":
            self.meta_filled.loc[to_update, to_fill] = "filled_kalman"

        # leave non-eligible or non-estimated filtered points as-is (still 'filtered', NaN in filled)
        if plot:
            self.plot_analysed(to_fill)

    def fill_missing_arima(
        self,
        to_fill: str,
        arange: Optional[Tuple[object, object]] = None,
        *,
        order: Tuple[int, int, int] = (1, 0, 1),          # (p,d,q)
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,  # (P,D,Q,s) or None
        trend: Optional[Literal["n","c","t","ct"]] = None,
        max_gap: Optional[int] = None,                     # only fill filtered runs with length <= max_gap
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        plot: bool = False,
        clear: bool = False,
        fit_kwargs: Optional[dict] = None,                 # e.g. {"disp": False, "maxiter": 200}
    ) -> None:
        """
        Fill filtered values in `to_fill` using SARIMAX (ARIMA/SARIMA) in-sample predictions.

        Behavior
        --------
        - Keeps original values untouched in self.filled[to_fill].
        - Identifies 'filtered' points via self.meta_valid[to_fill] (optionally limited by `max_gap`).
        - Fits a SARIMAX model on the selected window (NaNs handled via state-space).
        - Writes predictions at eligible indices; tags them as 'filled_arima' in meta_filled.

        Parameters
        ----------
        to_fill : str
            Column to fill.
        arange : (start, end) in index dtype; None = full index.
        order : (p,d,q)
            Non-seasonal ARIMA order.
        seasonal_order : (P,D,Q,s) or None
            Seasonal ARIMA order. Use None for non-seasonal.
        trend : {'n','c','t','ct'} or None
            Trend spec for SARIMAX.
        max_gap : int, optional
            Only fill filtered runs with length <= max_gap. None = no cap.
        enforce_stationarity, enforce_invertibility : bool
            Passed to SARIMAX.
        plot : bool
            If True, call plot_analysed(to_fill) after filling.
        clear : bool
            If True, reset meta_filled for this column before filling.
        fit_kwargs : dict
            Extra args for `model.fit(...)`.
        """
        # --- setup & checks ---
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")

        if clear:
            self._reset_meta_filled(to_fill)

        # Seed: originals into self.filled[to_fill], filtered → NaN
        self._add_to_meta(to_fill)

        # Select window
        if arange is None:
            win_idx = self.data.index
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` bounds must match the index dtype.") from e
            if len(win_idx) == 0:
                if plot:
                    self.plot_analysed(to_fill)
                return

        # Filter mask from meta_valid (this is where you tag)
        # tags_valid = (
        #     self.meta_valid[to_fill]
        #     .reindex(self.data.index)
        #     .fillna("original")
        #     .replace({"!!": "original"})
        # )
        tags_valid = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )

        filtered_all = (tags_valid == "filtered")
        filtered_win = filtered_all.reindex(win_idx, fill_value=False)
        if not filtered_win.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        # Optional: limit to short runs
        eligible = filtered_win.copy()
        if max_gap is not None and max_gap > 0:
            change = filtered_win.ne(filtered_win.shift(1)).cumsum()
            run_lengths = change.map(change.value_counts())
            eligible = filtered_win & (run_lengths <= max_gap)
        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        # Endogenous series for the model (filled: originals present, filtered NaN)
        y_full = pd.to_numeric(self.filled[to_fill], errors="coerce")
        y = y_full.loc[win_idx]

        # Ensure monotonic index (SARIMAX expects ordered data). Keep alignment.
        if isinstance(y.index, pd.DatetimeIndex) and not y.index.is_monotonic_increasing:
            y = y.sort_index()
            eligible = eligible.reindex(y.index, fill_value=False)
            win_idx = y.index

        # If DatetimeIndex has no freq and you're using seasonal, try inferring:
        if isinstance(win_idx, pd.DatetimeIndex) and seasonal_order is not None and win_idx.freq is None:
            try:
                win_idx = win_idx.inferred_freq and win_idx
            except Exception:
                pass  # SARIMAX can still run; seasonality uses period `s`

        # --- Fit SARIMAX ---
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except Exception as e:
            raise ImportError("statsmodels is required for ARIMA filling. `pip install statsmodels`.") from e

        fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
        fit_kwargs.setdefault("disp", False)

        model = SARIMAX(
            endog=y,
            order=order,
            seasonal_order=(seasonal_order or (0, 0, 0, 0)),
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            # measurement_error=False, simple_differencing=False, time_varying_regression=False, mle_regression=True
        )

        try:
            res = model.fit(**fit_kwargs)
        except Exception as err:
            # Gentle fallback: simplify the model progressively
            try:
                model2 = SARIMAX(endog=y, order=(max(order[0],1), 0, 0), seasonal_order=(0,0,0,0), trend=None)
                res = model2.fit(disp=False)
            except Exception as err2:
                raise RuntimeError(f"SARIMAX fit failed: {err} / fallback: {err2}")

        # In-sample smoothed/predicted mean for the window (includes NaNs at missing points)
        try:
            pred = res.get_prediction()
            y_hat = pred.predicted_mean
        except Exception:
            y_hat = res.fittedvalues
        y_hat = pd.to_numeric(y_hat, errors="coerce").reindex(win_idx)

        # Update only eligible & successfully estimated indices
        to_update = eligible.index[eligible & y_hat.notna()]
        if len(to_update) > 0:
            self.filled.loc[to_update, to_fill] = y_hat.loc[to_update]
            # Tag as filled_arima
            self.meta_filled[to_fill] = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            self.meta_filled.loc[to_update, to_fill] = "filled_arima"

        # Leave others as filtered (NaN in filled)
        if plot:
            self.plot_analysed(to_fill)

    def fill_missing_gaussian(
        self,
        to_fill: str,
        arange: Optional[Tuple[object, object]] = None,   # None → whole dataset
        *,
        only_checked: bool = True,
        clear: bool = False,
        plot: bool = False,
        # efficiency guards
        max_train_points: int = 5000,
        stride: int = 1,
        # GP config
        kernel: Optional[object] = None,
        seasonal_period: Optional[Union[pd.Timedelta, float]] = None,  # e.g. pd.Timedelta(days=1) or seconds float
        unit: str = "d",                          # {'sec','min','hr','d'} for datetime index conversion
        alpha: Optional[float] = None,            # measurement noise; if None, WhiteKernel carries it
        normalize_y: bool = True,
        n_restarts_optimizer: int = 2,
        random_state: Optional[int] = None,
        context_expand: Optional[pd.Timedelta] = None,  # expands training around arange (DatetimeIndex only)
        X_cols: Optional[Sequence[str]] = None,   # optional exogenous regressors
    ) -> None:
        """
        Fill gaps using a Gaussian Process regressor on time (and optional exogenous features).

        - Trains on 'original' points (from meta_valid) within `arange` (or whole data if None).
        - Predicts only indices tagged 'filtered' (unless only_checked=False).
        - Writes into self.filled[to_fill]; tags self.meta_filled[to_fill] = 'filled_gaussian'.

        Notes
        -----
        - GP training is O(n^3). Use `max_train_points` and/or `stride` to keep it tractable.
        - `seasonal_period` adds a periodic kernel (e.g., 1 day or 1 week).
        """
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")

        # Prepare meta_filled & filled scaffolding (seed filled with originals; filtered → NaN)
        if clear:
            self._reset_meta_filled(to_fill)
        self._add_to_meta(to_fill)

        # Resolve window
        if arange is None:
            win_idx = self.data.index
            train_slice = slice(self.data.index.min(), self.data.index.max())
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` must slice the DataFrame index.") from e
            train_slice = slice(arange[0], arange[1])

        if len(win_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return

        # Determine prediction targets
        if only_checked:
            mv = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            # mv = (
            #     self.meta_valid[to_fill]
            #     .reindex(self.data.index)
            #     .fillna("original")
            #     .replace({"!!": "original"})
            # )
            target_idx = win_idx[mv.loc[win_idx].eq("filtered")]
        else:
            target_idx = win_idx

        if len(target_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return

        # Optionally expand training window (only meaningful if arange is provided and index is datetime)
        if isinstance(self.data.index, pd.DatetimeIndex) and context_expand is not None and arange is not None:
            start = pd.to_datetime(arange[0]) - context_expand
            end   = pd.to_datetime(arange[1]) + context_expand
            train_slice = slice(start, end)

        # Build training y from originals
        y_full = pd.to_numeric(self.data[to_fill], errors="coerce")
        y_train = y_full.loc[train_slice].copy()
        if hasattr(self, "meta_valid") and to_fill in self.meta_valid:
            mv_train = (
                self.meta_valid[to_fill]
                .reindex(y_train.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            y_train = y_train.where(mv_train.eq("original"))

        y_train = y_train.dropna()
        # Downsample training for speed if requested
        if stride > 1 and len(y_train) > 0:
            y_train = y_train.iloc[::stride]

        # Cap training size
        if len(y_train) > max_train_points:
            take = np.linspace(0, len(y_train) - 1, max_train_points).astype(int)
            y_train = y_train.iloc[take]
            wn.warn(
                f"GP training truncated to {max_train_points} points (from {len(y_full.loc[train_slice])}).",
                RuntimeWarning, stacklevel=2
            )

        if len(y_train) < 5:
            wn.warn("Not enough original samples for GP training.", RuntimeWarning, stacklevel=2)
            if plot:
                self.plot_analysed(to_fill)
            return

        # Build design matrices (time → numeric; optionally add exogenous features)
        anchor_idx = y_train.index.union(target_idx)

        def _to_numeric_time(
            ix: pd.Index,
            origin: Union[pd.Timestamp, float],
            is_datetime: bool,
            unit: str = "d",
        ) -> np.ndarray:
            """
            Convert an index to a strictly NumPy float array of elapsed time in the requested unit.
            Works reliably across pandas versions.
            """
            if is_datetime:
                di = pd.DatetimeIndex(ix)
                # elapsed seconds as ndarray
                secs = (di - pd.Timestamp(origin)) / np.timedelta64(1, "s")
                arr = np.asarray(secs, dtype=float)
                if unit in ("sec", "s"):
                    return arr
                if unit in ("min", "m"):
                    return arr / 60.0
                if unit in ("hr", "h"):
                    return arr / 3600.0
                # default: days
                return arr / 86400.0
            else:
                # numeric index path
                return np.asarray(pd.Index(ix).to_numpy(dtype=float), dtype=float)

        is_dt = isinstance(anchor_idx, pd.DatetimeIndex)
        if is_dt:
            origin = anchor_idx.min()
            # x_train_time = _to_numeric_time(y_train.index, origin, True).reshape(-1, 1)
            # x_target_time = _to_numeric_time(target_idx, origin, True).reshape(-1, 1)
        else:
            # numeric index
            origin = 0.0
            # x_train_time = _to_numeric_time(y_train.index, origin, False).reshape(-1, 1)
            # x_target_time = _to_numeric_time(target_idx, origin, False).reshape(-1, 1)
        
        x_train_time = _to_numeric_time(y_train.index, origin, is_dt, unit).reshape(-1, 1)
        x_target_time = _to_numeric_time(target_idx, origin, is_dt, unit).reshape(-1, 1)

        X_train_list = [x_train_time]
        X_target_list = [x_target_time]

        # Optional exogenous features (aligned to y_train/target_idx)
        if X_cols:
            for col in X_cols:
                if col not in self.data.columns:
                    raise KeyError(f"Exogenous column '{col}' not found.")
                ex = pd.to_numeric(self.data[col], errors="coerce")
                X_train_list.append(ex.reindex(y_train.index).to_numpy().reshape(-1, 1))
                X_target_list.append(ex.reindex(target_idx).to_numpy().reshape(-1, 1))

        X_train = np.concatenate(X_train_list, axis=1)
        X_target = np.concatenate(X_target_list, axis=1)

        # Drop NaN rows in train (from exogenous alignment)
        mask_ok = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train.to_numpy())
        X_train = X_train[mask_ok]
        y_train = y_train.iloc[mask_ok]

        if len(y_train) < 5:
            wn.warn("Not enough valid rows after assembling GP training matrix.", RuntimeWarning, stacklevel=2)
            if plot:
                self.plot_analysed(to_fill)
            return

        # Construct kernel if not provided
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared
        except ImportError as e:
            raise ImportError("scikit-learn is required for fill_missing_gaussian. Install with `pip install scikit-learn`.") from e

        if kernel is None:
            k_base = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
            if seasonal_period is not None:
                if isinstance(seasonal_period, pd.Timedelta):
                    period_sec = seasonal_period.total_seconds()
                else:
                    period_sec = float(seasonal_period)  # assume seconds
                if is_dt:
                    if unit in ("sec","s"):      p = period_sec
                    elif unit in ("min","m"):    p = period_sec / 60.0
                    elif unit in ("hr","h"):     p = period_sec / 3600.0
                    else:                        p = period_sec / 86400.0
                else:
                    p = period_sec  # numeric index: assume same units
                k_season = ExpSineSquared(length_scale=1.0, periodicity=p,
                                        periodicity_bounds=(max(1e-6, p/10), p*10))
                k = ConstantKernel(1.0, (1e-3, 1e3)) * (k_base + k_season) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
            else:
                k = ConstantKernel(1.0, (1e-3, 1e3)) * k_base + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        else:
            k = kernel

        gp = GaussianProcessRegressor(
            kernel=k,
            alpha=0.0 if alpha is None else float(alpha),
            normalize_y=normalize_y,
            n_restarts_optimizer=int(n_restarts_optimizer),
            random_state=random_state,
        )

        # Fit & predict
        try:
            gp.fit(X_train, y_train.to_numpy().astype(float))
        except Exception as e:
            wn.warn(f"GP fitting failed: {e}", RuntimeWarning, stacklevel=2)
            if plot:
                self.plot_analysed(to_fill)
            return

        try:
            y_pred, y_std = gp.predict(X_target, return_std=True)
        except Exception as e:
            wn.warn(f"GP prediction failed: {e}", RuntimeWarning, stacklevel=2)
            if plot:
                self.plot_analysed(to_fill)
            return

        y_pred = pd.Series(pd.to_numeric(y_pred, errors="coerce"), index=target_idx).dropna()
        if y_pred.empty:
            if plot:
                self.plot_analysed(to_fill)
            return

        # Write results & tag
        if to_fill not in self.filled.columns:
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce")

        self.filled.loc[y_pred.index, to_fill] = y_pred.values
        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )
        self.meta_filled.loc[y_pred.index, to_fill] = "filled_gaussian"

        if plot:
            self.plot_analysed(to_fill)


    

    def fill_missing_ratio(
        self,
        to_fill: str,
        to_use: str,
        ratio: Optional[float] = None,
        arange: Optional[Tuple[object, object]] = None,
        *,
        intercept: float = 0.0,
        estimate_params: bool = False,
        train_range: Optional[Tuple[object, object]] = None,
        train_only_original: bool = True,
        zero_intercept: bool = False,
        robust: bool = False,
        min_train_points: int = 20,
        only_checked: bool = True,
        plot: bool = False,
        clear: bool = False,
        return_params: bool = False,
    ) -> Optional[Dict[str, float]]:
        """
        Fill values in `to_fill` using a linear relation with `to_use`:
            new = ratio * data[to_use] + intercept

        Enhancements:
        - Auto-estimate (ratio, intercept) from data when `estimate_params=True`.
        * If `zero_intercept=True`, estimate ratio with intercept forced to 0.
        * If `robust=True`, try RANSAC (needs scikit-learn); else fallback to least squares.
        * Train on `train_range` (or `arange` if None; else full index), using only original
            points if `train_only_original=True`.
        - Fill only eligible indices (by default, those tagged 'filtered' in meta_valid).

        Parameters
        ----------
        to_fill : str
            Column to fill.
        to_use : str
            Driver column.
        ratio : float, optional
            If not estimating, provide ratio.
        arange : (start, end), optional
            Apply fill within this index window. None = full index.
        intercept : float, default 0.0
            Additive constant; ignored if `zero_intercept=True` and `estimate_params=True`.
        estimate_params : bool, default False
            If True, estimate (ratio, intercept) from training data.
        train_range : (start, end), optional
            Window to fit the relation. Default: `arange` if provided, else full index.
        train_only_original : bool, default True
            Train only on points tagged 'original' (recommended).
        zero_intercept : bool, default False
            Force intercept=0 in estimation.
        robust : bool, default False
            Use a robust linear fit (RANSAC) if available.
        min_train_points : int, default 20
            Minimum number of paired samples required for estimation.
        only_checked : bool, default True
            If True, only fill points tagged 'filtered' in meta_valid[to_fill]; else fill all rows in window.
        plot : bool, default False
            Plot after filling.
        clear : bool, default False
            Reset meta_filled for this column before filling.
        return_params : bool, default False
            If True, return dict with {'ratio': ..., 'intercept': ...}.

        Returns
        -------
        dict or None
            If `return_params=True`, returns fitted/used params; otherwise None.
        """
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        # --- Checks ---
        for col in (to_fill, to_use):
            if col not in self.data.columns:
                raise KeyError(f"Column '{col}' not found in data.")

        if clear:
            self._reset_meta_filled(to_fill)
        self._add_to_meta(to_fill)  # seeds filled[to_fill] from originals

        # Windows
        def _slice_index(bounds: Optional[Tuple[object, object]]) -> pd.Index:
            if bounds is None:
                return self.data.index
            try:
                return self.data.loc[bounds[0]:bounds[1]].index
            except Exception as e:
                raise TypeError("Bounds must match index dtype.") from e

        win_idx = _slice_index(arange)
        if len(win_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return None

        train_idx = _slice_index(train_range if train_range is not None else arange)
        if len(train_idx) == 0:
            train_idx = self.data.index  # fallback to all data

        # --- Build training set for estimation (if requested) ---
        used_ratio = ratio
        used_intercept = 0.0 if zero_intercept and estimate_params else intercept

        if estimate_params:
            # Choose training mask: points with valid pairs (to_fill & to_use)
            y_train = pd.to_numeric(self.data[to_fill].reindex(train_idx), errors="coerce")
            x_train = pd.to_numeric(self.data[to_use].reindex(train_idx), errors="coerce")

            # Optionally restrict to 'original' points (recommended)
            if train_only_original:
                tags = (
                    self.meta_valid[to_fill]
                    .reindex(self.data.index)
                    .fillna("original")
                    .replace({"!!": "original"})
                )
                orig_mask = tags.reindex(train_idx, fill_value="original").eq("original")
            else:
                orig_mask = pd.Series(True, index=train_idx)

            mask = orig_mask & y_train.notna() & x_train.notna()
            x = x_train[mask].astype(float).values.reshape(-1, 1)
            y = y_train[mask].astype(float).values

            if x.shape[0] < max(2, min_train_points):
                wn.warn(
                    f"Not enough training points for estimation (have {x.shape[0]}, need ≥ {min_train_points}). "
                    "Falling back to provided ratio/intercept.",
                    RuntimeWarning,
                )
            else:
                # Robust or OLS fit
                try:
                    if robust:
                        # Try RANSAC; fallback to OLS if sklearn unavailable
                        from sklearn.linear_model import RANSACRegressor, LinearRegression  # type: ignore
                        base = LinearRegression(fit_intercept=not zero_intercept)
                        ransac = RANSACRegressor(base_estimator=base, random_state=0)
                        ransac.fit(x, y)
                        coef = getattr(ransac.estimator_, "coef_", np.array([np.nan]))[0]
                        intercept_est = getattr(ransac.estimator_, "intercept_", 0.0)
                        used_ratio = float(coef)
                        used_intercept = 0.0 if zero_intercept else float(intercept_est)
                    else:
                        # OLS via numpy
                        if zero_intercept:
                            # y = a * x (no intercept)
                            denom = float(np.dot(x.ravel(), x.ravel()))
                            used_ratio = float(np.dot(x.ravel(), y) / denom) if denom != 0.0 else 0.0
                            used_intercept = 0.0
                        else:
                            # y = a * x + b
                            # polyfit handles numerics well; returns [a, b]
                            a, b = np.polyfit(x.ravel(), y, 1)
                            used_ratio = float(a)
                            used_intercept = float(b)
                except Exception as e:
                    wn.warn(f"Parameter estimation failed ({e}). Falling back to provided ratio/intercept.", RuntimeWarning)

        # Final sanity for params if still None
        if used_ratio is None or not np.isfinite(used_ratio):
            raise ValueError("`ratio` must be provided or successfully estimated.")
        if not np.isfinite(used_intercept):
            used_intercept = 0.0

        # --- Determine fill targets in the application window ---
        if only_checked:
            # tags_valid = (
            #     self.meta_valid[to_fill]
            #     .reindex(self.data.index)
            #     .fillna("original")
            #     .replace({"!!": "original"})
            # )
            tags_valid = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            eligible = tags_valid.reindex(win_idx, fill_value="original").eq("filtered")
        else:
            eligible = pd.Series(True, index=win_idx)

        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return {"ratio": used_ratio, "intercept": used_intercept} if return_params else None

        # Compute replacements from driver
        x_driver = pd.to_numeric(self.data[to_use].reindex(win_idx), errors="coerce")
        replacements = used_ratio * x_driver + used_intercept

        target_idx = eligible.index[eligible & replacements.notna()]
        if len(target_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return {"ratio": used_ratio, "intercept": used_intercept} if return_params else None

        # --- Apply & tag ---
        if to_fill not in self.filled.columns:
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce")

        self.filled.loc[target_idx, to_fill] = replacements.loc[target_idx]

        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )
        self.meta_filled.loc[target_idx, to_fill] = "filled_ratio"

        if plot:
            self.plot_analysed(to_fill)

        if return_params:
            return {"ratio": used_ratio, "intercept": used_intercept}
        return None



    def fill_missing_standard(
        self,
        to_fill: str,
        arange: Optional[Tuple[object, object]] = None,
        *,
        only_checked: bool = True,
        plot: bool = False,
        clear: bool = False,
    ) -> None:
        """
        Fill missing/filtered values in `to_fill` using the average daily profile
        computed by `calc_daily_profile()`.

        - Uses profile['avg'] matched by time-of-day.
        - Preserves original values; tags filled points in meta_filled as 'filled_average_profile'.
        """
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")
        if to_fill not in getattr(self, "daily_profile", {}):
            raise KeyError(
                f"No daily profile found for '{to_fill}'. Run calc_daily_profile() first."
            )

        if clear:
            self._reset_meta_filled(to_fill)
        self._add_to_meta(to_fill)  # seeds self.filled[to_fill] with originals; filtered -> NaN

        # Window to operate in
        if arange is None:
            win_idx = self.data.index
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` bounds must match the index dtype.") from e
            if len(win_idx) == 0:
                if plot:
                    self.plot_analysed(to_fill)
                return

        # Eligibility
        if only_checked:
            # tags_valid = (
            #     self.meta_valid[to_fill]
            #     .reindex(self.data.index)
            #     .fillna("original")
            #     .replace({"!!": "original"})
            # )
            tags_valid = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            eligible = tags_valid.reindex(win_idx, fill_value="original").eq("filtered")
        else:
            eligible = pd.Series(True, index=win_idx)

        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        # ---- Build profile seconds from 'time_of_day' (strings like 'HH:MM:SS') ----
        prof = self.daily_profile[to_fill][["avg"]].copy()
        # Convert index (time-of-day) to Timedelta and then to seconds (float)
        td = pd.to_timedelta(pd.Index(prof.index.astype(str)), errors="coerce")
        prof = prof.assign(secs=pd.Series(td.total_seconds(), index=prof.index))
        prof = prof.dropna(subset=["secs", "avg"]).sort_values("secs")
        if prof.empty:
            wn.warn("Daily profile has no valid time-of-day entries.", RuntimeWarning, stacklevel=2)
            return

        prof_secs = prof["secs"].to_numpy(dtype=float)
        prof_vals = pd.to_numeric(prof["avg"], errors="coerce").to_numpy(dtype=float)
        mask_ok = np.isfinite(prof_secs) & np.isfinite(prof_vals)
        prof_secs = prof_secs[mask_ok]
        prof_vals = prof_vals[mask_ok]
        if prof_secs.size == 0:
            wn.warn("Daily profile contains no finite (secs, avg) pairs.", RuntimeWarning, stacklevel=2)
            return

        # ---- Compute target seconds for eligible indices ----
        elig_idx = eligible.index[eligible]
        if isinstance(self.data.index, pd.DatetimeIndex):
            # seconds since midnight (float, includes microseconds)
            td_since_midnight = elig_idx - elig_idx.normalize()
            tgt_secs = td_since_midnight.total_seconds().astype(float)
        else:
            # numeric index assumed to be day + fraction
            vals = pd.Index(elig_idx).astype(float)
            frac = vals - np.floor(vals)
            tgt_secs = (frac * 86400.0).astype(float)

        # ---- Interpolate profile onto target seconds ----
        # Outside [min(prof_secs), max(prof_secs)] → NaN
        filled_vals = np.interp(
            tgt_secs,
            prof_secs,
            prof_vals,
            left=np.nan,
            right=np.nan,
        )
        # Keep only finite results
        ok = np.isfinite(filled_vals)
        if not ok.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        to_write_idx = elig_idx[ok]
        to_write_vals = filled_vals[ok]

        # ---- Write to filled & tag meta ----
        if to_fill not in self.filled.columns:
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce")

        self.filled.loc[to_write_idx, to_fill] = to_write_vals

        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )
        self.meta_filled.loc[to_write_idx, to_fill] = "filled_average_profile"

        if plot:
            self.plot_analysed(to_fill)
    

    

    def fill_missing_model(
        self,
        to_fill: str,
        to_use: Union[pd.Series, pd.DataFrame],
        arange: Optional[Tuple[object, object]] = None,
        *,
        only_checked: bool = True,
        unit: str = "d",
        plot: bool = False,
        clear: bool = False,
        tolerance: Optional[Union[pd.Timedelta, float]] = None,
    ) -> None:
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")

        if isinstance(to_use, pd.DataFrame):
            if to_use.shape[1] != 1:
                raise ValueError("`to_use` DataFrame must have exactly one column.")
            model_series = to_use.iloc[:, 0].copy()
        elif isinstance(to_use, pd.Series):
            model_series = to_use.copy()
        else:
            raise TypeError("`to_use` must be a pandas Series or single-column DataFrame.")
        model_series = pd.to_numeric(model_series, errors="coerce")

        if clear:
            self._reset_meta_filled(to_fill)
        self._add_to_meta(to_fill)

        if arange is None:
            win_idx = self.data.index
        else:
            try:
                win_idx = self.data.loc[arange[0]:arange[1]].index
            except Exception as e:
                raise TypeError("`arange` bounds must match the index dtype.") from e
            if len(win_idx) == 0:
                if plot:
                    self.plot_analysed(to_fill)
                return

        if only_checked:
            # tags_valid = (
            #     self.meta_valid[to_fill]
            #     .reindex(self.data.index)
            #     .fillna("original")
            #     .replace({"!!": "original"})
            # )
            tags_valid = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            eligible = tags_valid.reindex(win_idx, fill_value="original").eq("filtered")
        else:
            eligible = pd.Series(True, index=win_idx)
        if not eligible.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        def _to_relative_numeric(tidx: pd.Index, origin: Optional[pd.Timestamp] = None) -> np.ndarray:
            if not isinstance(tidx, pd.DatetimeIndex):
                raise TypeError("_to_relative_numeric expects a DatetimeIndex.")
            if origin is None:
                origin = tidx.min().normalize()
            delta = (tidx - origin)
            secs = delta.total_seconds()
            if unit in ("sec", "s"):
                return secs
            if unit in ("min", "m"):
                return secs / 60.0
            if unit in ("hr", "h"):
                return secs / 3600.0
            if unit in ("d", "day", "days"):
                return secs / 86400.0
            raise ValueError("`unit` must be one of {'sec','min','hr','d'}.")

        data_idx = self.data.index
        left_idx = eligible.index[eligible]
        left_is_dt = isinstance(data_idx, pd.DatetimeIndex)
        right_is_dt = isinstance(model_series.index, pd.DatetimeIndex)

        # Helper to robustly reset/restore original index without relying on a column literally named "index"
        def _with_orig_index(df: pd.DataFrame, name="_orig_index") -> pd.DataFrame:
            out = df.copy()
            out[name] = out.index
            return out.reset_index(drop=True)

        # ===== Matching cases =====
        if left_is_dt and right_is_dt:
            # Datetime ↔ Datetime
            left_df = pd.DataFrame({"ts": pd.to_datetime(left_idx)}, index=left_idx)
            left_df = _with_orig_index(left_df, name="_orig_index").sort_values("ts")

            right_df = pd.DataFrame(
                {"ts": pd.to_datetime(model_series.index), "yhat": model_series.values}
            ).sort_values("ts")

            tol = tolerance if (isinstance(tolerance, pd.Timedelta) or tolerance is None) else pd.to_timedelta(tolerance)

            matched = pd.merge_asof(
                left_df, right_df, on="ts", direction="nearest", tolerance=tol
            )
            matched = matched.set_index("_orig_index")
            yhat = pd.Series(matched["yhat"].values, index=matched.index)

        elif (not left_is_dt) and (not right_is_dt):
            # Numeric ↔ Numeric
            left_vals = pd.Series(left_idx, index=left_idx).astype(float)
            left_df = pd.DataFrame({"x": left_vals.values}, index=left_vals.index)
            left_df = _with_orig_index(left_df, name="_orig_index").sort_values("x")

            right_vals = pd.Series(model_series.index).astype(float)
            right_df = pd.DataFrame({"x": right_vals.values, "yhat": model_series.values}).sort_values("x")

            tol = None if tolerance is None else float(tolerance)

            matched = pd.merge_asof(
                left_df, right_df, on="x", direction="nearest", tolerance=tol
            )
            matched = matched.set_index("_orig_index")
            yhat = pd.Series(matched["yhat"].values, index=matched.index)

        else:
            # Cross-type
            if left_is_dt:
                origin = pd.to_datetime(
                    min(left_idx.min(),
                        model_series.index.min() if right_is_dt else pd.Timestamp.utcnow())
                )
                left_x = _to_relative_numeric(pd.DatetimeIndex(left_idx), origin=origin)
                if right_is_dt:
                    right_x = _to_relative_numeric(pd.DatetimeIndex(model_series.index), origin=origin)
                else:
                    right_x = pd.Index(model_series.index).astype(float).to_numpy()
            else:
                origin = pd.to_datetime(
                    min(model_series.index.min(),
                        left_idx.min() if left_is_dt else pd.Timestamp.utcnow())
                )
                right_x = _to_relative_numeric(pd.DatetimeIndex(model_series.index), origin=origin)
                left_x = pd.Index(left_idx).astype(float).to_numpy()

            left_df = pd.DataFrame({"x": left_x}, index=left_idx)
            left_df = _with_orig_index(left_df, name="_orig_index").sort_values("x")
            right_df = pd.DataFrame({"x": right_x, "yhat": model_series.values}).sort_values("x")

            tol = None
            if tolerance is not None:
                if isinstance(tolerance, pd.Timedelta):
                    total_sec = tolerance.total_seconds()
                    tol = {"sec": total_sec, "s": total_sec,
                        "min": total_sec/60, "m": total_sec/60,
                        "hr": total_sec/3600, "h": total_sec/3600,
                        "d": total_sec/86400, "day": total_sec/86400, "days": total_sec/86400}[unit]
                else:
                    tol = float(tolerance)

            matched = pd.merge_asof(
                left_df, right_df, on="x", direction="nearest", tolerance=tol
            )
            matched = matched.set_index("_orig_index")
            yhat = pd.Series(matched["yhat"].values, index=matched.index)

        # keep only finite predictions
        yhat = pd.to_numeric(yhat, errors="coerce")
        valid_targets = yhat.index[yhat.notna()]
        if len(valid_targets) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return

        if to_fill not in self.filled.columns:
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce")

        self.filled.loc[valid_targets, to_fill] = yhat.loc[valid_targets].values

        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )
        self.meta_filled.loc[valid_targets, to_fill] = "filled_model"

        if plot:
            self.plot_analysed(to_fill)

   

    def fill_missing_daybefore(
        self,
        to_fill: str,
        arange: Tuple[object, object],
        range_to_replace: List[float] = [1.0, 4.0],  # min & max gap length in *days*
        *,
        only_checked: bool = True,
        plot: bool = False,
        clear: bool = False,
    ) -> None:
        """
        Fill 'filtered' gaps in `to_fill` by copying values from the *previous day*
        (same time-of-day), using the best available source:
        - Prefer `self.filled[to_fill]` in the previous-day window if it has data
        - Otherwise fall back to `self.data[to_fill]` (optionally only 'original' points)

        Requires equidistant sampling.

        Parameters
         ----------
         to_fill : str
             Column to fill.
         arange : (start, end)
             Window to apply the method to. Must start at least one day after the series start.
         range_to_replace : [min_days, max_days], default [1, 4]
             Minimum and maximum consecutive gap length (in *days*) that will be filled.
             Converted to number of points using the previous-day sample count.
         only_checked : bool, default True
             If True, only fill rows tagged 'filtered' in meta_valid[to_fill].
             If False, consider all rows in `arange`.
         plot : bool, default False
             Plot diagnostic chart with plot_analysed.
         clear : bool, default False
             Reset meta_filled[to_fill] from meta_valid[to_fill] before filling.
        """
        self._plot = "filled"
        try:
            self._filling_warning()
        except TypeError:
            pass

        if to_fill not in self.data.columns:
            raise KeyError(f"Column '{to_fill}' not found in data.")
        if arange is None or len(arange) != 2:
            raise ValueError("`arange` must be a (start, end) tuple; this method requires a bounded window.")

        # Ensure scaffolding: meta_filled & filled
        if clear:
            self._reset_meta_filled(to_fill)
        self._add_to_meta(to_fill)  # seeds self.filled[to_fill] with originals; filtered → NaN

        idx = self.data.index
        start, end = arange
        try:
            win_idx = self.data.loc[start:end].index
        except Exception as e:
            raise TypeError("`arange` bounds must be sliceable on the current index.") from e
        if len(win_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return

        # at least one day before `start`
        if isinstance(idx, pd.DatetimeIndex):
            start_ts = pd.to_datetime(start)
            if start_ts - pd.Timedelta(days=1) < idx.min():
                raise IndexError("No previous-day data available; choose a later `arange` start.")
            prev_slice = slice(start_ts - pd.Timedelta(days=1), start_ts)
        else:
            start_num = float(start)
            if start_num - 1.0 < float(idx.min()):
                raise IndexError("No previous-day data available; choose a later `arange` start.")
            prev_slice = slice(start_num - 1.0, start_num)

        # --- choose source series for previous-day profile ---
        # prefer self.filled[to_fill] if it has non-NaN in the prev window; else use self.data[to_fill]
        src_filled = self.filled.get(to_fill, pd.Series(index=self.data.index, dtype=float)).loc[prev_slice]
        use_filled = src_filled.notna().any()

        if use_filled:
            prev_series = src_filled
        else:
            src_data = pd.to_numeric(self.data[to_fill], errors="coerce").loc[prev_slice].copy()
            # if meta_valid exists, prefer original points to build the profile
            if hasattr(self, "meta_valid") and to_fill in self.meta_valid:
                mv_prev = self.meta_valid[to_fill].reindex(src_data.index)
                prev_series = src_data.where(mv_prev.eq("original"))
                # if that wiped everything, fall back to raw data
                if prev_series.dropna().empty:
                    prev_series = src_data
            else:
                prev_series = src_data

        if prev_series.dropna().empty:
            raise ValueError("Previous-day window has no usable samples to form a profile.")

        # Normalize to per-day key & compute day_size
        if isinstance(idx, pd.DatetimeIndex):
            prev_series = prev_series.dropna()
            prev_key = prev_series.index.time
            day_before = pd.DataFrame({"data": prev_series.values}, index=pd.Index(prev_key, name="tod"))
            day_before = day_before[~day_before.index.duplicated(keep="first")]
            day_size = len(day_before)
        else:
            prev_series = prev_series.dropna()
            prev_vals = pd.Index(prev_series.index).astype(float).to_numpy()
            frac_prev = (prev_vals - np.floor(prev_vals))
            order = np.argsort(frac_prev)
            frac_sorted = frac_prev[order]
            vals_sorted = prev_series.values[order]
            _, uniq_idx = np.unique(frac_sorted, return_index=True)
            day_before = pd.DataFrame({"frac": frac_sorted[uniq_idx], "data": vals_sorted[uniq_idx]}) \
                            .set_index("frac").sort_index()
            day_size = len(day_before)

        if day_size == 0:
            raise ValueError("Previous-day profile has no valid samples.")

        # gap-size thresholds in points
        min_pts = int(np.floor(range_to_replace[0] * day_size))
        max_pts = int(np.floor(range_to_replace[1] * day_size))
        if max_pts < 1:
            wn.warn("`range_to_replace` converts to <1 point; nothing will be filled.", RuntimeWarning, stacklevel=2)
            if plot:
                self.plot_analysed(to_fill)
            return

        # Eligible indices within window
        if only_checked:
            # mv = (
            #     self.meta_valid[to_fill]
            #     .reindex(self.data.index)
            #     .fillna("original")
            #     .replace({"!!": "original"})
            # )
            mv = (
                self.meta_filled[to_fill]
                .reindex(self.data.index)
                .fillna("original")
                .replace({"!!": "original"})
            )
            cand = mv.loc[start:end].eq("filtered")
        else:
            cand = pd.Series(True, index=win_idx)

        if not cand.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        # Run-length filter for gap lengths
        labels = (
            self.meta_valid[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
            .loc[start:end]
        )
        grp_ids = (labels != labels.shift()).cumsum()
        run_sizes = grp_ids.map(grp_ids.value_counts())
        mask_len_ok = (run_sizes >= min_pts) & (run_sizes <= max_pts)
        to_replace_idx = labels.index[cand & mask_len_ok]
        if len(to_replace_idx) == 0:
            if plot:
                self.plot_analysed(to_fill)
            return

        # Build replacement values by aligning time-of-day / fraction-of-day to `day_before`
        if isinstance(idx, pd.DatetimeIndex):
            target_tod = pd.Index(to_replace_idx).time
            src = pd.Series(day_before["data"].values, index=day_before.index, dtype=float)
            fill_vals = pd.Series(index=to_replace_idx, dtype=float)
            fill_vals.loc[:] = pd.Series(target_tod, index=to_replace_idx).map(src).to_numpy()
        else:
            tgt_vals = pd.Index(to_replace_idx).astype(float).to_numpy()
            tgt_frac = (tgt_vals - np.floor(tgt_vals))
            src_x = day_before.index.to_numpy(dtype=float)
            src_y = day_before["data"].to_numpy(dtype=float)
            fill_arr = np.interp(tgt_frac, src_x, src_y, left=np.nan, right=np.nan)
            fill_vals = pd.Series(fill_arr, index=to_replace_idx, dtype=float)

        ok = np.isfinite(fill_vals.values)
        if not ok.any():
            if plot:
                self.plot_analysed(to_fill)
            return

        valid_idx = fill_vals.index[ok]
        valid_vals = fill_vals.values[ok]

        # Write & tag
        if to_fill not in self.filled.columns:
            self.filled[to_fill] = pd.to_numeric(self.data[to_fill], errors="coerce")

        self.filled.loc[valid_idx, to_fill] = valid_vals
        self.meta_filled[to_fill] = (
            self.meta_filled[to_fill]
            .reindex(self.data.index)
            .fillna("original")
            .replace({"!!": "original"})
        )
        self.meta_filled.loc[valid_idx, to_fill] = "filled_profile_day_before"

        if plot:
            self.plot_analysed(to_fill)




#     ###############################################################################
#     ##                          RELIABILITY FUNCTIONS                            ##
#     ###############################################################################
    def _create_gaps(
        self,
        data_name: str,
        range_: tuple,
        number: int,
        max_size: int,
        *,
        reset: bool = False,
        user_output: bool = False,
        random_state: Optional[int] = None,
    ) -> pd.Index:
        """
        Randomly creates artificial gaps by tagging 'filtered' in meta_valid[data_name]
        and setting the *values of that column* to 0 at those indices.

        Returns
        -------
        pd.Index
            The index labels that were tagged as gaps.
        """
       

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")

        if reset:
            self._reset_meta_valid(data_name)

        # Ensure meta_valid exists and has the target column
        if not hasattr(self, "meta_valid") or not isinstance(self.meta_valid, pd.DataFrame):
            self.meta_valid = pd.DataFrame(index=self.data.index)
        self.meta_valid = self.meta_valid.reindex(self.data.index)
        if data_name not in self.meta_valid.columns:
            self.meta_valid[data_name] = "original"

        # Slice window by labels safely
        try:
            window = self.data.loc[range_[0]:range_[1]]
        except Exception as e:
            raise TypeError(
                "Slicing not possible for given index type and range_. "
                "Ensure range_ matches the index label type."
            ) from e

        if window.empty:
            raise ValueError("Selected `range_` yields an empty window; adjust the bounds.")

        # Convert window labels to absolute integer positions
        idx_all = self.data.index
        pos_window = idx_all.get_indexer(window.index)
        pos_window = pos_window[pos_window >= 0]

        if number <= 0 or max_size <= 0 or len(pos_window) < 2:
            return pd.Index([])

        # Random engine (support both RNG and RandomState)
        rng = np.random.RandomState(random_state) if random_state is not None else np.random

        low_val = int(pos_window.min())
        high_val = int(pos_window.max())

        # Choose random start positions within window
        starts = rng.randint(low_val, high_val, size=number)

        # Random lengths (>=1 and <= max_size)
        lengths = rng.randint(1, max_size + 1, size=number)

        # Build absolute integer locations, clip to window
        locs_list = [np.arange(s, s + L, dtype=int) for s, L in zip(starts, lengths)]
        if not locs_list:
            return pd.Index([])

        locs = np.unique(np.clip(np.concatenate(locs_list), low_val, high_val))
        gap_index = idx_all[locs]

        # Mutate only the target column (like your original)
        self.data.loc[gap_index, data_name] = 0
        self.meta_valid.loc[gap_index, data_name] = "filtered"

        if user_output:
            counts = self.meta_valid[data_name].value_counts(dropna=False)
            n_total = len(self.meta_valid)
            n_orig = int(counts.get("original", 0))
            left_pct = (n_orig * 100.0) / max(1, n_total)
            print(f"{left_pct:.2f}% of datapoints left after creating gaps")

        return pd.Index(gap_index)
   

    def _calculate_filling_error(
        self,
        data_name: str,
        filling_function: Union[str, Callable[..., Any]],
        test_data_range: Tuple[object, object],
        *,
        nr_small_gaps: int = 0,
        max_size_small_gaps: int = 0,
        nr_large_gaps: int = 0,
        max_size_large_gaps: int = 0,
        random_state: Optional[int] = None,
        **options: Dict[str, Any],
    ) -> Optional[float]:
        """
        Create artificial gaps in a copy, run a filling method, compare to original, return % error.
        """
        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found.")
        
        if "to_fill" not in options or options["to_fill"] is None:
            options["to_fill"] = data_name

        if "arange" not in options or not isinstance(options["arange"], (tuple, list)) or len(options["arange"]) != 2:
            # raise ValueError("`options` must include 'arange' = (start, end).")
            options['arange'] = (self.data.index[0], self.data.index[-1])

        to_fill: str = options.get("to_fill", data_name)

        # Build test copies
        s, e = test_data_range
        try:
            orig = self.__class__(self.data.loc[s:e].copy(),
                                timedata_column=getattr(self, "timename", "index"),
                                data_type=getattr(self, "data_type", None),
                                experiment_tag=getattr(self, "tag", None),
                                time_unit=getattr(self, "time_unit", None))
            gaps = self.__class__(self.data.loc[s:e].copy(),
                                timedata_column=getattr(self, "timename", "index"),
                                data_type=getattr(self, "data_type", None),
                                experiment_tag=getattr(self, "tag", None),
                                time_unit=getattr(self, "time_unit", None))
        except Exception as ex:
            raise TypeError("`test_data_range` does not align with index labels.") from ex

        if orig.data.empty:
            return None

        # Ensure meta frames exist & aligned
        for attr in ("meta_valid", "meta_filled"):
            frame = getattr(gaps, attr, None)
            if not isinstance(frame, pd.DataFrame):
                frame = pd.DataFrame(index=gaps.data.index)
            else:
                frame = frame.reindex(gaps.data.index)
            setattr(gaps, attr, frame)

        if to_fill not in gaps.meta_valid.columns:
            gaps.meta_valid[to_fill] = "original"
        if to_fill not in gaps.meta_filled.columns:
            gaps.meta_filled[to_fill] = gaps.meta_valid[to_fill].copy()

        # Create highs info if needed by your warnings (kept for parity with original)
        try:
            gaps.get_highs(data_name, 0.9, [s, e])
        except Exception:
            pass

        # Create gaps (small and/or large)
        tagged_all = pd.Index([])
        a_start, a_end = options["arange"]
        if nr_small_gaps > 0:
            idx_small = gaps._create_gaps(
                data_name, (a_start, a_end), nr_small_gaps, max_size_small_gaps,
                reset=True, user_output=False, random_state=random_state
            )
            tagged_all = tagged_all.union(idx_small)
        if nr_large_gaps > 0:
            idx_large = gaps._create_gaps(
                data_name, (a_start, a_end), nr_large_gaps, max_size_large_gaps,
                reset=False, user_output=False, random_state=None if random_state is None else random_state + 1
            )
            tagged_all = tagged_all.union(idx_large)

        if tagged_all.empty:
            return None

        # Build filled as a full copy of data, then plant NaNs at artificial gaps (for to_fill only)
        gaps.filled = gaps.data.apply(pd.to_numeric, errors="coerce").copy()
        gaps.filled.loc[tagged_all, to_fill] = np.nan
        gaps.meta_filled[to_fill] = gaps.meta_valid[to_fill].copy()

        # Run the filler (callable or method name)
        if callable(filling_function):
            filling_function(gaps, **options)
        else:
            if not hasattr(gaps, filling_function):
                raise ValueError(f"Filling method '{filling_function}' not found.")
            getattr(gaps, filling_function)(**options)

        # Choose indices to score
        if to_fill in gaps.meta_filled.columns:
            mf = gaps.meta_filled[to_fill].astype(str)
            filled_idx = mf.index[mf.str.startswith("filled_")]
            score_idx = tagged_all.intersection(filled_idx)
        else:
            score_idx = pd.Index([])

        # Fallback: any artificial gaps that became non-NaN in the result
        if score_idx.empty:
            non_nan = gaps.filled.loc[tagged_all, to_fill].dropna().index
            score_idx = non_nan

        if score_idx.empty:
            return None

        # Compute percent error
        o = pd.to_numeric(orig.data.loc[score_idx, to_fill], errors="coerce").astype(float)
        p = pd.to_numeric(gaps.filled.loc[score_idx, to_fill], errors="coerce").astype(float)
        valid = (~o.replace([np.inf, -np.inf], np.nan).isna()) & (~p.replace([np.inf, -np.inf], np.nan).isna())
        if not valid.any():
            return None

        o = o[valid]; p = p[valid]

        if (o != 0).any():
            err = (np.abs(p - o) / np.where(o == 0, np.nan, np.abs(o))) * 100.0
            err = err.replace([np.inf, -np.inf], np.nan).dropna()
            return float(err.mean()) if not err.empty else None
        else:
            eps = max(1e-9, float(np.nanmean(np.abs(o))))
            err = (np.abs(p - o) / eps) * 100.0
            return float(np.nanmean(err))


    def check_filling_error(
        self,
        nr_iterations: int,
        data_name: str,
        filling_function: Union[str, Callable[..., Any]],
        test_data_range: Tuple[object, object],
        *,
        nr_small_gaps: int = 0,
        max_size_small_gaps: int = 0,
        nr_large_gaps: int = 0,
        max_size_large_gaps: int = 0,
        random_state: Optional[int] = None,
        **options,
    ) -> None:
        """
        Run repeated artificial-gap tests and store the average filling error (%) in self.filling_error.
        """
        if nr_small_gaps == 0 and nr_large_gaps == 0:
            raise ValueError("Specify nr_small_gaps and/or nr_large_gaps > 0.")

        errors: list[float] = []

        prev_fill_warn = getattr(self, "_filling_warning_issued", False)
        prev_rain_warn = getattr(self, "_rain_warning_issued", False)
        self._filling_warning_issued = True
        self._rain_warning_issued = True

        try:
            for i in range(nr_iterations):
                seed = None if random_state is None else int(random_state + i)
                err = self._calculate_filling_error(
                    data_name,
                    filling_function,
                    test_data_range,
                    nr_small_gaps=nr_small_gaps,
                    max_size_small_gaps=max_size_small_gaps,
                    nr_large_gaps=nr_large_gaps,
                    max_size_large_gaps=max_size_large_gaps,
                    random_state=seed,
                    **options,
                )
                if err is not None and np.isfinite(err):
                    errors.append(float(err))

            if len(errors) == 0:
                raise ValueError(
                    "No valid filling error could be computed. Check `arange`, gap sizes, "
                    "method parameters, and ensure your filling method actually imputes values."
                )

            avg = float(np.mean(errors))

            # ensure table exists and row exists
            if not hasattr(self, "filling_error") or not isinstance(self.filling_error, pd.DataFrame):
                self.filling_error = pd.DataFrame(columns=["imputation error [%]"])
            if "imputation error [%]" not in self.filling_error.columns:
                self.filling_error["imputation error [%]"] = np.nan

            self.filling_error.loc[data_name, "imputation error [%]"] = avg
            # print(
            #     "Average deviation of imputed points from the original ones is "
            #     f"{avg:.2f}%. This value is saved in self.filling_error."
            # )
        finally:
            # restore flags
            self._filling_warning_issued = prev_fill_warn
            self._rain_warning_issued = prev_rain_warn

            n_attempts = nr_iterations
            n_success = len(errors)
            mean_err = float(np.mean(errors)) if errors else np.nan
            std_err = float(np.std(errors, ddof=1)) if len(errors) > 1 else np.nan
            success_rate = (n_success / n_attempts) if n_attempts > 0 else np.nan

            row = {
                "method": filling_function,
                "mean_error_pct": mean_err,
                "std_error_pct": std_err,
                "n_success": n_success,
                "n_attempts": n_attempts,
                "success_rate": success_rate,
            }
            summary = pd.Series(row)

            return summary



    def compare_filling_methods(
        self,
        data_name: str,
        method_specs: Dict[str, Dict[str, Any]],
        test_data_range: Tuple[object, object],
        *,
        nr_iterations: int = 5,
        nr_small_gaps: int = 0,
        max_size_small_gaps: int = 0,
        nr_large_gaps: int = 0,
        max_size_large_gaps: int = 0,
        random_state: Optional[int] = None,
        return_errors: bool = False,
        plot: bool = False,
    ) -> pd.DataFrame:
        """
        Compare multiple filling methods by computing imputation error (in %) using
        the artificial-gap test harness.

        Parameters
        ----------
        data_name : str
            Target column name whose gap-filling performance is evaluated.
        method_specs : dict[str, dict]
            Mapping from a human-readable method label to a spec dict.
            Each spec dict must contain:
            - "filling_function": str | callable
                The name of the filling method (on `self`) or a callable.
            - Any additional keyword options required by that filling method
                (e.g., arange, range_, to_use, ratio, etc.).
            Notes:
            - If a method requires `to_fill` and you don’t supply it, this
                function auto-sets `to_fill = data_name`.
            - If a method *requires* `arange` (no default), you must supply it
                in that method’s options; otherwise it is *not* required here.
        test_data_range : (start, end)
            Label-based bounds (like in `.loc[start:end]`) to define the window
            where artificial gaps are created and evaluation happens.
        nr_iterations : int, default 5
            How many random trials (new gap patterns) per method.
        nr_small_gaps, max_size_small_gaps : int
            Number and maximum size (in samples) of small gaps per iteration.
        nr_large_gaps, max_size_large_gaps : int
            Number and maximum size (in samples) of large gaps per iteration.
        random_state : Optional[int], default None
            Seed base for reproducibility across iterations/methods.
        return_errors : bool, default False
            If True, includes a column with the list of per-iteration errors.
        plot : bool, default False
            If True, show a simple bar chart of average error (lower is better).

        Returns
        -------
        pd.DataFrame
            Columns:
            - method: label
            - mean_error_pct
            - std_error_pct
            - n_success
            - n_attempts
            - success_rate
            - (optional) errors: list of per-iteration errors
            Sorted by mean_error_pct ascending (best first).
        """
        if nr_small_gaps == 0 and nr_large_gaps == 0:
            raise ValueError(
                "Please specify at least one of nr_small_gaps or nr_large_gaps > 0."
            )

        rows: List[Dict[str, Any]] = []

        # Iterate methods
        for label, spec in method_specs.items():
            if "filling_function" not in spec:
                raise ValueError(f"method_specs['{label}'] must include 'filling_function'.")

            filling_function = spec["filling_function"]

            # Prepare per-iteration results
            errors: List[float] = []
            # Iterate trials
            for i in range(nr_iterations):
                # ensure a changing seed per method/iteration (but deterministic if base provided)
                rs = (None if random_state is None else (random_state + hash(label) + i) % (2**31 - 1))

                try:
                    err = self._calculate_filling_error(
                        data_name=data_name,
                        filling_function=filling_function,
                        test_data_range=test_data_range,
                        nr_small_gaps=nr_small_gaps,
                        max_size_small_gaps=max_size_small_gaps,
                        nr_large_gaps=nr_large_gaps,
                        max_size_large_gaps=max_size_large_gaps,
                        random_state=rs,
                        **{k: v for k, v in spec.items() if k != "filling_function"},
                    )
                except Exception:
                    # If a single iteration fails hard (e.g., bad options),
                    # treat as no result for this iteration.
                    err = None

                if err is not None and np.isfinite(err):
                    errors.append(float(err))

            n_attempts = nr_iterations
            n_success = len(errors)
            mean_err = float(np.mean(errors)) if errors else np.nan
            std_err = float(np.std(errors, ddof=1)) if len(errors) > 1 else np.nan
            success_rate = (n_success / n_attempts) if n_attempts > 0 else np.nan

            row = {
                "method": label,
                "mean_error_pct": mean_err,
                "std_error_pct": std_err,
                "n_success": n_success,
                "n_attempts": n_attempts,
                "success_rate": success_rate,
            }
            if return_errors:
                row["errors"] = errors
            rows.append(row)

        summary = pd.DataFrame(rows).sort_values("mean_error_pct", na_position="last").reset_index(drop=True)

        if plot and not summary.empty and summary["mean_error_pct"].notna().any():
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(summary))
            ax.bar(x, summary["mean_error_pct"].fillna(0.0))
            ax.set_xticks(x)
            ax.set_xticklabels(summary["method"], rotation=30, ha="right")
            ax.set_ylabel("Mean error (%)")
            ax.set_title(f"Filling method comparison for '{data_name}'")
            # Optional error bars if std available
            if summary["std_error_pct"].notna().any():
                ax.errorbar(
                    x,
                    summary["mean_error_pct"].fillna(0.0),
                    yerr=summary["std_error_pct"].fillna(0.0),
                    fmt="none",
                    capsize=4,
                )
            ax.grid(alpha=0.3, axis="y")
            plt.tight_layout()

        return summary
