# -*- coding: utf-8 -*-
"""
Class_HydroData provides functionalities for handling data obtained in the context of (waste)water treatment.
Copyright (C) 2025 Chaim De Mulder, Saba Daneshgar, BIOMATH, Ghent University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.
"""
from __future__ import annotations


import os
import pandas as pd
import scipy as sp
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt   
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings as wn
from pathlib import Path

from matplotlib.colors import ListedColormap, BoundaryNorm


from typing import Dict, Optional, Sequence, Union, Tuple, Literal, Any, Hashable, Iterable, List
import logging
import json
import time

try:
    import yaml  
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

logger = logging.getLogger(__name__)

class HydroData:
    """
    Class for handling (waste)water related data

    Parameters
    ----------
    data : pd.DataFrame | dict | sequence
        Tabular data. If `timedata_column='index'`, the DataFrame index is used as time;
        otherwise `timedata_column` must exist in the input.
    timedata_column : str, default 'index'
        Name of the column containing time data, or 'index' to use the DataFrame index.
    data_type : str, default 'WWTP'
        Type/category of the dataset.
    experiment_tag : str, default 'No tag given'
        Tag identifying the experiment (date, code, etc.).
    time_unit : str | None, default None
        Unit of the provided time values (informational; not enforced).
    units : dict[str, str], default {}
        Mapping of column name -> unit string.
    copy : bool, default True
        If True, the input data is defensively copied.

    Attributes
    ----------
    data : pd.DataFrame
        Stored data (copied if requested). The time column is NOT converted to index here.
    timename : str
        'index' or the name of the column used for time.
    time : pd.Index | np.ndarray
        The raw time vector: the DataFrame index (if `timedata_column='index'`)
        or the values of the specified column.
    columns : np.ndarray
        Array of column names from `self.data`.
    data_type : str
        Dataset type/category.
    tag : str
        Experiment tag.
    time_unit : str | None
        Time unit (informational).
    meta_valid : pd.DataFrame
        Empty per-timestamp metadata frame aligned to `self.data.index`.
    units : dict[str, str]
        Units by column.
    """

    data: pd.DataFrame
    timename: str
    time: Union[pd.Index, np.ndarray]
    columns: np.ndarray
    data_type: str
    tag: str
    time_unit: Optional[str]
    meta_valid: pd.DataFrame
    units: Dict[str, str]

    def __init__(
        self,
        data: Union[pd.DataFrame, Dict, Sequence],
        timedata_column: str = "index",
        data_type: str = "WWTP",
        experiment_tag: str = "No tag given",
        time_unit: Optional[str] = None,
        units: Optional[Dict[str, str]] = None,
        *,
        copy: bool = True,
    ) -> None:
        if isinstance(data, pd.DataFrame):
            df = data.copy() if copy else data
        else:
            try:
                df = pd.DataFrame(data)
                if copy:
                    df = df.copy()
            except Exception as e:
                raise TypeError("Input data is not convertible to a pandas DataFrame.") from e

        if df.empty:
            logger.warning("HydroData initialized with an empty DataFrame.")
        self.data = df

        if timedata_column == "index":
            self.timename = "index"
            self.time = self.data.index
        else:
            if timedata_column not in self.data.columns:
                raise KeyError(
                    f"timedata_column '{timedata_column}' not found in data columns: "
                    f"{list(self.data.columns)}"
                )
            self.timename = timedata_column
            self.time = self.data[timedata_column].to_numpy().ravel()

        self.columns = np.array(self.data.columns, dtype=object)

        self.data_type = data_type
        self.tag = experiment_tag
        self.time_unit = time_unit
        self.meta_valid = pd.DataFrame(index=self.data.index)
        self.units = dict(units or {})


    def __repr__(self) -> str:
        nrows, ncols = self.data.shape
        tname = self.timename
        idx_name = getattr(self.data.index, "name", None)
        return (
            f"HydroData(n_rows={nrows}, n_cols={ncols}, time='{tname}', "
            f"index_name='{idx_name}', data_type='{self.data_type}', tag='{self.tag}')"
        )

    @property
    def index(self) -> pd.Index:
        """Return the current DataFrame index."""
        return self.data.index

    def copy(self) -> "HydroData":
        """Deep copy the HydroData (data & shallow metadata copy)."""
        return HydroData(
            data=self.data.copy(),
            timedata_column=self.timename,
            data_type=self.data_type,
            experiment_tag=self.tag,
            time_unit=self.time_unit,
            units=self.units.copy(),
            copy=True,
        )


    def set_tag(self, tag: str) -> None:
        """
        Set the experiment tag for this HydroData object.

        Parameters
        ----------
        tag : str
            New experiment tag.

        Returns
        -------
        None
        """
        if not isinstance(tag, str):
            raise TypeError("tag must be a string.")
        self.tag = tag

    def set_units(self, units: Union[pd.DataFrame, dict]) -> None:
        """
        Set the units for the variables.

        Parameters
        ----------
        units : pd.DataFrame or dict
            Units information. Can be provided as a DataFrame (copied directly)
            or as a dict, which will be converted into a DataFrame.

        Returns
        -------
        None
        """
        if isinstance(units, pd.DataFrame):
            self.units = units.copy()
        elif isinstance(units, dict):
            self.units = pd.DataFrame([units])  
        else:
            raise TypeError("units must be a pandas DataFrame or a dict.")

    def set_time_unit(self, unit: str) -> None:
        """
        Set the time unit metadata for this HydroData object.

        Parameters
        ----------
        unit : str
            New time unit.

        Returns
        -------
        None
        """
        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")
        self.time_unit = unit


    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Return the first `n` rows of the data.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return.

        Returns
        -------
        pd.DataFrame
            First `n` rows of the data.
        """
        return self.data.head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """
        Return the last `n` rows of the data.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return.

        Returns
        -------
        pd.DataFrame
            Last `n` rows of the data.
        """
        return self.data.tail(n)

    

    #####################
    ###   FORMATTING
    #####################

    def _update_time(self) -> None:
        """
        Sync `self.time` and realign `meta_valid` to the current data index.
        """
        self.time = self.data.index
        self.meta_valid = pd.DataFrame(index=self.data.index)

    def fill_index(
        self,
        arange: Tuple[Union[pd.Timestamp, float, int], Union[pd.Timestamp, float, int]],
        index_type: Literal["datetime", "float", "int", "auto"] = "auto",
        *,
        freq: Optional[pd.Timedelta] = None,   
        step: Optional[float] = None,          
        inclusive: Literal["left", "right", "both", "neither"] = "left",
    ) -> None:
        """
        Fill missing index values within [start, end] assuming equidistant sampling.
        Leaves newly created rows as NaN. 
        """
        wn.warn(
            "fill_index assumes equidistant sampling and inserts missing index values accordingly.",
            RuntimeWarning, stacklevel=2
        )
        if self.data.empty:
            return

        idx = self.data.index
        original_name = idx.name  
        is_datetime = isinstance(idx, pd.DatetimeIndex)

        if index_type == "auto":
            if is_datetime:
                index_type = "datetime"
            else:
                index_type = "float" if np.issubdtype(idx.dtype, np.floating) else "int"

        start, end = arange

        if index_type == "datetime":
            if not is_datetime:
                raise TypeError("Index is not DatetimeIndex; choose index_type='float' or 'int'.")
            if freq is None:
                if len(idx) < 2:
                    raise ValueError("Cannot infer frequency from <2 index points. Provide `freq`.")
                deltas_ns = np.diff(idx.asi8)  
                med_ns = int(np.median(deltas_ns))
                if med_ns <= 0:
                    raise ValueError("Non-increasing timestamps; cannot infer frequency.")
                freq = pd.to_timedelta(med_ns, unit="ns")

            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end)
            fill_index = pd.date_range(start=start_ts, end=end_ts, freq=freq, inclusive=inclusive)
            fill_index = fill_index.rename(original_name)  

        elif index_type in ("float", "int"):
            if len(idx) < 2 and step is None:
                raise ValueError("Cannot infer numeric step from <2 index points. Provide `step`.")
            if step is None:
                diffs = np.diff(idx.values.astype(float))
                med = float(np.median(diffs))
                if med <= 0:
                    raise ValueError("Non-increasing numeric index; cannot infer step.")
                step = med

            start_f = float(start)
            end_f = float(end)
            stop = end_f + (step * 0.5 if inclusive in ("right", "both") else 0.0)
            fill_vals = np.arange(start_f, stop, step, dtype=float)
            if index_type == "int":
                fill_vals = np.rint(fill_vals).astype(int) 
            fill_index = pd.Index(fill_vals, name=original_name)  

        else:
            raise ValueError("index_type must be one of: 'datetime', 'float', 'int', 'auto'.")

        if len(fill_index) == 0:
            return

        fill_block = pd.DataFrame(index=fill_index)
        fill_block = fill_block.reindex(columns=self.data.columns)

        left = self.data.loc[self.data.index < fill_index.min()]
        right = self.data.loc[self.data.index > fill_index.max()]
        mid_existing = self.data.loc[self.data.index.isin(fill_index)]

        combined = pd.concat([left, mid_existing, fill_block, right], axis=0)
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()

        combined.index.name = original_name

        self.data = combined
        self._update_time()
    

    def _reset_meta_valid(
        self,
        data_name: Optional[Union[str, Sequence[str]]] = None,
        *,
        fill_value: str = "original",
        reindex: bool = True,
    ) -> None:
        """
        Reset the `meta_valid` DataFrame.

        Parameters
        ----------
        data_name : str | sequence[str] | None, default None
            - None: reset the entire `meta_valid` to an empty frame aligned to `self.data.index`.
            - str or list of str: ensure those columns exist and set their tags to `fill_value`.
        fill_value : str, default 'original'
            Tag to set when resetting specific columns.
        reindex : bool, default True
            If True, align `meta_valid` index to `self.data.index` before resetting columns.
        """
        if getattr(self, "meta_valid", None) is None:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        elif reindex:
            self.meta_valid = self.meta_valid.reindex(self.data.index)

        if data_name is None:
            self.meta_valid = pd.DataFrame(index=self.data.index)
            return

        cols = [data_name] if isinstance(data_name, str) else list(data_name)

        for col in cols:
            if col not in self.meta_valid.columns:
                self.meta_valid[col] = fill_value
            else:
                self.meta_valid[col] = pd.Series(fill_value, index=self.meta_valid.index)


    
    def drop_index_duplicates(
        self,
        *,
        keep: Literal["first", "last"] = "first",
        sort: bool = False,
        print_number: bool = True,
    ) -> int:
        """
        Drop rows with duplicate index values (keep the first or last occurrence).

        Parameters
        ----------
        keep : {'first','last'}, default 'first'
            Which duplicate to keep for each index value.
        sort : bool, default False
            If True, sort the DataFrame by index after dropping duplicates.

        Returns
        -------
        int
            Number of rows dropped.

        Notes
        -----
        - This keeps the *relative* order of retained rows (unless `sort=True`).
        - If your index is strings/objects, the notion of "first" depends on current row order.
        Convert to a DatetimeIndex/number and/or `sort_index()` for deterministic results.
        """
        if self.data.empty:
            return 0

        idx = self.data.index
        if idx.dtype == "object" or pd.api.types.is_string_dtype(idx):
            wn.warn(
                "Index is object/string-typed; 'first'/'last' are based on current row order. "
                "Consider converting to datetime/numeric and/or using sort_index() for determinism.",
                RuntimeWarning,
                stacklevel=2,
            )

        keep_mask = ~idx.duplicated(keep=keep)

        before = len(self.data)
        self.data = self.data.loc[keep_mask]
        if hasattr(self, "meta_valid") and isinstance(self.meta_valid, pd.DataFrame):
            self.meta_valid = self.meta_valid.reindex(self.data.index)
        else:
            self.meta_valid = pd.DataFrame(index=self.data.index)

        if sort:
            self.data = self.data.sort_index()
            self.meta_valid = self.meta_valid.sort_index()

        self._update_time()

        dropped = before - len(self.data)
        if print_number:
            print(f'{dropped} rows of duplicate indices have been dropped.')
        # return dropped

    
    def replace(
        self,
        to_replace: Any = None,
        value: Any = None,
        *,
        inplace: bool = False,
        limit: Optional[int] = None,
        regex: bool | dict[str, Any] | None = None,
        method: Optional[str] = None,
    ) -> Optional["HydroData"]:
        """
        Wrapper for pandas.DataFrame.replace with metadata preservation.

        Parameters
        ----------
        to_replace, value, limit, regex, method
            Same semantics as pandas.DataFrame.replace.
        inplace : bool, default False
            If True, modify `self.data` in place and return None.
            If False, return a new HydroData with the replaced data and
            the same metadata (timename, data_type, tag, time_unit, units).

        Returns
        -------
        HydroData | None
            New HydroData when `inplace=False`, otherwise None.
        """
        if inplace:
            self.data.replace(
                to_replace=to_replace,
                value=value,
                limit=limit,
                regex=regex,
                method=method,
                inplace=True,
            )
            
            if hasattr(self, "meta_valid") and isinstance(self.meta_valid, pd.DataFrame):
                self.meta_valid = self.meta_valid.reindex(self.data.index)
            return None

        new_df = self.data.replace(
            to_replace=to_replace,
            value=value,
            limit=limit,
            regex=regex,
            method=method,
            inplace=False,
        )
        return self.__class__(
            data=new_df,
            timedata_column=self.timename,
            data_type=self.data_type,
            experiment_tag=self.tag,
            time_unit=self.time_unit,
            units=self.units.copy() if isinstance(self.units, dict) else self.units,
            copy=True,
        )
    def set_index(
        self,
        keys: Union[Hashable, Iterable[Hashable]],
        *,
        key_is_time: bool = False,
        drop: bool = True,
        inplace: bool = False,
        verify_integrity: bool = False,
        save_prev_index: bool = True,
    ) -> Optional["HydroData"]:
        """
        Extended wrapper around pandas.DataFrame.set_index.

        Parameters
        ----------
        keys : label or list of labels
            Column(s) to set as the new index (passed to pandas).
        key_is_time : bool, default False
            If True, the new index is considered the time axis from now on.
            (We do not coerce strings to datetime automatically here.)
        drop : bool, default True
            Drop the column(s) used as the new index.
        inplace : bool, default False
            If True, modify this object in place and return None.
            If False, return a new HydroData with the updated index.
        verify_integrity : bool, default False
            Check for duplicates in the new index.
        save_prev_index : bool, default True
            If True, saves the current index to `self.prev_index` before changing it.

        Returns
        -------
        HydroData | None
            - Returns a new HydroData if `inplace=False`.
            - Returns None if `inplace=True`.

        Notes
        -----
        - If `key_is_time=True`, this method will reject string-typed time values.
        Convert to datetime first if needed.
        """
        if save_prev_index:
            self.prev_index = self.data.index

        if key_is_time:
            if inplace and self.timename == "index":
                raise IndexError("A time series already resides in the DataFrame index.")
            if isinstance(keys, (str, int)) and keys in self.data.columns:
                sample = self.data[keys].iloc[:1]
                if len(sample) and isinstance(sample.iloc[0], str):
                    raise ValueError('Time values of type "str" cannot be used as index. Convert to datetime first.')

        new_df = self.data.set_index(
            keys=keys,
            drop=drop,
            verify_integrity=verify_integrity,
            inplace=False,
        )

        if inplace:
            self.data = pd.DataFrame(new_df)
            self.columns = np.array(self.data.columns, dtype=object)

            if key_is_time:
                self.timename = "index"
                self.time = self.data.index
            self._update_time()
            return None

        timedata_column = "index" if key_is_time else self.timename

        return self.__class__(
            data=pd.DataFrame(new_df),
            timedata_column=timedata_column,
            data_type=self.data_type,
            experiment_tag=self.tag,
            time_unit=self.time_unit,
            units=self.units.copy() if isinstance(self.units, dict) else self.units,
            copy=True,
        )
    

    def to_float(self, columns: Union[str, Sequence[str]] = "all") -> None:
        """
        Convert values in the given columns to float dtype.

        Parameters
        ----------
        columns : {'all'} or sequence of str, default 'all'
            - 'all': attempt to convert all columns.
            - list/sequence: specific column names to convert.

        Notes
        -----
        - Non-convertible values are coerced to NaN.
        - Columns not found in the DataFrame are ignored with a warning.
        - Updates self.data in place.
        """
        if columns == "all":
            target_cols = list(self.data.columns)
        elif isinstance(columns, str):
            target_cols = [columns]
        else:
            target_cols = list(columns)

        missing = [c for c in target_cols if c not in self.data.columns]
        if missing:
            import warnings
            warnings.warn(
                f"Columns not found in data and will be ignored: {missing}",
                RuntimeWarning,
                stacklevel=2,
            )
            target_cols = [c for c in target_cols if c in self.data.columns]

        for col in target_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce").astype(float)

        self._update_time()

    def to_datetime(
        self,
        time_column: str = "index",
        *,
        fmt: Optional[str] = None,                  
        unit: Optional[str] = None,
        errors: Literal["raise", "coerce", "ignore"] = "raise",
        utc: bool = False,
        dayfirst: bool = False,
        yearfirst: bool = False,
        set_index: bool = True,                     
        drop: bool = True,                          
        index_name: str = "Datetime",
    ) -> None:
        """
        Convert time values to pandas datetime and (optionally) set as the index.

        Parameters
        ----------
        time_column : str, default 'index'
            - 'index': convert the existing DataFrame index.
            - any other column name: convert that column; if set_index=True, make it the index.
        fmt : str | None, default None
            Explicit datetime format for string parsing (passed to pandas to_datetime as `format`).
            If None, pandas will try to infer the format.
        unit : str | None, default None
            Unit for integer/float epoch-like values (e.g., 's','ms','us','ns','D').
        errors : {'raise','coerce','ignore'}, default 'raise'
            Error handling for invalid parsing.
        utc : bool, default False
            If True, return timezone-aware UTC timestamps.
        dayfirst : bool, default False
        yearfirst : bool, default False
        set_index : bool, default True
            If `time_column != 'index'` and True, set that column as the new index after conversion.
        drop : bool, default True
            If `set_index=True`, drop the column after moving it to the index.
        index_name : str, default 'Datetime'
            Name to assign to the datetime index.

        Returns
        -------
        None
            Modifies self.data in place and updates internal time/metadata.
        """
        if self.data.empty:
            self._update_time()
            return

        if time_column == "index":
            idx_vals = self.data.index
            use_unit = unit if np.issubdtype(idx_vals.dtype, np.number) else None
            new_index = pd.to_datetime(
                idx_vals,
                errors=errors,
                format=fmt,       
                unit=use_unit,    
                utc=utc,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                exact=False if fmt is None else True,  
            )
            self.data.index = pd.DatetimeIndex(new_index, name=index_name)
            self.data.sort_index(inplace=True)

        else:
            if time_column not in self.data.columns:
                raise KeyError(f"time_column '{time_column}' not found in data.")
            s = self.data[time_column]
            use_unit = unit if np.issubdtype(s.dtype, np.number) else None

            converted = pd.to_datetime(
                s,
                errors=errors,
                format=fmt,
                unit=use_unit,
                utc=utc,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                exact=False if fmt is None else True,
            )
            self.data[time_column] = converted

            if set_index:
                self.data.set_index(time_column, drop=drop, inplace=True)
                self.data.index = pd.DatetimeIndex(self.data.index, name=index_name)
                self.data.sort_index(inplace=True)
            else:
                self.data.sort_values(by=time_column, inplace=True, kind="mergesort") 

        self._update_time()


    def absolute_to_relative(
        self,
        time_data: str = "index",
        unit: Literal["sec", "s", "min", "m", "hr", "h", "d", "day", "days"] = "d",
        *,
        inplace: bool = True,
        save_abs: bool = True,
        decimals: int = 5,
    ) -> Optional["HydroData"]:
        """
        Convert absolute datetime values to relative numeric time (starting at 0) in the given unit.

        Parameters
        ----------
        time_data : str, default 'index'
            Name of the column containing the time data; use 'index' to convert the index.
        unit : {'sec','s','min','m','hr','h','d','day','days'}, default 'd'
            Target unit for relative time.
        inplace : bool, default True
            If True, modify this object and return None. If False, return a new HydroData.
        save_abs : bool, default True
            If True, preserve the absolute time in a 'time_abs' column before conversion.
            (For index-based time, 'time_abs' is added as a column.)
        decimals : int, default 5
            Round the relative values to this many decimals.

        Returns
        -------
        HydroData | None
            New HydroData when inplace=False; otherwise None.

        Notes
        -----
        - The first timestamp is considered t=0.
        - If the time source is not datetime-like, the method attempts to convert it using
        pandas.to_datetime (raises on failure).
        """
        if self.data.empty:
            return None if inplace else self.__class__(
                data=self.data.copy(),
                timedata_column=self.timename,
                data_type=self.data_type,
                experiment_tag=self.tag,
                time_unit=unit,
                units=self.units.copy() if isinstance(self.units, dict) else self.units,
                copy=True,
            )

        if time_data == "index":
            tser = pd.Series(self.data.index, index=self.data.index)
        else:
            if time_data not in self.data.columns:
                raise KeyError(f"time_data column '{time_data}' not found in DataFrame.")
            tser = pd.Series(self.data[time_data].values, index=self.data.index, name=time_data)

        if not np.issubdtype(pd.Series(tser).dtype, np.datetime64):
            try:
                tser = pd.to_datetime(tser, errors="raise")
            except Exception as e:
                raise ValueError(
                    f"Time data in '{time_data}' is not datetime-like and could not be converted."
                ) from e

        if len(tser) == 0:
            rel = pd.Series(dtype="float64", index=self.data.index, name="time_rel")
        else:
            t0 = tser.iloc[0]
            deltas = (tser - t0)  
            seconds = deltas.dt.total_seconds()

            unit_map = {
                "sec": 1.0, "s": 1.0,
                "min": 60.0, "m": 60.0,
                "hr": 3600.0, "h": 3600.0,
                "d": 86400.0, "day": 86400.0, "days": 86400.0,
            }
            if unit not in unit_map:
                raise ValueError("unit must be one of {'sec','s','min','m','hr','h','d','day','days'}.")

            scale = unit_map[unit]
            rel = (seconds / scale).round(decimals)
            rel.name = "time_rel"

        if not inplace:
            new_df = self.data.copy()
            if save_abs:
                new_df["time_abs"] = tser.values
            if time_data == "index":
                new_df.index = pd.Index(rel.values, name="time_rel")
            else:
                new_df[time_data] = rel.values

            return self.__class__(
                data=new_df,
                timedata_column=self.timename if time_data != "index" else "index",
                data_type=self.data_type,
                experiment_tag=self.tag,
                time_unit=str(unit),
                units=self.units.copy() if isinstance(self.units, dict) else self.units,
                copy=True,
            )

        if save_abs:
            self.data["time_abs"] = tser.values
            self.columns = np.array(self.data.columns, dtype=object)

        if time_data == "index":
            self.data.index = pd.Index(rel.values, name="time_rel")
            self.timename = "index"
            self.time_unit = str(unit)
            self._update_time()
            self.columns = np.array(self.data.columns, dtype=object)
            return None

        self.data[time_data] = rel.values
        self.time_unit = str(unit)
        self._update_time()
        return None
    
    def write(
        self,
        filename: str,
        filepath: str | os.PathLike = os.getcwd(),
        *,
        method: Literal["all", "filtered", "filled"] = "all",
        sep: str = "auto",                    
        na_rep: str = "",
        float_format: Optional[str] = None,    
        index: bool = True,
        index_label: Optional[str] = None,     
        include_units: bool = False,
        default_ext: str = ".csv",             
    ) -> Path:
        """
        Write a data export to disk (optionally compressed based on filename).

        - If `filename` has no extension, `default_ext` is appended.
        - Compression is inferred from the filename suffix (.gz, .bz2, .xz, .zip).
        - If `sep='auto'`, uses ',' for .csv (or .csv.gz, etc.) else '\\t'.
        """
        if method == "all":
            df = self.data
        elif method == "filtered":
            if self.meta_valid is None or self.meta_valid.empty:
                df = self.data.copy()
            else:
                df = self.data.copy()
                for col in self.meta_valid.columns:
                    if col in df.columns:
                        mask = (self.meta_valid[col] == "original")
                        df.loc[~mask, col] = np.nan
        elif method == "filled":
            if not hasattr(self, "filled") or self.filled is None or self.filled.empty:
                df = pd.DataFrame(index=self.data.index, columns=self.data.columns)
            else:
                df = self.filled
        else:
            raise ValueError("`method` must be one of {'all','filtered','filled'}.")

        out_dir = Path(filepath).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        p = Path(filename)
        if not p.suffixes:  
            p = p.with_suffix(default_ext)

        suffixes = p.suffixes  
        compression_map = {".gz": "gzip", ".bz2": "bz2", ".xz": "xz", ".zip": "zip"}
        compression = None
        if suffixes:
            last = suffixes[-1].lower()
            compression = compression_map.get(last)  

        base_ext = suffixes[0].lower() if suffixes else default_ext.lower()
        if sep == "auto":
            sep_to_use = "," if base_ext == ".csv" else "\t"
        else:
            sep_to_use = sep

        out_path = (out_dir / p)

        pre_rows: list[pd.DataFrame] = []
        if include_units:
            units_row = {}
            if isinstance(self.units, dict):
                for c in df.columns:
                    units_row[c] = self.units.get(c, "")
            else:
                try:
                    units_row = {c: str(self.units[c].iloc[0]) if c in self.units else "" for c in df.columns}  # type: ignore[index]
                except Exception:
                    units_row = {c: "" for c in df.columns}
            idx_name = index_label if index_label is not None else (df.index.name or "")
            units_df = pd.DataFrame([units_row], columns=df.columns)
            units_df.index = pd.Index(["[units]"], name=(idx_name if index else None))
            pre_rows.append(units_df)

        to_write = pd.concat([pd.concat(pre_rows).reindex(columns=df.columns)] + [df], axis=0) if pre_rows else df

        to_write.to_csv(
            out_path,
            sep=sep_to_use,
            na_rep=na_rep,
            float_format=float_format,
            index=index,
            index_label=index_label,
            compression=compression,  
        )

        return out_path


#     #######################
#     ### DATA EXPLORATION
#     #######################

    def get_avg(
        self,
        name: Optional[Union[str, Sequence[str]]] = None,
        *,
        only_checked: bool = True,
        return_scalar: bool = False,
    ) -> Union[pd.Series, float, None]:
        """
        Compute column averages (mean) for selected or all columns.

        Parameters
        ----------
        name : str or sequence of str, optional
            Column name(s) to compute averages for.
            - None (default): compute averages for all numeric columns.
            - str: single column.
            - list/tuple: multiple columns.
        only_checked : bool, default True
            If True, values marked as 'filtered' in `self.meta_valid` are ignored (set to NaN).
        return_scalar : bool, default False
            If True and a single column is requested, return the float instead of a 1-element Series.

        Returns
        -------
        pd.Series or float or None
            - `pd.Series` of column means when multiple columns are requested or default (None).
            - `float` when a single column is requested with return_scalar=True.
            - None if no valid numeric columns were found.
        """
        if name is None:
            cols = list(self.data.columns)
        elif isinstance(name, str):
            cols = [name]
        else:
            cols = list(name)

        
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            import warnings
            warnings.warn(f"Columns not found and ignored: {missing}", RuntimeWarning, stacklevel=2)
            cols = [c for c in cols if c in self.data.columns]

        if not cols:
            return None

        if only_checked and self.meta_valid is not None and not self.meta_valid.empty:
            df = self.data.copy()
            for c in cols:
                if c in self.meta_valid.columns:
                    mask = self.meta_valid[c] == "filtered"
                    df.loc[mask, c] = np.nan
        else:
            df = self.data

        means = df[cols].mean(numeric_only=True)

        
        if return_scalar and len(means) == 1:
            return float(means.iloc[0])

        return means

   
    def get_std(
        self,
        name: Optional[Union[str, Sequence[str]]] = None,
        *,
        only_checked: bool = True,
        return_scalar: bool = False,
    ) -> Union[pd.Series, float, None]:
        """
        Compute column standard deviations for selected or all columns.

        Parameters
        ----------
        name : str or sequence of str, optional
            Column name(s) to compute std for.
            - None (default): compute std for all numeric columns.
            - str: single column.
            - list/tuple: multiple columns.
        only_checked : bool, default True
            If True, values marked as 'filtered' in `self.meta_valid` are ignored (set to NaN).
        return_scalar : bool, default False
            If True and a single column is requested, return the float instead of a 1-element Series.

        Returns
        -------
        pd.Series or float or None
            - `pd.Series` of column std when multiple columns are requested or default (None).
            - `float` when a single column is requested with return_scalar=True.
            - None if no valid numeric columns were found.
        """
        if name is None:
            cols = list(self.data.columns)
        elif isinstance(name, str):
            cols = [name]
        else:
            cols = list(name)

        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            import warnings
            warnings.warn(f"Columns not found and ignored: {missing}", RuntimeWarning, stacklevel=2)
            cols = [c for c in cols if c in self.data.columns]

        if not cols:
            return None

        if only_checked and self.meta_valid is not None and not self.meta_valid.empty:
            df = self.data.copy()
            for c in cols:
                if c in self.meta_valid.columns:
                    mask = (self.meta_valid[c] == "filtered")
                    df.loc[~mask, c] = df.loc[~mask, c]  
                    df.loc[mask, c] = np.nan             
        else:
            df = self.data

        stds = df[cols].std(numeric_only=True)

        if return_scalar and len(stds) == 1:
            return float(stds.iloc[0])

        return stds


    def get_highs(
        self,
        data_name: str,
        bound_value: float,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        *,
        method: Literal["value", "percentile"] = "percentile",
        plot: bool = False,
    ) -> None:
        """
        Tag indices where a column's values exceed a threshold (absolute or percentile).

        Parameters
        ----------
        data_name : str
            Column to evaluate.
        bound_value : float
            - If method='value': the absolute threshold.
            - If method='percentile': the percentile in [0, 1].
        arange : (start, end), optional
            Slice bounds along the index (e.g., timestamps). Either bound may be None.
            If None (default), no slicing is applied (use full index).
        method : {'value', 'percentile'}, default 'percentile'
            Threshold interpretation.
        plot : bool, default False
            If True, plot the series with high points highlighted.

        Returns
        -------
        None
            Updates/creates `self.highs` with a column 'highs' (0/1) aligned to `self.data.index`.
            Stores the actual numeric threshold used in `self._last_highs_threshold[data_name]`.
        """
        if self.data.empty:
            wn.warn("get_highs: data is empty; nothing to tag.", RuntimeWarning, stacklevel=2)
            self.highs = pd.DataFrame({"highs": pd.Series(0, index=self.data.index, dtype=int)})
            return

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")

        
        self.highs = pd.DataFrame(index=self.data.index, data={"highs": 0}, dtype=int)

        
        if arange is None:
            data_to_use = self.data[data_name].copy()
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list; either may be None.")
            start, end = arange
            try:
                data_to_use = self.data.loc[start:end, data_name].copy()
            except TypeError as e:
                raise TypeError(
                    "Slicing not possible for index type "
                    f"{type(self.data.index[0])} and arange argument type {type(start)}."
                ) from e

        if data_to_use.empty:
            wn.warn("get_highs: selected data is empty; no highs tagged.", RuntimeWarning, stacklevel=2)
            return

        if method == "value":
            thresh = float(bound_value)
        elif method == "percentile":
            if not (0.0 <= bound_value <= 1.0):
                raise ValueError("For method='percentile', bound_value must be in [0, 1].")
            thresh = float(data_to_use.dropna().quantile(bound_value))
        else:
            raise ValueError("`method` must be 'value' or 'percentile'.")

        idx_high = data_to_use.index[data_to_use > thresh]
        self.highs.loc[idx_high, "highs"] = 1

        if not hasattr(self, "_last_highs_threshold") or not isinstance(self._last_highs_threshold, dict):
            self._last_highs_threshold = {}
        self._last_highs_threshold[data_name] = thresh

        if plot:
            import matplotlib.pyplot as plt

            highs_mask = self.highs.loc[data_to_use.index, "highs"].astype(bool)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(data_to_use.index[~highs_mask], data_to_use[~highs_mask], "-", color='blue', label=f"{data_name} (normal)")
            ax.plot(data_to_use.index[highs_mask], data_to_use[highs_mask], ".", color='red', label=f"{data_name} (high)")
            ax.axhline(thresh, linestyle="--", color="firebrick", label=f"threshold={thresh:.3g}")
            ax.set_title(f"Highs for '{data_name}' ({method}: {bound_value})")
            ax.set_xlabel(self.data.index.name or "Index")
            ax.set_ylabel(data_name)
            ax.legend()
            fig.tight_layout()
            plt.show()

    def get_summary(self) -> Dict[str, pd.DataFrame]:
        """
        Extended summary of the dataset.

        Returns
        -------
        dict with keys:
        - 'numeric': describe() + missing/filtered rows
        - 'non_numeric': counts & missing for non-numeric columns (if any)
        - 'index': a 1-row summary about the index (start/end/freq/rows)
        """
        out: Dict[str, pd.DataFrame] = {}

        num_desc = self.data.describe(include=[np.number])
        miss = self.data.isna().sum(numeric_only=False)
        miss = miss.reindex(num_desc.columns).fillna(0).astype(int)
        miss_pct = (miss / len(self.data) * 100.0).round(2)
        if getattr(self, "meta_valid", None) is not None and not self.meta_valid.empty:
            filt = pd.Series(0, index=num_desc.columns, dtype=int)
            for c in num_desc.columns:
                if c in self.meta_valid.columns:
                    filt[c] = (self.meta_valid[c] == "filtered").sum()
            filt_pct = (filt / len(self.data) * 100.0).round(2)
        else:
            filt = pd.Series(0, index=num_desc.columns, dtype=int)
            filt_pct = pd.Series(0.0, index=num_desc.columns)

        num_desc.loc["missing_count"] = miss
        num_desc.loc["missing_pct"] = miss_pct
        num_desc.loc["filtered_count"] = filt
        num_desc.loc["filtered_pct"] = filt_pct
        out["numeric"] = num_desc

        non_num_cols = self.data.columns.difference(num_desc.columns)
        if len(non_num_cols) > 0:
            nn = pd.DataFrame(index=non_num_cols, columns=["count", "missing_count", "missing_pct"], dtype=float)
            nn["count"] = len(self.data)
            nn["missing_count"] = self.data[non_num_cols].isna().sum()
            nn["missing_pct"] = (nn["missing_count"] / len(self.data) * 100.0).round(2)
            out["non_numeric"] = nn

        if len(self.data.index) > 0:
            start = self.data.index.min()
            end = self.data.index.max()
            try:
                if isinstance(self.data.index, pd.DatetimeIndex) and len(self.data.index) >= 3:
                    freq = pd.to_timedelta(int(np.median(np.diff(self.data.index.view("int64")))), unit="ns")
                else:
                    freq = pd.NaT
            except Exception:
                freq = pd.NaT
        else:
            start = pd.NaT
            end = pd.NaT
            freq = pd.NaT

        idx_df = pd.DataFrame(
            {"n_rows": [len(self.data)], "start": [start], "end": [end], "median_freq": [freq]}
        )
        out["index"] = idx_df

        return out
    
    def missing_stats(self, *, treat_filtered_as_missing: bool = True) -> pd.DataFrame:
        """
        Per-column missing stats (counts & percentages). Optionally treat 'filtered' as missing.

        Returns
        -------
        pd.DataFrame with columns:
        ['total', 'missing', 'missing_pct', 'filtered', 'filtered_pct', 'effective_missing', 'effective_missing_pct']
        """
        n = len(self.data)
        cols = list(self.data.columns)
        df = self.data

        missing = df.isna().sum()
        filtered = pd.Series(0, index=cols, dtype=int)
        if treat_filtered_as_missing and getattr(self, "meta_valid", None) is not None and not self.meta_valid.empty:
            for c in cols:
                if c in self.meta_valid.columns:
                    filtered[c] = (self.meta_valid[c] == "filtered").sum()

        effective_missing = missing.add(filtered, fill_value=0).clip(upper=n)

        out = pd.DataFrame(
            {
                "total": n,
                "missing": missing,
                "missing_pct": (missing / n * 100.0).round(2),
                "filtered": filtered,
                "filtered_pct": (filtered / n * 100.0).round(2),
                "effective_missing": effective_missing,
                "effective_missing_pct": (effective_missing / n * 100.0).round(2),
            }
        ).reindex(index=cols)
        return out
    
    def gap_lengths(
        self,
        *,
        treat_filtered_as_missing: bool = True,
        min_gap_len: int = 1,
    ) -> pd.DataFrame:
        """
        Summarize contiguous missing segments (gaps) per column.

        Parameters
        ----------
        treat_filtered_as_missing : bool, default True
            Count meta_valid == 'filtered' as missing.
        min_gap_len : int, default 1
            Only gaps with length >= min_gap_len are counted.

        Returns
        -------
        pd.DataFrame with columns:
        ['n_gaps', 'max_gap', 'mean_gap', 'total_missing', 'coverage_pct']
        Where:
        - For DatetimeIndex: gap units are Timedelta
        - For other indexes: gap units are number of samples (int)
        """
        idx = self.data.index
        is_dt = isinstance(idx, pd.DatetimeIndex)

        def _run_lengths(mask: pd.Series) -> list[int]:
            if mask.empty:
                return []
            change = mask.ne(mask.shift(fill_value=False))
            groups = change.cumsum()
            runs = mask.groupby(groups).sum()  
            return [int(v) for v, m in zip(runs.values, mask.groupby(groups).first().values) if m]

        rows = []
        n = len(self.data)

        for c in self.data.columns:
            miss = self.data[c].isna()
            if treat_filtered_as_missing and getattr(self, "meta_valid", None) is not None and c in self.meta_valid.columns:
                miss = miss | (self.meta_valid[c] == "filtered")

            lengths = [L for L in _run_lengths(miss) if L >= min_gap_len]
            total_missing = int(miss.sum())

            if is_dt and len(idx) >= 2:
                try:
                    step_ns = int(np.median(np.diff(idx.view("int64"))))
                    step = pd.to_timedelta(step_ns, unit="ns") if step_ns > 0 else pd.NaT
                except Exception:
                    step = pd.NaT
                if pd.isna(step):
                    max_gap = pd.NaT
                    mean_gap = pd.NaT
                else:
                    max_gap = max(lengths) * step if lengths else pd.to_timedelta(0)
                    mean_gap = (np.mean(lengths) * step) if lengths else pd.to_timedelta(0)
            else:
                max_gap = int(max(lengths)) if lengths else 0
                mean_gap = float(np.mean(lengths)) if lengths else 0.0

            coverage = (1.0 - total_missing / n) * 100.0 if n > 0 else 0.0

            rows.append(
                {
                    "column": c,
                    "n_gaps": int(len(lengths)),
                    "max_gap": max_gap,
                    "mean_gap": mean_gap,
                    "total_missing": int(total_missing),
                    "coverage_pct": round(coverage, 2),
                }
            )

        return pd.DataFrame(rows).set_index("column")
    
    def plot_missing_matrix(
        self,
        *,
        treat_filtered_as_missing: bool = True,
        sort_columns_by_missing: bool = True,
        figsize: tuple[float, float] = (10, 5),
    ) -> None:
        """
        Visualize missingness over time (rows) and columns (cols).
        Missing (NaN or filtered) = 1, present = 0.
        """
        import matplotlib.pyplot as plt

        if self.data.empty:
            plt.figure(figsize=figsize)
            plt.title("No data")
            plt.show()
            return

        miss = self.data.isna()
        if treat_filtered_as_missing and getattr(self, "meta_valid", None) is not None and not self.meta_valid.empty:
            for c in self.data.columns:
                if c in self.meta_valid.columns:
                    miss[c] = miss[c] | (self.meta_valid[c] == "filtered")

        mat = miss.astype(int)
        if sort_columns_by_missing:
            order = mat.sum(axis=0).sort_values(ascending=False).index
            mat = mat[order]

        cmap = ListedColormap(["#70c66d", "#ed494c"])  
        bounds = [-0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat.T.values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

        ax.set_yticks(range(len(mat.columns)))
        ax.set_yticklabels(mat.columns)
        ax.set_xlabel(self.data.index.name or "Index")
        ax.set_title("Missing data matrix")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
        cbar.ax.set_yticklabels(["present", "missing"])

        plt.tight_layout()
        plt.show()

    def plot_histogram(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        *,
        bins: int = 30,
        density: bool = False,
        log: bool = False,
        figsize: Tuple[float, float] = (10, 6),
        treat_filtered_as_missing: bool = True,
    ) -> None:
        """
        Plot histograms for selected numeric columns.

        Parameters
        ----------
        columns : str | sequence[str] | None, default None
            Which columns to plot. None = all numeric columns.
        bins : int, default 30
        density : bool, default False
        log : bool, default False
        figsize : (w, h), default (10, 6)
        treat_filtered_as_missing : bool, default True
            If True and `meta_valid` exists, rows where meta_valid[col] == 'filtered'
            are treated as missing for that column.
        """
        import matplotlib.pyplot as plt

        df = self.data.copy()
        if columns is None:
            cols = list(df.select_dtypes(include=[np.number]).columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)

        if treat_filtered_as_missing and getattr(self, "meta_valid", None) is not None:
            for c in cols:
                if c in self.meta_valid.columns:
                    df.loc[self.meta_valid[c] == "filtered", c] = np.nan

        if not cols:
            raise ValueError("No numeric columns selected for histogram.")

        n = len(cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.ravel()

        for i, c in enumerate(cols):
            ax = axes[i]
            series = pd.to_numeric(df[c], errors="coerce").dropna()
            ax.hist(series.values, bins=bins, density=density, log=log)
            ax.set_title(str(c))
            ax.set_ylabel("Density" if density else "Count")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.tight_layout()
        plt.show()

    def plot_boxplot(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        *,
        showfliers: bool = True,
        figsize: Tuple[float, float] = (10, 6),
        treat_filtered_as_missing: bool = True,
    ) -> None:
        """
        Plot boxplots for selected numeric columns.

        Parameters
        ----------
        columns : str | sequence[str] | None, default None
            Which columns to plot. None = all numeric columns.
        showfliers : bool, default True
        figsize : (w, h), default (10, 6)
        treat_filtered_as_missing : bool, default True
            If True and `meta_valid` exists, rows where meta_valid[col] == 'filtered'
            are treated as missing for that column.
        """
        import matplotlib.pyplot as plt

        df = self.data.copy()
        if columns is None:
            cols = list(df.select_dtypes(include=[np.number]).columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)

        if not cols:
            raise ValueError("No numeric columns selected for boxplot.")

        if treat_filtered_as_missing and getattr(self, "meta_valid", None) is not None:
            for c in cols:
                if c in self.meta_valid.columns:
                    df.loc[self.meta_valid[c] == "filtered", c] = np.nan

        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(
            [pd.to_numeric(df[c], errors="coerce").dropna().values for c in cols],
            showfliers=showfliers,
            labels=[str(c) for c in cols],
        )
        ax.set_ylabel("Value")
        ax.set_title("Boxplot")
        fig.tight_layout()
        plt.show()

    def correlation_matrix(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        *,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        treat_filtered_as_missing: bool = True,
        plot_heatmap: bool = False,
        figsize: Tuple[float, float] = (8, 6),
        annotate: bool = True,
        vmin: float = -1.0,
        vmax: float = 1.0,
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for selected numeric columns and optionally plot a heatmap.

        Parameters
        ----------
        columns : str | sequence[str] | None, default None
            Which columns to compute correlations for. None = all numeric columns.
        method : {'pearson','spearman','kendall'}, default 'pearson'
        treat_filtered_as_missing : bool, default True
            If True and `meta_valid` exists, rows where meta_valid[col] == 'filtered'
            are treated as missing for that column.
        plot_heatmap : bool, default False
            If True, show a heatmap of the correlation matrix.
        figsize : (w, h), default (8, 6)
        annotate : bool, default True
            If True, overlay correlation values on the heatmap.
        vmin, vmax : float, default -1.0, 1.0
            Color scale limits for the heatmap.

        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        import matplotlib.pyplot as plt

        df = self.data.copy()
        if columns is None:
            cols = list(df.select_dtypes(include=[np.number]).columns)
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)

        if not cols:
            raise ValueError("No numeric columns selected for correlation.")

        if treat_filtered_as_missing and getattr(self, "meta_valid", None) is not None:
            for c in cols:
                if c in self.meta_valid.columns:
                    df.loc[self.meta_valid[c] == "filtered", c] = np.nan

        corr = df[cols].corr(method=method, numeric_only=True)

        if plot_heatmap:
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(corr.values, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha="right")
            ax.set_yticklabels(cols)

            if annotate:
                for i in range(len(cols)):
                    for j in range(len(cols)):
                        val = corr.values[i, j]
                        if np.isfinite(val):
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"{method.capitalize()} correlation")
            ax.set_title("Correlation heatmap")
            fig.tight_layout()
            plt.show()

        return corr
    

#     def _reset_highs(self):
#         """
#         """
#         self.highs = pd.DataFrame(data=0,columns=['highs'],index=self.index())

#     ########################
#     ### FILTERING UNIVARIATE
#     ########################


    def add_to_meta_valid(self, column_names: Union[str, Sequence[str]]) -> None:
        """
        Ensure one or more columns exist in `self.meta_valid` with all tags set to 'original'.

        Parameters
        ----------
        column_names : str or sequence of str
            Column name(s) to add to `self.meta_valid`. If a column already exists,
            it is left unchanged and a warning is issued.

        Notes
        -----
        - Ensures that `self.meta_valid` is aligned with the current index of `self.data`.
        - Useful to mark data as reliable for further processing (e.g., filling, filtering).
        """
        if isinstance(column_names, str):
            cols = [column_names]
        else:
            cols = list(column_names)

        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)

        for col in cols:
            if col not in self.meta_valid.columns:
                self.meta_valid[col] = "original"
            else:
                wn.warn(
                    f"meta_valid already contains a column named '{col}'. Keeping existing values.",
                    UserWarning,
                    stacklevel=2,
                )
                # self.meta_valid[col] = self.meta_valid[col].where(
                # self.meta_valid[col].isin(["original", "filtered"] + [c for c in self.meta_valid[col].unique() if str(c).startswith("filled_")]),
                # "original")

    def tag_nan(
        self,
        data_name: str,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        *,
        clear: bool = False,
    ) -> None:
        """
        Tag NaN values in a column as 'filtered' in `self.meta_valid`.

        Parameters
        ----------
        data_name : str
            Column name to apply the function to.
        arange : (start, end), optional
            Index range within which NaN values should be tagged.
            If None (default), the whole column is checked.
        clear : bool, default False
            If True, reset all tags in meta_valid for this column before applying new ones.

        Notes
        -----
        - NaN values → 'filtered'
        - Non-NaN values → 'original'
        - If the column is missing in `meta_valid`, it will be created.
        """
        self._plot = "valid"

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")

        if clear:
            self._reset_meta_valid(data_name)

        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index, fill_value="!!")

        if data_name not in self.meta_valid.columns:
            self.add_to_meta_valid([data_name])

        if arange is None:
            series = self.data[data_name]
            len_orig = len(series)
            self.meta_valid[data_name] = np.where(series.isna(), "filtered", "original")
            len_new = series.count()
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list.")
            start, end = arange
            try:
                series = self.data.loc[start:end, data_name]
            except TypeError as e:
                raise TypeError(
                    f"Slicing not possible for index type {type(self.data.index[0])} "
                    f"with arange {arange}. Adjust `arange` to match index type."
                ) from e

            len_orig = len(series)
            self.meta_valid.loc[start:end, data_name] = np.where(series.isna(), "filtered", "original")
            len_new = series.count()

        
        _print_removed_output(len_orig, len_new, "NaN tagging")

    
    def tag_doubles(
        self,
        data_name: str,
        bound: float,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        *,
        clear: bool = False,
        inplace: bool = False,
        log_file: Optional[str] = None,
        plot: bool = False,
        final: bool = False,
    ):
        """
        Tag subsequent 'double' values (near-constant segments) in a series.

        A point is tagged as 'filtered' when |x[t] - x[t-1]| < bound
        (i.e., the increment is smaller than the threshold). Outside the optional
        `arange`, data is left untouched.

        Parameters
        ----------
        data_name : str
            Column to analyze.
        bound : float
            Threshold for absolute first-difference. Values with diff < bound are tagged.
        arange : (start, end) or None, default None
            Index slice to apply the tagging within. If None, use the full index.
            Either bound may be None.
        clear : bool, default False
            If True, reset meta_valid tags for this column to 'original' before tagging.
        inplace : bool, default False
            Only used when `final=True`. If True, overwrite this object's data;
            otherwise return a new HydroData with the tagged values set to NaN.
        log_file : str | None, default None
            If provided, append a log line with the number of tagged points.
        plot : bool, default False
            If True, calls `self.plot_analysed(data_name)` after tagging (if available).
        final : bool, default False
            If True, convert tagged points to NaN in the output (inplace or returned).
            If False, only update `meta_valid` (no data modification).

        Returns
        -------
        None or HydroData
            - If `final and inplace`, returns None and modifies `self`.
            - If `final and not inplace`, returns a new HydroData with NaNs at tagged points.
            - If `not final`, returns None (only tags in `meta_valid`).
        """
        self._plot = "valid"

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")

        series0 = self.data[data_name]
        len_orig = series0.count()

        df_temp = self.__class__(
            self.data.copy(),
            timedata_column=self.timename,
            data_type=self.data_type,
            experiment_tag=self.tag,
            time_unit=self.time_unit,
        )

        s = pd.to_numeric(self.data[data_name], errors="coerce")
        diff = s.diff().abs()
        bound_mask = diff >= bound
        bound_mask = bound_mask.reindex(df_temp.data.index).fillna(True)

        if arange is None:
            mask_keep = bound_mask
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list; either value can be None.")
            start, end = arange
            try:
                idx = self.data.index
                range_keep = (idx < start) | (idx > end)
            except TypeError as e:
                raise TypeError(
                    "Slicing not possible for index type "
                    f"{type(self.data.index[0])} with arange {arange}. "
                    "Adjust `arange` to match the index type."
                ) from e
            mask_keep = bound_mask | range_keep

        if clear:
            self._reset_meta_valid(data_name)
        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)

        if data_name not in self.meta_valid.columns:
            self.add_to_meta_valid([data_name])

        self.meta_valid.loc[~mask_keep, data_name] = "filtered"
        self.meta_valid.loc[mask_keep, data_name] = self.meta_valid.loc[mask_keep, data_name].fillna("original")

        if final:
            kept_series = df_temp.data[data_name].drop(df_temp.data.index[~mask_keep])
            df_temp.data.loc[:, data_name] = kept_series  

        len_new = (df_temp.data[data_name].count() if final else series0[mask_keep].count())
        removed = _print_removed_output(len_orig, len_new, "double value tagging")

        if isinstance(log_file, str):
            _log_removed_output(log_file, len_orig, len_new, tag="filtered")
        elif log_file is not None:
            raise TypeError("Provide the location of the log file as a string, or omit the argument.")

        if plot:
            try:
                self.plot_analysed(data_name)
            except Exception:
                wn.warn("plot_analysed failed; continuing.", RuntimeWarning, stacklevel=2)

        if final:
            if inplace:
                self.data.loc[:, data_name] = df_temp.data[data_name]
                self._update_time()
                return None
            else:
                return df_temp

        return None
    
    

    def tag_extremes(
        self,
        data_name: str,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        *,
        limit: float = 0.0,
        method: Literal["below", "above"] = "below",
        clear: bool = False,
        plot: bool = False,
    ) -> None:
        """
        Tag values above or below a given limit as 'filtered' in `self.meta_valid`.

        Parameters
        ----------
        data_name : str
            Column to tag.
        arange : (start, end) or None, default None
            Index slice within which to apply tagging. If None, use the full index.
            Either bound may be None.
        limit : float, default 0.0
            Threshold for tagging (strict inequality).
        method : {'below','above'}, default 'below'
            - 'below' → tag points with value < limit
            - 'above' → tag points with value > limit
        clear : bool, default False
            If True, reset tags for this column to 'original' before applying.
        plot : bool, default False
            If True, call `self.plot_analysed(data_name)` after tagging (if available).

        Returns
        -------
        None
        """
        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")

        if method not in ("below", "above"):
            raise ValueError("`method` must be one of {'below','above'}.")

        if clear:
            self._reset_meta_valid(data_name)

        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)

        if data_name not in self.meta_valid.columns:
            self.add_to_meta_valid([data_name])

        if arange is None:
            idx_slice = self.data.index
            s = pd.to_numeric(self.data[data_name], errors="coerce")
            meta_col = self.meta_valid[data_name]
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list; either may be None.")
            start, end = arange
            try:
                s = pd.to_numeric(self.data.loc[start:end, data_name], errors="coerce")
                meta_col = self.meta_valid.loc[start:end, data_name]
                idx_slice = s.index
            except TypeError as e:
                raise TypeError(
                    "Slicing not possible for index type "
                    f"{type(self.data.index[0])} with arange {arange}. "
                    "Adjust `arange` to match the index type."
                ) from e

        if s.empty:
            wn.warn("tag_extremes: selected data is empty; no tagging performed.", RuntimeWarning, stacklevel=2)
            return

        if method == "below":
            mask_tagging = s < limit
        else:  # method == "above"
            mask_tagging = s > limit

        len_orig = len(s)
        len_new = len_orig - int(mask_tagging.sum())

        existing_filtered = (meta_col == "filtered")
        combined = existing_filtered | mask_tagging

        self.meta_valid.loc[idx_slice, data_name] = np.where(combined, "filtered", "original")

        _print_removed_output(len_orig, len_new, f"tagging of extremes ({method})")

        if plot:
            try:
                self.plot_analysed(data_name)
            except Exception:
                wn.warn("plot_analysed failed; continuing.", RuntimeWarning, stacklevel=2)

        return None

    
    def calc_slopes(
        self,
        xdata: str,
        ydata: str,
        time_unit: Optional[Literal["sec", "min", "hr", "d"]] = None,
        slope_range: None = None,   
        *,
        window: int = 1,            
    ) -> pd.Series:
        """
        Calculate point-to-point (or windowed) slopes dy/dx.

        Parameters
        ----------
        xdata : str
            Column name for x; use 'index' to use DataFrame index.
            If datetime-like, provide `time_unit` to express dx in that unit.
        ydata : str
            Column name for y.
        time_unit : {'sec','min','hr','d'} or None
            Required if x is datetime-like. Ignored for numeric x.
        window : int, default 1
            Use y.diff(window) / x.diff(window) for windowed slopes (>=1).

        Returns
        -------
        pd.Series
            Slopes aligned with the original index (first `window` rows will be NaN).
        """
        if window < 1:
            raise ValueError("`window` must be >= 1.")
        if ydata not in self.data.columns:
            raise KeyError(f"Column '{ydata}' not found in data.")

        if xdata == "index":
            x = pd.Series(self.data.index, index=self.data.index, name="__x__")
        else:
            if xdata not in self.data.columns:
                raise KeyError(f"Column '{xdata}' not found in data.")
            x = pd.Series(self.data[xdata].values, index=self.data.index, name=xdata)

        y = pd.to_numeric(self.data[ydata], errors="coerce")

        is_datetime = pd.api.types.is_datetime64_any_dtype(x) or \
                    (xdata == "index" and isinstance(self.data.index, pd.DatetimeIndex))

        if is_datetime:
            if time_unit is None:
                raise ValueError("xdata is datetime-like; please provide `time_unit` {'sec','min','hr','d'}.")
            dx_seconds = pd.to_timedelta(x.diff(window)).dt.total_seconds()
            unit_div = {"sec": 1.0, "min": 60.0, "hr": 3600.0, "d": 86400.0}
            if time_unit not in unit_div:
                raise ValueError("time_unit must be one of {'sec','min','hr','d'}.")
            dx = dx_seconds / unit_div[time_unit]
        else:
            dx = pd.to_numeric(x.diff(window), errors="coerce")

        dy = y.diff(window)
        slopes = dy / dx
        slopes.name = f"slope[{ydata}]/d{xdata}@{window}"
        return slopes
    
    def moving_slope_filter(
        self,
        data_name: str,
        cutoff: float,
        xdata: str = 'index',
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        *,
        time_unit: Optional[str] = None,
        window: int = 1,          # NEW
        clear: bool = False,
        inplace: bool = False,
        log_file: Optional[str] = None,
        plot: bool = False,
        final: bool = False,
        max_iters: int = 20,
    ):
        """
        Iteratively tag/remove points where |windowed slope| > cutoff.

        - Slope computed as y.diff(window) / x.diff(window)
        - If x is datetime-like, pass `time_unit`.
        - If `final=True`, filtered points become NaN (inplace or returned copy).
        """
        self._plot = "valid"
        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")
        if window < 1:
            raise ValueError("`window` must be >= 1.")

        if arange is None:
            work_df = self.data.copy()
            idx_slice = work_df.index
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list.")
            start, end = arange
            try:
                work_df = self.data.loc[start:end].copy()
                idx_slice = work_df.index
            except TypeError as e:
                raise TypeError(
                    f"Slicing not possible for index type {type(self.data.index[0])} with arange {arange}."
                ) from e

        len_orig = work_df[data_name].count()

        if clear:
            self._reset_meta_valid(data_name)
        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)
        if data_name not in self.meta_valid.columns:
            self.add_to_meta_valid([data_name])

        df_temp = self.__class__(
            data=work_df.copy(),
            timedata_column=self.timename,
            data_type=self.data_type,
            experiment_tag=self.tag,
            time_unit=self.time_unit,
        )

        it = 0
        while True:
            slopes = df_temp.calc_slopes(xdata=xdata, ydata=data_name, time_unit=time_unit, window=window)
            big = slopes.abs() > cutoff
            big = big.fillna(False)
            to_drop = big[big].index
            if len(to_drop) == 0 or it >= max_iters:
                break
            df_temp.data.loc[to_drop, data_name] = np.nan
            df_temp.data[data_name] = df_temp.data[data_name].dropna()
            it += 1

        len_new = df_temp.data[data_name].count()
        _print_removed_output(len_orig, len_new, f"moving slope filter (window={window})")
        if isinstance(log_file, str):
            _log_removed_output(log_file, len_orig, len_new, type_="filtered")
        elif log_file is not None:
            raise TypeError("`log_file` must be a string path or None.")

        removed_idx = work_df.index.difference(df_temp.data.index)
        self.meta_valid.loc[removed_idx, data_name] = "filtered"

        if plot and hasattr(self, "plot_analysed"):
            try:
                self.plot_analysed(data_name)
            except Exception:
                wn.warn("plot_analysed failed; continuing.", RuntimeWarning, stacklevel=2)

        if final:
            if inplace:
                s_new = self.data[data_name].copy()
                s_new.loc[removed_idx] = np.nan
                self.data.loc[:, data_name] = s_new
                self._update_time()
                return None
            else:
                out = self.__class__(
                    data=self.data.copy(),
                    timedata_column=self.timename,
                    data_type=self.data_type,
                    experiment_tag=self.tag,
                    time_unit=self.time_unit,
                )
                s_new = out.data[data_name].copy()
                s_new.loc[removed_idx] = np.nan
                out.data.loc[:, data_name] = s_new
                out._update_time()
                return out

        return None

    def calc_moving_average(
        self,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        window: int = 10,
        data_name: Optional[Union[str, Sequence[str]]] = None,
        *,
        kind: Literal["mean", "median", "ema"] = "mean",   
        ema_alpha: Optional[float] = None,                
        ema_adjust: bool = True,                           
        ema_min_periods: Optional[int] = None,            
        inplace: bool = False,
        plot: bool = True,
    ):
        """
        Smooth columns via mean/median rolling window or EMA.

        Parameters
        ----------
        arange : (start, end) or None
        window : int
            - mean/median: rolling window size (centered)
            - ema: if `ema_alpha` is None, this is used as the EMA span
        data_name : str | sequence[str] | None
        kind : {'mean','median','ema'}
            'ema' uses pd.Series.ewm(alpha=..., adjust=..., min_periods=...)
        ema_alpha : float in (0,1], optional
            If provided, EMA will use this alpha directly. If None, `window` is used as span.
        ema_adjust : bool, default True
            Pass-through to pandas ewm(..., adjust=...).
        ema_min_periods : int or None
            Pass-through to pandas ewm(..., min_periods=...).
        inplace, plot : see previous docs
        """
        if window < 1:
            raise ValueError("`window` must be >= 1.")
        if kind not in ("mean", "median", "ema"):
            raise ValueError("`kind` must be 'mean', 'median', or 'ema'.")

        if arange is None:
            work_df = self.data.copy()
            start = work_df.index[0] if len(work_df) else None
            end = work_df.index[-1] if len(work_df) else None
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list.")
            start, end = arange
            work_df = self.data.loc[start:end].copy()

        if work_df.empty:
            return None if inplace else self.__class__(
                self.data.copy(), timedata_column=self.timename,
                data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit
            )

        if data_name is None:
            cols = list(work_df.select_dtypes(include=[np.number]).columns)
            if not cols:
                raise ValueError("No numeric columns available to smooth.")
        elif isinstance(data_name, str):
            if data_name not in work_df.columns:
                raise KeyError(f"Column '{data_name}' not found in selected range.")
            cols = [data_name]
        else:
            cols = [c for c in data_name if c in work_df.columns]
            if not cols:
                raise KeyError("None of the requested columns were found in the data.")

        averaged_slice = work_df.copy()
        for c in cols:
            series = pd.to_numeric(averaged_slice[c], errors="coerce").interpolate()

            if kind == "mean":
                averaged_slice[c] = series.rolling(window=window, center=True).mean()
            elif kind == "median":
                averaged_slice[c] = series.rolling(window=window, center=True).median()
            else:  
                if ema_alpha is not None:
                    averaged_slice[c] = series.ewm(alpha=ema_alpha,
                                                adjust=ema_adjust,
                                                min_periods=ema_min_periods).mean()
                else:
                    averaged_slice[c] = series.ewm(span=window,
                                                adjust=ema_adjust,
                                                min_periods=ema_min_periods).mean()

        if inplace:
            out_df = self.data.copy()
            out_df.loc[averaged_slice.index, cols] = averaged_slice[cols]
            self.data = out_df
            self._update_time()

            if plot and len(cols) == 1 and start is not None and end is not None:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(16, 6))
                ax.plot(self.data.loc[start:end].index, self.data.loc[start:end, cols[0]], "b-",
                        label=f"{kind}({('alpha='+str(ema_alpha)) if kind=='ema' and ema_alpha is not None else 'window='+str(window)})")
                ax.set_title(f"{kind.upper()} smoothing on {cols[0]}")
                ax.set_xlabel(self.timename); ax.set_ylabel(cols[0]); ax.legend(); fig.tight_layout(); plt.show()
            return None

        new_df = self.data.copy()
        new_df.loc[averaged_slice.index, cols] = averaged_slice[cols]
        hd_new = self.__class__(new_df, timedata_column=self.timename, data_type=self.data_type,
                                experiment_tag=self.tag, time_unit=self.time_unit)

        if plot and len(cols) == 1 and start is not None and end is not None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.plot(self.data.loc[start:end].index, self.data.loc[start:end, cols[0]], "r--", label="original")
            ax.plot(hd_new.data.loc[start:end].index, hd_new.data.loc[start:end, cols[0]], "b-",
                    label=f"{kind}({('alpha='+str(ema_alpha)) if kind=='ema' and ema_alpha is not None else 'window='+str(window)})")
            ax.set_title(f"{kind.upper()} smoothing on {cols[0]}")
            ax.set_xlabel(self.timename); ax.set_ylabel(cols[0]); ax.legend(); fig.tight_layout(); plt.show()

        return hd_new
    
    def moving_average_filter(
        self,
        data_name: str,
        window: int,
        cutoff_frac: float,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        *,
        kind: Literal["mean", "median", "ema"] = "mean",     
        ema_alpha: Optional[float] = None,                   
        ema_adjust: bool = True,
        ema_min_periods: Optional[int] = None,
        absolute_cutoff: Optional[float] = None,
        clear: bool = False,
        inplace: bool = False,
        log_file: Optional[str] = None,
        plot: bool = False,
        final: bool = False,
    ):
        """
        Filter outliers by comparing data to its smoother (mean/median/EMA).

        A point is filtered if:
        - `abs(data - smooth) / smooth >= cutoff_frac` where smooth != 0 & not NaN
        - OR (if `absolute_cutoff` is provided) `abs(data) >= absolute_cutoff`
            at positions where smooth == 0 or NaN.
        """
        self._plot = "valid"

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")
        if window < 1:
            raise ValueError("`window` must be >= 1.")
        if cutoff_frac < 0:
            raise ValueError("`cutoff_frac` must be >= 0.")
        if kind not in ("mean", "median", "ema"):
            raise ValueError("`kind` must be 'mean', 'median', or 'ema'.")

        if arange is None:
            work_df = self.data.copy()
            idx_slice = work_df.index
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list.")
            start, end = arange
            work_df = self.data.loc[start:end].copy()
            idx_slice = work_df.index

        if work_df.empty:
            return None

        len_orig = work_df[data_name].count()

        if clear:
            self._reset_meta_valid(data_name)
        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)
        if data_name not in self.meta_valid.columns:
            self.add_to_meta_valid([data_name])

        smooth_hd = self.calc_moving_average(
            arange=arange,
            window=window,
            data_name=data_name,
            kind=kind,
            ema_alpha=ema_alpha,
            ema_adjust=ema_adjust,
            ema_min_periods=ema_min_periods,
            inplace=False,
            plot=False,
        )
        smooth_series = pd.to_numeric(smooth_hd.data.loc[idx_slice, data_name], errors="coerce")
        orig_series = pd.to_numeric(work_df[data_name], errors="coerce")

        valid_sm = smooth_series.notna() & (smooth_series != 0)
        denom = smooth_series.where(valid_sm, np.nan)
        rel_dev = (orig_series - smooth_series).abs() / denom
        rel_mask = rel_dev >= cutoff_frac
        rel_mask = rel_mask & rel_dev.notna()

        if absolute_cutoff is not None:
            abs_mask = (~valid_sm) & (orig_series.abs() >= absolute_cutoff)
        else:
            abs_mask = (~valid_sm) & pd.Series(False, index=orig_series.index)

        to_filter = (rel_mask | abs_mask).fillna(False)
        idx_remove = to_filter[to_filter].index

        df_temp = self.__class__(
            work_df.copy(), timedata_column=self.timename,
            data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit
        )
        df_temp.data.loc[idx_remove, data_name] = np.nan
        df_temp.data[data_name] = df_temp.data[data_name].dropna()
        len_new = df_temp.data[data_name].count()

        _print_removed_output(len_orig, len_new, f"moving average filter (kind={kind})")
        if isinstance(log_file, str):
            _log_removed_output(log_file, len_orig, len_new, type_="filtered")
        elif log_file is not None:
            raise TypeError("`log_file` must be a string path or None.")

        self.meta_valid.loc[idx_remove, data_name] = "filtered"

        if plot and hasattr(self, "plot_analysed"):
            try:
                self.plot_analysed(data_name)
            except Exception:
                wn.warn("plot_analysed failed; continuing.", RuntimeWarning, stacklevel=2)

        if final:
            if inplace:
                s_new = self.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                self.data.loc[:, data_name] = s_new
                self._update_time()
                return None
            else:
                out = self.__class__(
                    data=self.data.copy(), timedata_column=self.timename,
                    data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit
                )
                s_new = out.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                out.data.loc[:, data_name] = s_new
                out._update_time()
                return out

        return None


    def zscore_filter(
        self,
        data_name: str,
        *,
        k: float = 3.0,
        robust: bool = False,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        clear: bool = False,
        inplace: bool = False,
        log_file: Optional[str] = None,
        plot: bool = False,
        final: bool = False,
    ) -> Optional["HydroData"]:
        """
        Tag outliers using (robust) z-score.

        Parameters
        ----------
        data_name : str
            Column to filter.
        k : float, default 3.0
            Threshold |z| >= k is filtered.
        robust : bool, default False
            Use robust z-score based on median and MAD (scaled by 1.4826).
        arange : (start, end) or None
            Index slice to apply the filter. None -> full index.
        clear, inplace, log_file, plot, final : see other filters.

        Returns
        -------
        None or HydroData
            If final & not inplace: a new HydroData with NaNs at filtered points.
            Else None.
        """
        self._plot = "valid"

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")

        if arange is None:
            work_df = self.data.copy()
            idx_slice = work_df.index
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list.")
            start, end = arange
            try:
                work_df = self.data.loc[start:end].copy()
                idx_slice = work_df.index
            except Exception as e:
                raise TypeError(
                    f"Slicing not possible for index type {type(self.data.index[0])} with arange {arange}."
                ) from e

        if work_df.empty:
            return None

        s = pd.to_numeric(work_df[data_name], errors="coerce")
        len_orig = s.count()

        if robust:
            med = s.median()
            mad = (s - med).abs().median()
            scale = 1.4826 * mad if mad and np.isfinite(mad) else np.nan
            z = (s - med) / scale if scale and np.isfinite(scale) and scale != 0 else pd.Series(np.nan, index=s.index)
        else:
            mu = s.mean()
            sd = s.std(ddof=0)
            z = (s - mu) / sd if sd and np.isfinite(sd) and sd != 0 else pd.Series(np.nan, index=s.index)

        to_filter = z.abs() >= k
        to_filter = to_filter & z.notna()
        idx_remove = to_filter[to_filter].index

        if clear:
            self._reset_meta_valid(data_name)
        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)
        if data_name not in self.meta_valid.columns:
            self.add_to_meta_valid([data_name])

        self.meta_valid.loc[idx_remove, data_name] = "filtered"

        df_temp = self.__class__(work_df.copy(), timedata_column=self.timename,
                                data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit)
        df_temp.data.loc[idx_remove, data_name] = np.nan
        df_temp.data[data_name] = df_temp.data[data_name].dropna()
        len_new = df_temp.data[data_name].count()

        _print_removed_output(len_orig, len_new, f"z-score filter ({'robust' if robust else 'standard'})")
        if isinstance(log_file, str):
            _log_removed_output(log_file, len_orig, len_new, type_="filtered")
        elif log_file is not None:
            raise TypeError("`log_file` must be a string path or None.")

        if plot and hasattr(self, "plot_analysed"):
            try:
                self.plot_analysed(data_name)
            except Exception:
                wn.warn("plot_analysed failed; continuing.", RuntimeWarning, stacklevel=2)

        if final:
            if inplace:
                s_new = self.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                self.data.loc[:, data_name] = s_new
                self._update_time()
                return None
            else:
                out = self.__class__(self.data.copy(), timedata_column=self.timename,
                                    data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit)
                s_new = out.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                out.data.loc[:, data_name] = s_new
                out._update_time()
                return out

        return None
    
    def stl_residual_filter(
        self,
        data_name: str,
        *,
        period: int,
        k: float = 3.0,
        robust: bool = True,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        low_pass_deg: int = 1,
        seasonal: int = 7,
        trend: Optional[int] = None,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        clear: bool = False,
        inplace: bool = False,
        log_file: Optional[str] = None,
        plot: bool = False,
        final: bool = False,
    ) -> Optional["HydroData"]:
        """
        Tag outliers using STL decomposition residuals:
        decompose y = trend + seasonal + resid,
        compute z-score on resid and tag |z| >= k.

        Parameters
        ----------
        data_name : str
            Column to filter (numeric).
        period : int
            Seasonal period (samples per season). E.g., 288 for 5-min data with daily cycle.
        k : float, default 3.0
            Z-score threshold (|z| >= k → filtered).
        robust : bool, default True
            Use robust z-score (median/MAD) on residuals. If False, mean/std.
        seasonal_deg, trend_deg, low_pass_deg : int
            Polynomial degrees for STL (statsmodels STL parameters).
        seasonal : int, default 7
            Length of seasonal smoother (must be odd); see statsmodels STL.
        trend : int | None
            Length of trend smoother (must be odd); if None, STL chooses automatically.
        arange, clear, inplace, log_file, plot, final : see other filters.

        Returns
        -------
        None or HydroData
        """
        
        try:
            from statsmodels.tsa.seasonal import STL
        except Exception:
            wn.warn("statsmodels is required for STL filtering. Skipping.", RuntimeWarning, stacklevel=2)
            return None

        self._plot = "valid"

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("stl_residual_filter requires a DatetimeIndex.")
        if period < 2:
            raise ValueError("`period` must be >= 2.")

        if arange is None:
            work_df = self.data.copy()
            idx_slice = work_df.index
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list.")
            start, end = arange
            try:
                work_df = self.data.loc[start:end].copy()
                idx_slice = work_df.index
            except Exception as e:
                raise TypeError(
                    f"Slicing not possible for index type {type(self.data.index[0])} with arange {arange}."
                ) from e

        if work_df.empty:
            return None

        s = pd.to_numeric(work_df[data_name], errors="coerce")
        len_orig = s.count()

        s_ = s.dropna()
        if s_.empty:
            return None

        res = STL(
            s_,
            period=period,
            seasonal=seasonal,
            trend=trend,
            seasonal_deg=seasonal_deg,
            trend_deg=trend_deg,
            low_pass_deg=low_pass_deg,
            robust=True,  
        ).fit()

        resid = res.resid.reindex(s.index)  

        if robust:
            med = resid.median()
            mad = (resid - med).abs().median()
            scale = 1.4826 * mad if mad and np.isfinite(mad) else np.nan
            z = (resid - med) / scale if scale and np.isfinite(scale) and scale != 0 else pd.Series(np.nan, index=resid.index)
        else:
            mu = resid.mean()
            sd = resid.std(ddof=0)
            z = (resid - mu) / sd if sd and np.isfinite(sd) and sd != 0 else pd.Series(np.nan, index=resid.index)

        to_filter = z.abs() >= k
        to_filter = to_filter & z.notna()
        idx_remove = to_filter[to_filter].index

        if clear:
            self._reset_meta_valid(data_name)
        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)
        if data_name not in self.meta_valid.columns:
            self.add_to_meta_valid([data_name])
        self.meta_valid.loc[idx_remove, data_name] = "filtered"

        df_temp = self.__class__(work_df.copy(), timedata_column=self.timename,
                                data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit)
        df_temp.data.loc[idx_remove, data_name] = np.nan
        df_temp.data[data_name] = df_temp.data[data_name].dropna()
        len_new = df_temp.data[data_name].count()

        _print_removed_output(len_orig, len_new, f"STL residual filter (period={period}, k={k}, robust={robust})")
        if isinstance(log_file, str):
            _log_removed_output(log_file, len_orig, len_new, type_="filtered")
        elif log_file is not None:
            raise TypeError("`log_file` must be a string path or None.")

        if plot and hasattr(self, "plot_analysed"):
            try:
                self.plot_analysed(data_name)
            except Exception:
                wn.warn("plot_analysed failed; continuing.", RuntimeWarning, stacklevel=2)

        if final:
            if inplace:
                s_new = self.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                self.data.loc[:, data_name] = s_new
                self._update_time()
                return None
            else:
                out = self.__class__(
                    data=self.data.copy(), timedata_column=self.timename,
                    data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit
                )
                s_new = out.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                out.data.loc[:, data_name] = s_new
                out._update_time()
                return out

        return None
    
    def iqr_filter(
        self,
        data_name: str,
        *,
        k: float = 1.5,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        clear: bool = False,
        inplace: bool = False,
        log_file: Optional[str] = None,
        plot: bool = False,
        final: bool = False,
    ) -> Optional["HydroData"]:
        """
        Tag outliers outside Tukey IQR fences:
        value < Q1 - k*IQR  or  value > Q3 + k*IQR

        Parameters
        ----------
        data_name : str
            Column to filter.
        k : float, default 1.5
            Multiplier for the IQR (3.0 is more conservative).
        arange, clear, inplace, log_file, plot, final : see other filters.

        Returns
        -------
        None or HydroData
        """
        self._plot = "valid"

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")

        
        if arange is None:
            work_df = self.data.copy()
            idx_slice = work_df.index
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list.")
            start, end = arange
            try:
                work_df = self.data.loc[start:end].copy()
                idx_slice = work_df.index
            except Exception as e:
                raise TypeError(
                    f"Slicing not possible for index type {type(self.data.index[0])} with arange {arange}."
                ) from e

        if work_df.empty:
            return None

        s = pd.to_numeric(work_df[data_name], errors="coerce")
        len_orig = s.count()

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            idx_remove = s.index[s.notna() & False]
        else:
            low = q1 - k * iqr
            high = q3 + k * iqr
            idx_remove = s.index[(s < low) | (s > high)]

        if clear:
            self._reset_meta_valid(data_name)
        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)
        if data_name not in self.meta_valid.columns:
            self.add_to_meta_valid([data_name])
        self.meta_valid.loc[idx_remove, data_name] = "filtered"

        df_temp = self.__class__(work_df.copy(), timedata_column=self.timename,
                                data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit)
        df_temp.data.loc[idx_remove, data_name] = np.nan
        df_temp.data[data_name] = df_temp.data[data_name].dropna()
        len_new = df_temp.data[data_name].count()

        _print_removed_output(len_orig, len_new, f"IQR filter (k={k})")
        if isinstance(log_file, str):
            _log_removed_output(log_file, len_orig, len_new, type_="filtered")
        elif log_file is not None:
            raise TypeError("`log_file` must be a string path or None.")

        if plot and hasattr(self, "plot_analysed"):
            try:
                self.plot_analysed(data_name)
            except Exception:
                wn.warn("plot_analysed failed; continuing.", RuntimeWarning, stacklevel=2)

        if final:
            if inplace:
                s_new = self.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                self.data.loc[:, data_name] = s_new
                self._update_time()
                return None
            else:
                out = self.__class__(self.data.copy(), timedata_column=self.timename,
                                    data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit)
                s_new = out.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                out.data.loc[:, data_name] = s_new
                out._update_time()
                return out

        return None
    
    def rolling_iqr_filter(
        self,
        data_name: str,
        *,
        window: int = 25,
        q_low: float = 0.25,
        q_high: float = 0.75,
        k: float = 1.5,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        clear: bool = False,
        inplace: bool = False,
        log_file: Optional[str] = None,
        plot: bool = False,
        final: bool = False,
    ) -> Optional["HydroData"]:
        """
        Tag outliers using a rolling IQR fence:
        value < Q1(w) - k*IQR(w)  or  value > Q3(w) + k*IQR(w),
        where Q1,Q3 are computed within a centered rolling window.

        Parameters
        ----------
        data_name : str
            Column to filter.
        window : int, default 25
            Centered rolling window size (should be >= 5 or so).
        q_low, q_high : float, default 0.25, 0.75
            Quantiles for lower/upper rolling bounds.
        k : float, default 1.5
            IQR multiplier (3.0 is more conservative).
        arange, clear, inplace, log_file, plot, final : see other filters.

        Returns
        -------
        None or HydroData
        """
        self._plot = "valid"

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")
        if window < 3:
            raise ValueError("`window` must be >= 3.")
        if not (0 < q_low < q_high < 1):
            raise ValueError("q_low and q_high must satisfy 0 < q_low < q_high < 1.")

        if arange is None:
            work_df = self.data.copy()
            idx_slice = work_df.index
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list.")
            start, end = arange
            try:
                work_df = self.data.loc[start:end].copy()
                idx_slice = work_df.index
            except Exception as e:
                raise TypeError(
                    f"Slicing not possible for index type {type(self.data.index[0])} with arange {arange}."
                ) from e

        if work_df.empty:
            return None

        s = pd.to_numeric(work_df[data_name], errors="coerce")
        len_orig = s.count()

        q1 = s.rolling(window, center=True).quantile(q_low)
        q3 = s.rolling(window, center=True).quantile(q_high)
        iqr = q3 - q1

        low = q1 - k * iqr
        high = q3 + k * iqr

        to_filter = (s < low) | (s > high)
        to_filter = to_filter & q1.notna() & q3.notna() & iqr.notna()
        idx_remove = to_filter[to_filter].index

        if clear:
            self._reset_meta_valid(data_name)
        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)
        if data_name not in self.meta_valid.columns:
            self.add_to_meta_valid([data_name])

        self.meta_valid.loc[idx_remove, data_name] = "filtered"

        df_temp = self.__class__(work_df.copy(), timedata_column=self.timename,
                                data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit)
        df_temp.data.loc[idx_remove, data_name] = np.nan
        df_temp.data[data_name] = df_temp.data[data_name].dropna()
        len_new = df_temp.data[data_name].count()

        _print_removed_output(len_orig, len_new, f"rolling IQR filter (window={window}, k={k})")
        if isinstance(log_file, str):
            _log_removed_output(log_file, len_orig, len_new, type_="filtered")
        elif log_file is not None:
            raise TypeError("`log_file` must be a string path or None.")

        if plot and hasattr(self, "plot_analysed"):
            try:
                self.plot_analysed(data_name)
            except Exception:
                wn.warn("plot_analysed failed; continuing.", RuntimeWarning, stacklevel=2)

        if final:
            if inplace:
                s_new = self.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                self.data.loc[:, data_name] = s_new
                self._update_time()
                return None
            else:
                out = self.__class__(self.data.copy(), timedata_column=self.timename,
                                    data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit)
                s_new = out.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                out.data.loc[:, data_name] = s_new
                out._update_time()
                return out

        return None
    
    def hampel_filter(
        self,
        data_name: str,
        *,
        window: int = 11,
        k: float = 3.0,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        clear: bool = False,
        inplace: bool = False,
        log_file: Optional[str] = None,
        plot: bool = False,
        final: bool = False,
    ) -> Optional["HydroData"]:
        """
        Hampel spike detector: tag points where
        |x - rolling_median| > k * 1.4826 * rolling_MAD

        Parameters
        ----------
        data_name : str
            Column to filter.
        window : int, default 11
            Rolling window size (should be odd; function will handle even by using center=True anyway).
        k : float, default 3.0
            Threshold multiplier on MAD.
        arange, clear, inplace, log_file, plot, final : see other filters.

        Returns
        -------
        None or HydroData
        """
        self._plot = "valid"

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not found in data.")
        if window < 3:
            raise ValueError("`window` should be >= 3 for Hampel filtering.")

        if arange is None:
            work_df = self.data.copy()
            idx_slice = work_df.index
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list.")
            start, end = arange
            try:
                work_df = self.data.loc[start:end].copy()
                idx_slice = work_df.index
            except Exception as e:
                raise TypeError(
                    f"Slicing not possible for index type {type(self.data.index[0])} with arange {arange}."
                ) from e

        s = pd.to_numeric(work_df[data_name], errors="coerce")
        len_orig = s.count()

        med = s.rolling(window=window, center=True).median()
        mad = (s - med).abs().rolling(window=window, center=True).median()
        robust_sigma = 1.4826 * mad
        thresh = k * robust_sigma

        to_filter = (s - med).abs() > thresh
        to_filter = to_filter & med.notna() & robust_sigma.notna()
        idx_remove = to_filter[to_filter].index

        if clear:
            self._reset_meta_valid(data_name)
        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)
        if data_name not in self.meta_valid.columns:
            self.add_to_meta_valid([data_name])
        self.meta_valid.loc[idx_remove, data_name] = "filtered"

        df_temp = self.__class__(work_df.copy(), timedata_column=self.timename,
                                data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit)
        df_temp.data.loc[idx_remove, data_name] = np.nan
        df_temp.data[data_name] = df_temp.data[data_name].dropna()
        len_new = df_temp.data[data_name].count()

        _print_removed_output(len_orig, len_new, f"Hampel filter (window={window}, k={k})")
        if isinstance(log_file, str):
            _log_removed_output(log_file, len_orig, len_new, type_="filtered")
        elif log_file is not None:
            raise TypeError("`log_file` must be a string path or None.")

        if plot and hasattr(self, "plot_analysed"):
            try:
                self.plot_analysed(data_name)
            except Exception:
                wn.warn("plot_analysed failed; continuing.", RuntimeWarning, stacklevel=2)

        if final:
            if inplace:
                s_new = self.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                self.data.loc[:, data_name] = s_new
                self._update_time()
                return None
            else:
                out = self.__class__(self.data.copy(), timedata_column=self.timename,
                                    data_type=self.data_type, experiment_tag=self.tag, time_unit=self.time_unit)
                s_new = out.data[data_name].copy()
                s_new.loc[idx_remove] = np.nan
                out.data.loc[:, data_name] = s_new
                out._update_time()
                return out

        return None
    
#     ##########################
#     ### FILTERING MULTIVARIATE
#     ##########################

    def isolation_forest_filter(
        self,
        columns: Sequence[str],
        *,
        contamination: float = 0.01,
        n_estimators: int = 200,
        max_samples: Union[int, float, str] = "auto",
        random_state: Optional[int] = 0,
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        clear: bool = False,
        inplace: bool = False,
        log_file: Optional[str] = None,
        plot: bool = False,
        final: bool = False,
        ):
        """
        Multivariate outlier detection using Isolation Forest.

        Parameters
        ----------
        columns : sequence of str
            Columns to use (must exist).
        contamination : float, default 0.01
            Expected outlier fraction.
        n_estimators, max_samples, random_state : IsolationForest params.
        arange : (start, end) or None
            Index slice to apply the detector. None -> full index.
        clear : bool
            If True, reset tags for these columns before tagging.
        inplace, log_file, plot, final : like other filters.

        Returns
        -------
        None or HydroData
            If `final and not inplace`: returns a new HydroData with NaNs at outlier rows for given columns.
            Else None.
        """
        try:
            from sklearn.ensemble import IsolationForest
        except Exception:
            wn.warn("sklearn is required for isolation forest filtering. Skipping.", RuntimeWarning, stacklevel=2)
            return None
        
        self._plot = "valid"

        if not columns:
            raise ValueError("`columns` must be a non-empty sequence of column names.")
        missing = [c for c in columns if c not in self.data.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        if arange is None:
            work_df = self.data.copy()
            idx_slice = work_df.index
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list.")
            start, end = arange
            try:
                work_df = self.data.loc[start:end].copy()
                idx_slice = work_df.index
            except Exception as e:
                raise TypeError(
                    f"Slicing not possible for index type {type(self.data.index[0])} with arange {arange}."
                ) from e

        if work_df.empty:
            return None

        X = work_df[columns].apply(pd.to_numeric, errors="coerce")
        X_fit = X.dropna(axis=0, how="any")
        if X_fit.empty:
            wn.warn("No complete rows for the selected columns in the chosen range.", RuntimeWarning, stacklevel=2)
            return None

        clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        y_pred = clf.fit_predict(X_fit)  
        out_idx = X_fit.index[y_pred == -1]

        len_orig = len(X_fit)
        len_new = len_orig - len(out_idx)

        if clear:
            self._reset_meta_valid(columns)
        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)

        for col in columns:
            if col not in self.meta_valid.columns:
                self.add_to_meta_valid([col])
            self.meta_valid.loc[out_idx, col] = "filtered"

        _print_removed_output(len_orig, len_new, f"IsolationForest (contamination={contamination})")
        if isinstance(log_file, str):
            _log_removed_output(log_file, len_orig, len_new, type_="filtered")
        elif log_file is not None:
            raise TypeError("`log_file` must be a string path or None.")

        if plot and hasattr(self, "plot_analysed") and len(columns) == 1:
            try:
                self.plot_analysed(columns[0])
            except Exception:
                wn.warn("plot_analysed failed; continuing.", RuntimeWarning, stacklevel=2)

        if final:
            if inplace:
                for col in columns:
                    s = self.data[col].copy()
                    s.loc[out_idx] = np.nan
                    self.data.loc[:, col] = s
                self._update_time()
                return None
            else:
                out = self.__class__(
                    data=self.data.copy(),
                    timedata_column=self.timename,
                    data_type=self.data_type,
                    experiment_tag=self.tag,
                    time_unit=self.time_unit,
                )
                for col in columns:
                    s = out.data[col].copy()
                    s.loc[out_idx] = np.nan
                    out.data.loc[:, col] = s
                out._update_time()
                return out

        return None
    
    def pca_filter(
        self,
        columns: Sequence[str],
        *,
        standardize: bool = True,
        retain_var: float = 0.95,
        method: Literal["percentile", "zscore"] = "percentile",
        q: float = 0.995,                
        k: float = 3.0,                 
        arange: Optional[Union[Sequence[object], Tuple[object, object]]] = None,
        clear: bool = False,
        inplace: bool = False,
        log_file: Optional[str] = None,
        plot: bool = False,
        final: bool = False,
        ):
        """
        Multivariate outlier detection via PCA reconstruction error.

        Parameters
        ----------
        columns : sequence of str
            Columns to use (must exist).
        standardize : bool, default True
            Standardize features before PCA (recommended).
        retain_var : float in (0,1], default 0.95
            Retain enough components to explain this fraction of variance.
        method : {'percentile','zscore'}
            Threshold rule for reconstruction error.
        q : float, default 0.995
            Percentile for the error threshold (used when method='percentile').
        k : float, default 3.0
            Z-score threshold for the error (used when method='zscore').
        arange, clear, inplace, log_file, plot, final : like other filters.

        Returns
        -------
        None or HydroData
            If `final and not inplace`: new HydroData with NaNs at outlier rows/columns.
            Else None.
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except Exception:
            wn.warn("sklearn is required for pca filtering. Skipping.", RuntimeWarning, stacklevel=2)
            return None

        self._plot = "valid"

        if not columns:
            raise ValueError("`columns` must be a non-empty sequence of column names.")
        missing = [c for c in columns if c not in self.data.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        if not (0 < retain_var <= 1.0):
            raise ValueError("`retain_var` must be in (0, 1].")
        if method not in ("percentile", "zscore"):
            raise ValueError("`method` must be 'percentile' or 'zscore'.")

        if arange is None:
            work_df = self.data.copy()
            idx_slice = work_df.index
        else:
            if not isinstance(arange, (tuple, list)) or len(arange) != 2:
                raise ValueError("`arange` must be a (start, end) tuple/list.")
            start, end = arange
            try:
                work_df = self.data.loc[start:end].copy()
                idx_slice = work_df.index
            except Exception as e:
                raise TypeError(
                    f"Slicing not possible for index type {type(self.data.index[0])} with arange {arange}."
                ) from e

        if work_df.empty:
            return None

        X = work_df[columns].apply(pd.to_numeric, errors="coerce")
        X_fit = X.dropna(axis=0, how="any")
        if X_fit.empty:
            wn.warn("No complete rows for the selected columns in the chosen range.", RuntimeWarning, stacklevel=2)
            return None

        if standardize:
            scaler = StandardScaler()
            Z = scaler.fit_transform(X_fit.values)
        else:
            scaler = None
            Z = X_fit.values

        pca_full = PCA().fit(Z)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumsum, retain_var) + 1)
        n_components = max(1, min(n_components, Z.shape[1]))

        pca = PCA(n_components=n_components).fit(Z)
        Z_proj = pca.transform(Z)
        Z_recon = pca.inverse_transform(Z_proj)
        err = ((Z - Z_recon) ** 2).mean(axis=1)
        err_series = pd.Series(err, index=X_fit.index, name="pca_recon_error")

        if method == "percentile":
            thr = np.nanquantile(err_series, q)
            out_idx = err_series.index[err_series >= thr]
            label = f"PCA recon (retain={retain_var:.2f}, q={q})"
        else:
            mu = err_series.mean(); sd = err_series.std(ddof=0)
            if not (np.isfinite(sd) and sd > 0):
                out_idx = err_series.index[[]]
            else:
                z = (err_series - mu) / sd
                out_idx = err_series.index[(z >= k)]
            label = f"PCA recon (retain={retain_var:.2f}, z≥{k})"

        len_orig = len(X_fit)
        len_new = len_orig - len(out_idx)

        if clear:
            self._reset_meta_valid(columns)
        if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            self.meta_valid = self.meta_valid.reindex(self.data.index)

        for col in columns:
            if col not in self.meta_valid.columns:
                self.add_to_meta_valid([col])
            self.meta_valid.loc[out_idx, col] = "filtered"

        _print_removed_output(len_orig, len_new, label)
        if isinstance(log_file, str):
            _log_removed_output(log_file, len_orig, len_new, type_="filtered")
        elif log_file is not None:
            raise TypeError("`log_file` must be a string path or None.")

        if plot:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(np.sort(err_series.values))
                ax.set_title(label)
                ax.set_ylabel("reconstruction error")
                ax.set_xlabel("sorted samples")
                fig.tight_layout()
            except Exception:
                wn.warn("Plot failed; continuing.", RuntimeWarning, stacklevel=2)

        if final:
            if inplace:
                for col in columns:
                    s = self.data[col].copy()
                    s.loc[out_idx] = np.nan
                    self.data.loc[:, col] = s
                self._update_time()
                return None
            else:
                out = self.__class__(
                    data=self.data.copy(),
                    timedata_column=self.timename,
                    data_type=self.data_type,
                    experiment_tag=self.tag,
                    time_unit=self.time_unit,
                )
                for col in columns:
                    s = out.data[col].copy()
                    s.loc[out_idx] = np.nan
                    out.data.loc[:, col] = s
                out._update_time()
                return out

        return None
    
  


# #==============================================================================
# # DATA (COR)RELATION
# #==============================================================================

    def calc_ratio(
        self,
        data_1: str,
        data_2: str,
        arange: tuple,
        *,
        only_checked: bool = False,
    ) -> tuple[float, float]:
        """
        Calculate the average ratio between two time series (data_1 / data_2),
        within a given range, with optional exclusion of previously filtered values.

        Parameters
        ----------
        data_1 : str
            Column name for numerator.
        data_2 : str
            Column name for denominator.
        arange : (start, end)
            The index range (inclusive slicing) within which the ratio is calculated.
        only_checked : bool, default False
            If True, only values marked as 'original' in self.meta_valid are used.

        Returns
        -------
        (mean, std) : tuple of float
            Mean ratio and standard deviation of (data_1 / data_2) within the range.
        """
        for col in (data_1, data_2):
            if col not in self.data.columns:
                raise KeyError(f"Column '{col}' not found in data.")

        try:
            _ = self.data.loc[arange[0]:arange[1]]
        except TypeError:
            raise TypeError(
                f"Slicing not possible for index type {type(self.data.index[0])} "
                f"with arange element types {type(arange[0])}, {type(arange[1])}. "
                "Ensure arange values are compatible with the index."
            )

        if arange[0] < self.index()[0] or arange[1] > self.index()[-1]:
            raise IndexError(
                "Index out of bounds. Ensure that 'arange' values fall within the data index range."
            )

        s1 = self.data.loc[arange[0]:arange[1], data_1]
        s2 = self.data.loc[arange[0]:arange[1], data_2]

        if only_checked:
            mask1 = self.meta_valid[data_1].loc[arange[0]:arange[1]] == "original"
            mask2 = self.meta_valid[data_2].loc[arange[0]:arange[1]] == "original"
            mask = mask1 & mask2
            s1 = s1[mask]
            s2 = s2[mask]

        ratios = (s1 / s2).replace([np.inf, -np.inf], np.nan).dropna()

        if ratios.empty:
            mean, std = np.nan, np.nan
        else:
            mean, std = ratios.mean(), ratios.std()

        return mean, std



    def compare_ratio(
        self,
        data_1: str,
        data_2: str,
        arange: int,
        *,
        only_checked: bool = False,
        verbose: bool = False,
    ) -> Tuple[float, float]:
        """
        Compare average ratios (data_1/data_2) over multiple non-overlapping
        windows and return the most reliable one (min relative std = std/|mean|).

        Parameters
        ----------
        data_1 : str
            Column name for numerator.
        data_2 : str
            Column name for denominator.
        arange : int
            Window length. If index is DatetimeIndex, interpreted as *days*.
            If index is numeric, interpreted as the same unit as the index.
        only_checked : bool, default False
            If True, use only values tagged as 'original' in `meta_valid`.
        verbose : bool, default False
            If True, prints the winning window [start, end].

        Returns
        -------
        (best_mean, best_std) : tuple[float, float]
            The average ratio and standard deviation for the best window.
            Returns (nan, nan) if no valid window produced finite results.
        """
        for col in (data_1, data_2):
            if col not in self.data.columns:
                raise KeyError(f"Column '{col}' not found in data.")

        if arange <= 0:
            raise ValueError("`arange` must be a positive integer.")

        if len(self.data.index) == 0:
            return (np.nan, np.nan)

        idx = self.data.index
        start_full = idx[0]
        end_full = idx[-1]

        windows: list[Tuple[object, object]] = []

        if isinstance(idx, pd.DatetimeIndex):
            delta = pd.Timedelta(days=arange)
            cur_start = start_full
            while cur_start < end_full:
                cur_end = cur_start + delta
                if cur_end > end_full:
                    break
                windows.append((cur_start, cur_end))
                cur_start = cur_end
        else:
            cur_start_val = float(start_full)
            last_val = float(end_full)
            while cur_start_val < last_val:
                cur_end_val = cur_start_val + float(arange)
                if cur_end_val > last_val:
                    break
                windows.append((cur_start_val, cur_end_val))
                cur_start_val = cur_end_val

        if not windows:
            return (np.nan, np.nan)

        best_rel_std = np.inf
        best_mean = np.nan
        best_std = np.nan
        best_win: Optional[Tuple[object, object]] = None

        for w in windows:
            try:
                mean, std = self.calc_ratio(data_1, data_2, w, only_checked=only_checked)
            except Exception:
                continue

            if not (np.isfinite(mean) and np.isfinite(std)):
                continue
            if mean == 0:
                continue

            rel_std = float(std) / abs(float(mean))
            if rel_std < best_rel_std:
                best_rel_std = rel_std
                best_mean = float(mean)
                best_std = float(std)
                best_win = w

        if verbose and best_win is not None:
            print(f"Best ratio ({best_mean} ± {best_std}) in range: {best_win}")

        return (best_mean, best_std)
    

    def get_correlation(
        self,
        data_1: str,
        data_2: str,
        arange: Tuple[object, object],
        *,
        zero_intercept: bool = False,
        only_checked: bool = False,
        plot: bool = False,
    ) -> Union[Tuple[float, float, float], Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]]:
        """
        Linear relationship between two series: data_2 ~ a * data_1 (+ b).
        Returns slope (a), intercept (b), and R^2. If plot=True, returns (fig, ax).

        Parameters
        ----------
        data_1 : str
            Column name for the independent variable (X).
        data_2 : str
            Column name for the dependent variable (Y).
        arange : (start, end)
            Index range (inclusive slicing) over which to compute the correlation.
        zero_intercept : bool, default False
            If True, fit with intercept fixed at 0.
        only_checked : bool, default False
            If True, use only rows tagged 'original' in meta_valid for BOTH columns.
        plot : bool, default False
            If True, plot scatter with fitted line and 95% prediction interval.

        Returns
        -------
        (slope, intercept, r_sq) or (fig, ax) if plot=True
        """
        for col in (data_1, data_2):
            if col not in self.data.columns:
                raise KeyError(f"Column '{col}' not found in data.")

        try:
            df_slice = self.data.sort_index().loc[arange[0]:arange[1]].copy()
        except TypeError as e:
            raise TypeError(
                f"Slicing not possible for index type {type(self.data.index[0])} "
                f"with arange element types {type(arange[0])}, {type(arange[1])}. "
                "Ensure arange values are compatible with the index."
            ) from e

        if df_slice.empty:
            wn.warn("Selected range is empty; returning NaNs.", RuntimeWarning, stacklevel=2)
            return (np.nan, np.nan, np.nan) if not plot else (None, None)

        if only_checked:
            if getattr(self, "meta_valid", None) is None or self.meta_valid.empty:
                wn.warn("meta_valid is empty; only_checked=True has no effect.", RuntimeWarning, stacklevel=2)
            else:
                mv = self.meta_valid.reindex(df_slice.index)
                m1 = (mv.get(data_1, pd.Series(index=df_slice.index, dtype=object)) == "original")
                m2 = (mv.get(data_2, pd.Series(index=df_slice.index, dtype=object)) == "original")
                mask = m1 & m2
                df_slice = df_slice.loc[mask]

        X_raw = pd.to_numeric(df_slice[data_1], errors="coerce")
        Y_raw = pd.to_numeric(df_slice[data_2], errors="coerce")
        valid = X_raw.notna() & Y_raw.notna()
        X = X_raw[valid]
        Y = Y_raw[valid]

        if len(X) < 2:
            wn.warn("Not enough valid points to fit a regression; returning NaNs.", RuntimeWarning, stacklevel=2)
            return (np.nan, np.nan, np.nan) if not plot else (None, None)

        try:
            import statsmodels.api as sm
        except Exception as e:
            raise ImportError("statsmodels is required for get_correlation.") from e

        if zero_intercept:
            exog = X.values  
            exog = exog.reshape(-1, 1)
            model = sm.OLS(Y.values, exog)
            results = model.fit()
            slope = float(results.params[0])
            intercept = 0.0
            r_sq = float(results.rsquared)
        else:
            exog = sm.add_constant(X.values)  
            model = sm.OLS(Y.values, exog)
            results = model.fit()
            intercept = float(results.params[0])
            slope = float(results.params[1])
            r_sq = float(results.rsquared)

        if not plot:
            return (slope, intercept, r_sq)

        import matplotlib.pyplot as plt

        x_sorted = np.sort(X.values)
        if zero_intercept:
            exog_pred = x_sorted.reshape(-1, 1)
        else:
            exog_pred = sm.add_constant(x_sorted)

        pred = results.get_prediction(exog_pred)
        pred_summary = pred.summary_frame(alpha=0.05)  

        y_fit = pred_summary["mean"].values
        y_lo = pred_summary["obs_ci_lower"].values  
        y_hi = pred_summary["obs_ci_upper"].values

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(X.values, Y.values, "o", markerfacecolor="none", markeredgewidth=1, markeredgecolor="b",
                markersize=4, label="Data")
        ax.plot(x_sorted, y_fit, "k", label="Linear fit")
        ax.fill_between(x_sorted.astype(float), y_lo, y_hi, alpha=0.2, label="Prediction interval (95%)")

        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        ax.set_xlabel(data_1, size=14)    
        ax.set_ylabel(data_2, size=14)     
        ax.set_title(f"slope={slope:.4g}  intercept={intercept:.4g}  R²={r_sq:.4f}", fontsize=12)
        fig.tight_layout()

        return fig, ax

# #==============================================================================
# # DAILY PROFILE CALCULATION
# #==============================================================================


    def calc_daily_profile(
        self,
        column_name: str,
        arange: Optional[Tuple[object, object]] = None,
        *,
        quantile: float = 0.9,
        plot: bool = False,
        plot_method: str = "quantile",  
        clear: bool = False,
        only_checked: bool = False,
    ):
        """
        Compute a typical daily profile (time-of-day statistics) for a series.

        Parameters
        ----------
        column_name : str
            Column to analyze.
        arange : (start, end) or None, default None
            Index range to include in the aggregation.
            - If None, uses the full dataset.
            - If DatetimeIndex: accepts datetime-like or (int,int) day offsets from start.
        quantile : float, default 0.9
            Upper quantile; lower band uses 1 - quantile.
        plot : bool, default False
            If True, returns (fig, ax) with mean curve and envelope.
        plot_method : {"quantile","stdev"}
            Which envelope to plot: quantile bands or mean ± std.
        clear : bool, default False
            If True and a profile for this column already exists, overwrite it.
        only_checked : bool, default False
            If True, use only rows marked 'original' in self.meta_valid[column_name].

        Returns
        -------
        dict or (fig, ax)
            Updates self.daily_profile[column_name] with a DataFrame indexed by
            time-of-day containing ['avg','std','Qupper','Qlower'].
            If `plot=True`, returns the figure and axis.
        """
        import datetime as dt
        import matplotlib.pyplot as plt
        import warnings as wn
        import numpy as np
        import pandas as pd

        # --- checks & setup ---
        if column_name not in self.data.columns:
            raise KeyError(f"Column '{column_name}' not found in data.")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("calc_daily_profile requires a DatetimeIndex.")
        if not (0 < quantile < 1):
            raise ValueError("`quantile` must be between 0 and 1.")

        if not hasattr(self, "daily_profile") or not isinstance(getattr(self, "daily_profile"), dict):
            self.daily_profile = {}
        if clear:
            self.daily_profile.pop(column_name, None)
        elif column_name in self.daily_profile:
            raise KeyError(
                f"daily_profile already contains '{column_name}'. "
                f"Pass clear=True to overwrite."
            )

        idx = self.data.index.sort_values()
        first_day = idx[0].normalize()
        last_day = idx[-1].normalize()

        if arange is None:
            start_dt = first_day
            end_dt_exclusive = last_day + pd.Timedelta(days=1)
        else:
            start, end = arange
            if isinstance(start, int) and isinstance(end, int):
                start_dt = first_day + pd.Timedelta(days=int(start))
                end_dt_exclusive = first_day + pd.Timedelta(days=int(end))
            else:
                try:
                    start_dt = pd.to_datetime(start)
                    end_dt = pd.to_datetime(end)
                except Exception as e:
                    raise TypeError("`arange` must be (int,int) or (datetime-like, datetime-like).") from e
                start_dt = start_dt.normalize()
                end_dt_exclusive = (end_dt + pd.Timedelta(days=1)).normalize()

        if start_dt < first_day or end_dt_exclusive > (last_day + pd.Timedelta(days=1)):
            wn.warn("`arange` extends beyond available data range; truncated.", RuntimeWarning, stacklevel=2)
            start_dt = max(start_dt, first_day)
            end_dt_exclusive = min(end_dt_exclusive, last_day + pd.Timedelta(days=1))

        df = self.data.loc[start_dt:end_dt_exclusive - pd.Timedelta(microseconds=1), [column_name]].copy()
        if df.empty:
            wn.warn("Selected range is empty; no daily profile computed.", RuntimeWarning, stacklevel=2)
            return self.daily_profile

        try:
            if getattr(self, "data_type", None) == "WWTP" and hasattr(self, "highs") and "highs" in self.highs:
                highs_in_window = self.highs.loc[start_dt:end_dt_exclusive, "highs"].sum()
                if highs_in_window > 0:
                    wn.warn("Rain/high events present; profile may not be representative.", RuntimeWarning, stacklevel=2)
        except Exception:
            pass

        if only_checked and hasattr(self, "meta_valid") and column_name in self.meta_valid:
            mv = self.meta_valid.reindex(df.index)
            df = df[mv[column_name] == "original"]

        s = pd.to_numeric(df[column_name], errors="coerce").dropna()
        if s.empty:
            wn.warn("No valid samples to aggregate in selected window.", RuntimeWarning, stacklevel=2)
            return self.daily_profile

        tod = s.index.time
        grp = s.groupby(tod)
        mean_day = pd.DataFrame(index=pd.Index(sorted(set(tod)), name="time_of_day"))
        mean_day["avg"] = grp.mean().reindex(mean_day.index)
        mean_day["std"] = grp.std().reindex(mean_day.index)
        mean_day["Qupper"] = grp.quantile(quantile).reindex(mean_day.index)
        mean_day["Qlower"] = grp.quantile(1 - quantile).reindex(mean_day.index)

        self.daily_profile[column_name] = mean_day

        if not plot:
            return self.daily_profile

        base_date = dt.date(2000, 1, 1)
        x = pd.to_datetime([dt.datetime.combine(base_date, t) for t in mean_day.index])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, mean_day["avg"], label="Average", linewidth=2)

        if plot_method.lower().startswith("q"):
            ax.plot(x, mean_day["Qupper"], alpha=0.6, label=f"Q{int(quantile*100)}")
            ax.plot(x, mean_day["Qlower"], alpha=0.6, label=f"Q{int((1-quantile)*100)}")
            ax.fill_between(x, mean_day["Qlower"], mean_day["Qupper"], alpha=0.25, label="Quantile band")
        else:
            upper = mean_day["avg"] + mean_day["std"]
            lower = mean_day["avg"] - mean_day["std"]
            ax.plot(x, upper, alpha=0.6, label="avg + std")
            ax.plot(x, lower, alpha=0.6, label="avg - std")
            ax.fill_between(x, lower, upper, alpha=0.25, label="±1 std")

        ax.set_xlim(x.min(), x.max())
        ax.set_xlabel("Time of day")
        ax.set_ylabel(column_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig, ax

   

#     ##############
#     ### PLOTTING
#     ##############
    def plot_analysed(
        self,
        data_name: str,
        time_range="default",
        only_checked: bool = False,
    ):
        """
        Plot values & their types (original, filtered, filled) for `data_name`
        over a given range. Works for both HydroData and OnlineSensorBased.

        - If self._plot == 'valid'  : uses self.meta_valid and self.data
        - If self._plot == 'filled' : uses self.meta_filled and self.filled (falls
        back to self.data if `filled` is missing)

        Parameters
        ----------
        data_name : str
        time_range : 'default' or [start, end]
            If 'default': full span. Otherwise slice using the provided bounds.
        only_checked : bool
            If True, exclude filtered values.

        Returns
        -------
        (fig, ax)
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        if data_name not in self.data.columns:
            raise KeyError(f"Column '{data_name}' not in data.")

        if getattr(self, "_plot", "valid") == "filled":
            meta = getattr(self, "meta_filled", None)
            values = getattr(self, "filled", None)
            if not isinstance(values, pd.DataFrame) or values.empty:
                values = self.data
            label_map = {
                "original": ("original", "g.", "Original"),
                "filtered": ("filtered", "r.", "Filtered"),
                "filled_interpol": ("filled_interpol", "b.", "Filled (interpolation)"),
                "filled_kalman": ("filled_kalman", "b.", "Filled (kalman)"),
                "filled_arima": ("filled_arima", "b.", "Filled (arima)"),
                "filled_gaussian": ("filled_gaussian", "b.", "Filled (gaussian process)"),
                "filled_ratio": ("filled_ratio", "m.", "Filled (ratio)"),
                "filled_correlation": ("filled_correlation", "k.", "Filled (corr)"),
                "filled_average_profile": ("filled_average_profile", "y.", "Filled (typical day)"),
                "filled_model": ("filled_model", "c.", "Filled (model)"),
                "filled_profile_day_before": ("filled_profile_day_before", ".", "Filled (prev day)"),
            }
        else:
            meta = getattr(self, "meta_valid", None)
            values = self.data
            label_map = {
                "original": ("original", "g.", "Original"),
                "filtered": ("filtered", "r.", "Filtered"),
            }

        idx = self.data.index
        series_vals = pd.to_numeric(values.get(data_name).reindex(idx), errors="coerce")
        if isinstance(meta, pd.DataFrame) and data_name in meta:
            series_meta = meta[data_name].reindex(idx)
        else:
            series_meta = pd.Series(index=idx, data="original")

        if time_range == "default":
            start_i, end_i = idx.min(), idx.max()
        else:
            start_i, end_i = time_range
        view_idx = self.data.loc[start_i:end_i].index
        s_val = series_vals.reindex(view_idx)
        s_meta = series_meta.reindex(view_idx)

        if only_checked:
            s_val = s_val.where(s_meta != "filtered")

        fig, ax = plt.subplots(figsize=(16, 6))

        order = [
            "original",
            "filled_interpol",
            "filled_kalman",
            "filled_arima",
            "filled_gaussian",
            "filled_ratio",
            "filled_correlation",
            "filled_average_profile",
            "filled_model",
            "filled_profile_day_before",
            "filtered",
        ]

        handles = []
        for key in order:
            if key not in label_map:
                continue
            tag, style, title = label_map[key]
            mask = (s_meta == tag)
            if mask.any():
                h = ax.plot(s_val.index[mask], s_val[mask], style, label=title)
                handles.append(h[0])

        if not handles:
            ax.plot(s_val.index, s_val.values, "-", label=data_name)

        ax.set_xlabel(self.timename if hasattr(self, "timename") else "Time")
        ax.set_ylabel(data_name)
        ax.tick_params(labelsize=14)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12)
        fig.tight_layout()

        counts = s_meta.value_counts(dropna=False).to_dict()
        summary = ", ".join(f"{k}:{v}" for k, v in counts.items())
        ax.set_title(f"{data_name}  —  {summary}")

        return fig, ax

   
# ##############################
# ###   NON-CLASS FUNCTIONS  ###
# ##############################


def _print_removed_output(original: int, new: int, function: str) -> int:
    """
    Print how many values were tagged/removed by a function and return that count.

    Parameters
    ----------
    original : int
        Original number of entries considered.
    new : int
        New number of non-missing/non-filtered entries after tagging.
    function : str
        Name/description of the function that performed the tagging.

    Returns
    -------
    int
        The number of values tagged (original - new).
    """
    removed = int(original) - int(new)
    print(f"{removed} values detected and tagged as filtered by function {function}")
    return removed



def _log_removed_output(
    log_file: str,
    original: int,
    new: int,
    type_: str = "removed",
    ) -> None:
    """
    Write information about removed/dropped datapoints to a log file.

    Parameters
    ----------
    log_file : str
        Path to the log file to append to.
    original : int
        Original number of datapoints.
    new : int
        Number of datapoints after removal.
    type_ : str, default "removed"
        Descriptor for the action (e.g., 'removed', 'dropped', 'filtered').

    Returns
    -------
    None
    """
    removed = original - new
    if removed < 0:
        removed = 0  
    message = (
        f"\nOriginal dataset: {original} datapoints; "
        f"new dataset: {new} datapoints; "
        f"{removed} datapoints {type_}."
    )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message)


def _ensure_parent_dir(path: Union[str, Path]) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def _export_params(
    steps: List[Dict[str, Any]],
    dest: Optional[Union[str, Path]] = None,
    *,
    fmt: Literal["json", "yaml"] = "json",
) -> Optional[str]:
    """
    Export pipeline parameters (fn + kwargs per step) to JSON/YAML.
    Returns the serialized string if dest is None, else writes file and returns dest path as str.
    """
    serializable = []
    for s in steps:
        serializable.append({
            "fn": s.get("fn"),
            "kwargs": s.get("kwargs", {})
        })
    if fmt == "yaml":
        if not _HAS_YAML:
            warnings.warn("PyYAML not installed; exporting as JSON instead.", RuntimeWarning, stacklevel=2)
            fmt = "json"

    if dest is None:
        return yaml.safe_dump(serializable, sort_keys=False) if fmt == "yaml" else json.dumps(serializable, indent=2)

    _ensure_parent_dir(dest)
    if fmt == "yaml":
        with open(dest, "w", encoding="utf-8") as f:
            yaml.safe_dump(serializable, f, sort_keys=False)
    else:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
    return str(dest)


def _export_summary(
    summary: pd.DataFrame,
    dest: Optional[Union[str, Path]] = None,
    *,
    fmt: Literal["csv", "parquet", "sqlite"] = "csv",
    sqlite_path: Optional[Union[str, Path]] = None,
    sqlite_table: str = "pipeline_summary",
) -> Optional[str]:
    """
    Export the summary table to CSV/Parquet or SQLite.
    Returns the path (or 'sqlite:<db>#<table>') if written, else None.
    """
    if summary is None or summary.empty:
        return None

    if fmt == "sqlite":
        if sqlite_path is None:
            raise ValueError("sqlite_path is required when fmt='sqlite'.")
        import sqlite3
        _ensure_parent_dir(sqlite_path)
        con = sqlite3.connect(str(sqlite_path))
        try:
            summary.to_sql(sqlite_table, con, if_exists="append", index=False)
        finally:
            con.close()
        return f"sqlite:{sqlite_path}#{sqlite_table}"

    if dest is None:
        raise ValueError("dest is required for csv/parquet export.")

    _ensure_parent_dir(dest)
    if fmt == "csv":
        summary.to_csv(dest, index=False)
    elif fmt == "parquet":
        summary.to_parquet(dest, index=False)
    else:
        raise ValueError("fmt must be one of {'csv','parquet','sqlite'}.")
    return str(dest)


def _snapshot_frame(
    df: pd.DataFrame,
    *,
    how: Literal["memory", "csv", "parquet", "sqlite"] = "memory",
    memory_store: Optional[Dict[str, pd.DataFrame]] = None,
    name: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    sqlite_path: Optional[Union[str, Path]] = None,
    sqlite_table: Optional[str] = None,
) -> Optional[str]:
    """
    Save a snapshot of the DataFrame in memory/CSV/Parquet/SQLite.
    Returns a locator string (path or sqlite:...) or None for memory.
    """
    if how == "memory":
        if memory_store is None:
            return None
        key = name or f"snapshot_{len(memory_store)+1}"
        memory_store[key] = df.copy()
        return None

    if how in ("csv", "parquet"):
        if path is None:
            raise ValueError("path is required for csv/parquet snapshots.")
        _ensure_parent_dir(path)
        if how == "csv":
            df.to_csv(path, index=True)
        else:
            df.to_parquet(path, index=True)
        return str(path)

    if how == "sqlite":
        if sqlite_path is None or sqlite_table is None:
            raise ValueError("sqlite_path and sqlite_table are required for sqlite snapshots.")
        import sqlite3
        _ensure_parent_dir(sqlite_path)
        con = sqlite3.connect(str(sqlite_path))
        try:
            df.to_sql(sqlite_table, con, if_exists="replace", index=True)
        finally:
            con.close()
        return f"sqlite:{sqlite_path}#{sqlite_table}"

    raise ValueError("how must be one of {'memory','csv','parquet','sqlite'}.")
