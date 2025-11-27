# -*- coding: utf-8 -*-
"""
Class_LabExperimBased provides functionalities for data handling of data obtained in lab experiments in the field of (waste)water treatment.
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
import sys

import matplotlib.pyplot as plt  
import warnings as wn
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Tuple, List, Literal, Union

from .Class_HydroData import HydroData

@dataclass
class ValidationMetrics:
    n: int
    mae: float
    rmse: float
    mbe: float            
    mape: float
    r: float              
    r2: float
    ccc: float            
    slope: float
    intercept: float

class LabExperimentBased(HydroData):

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        timedata_column: str = "index",
        data_type: str = "WWTP",
        experiment_tag: str = "No tag given",
        time_unit: Optional[str] = None,
        units: Optional[Dict[str, str]] = None,
        meta: Optional[pd.DataFrame] = None,       
        id_column: Optional[str] = None,           
    ):
        super().__init__(
            data=data,
            timedata_column=timedata_column,
            data_type=data_type,
            experiment_tag=experiment_tag,
            time_unit=time_unit,
            units=({} if units is None else units),
        )
        self.meta: pd.DataFrame = (
            meta.copy() if isinstance(meta, pd.DataFrame) else pd.DataFrame(index=self.data.index)
        )
        self.id_column: Optional[str] = id_column if id_column in self.data.columns else None

        self.meta_qc: pd.DataFrame = pd.DataFrame(index=self.data.index)


    def apply_dilutions_and_units(
        self,
        analytes: Iterable[str],
        *,
        dilution_factors: Union[float, Dict[str, float]] = 1.0,
        unit_map: Optional[Dict[str, Tuple[str, float]]] = None,
    ) -> None:
        """
        Apply dilution factors and unit conversions.
        unit_map: { 'NH4-N': ('mg/L as N', factor_to_apply) }
        """
        for a in analytes:
            factor = dilution_factors if isinstance(dilution_factors, (int, float)) else dilution_factors.get(a, 1.0)
            self.data[a] = pd.to_numeric(self.data[a], errors="coerce") * factor
            if unit_map and a in unit_map:
                new_unit, u_factor = unit_map[a]
                self.data[a] = self.data[a] * u_factor
                self.units[a] = new_unit

    def control_chart(
        self,
        analyte: str,
        *,
        subgroup: Optional[str] = None,
        window: int = 20,
    ) -> pd.DataFrame:
        """
        Compute basic Shewhart X-bar chart stats: mean and ±3std on rolling window (or by batch if subgroup provided).
        Returns a small DataFrame suitable for plotting.
        """
        s = pd.to_numeric(self.data[analyte], errors="coerce")
        if subgroup and subgroup in self.meta.columns:
            grp = self.meta.groupby(self.meta[subgroup])
            mu = grp.apply(lambda g: s.reindex(g.index).mean())
            sd = grp.apply(lambda g: s.reindex(g.index).std(ddof=1))
            out = pd.DataFrame({"center": mu, "ucl": mu + 3 * sd, "lcl": mu - 3 * sd})
        else:
            mu = s.rolling(window, min_periods=max(5, window//5)).mean()
            sd = s.rolling(window, min_periods=max(5, window//5)).std(ddof=1)
            out = pd.DataFrame({"center": mu, "ucl": mu + 3 * sd, "lcl": mu - 3 * sd})
        return out
    
    
    def align_with_reference(
        self,
        analyte: str,
        reference: Union[pd.Series, pd.DataFrame],
        *,
        method: Literal["nearest", "linear", "ffill"] = "nearest",
        tolerance: Optional[pd.Timedelta] = pd.Timedelta("30min"),
    ) -> pd.DataFrame:
        """
        Align lab analyte (discrete) with a continuous reference (sensor/model).
        Returns a DataFrame with columns ['lab','ref'] on lab timestamps.
        """
        if analyte not in self.data.columns:
            raise KeyError(f"'{analyte}' not in LabData.data columns")
        ref = _ensure_series(reference).copy()
        lab = pd.to_numeric(self.data[analyte], errors="coerce").dropna()
        if lab.empty:
            wn.warn("No lab samples to align.", stacklevel=2)
            return pd.DataFrame(columns=["lab","ref"])

        if method == "nearest":
            aligned = ref.reindex(lab.index, method="nearest", tolerance=tolerance)
        elif method == "linear":
            aligned = ref.reindex(ref.index.union(lab.index)).interpolate("time").reindex(lab.index)
        else:  # ffill
            aligned = ref.reindex(ref.index.union(lab.index)).ffill().reindex(lab.index)

        out = pd.DataFrame({"lab": lab, "ref": aligned})
        return out.dropna()
    

    def compute_validation_metrics(
        self,
        pairs: pd.DataFrame,
        *,
        bias_as: Literal["ref_minus_lab","lab_minus_ref"] = "ref_minus_lab",
        zero_guard: float = 1e-12,
    ) -> ValidationMetrics:
        """Compute common metrics on a ['lab','ref'] DataFrame."""
        df = pairs.dropna()
        n = len(df)
        if n == 0:
            return ValidationMetrics(0, *[np.nan]*10)

        lab = df["lab"].astype(float)
        ref = df["ref"].astype(float)

        err = (ref - lab) if bias_as == "ref_minus_lab" else (lab - ref)
        mae  = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        mbe  = float(np.mean(err))
        denom = np.maximum(np.abs(ref), zero_guard)
        mape = float(np.mean(np.abs(err) / denom) * 100.0)

        r = float(np.corrcoef(ref, lab)[0,1]) if n > 1 else np.nan
        r2 = r*r if np.isfinite(r) else np.nan
        x = ref.values
        y = lab.values
        x_mean, y_mean = np.mean(x), np.mean(y)
        sxy = np.sum((x - x_mean)*(y - y_mean))
        sxx = np.sum((x - x_mean)**2)
        slope = float(sxy / sxx) if sxx > 0 else np.nan
        intercept = float(y_mean - slope*x_mean) if np.isfinite(slope) else np.nan

        sx = np.var(x, ddof=1)
        sy = np.var(y, ddof=1)
        if n > 1 and np.isfinite(r) and sx > 0 and sy > 0:
            ccc = float(2*r*np.sqrt(sx*sy) / (sx + sy + (x_mean - y_mean)**2))
        else:
            ccc = np.nan

        return ValidationMetrics(n, mae, rmse, mbe, mape, r, r2, ccc, slope, intercept)
    

    def validate_against(
        self,
        analyte: str,
        reference: Union[pd.Series, pd.DataFrame],
        *,
        align_method: Literal["nearest","linear","ffill"] = "nearest",
        tolerance: Optional[pd.Timedelta] = pd.Timedelta("30min"),
        bias_as: Literal["ref_minus_lab","lab_minus_ref"] = "ref_minus_lab",
        return_pairs: bool = False,
    ) -> Union[ValidationMetrics, Tuple[ValidationMetrics, pd.DataFrame]]:
        """
        Validate lab analyte against a reference (sensor/model). Returns metrics, and optionally pairs.
        """
        pairs = self.align_with_reference(analyte, reference, method=align_method, tolerance=tolerance)
        metrics = self.compute_validation_metrics(pairs, bias_as=bias_as)
        return (metrics, pairs) if return_pairs else metrics
    
    def fit_calibration(
        self,
        analyte: str,
        reference: Union[pd.Series, pd.DataFrame],
        *,
        align_method: Literal["nearest","linear","ffill"] = "nearest",
        tolerance: Optional[pd.Timedelta] = pd.Timedelta("30min"),
        zero_intercept: bool = False,
    ) -> Dict[str, float]:
        """
        Fit lab = a*ref + b (or b=0 if zero_intercept). Returns {'slope','intercept','r2'}.
        """
        pairs = self.align_with_reference(analyte, reference, method=align_method, tolerance=tolerance)
        if pairs.empty:
            return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}

        x = pairs["ref"].values
        y = pairs["lab"].values
        if zero_intercept:
            sxx = np.sum(x*x)
            sxy = np.sum(x*y)
            slope = float(sxy / sxx) if sxx > 0 else np.nan
            intercept = 0.0
            yhat = slope * x
        else:
            x_mean, y_mean = np.mean(x), np.mean(y)
            sxx = np.sum((x - x_mean)**2)
            sxy = np.sum((x - x_mean)*(y - y_mean))
            slope = float(sxy / sxx) if sxx > 0 else np.nan
            intercept = float(y_mean - slope*x_mean) if np.isfinite(slope) else np.nan
            yhat = slope * x + intercept

        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan

        return {"slope": slope, "intercept": intercept, "r2": r2}

    def apply_calibration(
        self,
        reference: Union[pd.Series, pd.DataFrame],
        slope: float,
        intercept: float = 0.0,
        *,
        name: Optional[str] = None,
    ) -> pd.Series:
        """
        Apply lab = slope * reference + intercept, returned on the reference index.
        """
        ref = _ensure_series(reference).astype(float)
        est = slope * ref + intercept
        est.name = name or f"calibrated_{_name_or_default(ref,'ref')}"
        return est

    def rolling_validation(
        self,
        analyte: str,
        reference: Union[pd.Series, pd.DataFrame],
        *,
        window: int = 20,
        step: int = 5,
        align_method: Literal["nearest","linear","ffill"] = "nearest",
        tolerance: Optional[pd.Timedelta] = pd.Timedelta("30min"),
    ) -> pd.DataFrame:
        """
        Rolling window validation to monitor drift. Returns a DataFrame with window-end index and metrics.
        """
        pairs = self.align_with_reference(analyte, reference, method=align_method, tolerance=tolerance)
        if pairs.empty:
            return pd.DataFrame(columns=["n","mae","rmse","mbe","mape","r","r2","ccc","slope","intercept"])

        rows = []
        idxs = pairs.index
        for start in range(0, len(pairs)-window+1, step):
            win = pairs.iloc[start:start+window]
            m = self.compute_validation_metrics(win)
            rows.append({
                "end": idxs[min(start+window-1, len(pairs)-1)],
                **m.__dict__
            })
        out = pd.DataFrame(rows).set_index("end")
        return out


##############################
# Help functions
##############################

def _ensure_series(x: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    return x.squeeze() if isinstance(x, pd.DataFrame) else x

def _name_or_default(s: pd.Series, default: str) -> str:
    return s.name if s.name is not None else default