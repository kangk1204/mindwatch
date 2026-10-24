"""Small causal EMA state models; no survey outcomes or file loading.

The pooled model has a stationary Ornstein--Uhlenbeck deviation.  The
personalized model adds a static Gaussian random intercept, jointly filtered
with that deviation. Population parameters are estimated only from explicitly
allowed people and dates before ``cutoff``. Parameters minimize participant-
equal mean Gaussian innovation loss: this is empirical-Bayes shrinkage, not a
full Bayesian posterior over hyperparameters.

Dates are conservatively reduced to local calendar days. An anchor's entire
day is excluded. No EMA is interpolated, and no backward smoother is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mindwatch.features import DOMAINS

VERSION = "causal_ou_random_intercept_v1"
LN2 = math.log(2.0)
VAR_FLOOR = 1e-4


@dataclass(frozen=True)
class DomainParameters:
    mean: float
    half_life_days: float
    dynamic_variance: float
    observation_variance: float
    intercept_variance: float
    people: int
    observations: int
    repeated_people: int
    converged: bool
    iterations: int
    objective: float | None
    status: str
    boundary_parameters: tuple[str, ...] = ()


def _day(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None:
        raise ValueError("A nonmissing, timezone-resolved local date is required")
    return result.normalize()


def _days(values):
    result = pd.to_datetime(values, errors="raise")
    if result.isna().any() or result.dt.tz is not None:
        raise ValueError("Use nonmissing, timezone-resolved local dates")
    return result.dt.normalize()


def _identity(frame):
    if frame.columns.duplicated().any() or "subject_id" not in frame:
        raise ValueError("Unique columns and subject_id are required")
    if frame.subject_id.isna().any():
        raise ValueError("Missing subject key")


def _scoped_events(ema, subjects, cutoff, domains):
    """Filter dates/people before examining numeric values in the fit scope."""
    _identity(ema)
    if not {"date", *domains}.issubset(ema.columns):
        raise ValueError("EMA date and all declared domain columns are required")
    data = ema.loc[ema.subject_id.isin(subjects), ["subject_id", "date", *domains]].copy()
    data["date"] = _days(data.date)
    data = data.loc[data.date.lt(_day(cutoff))].copy()
    if data.duplicated(["subject_id", "date"]).any():
        raise ValueError("Duplicate subject/day EMA requires upstream resolution")
    for domain in domains:
        value = pd.to_numeric(data[domain], errors="raise").astype(float)
        valid = value.notna()
        if (valid & (~np.isfinite(value) | value.lt(0) | value.gt(10))).any():
            raise ValueError("EMA must be numeric 0--10 or missing")
        data[domain] = value
    # Do not stringify identifiers: integer and string keys are distinct.
    return data.sort_values("date", kind="stable").reset_index(drop=True)


def _propagate(mean, covariance, days, parameters):
    if days < 0 or not np.isfinite(days):
        raise ValueError("State propagation requires a nonnegative finite interval")
    decay = math.exp(-LN2 * days / parameters.half_life_days)
    transition = np.diag([decay, 1.0])
    process = np.diag([parameters.dynamic_variance * (1 - decay * decay), 0.0])
    return transition @ mean, transition @ covariance @ transition.T + process


def _update(mean, covariance, value, parameters):
    observation = np.ones(2)
    innovation = float(value - parameters.mean - observation @ mean)
    projected = covariance @ observation
    innovation_variance = float(observation @ projected + parameters.observation_variance)
    gain = projected / innovation_variance
    updated = mean + gain * innovation
    # Joseph form avoids loss of positive semidefiniteness in long histories.
    residual = np.eye(2) - np.outer(gain, observation)
    posterior = residual @ covariance @ residual.T + parameters.observation_variance * np.outer(gain, gain)
    return updated, (posterior + posterior.T) / 2, innovation, innovation_variance


def _panel(data, domain):
    groups = []
    for _, group in data.loc[data[domain].notna()].groupby("subject_id", sort=False):
        group = group.sort_values("date", kind="stable")
        groups.append((group.date.to_numpy(dtype="datetime64[D]").astype(np.int64),
                       group[domain].to_numpy(float)))
    if not groups:
        return None
    lengths = np.asarray([len(v) for _, v in groups], int)
    width = int(lengths.max())
    values = np.zeros((len(groups), width), float)
    intervals = np.zeros_like(values)
    valid = np.arange(width)[None, :] < lengths[:, None]
    for i, (dates, scores) in enumerate(groups):
        values[i, :len(scores)] = scores
        intervals[i, 1:len(scores)] = np.diff(dates)
    return values, intervals, valid, lengths


def _innovation_loss(log_parameters, panel, population_mean, personalized):
    """Vectorized over subjects; equal subject weight, then mean per event."""
    half, dynamic, noise = np.exp(log_parameters[:3])
    intercept = float(np.exp(log_parameters[3])) if personalized else 0.0
    values, intervals, valid, lengths = panel
    n = len(lengths)
    md = np.zeros(n); mb = np.zeros(n)
    pdd = np.full(n, dynamic); pbb = np.full(n, intercept); pdb = np.zeros(n)
    loss = np.zeros(n)
    for j in range(values.shape[1]):
        use = valid[:, j]
        a = np.exp(-LN2 * intervals[:, j] / half)
        md = a * md; pdd = a*a*pdd + dynamic*(1-a*a); pdb = a*pdb
        innovation = values[:, j] - population_mean - md - mb
        cross_d = pdd + pdb; cross_b = pbb + pdb
        variance = np.maximum(cross_d + cross_b + noise, VAR_FLOOR)
        loss += np.where(use, .5*(np.log(2*np.pi*variance) + innovation**2/variance), 0)
        kd = cross_d/variance; kb = cross_b/variance
        md = np.where(use, md+kd*innovation, md)
        mb = np.where(use, mb+kb*innovation, mb)
        pdd = np.where(use, np.maximum(pdd-kd*cross_d, 0), pdd)
        pbb = np.where(use, np.maximum(pbb-kb*cross_b, 0), pbb)
        pdb = np.where(use, pdb-kd*cross_b, pdb)
    return float(np.mean(loss/lengths))


class CausalStateSpace:
    """Fitted population dynamics plus strictly forward individual filtering.

    ``level`` is the posterior latent signal mean at the anchor. ``deviation``
    is the dynamic component relative to the shrunk individual baseline (or
    population mean for pooled models). ``uncertainty`` is the latent posterior
    standard deviation, not a calibrated clinical prediction interval.
    """

    def __init__(self, personalized=False, domains=DOMAINS, maxiter=60):
        self.personalized = bool(personalized)
        self.domains = tuple(domains)
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise ValueError("Declare nonempty unique EMA domains")
        self.maxiter = int(maxiter)
        if self.maxiter < 1:
            raise ValueError("maxiter must be positive")

    @property
    def feature_columns(self):
        return tuple(f"ss_{d}_{s}" for d in self.domains
                     for s in ("level", "deviation", "uncertainty"))

    def fit(self, ema, allowed_subjects, cutoff):
        allowed = set(allowed_subjects)
        if not allowed or any(pd.isna(x) for x in allowed):
            raise ValueError("Explicit nonmissing training subjects are required")
        data = _scoped_events(ema, allowed, cutoff, self.domains)
        self.cutoff_ = _day(cutoff)
        self.training_subjects_ = frozenset(data.subject_id.unique())
        self.parameters_ = {}
        self.diagnostics_ = {"version": VERSION, "personalized": self.personalized,
            "fit_cutoff_exclusive": str(self.cutoff_.date()),
            "fit_people": len(self.training_subjects_), "fit_event_rows": len(data),
            "fit_latest_date": None if data.empty else str(data.date.max().date()),
            "fit_earliest_date": None if data.empty else str(data.date.min().date()),
            "loss": "mean_over_people(mean_over_observations(Gaussian_innovation_NLL))",
            "uses_survey_outcomes": False, "uses_backward_smoothing": False, "domains": {}}
        # Hash fit content locally; never emit identities or individual values.
        self.fit_content_sha256_ = hashlib.sha256(
            pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes()).hexdigest()
        for domain in self.domains:
            panel = _panel(data, domain)
            if panel is None:
                self.parameters_[domain] = None
                self.diagnostics_["domains"][domain] = {"status": "unobserved_training_channel"}
                continue
            values, _, valid, lengths = panel
            means = (values*valid).sum(1)/lengths
            mean = float(means.mean())
            variance = float(np.mean((((values-mean)**2)*valid).sum(1)/lengths))
            within = float(np.mean((((values-means[:, None])**2)*valid).sum(1)/lengths))
            n, total, repeats = len(lengths), int(lengths.sum()), int((lengths >= 3).sum())
            if variance < 1e-8:
                params = DomainParameters(mean, 14., 0., VAR_FLOOR, 0., n, total,
                                          repeats, True, 0, None, "constant_training_channel")
            else:
                dynamic = max(within*.6 if self.personalized else variance*.6, .01)
                noise = max(within*.4, .01)
                between = max(variance-within, .01)
                x0 = [14., dynamic, noise] + ([between] if self.personalized else [])
                lower = [1., VAR_FLOOR, VAR_FLOOR] + ([VAR_FLOOR] if self.personalized else [])
                upper = [180., 50., 50.] + ([50.] if self.personalized else [])
                bounds = list(zip(np.log(lower), np.log(upper)))
                if repeats < 5:
                    optimum = np.log(x0); success=False; iterations=0
                    objective = _innovation_loss(optimum, panel, mean, self.personalized)
                    status = "sparse_history_fixed_moment_parameters"
                else:
                    result = minimize(_innovation_loss, np.log(x0), args=(panel,mean,self.personalized),
                        method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": self.maxiter, "ftol": 1e-8, "gtol": 1e-5, "maxls": 20})
                    initial_loss = _innovation_loss(np.log(x0), panel, mean, self.personalized)
                    if np.isfinite(result.fun) and result.fun <= initial_loss:
                        optimum=result.x;objective=float(result.fun)
                    else:
                        optimum=np.log(x0);objective=initial_loss
                    success=bool(result.success);iterations=int(result.nit)
                    status="fitted" if success else "bounded_fit_not_converged"
                fitted=np.exp(optimum)
                names=("half_life_days","dynamic_variance","observation_variance","intercept_variance")
                boundary=tuple(names[j] for j,(v,lo,hi) in enumerate(zip(fitted,lower,upper))
                               if v <= lo*1.01 or v >= hi*.99)
                params = DomainParameters(mean,float(fitted[0]),float(fitted[1]),float(fitted[2]),
                    float(fitted[3]) if self.personalized else 0.,n,total,repeats,success,iterations,
                    objective,status,boundary)
            self.parameters_[domain] = params
            self.diagnostics_["domains"][domain] = asdict(params)
        return self

    def _require_fit(self):
        if not hasattr(self,"parameters_"):
            raise ValueError("Fit population parameters before filtering")

    def _filter_history(self, history, anchor, domain):
        params=self.parameters_[domain]
        empty={"level":np.nan,"deviation":np.nan,"uncertainty":np.nan,
               "baseline":np.nan,"innovation":np.nan,"innovation_variance":np.nan,
               "n_observations":0,"last_date":None,"covariance":None,"status":"no_history"}
        if params is None:
            return {**empty,"status":"unobserved_training_channel"}
        observed=history.loc[history[domain].notna(),["date",domain]]
        if observed.empty:return empty
        mean=np.zeros(2);covariance=np.diag([params.dynamic_variance,params.intercept_variance])
        previous=observed.date.iloc[0]
        innovation=np.nan;innovation_variance=np.nan
        for date,value in observed.itertuples(index=False,name=None):
            mean,covariance=_propagate(mean,covariance,(date-previous).total_seconds()/86400.,params)
            mean,covariance,innovation,innovation_variance=_update(mean,covariance,float(value),params)
            previous=date
        mean,covariance=_propagate(mean,covariance,(anchor-previous).total_seconds()/86400.,params)
        variance=max(float(np.ones(2)@covariance@np.ones(2)),0.)
        return {"level":float(params.mean+mean.sum()),"deviation":float(mean[0]),
                "uncertainty":math.sqrt(variance),"baseline":float(params.mean+mean[1]),
                "innovation":float(innovation),"innovation_variance":float(innovation_variance),
                "n_observations":len(observed),"last_date":previous,"covariance":covariance.copy(),
                "status":params.status}

    def forecast_one(self, subject_id, anchor, ema):
        """EMA prior to a known prediction anchor; never infer future response dates."""
        self._require_fit();anchor=_day(anchor)
        history=_scoped_events(ema,{subject_id},anchor,self.domains)
        return {domain:self._filter_history(history,anchor,domain) for domain in self.domains}

    def transform(self, meta, ema):
        self._require_fit();_identity(meta)
        if "index_date" not in meta:
            raise ValueError("Prediction metadata requires index_date")
        anchors=_days(meta.index_date)
        result=np.full((len(meta),len(self.feature_columns)),np.nan)
        if not len(meta):return pd.DataFrame(result,index=meta.index,columns=self.feature_columns)
        data=_scoped_events(ema,set(meta.subject_id),anchors.max(),self.domains)
        groups={key:group for key,group in data.groupby("subject_id",sort=False)}
        empty=data.iloc[:0]
        for i,(subject,anchor) in enumerate(zip(meta.subject_id,anchors)):
            history=groups.get(subject,empty)
            history=history.loc[history.date.lt(anchor)]
            for j,domain in enumerate(self.domains):
                state=self._filter_history(history,anchor,domain)
                result[i,j*3:(j+1)*3]=[state[k] for k in ("level","deviation","uncertainty")]
        return pd.DataFrame(result,index=meta.index,columns=self.feature_columns)
