import pandas as pd, numpy as np, matplotlib.pyplot as plt
import statsmodels.api as sm

PEAK_START, PEAK_END = 1101, 1230

df = pd.read_csv("/content/cleaned_global_ecommerce_data.csv", parse_dates=["Date"])
df.head()

print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("Duplicate IDs:", df.Transaction_ID.duplicated().sum())

calc = df.Unit_Price * df.Quantity * (1 - df.Discount_Applied)
print("Revenue formula holds:", (abs(calc - df.Total_Revenue) < 0.05).mean())

print("Date range:", df.Date.min().date(), "to", df.Date.max().date())
print(df.groupby("Date").size().tail(3))   # last day is a partial day

df = df[df.Date < "2026-06-01"]   # drop the 1-day partial

daily = df.groupby("Date").agg(
    txns=("Transaction_ID", "count"),
    revenue=("Total_Revenue", "sum"),
    units=("Quantity", "sum"),
    avg_discount=("Discount_Applied", "mean"),
)
daily["aov"] = daily.revenue / daily.txns
daily["units_per_order"] = daily.units / daily.txns
md = daily.index.month * 100 + daily.index.day
daily["peak"] = ((md >= PEAK_START) & (md <= PEAK_END)).astype(int)

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
daily.txns.plot(ax=ax[0], title="Daily transactions")
daily.revenue.plot(ax=ax[1], title="Daily revenue")
plt.tight_layout();
plt.savefig("01_daily_series.png", dpi=150)

!pip install ruptures -q
import ruptures as rpt

sig = daily.txns.values.reshape(-1, 1)
bk = rpt.Binseg(model="l2").fit(sig).predict(n_bkps=6)
print([str(daily.index[b].date()) for b in bk if b < len(daily)])

"""Step 1: validate the peak window + check Poisson overdispersion.
Paste after `daily` is built (needs columns: txns, peak; DatetimeIndex)."""
import numpy as np, pandas as pd, statsmodels.api as sm, matplotlib.pyplot as plt


def peak_flag(idx, start_md, end_md):
    md = idx.month * 100 + idx.day
    return ((md >= start_md) & (md <= end_md)).astype(int)


def design(daily, peak=None):
    """Weekday dummies + peak flag + constant (original model ignored weekday)."""
    D = pd.get_dummies(daily.index.dayofweek, prefix="d", drop_first=True).astype(float)
    D = D.set_axis(daily.index)
    D["peak"] = daily.peak.values if peak is None else peak
    return sm.add_constant(D)


# ---- 1a. Is Nov 1 - Dec 30 actually the best window? -------------------------
def scan_windows(daily, starts, ends):
    rows = []
    for s in starts:
        for e in ends:
            X = design(daily, peak_flag(daily.index, s, e))
            f = sm.GLM(daily.txns, X, family=sm.families.Poisson()).fit()
            rows.append(dict(start=s, end=e, deviance=f.deviance,
                             multiplier=np.exp(f.params["peak"])))
    out = pd.DataFrame(rows).sort_values("deviance").reset_index(drop=True)
    out["delta_dev_vs_best"] = out.deviance - out.deviance.min()
    return out


starts = [1015, 1020, 1025, 1101, 1105, 1110, 1115, 1120]
ends = [1215, 1220, 1225, 1230, 1231]
scan = scan_windows(daily[daily.index < "2024-11-01"], starts, ends)
print(scan.head(8).round(3))
print("Your window (1101-1230):")
print(scan.query("start == 1101 and end == 1230").round(3))

# ---- 1b. Overdispersion: is Poisson appropriate? -----------------------------
X = design(daily)
pois = sm.GLM(daily.txns, X, family=sm.families.Poisson()).fit()
phi = pois.pearson_chi2 / pois.df_resid
print("Pearson dispersion (1 = Poisson-like):", round(phi, 2))

# Cameron-Trivedi auxiliary test: H0 var = mean
mu = np.asarray(pois.mu)
aux = ((daily.txns.values - mu) ** 2 - daily.txns.values) / mu
ct = sm.OLS(aux, mu).fit()
print("Cameron-Trivedi alpha = %.4f, t = %.2f, p = %.3g"
      % (ct.params[0], ct.tvalues[0], ct.pvalues[0]))

alpha_hat = max(ct.params[0], 1e-6)
nb = sm.GLM(daily.txns, X, family=sm.families.NegativeBinomial(alpha=alpha_hat)).fit()
pois_rob = sm.GLM(daily.txns, X, family=sm.families.Poisson()).fit(cov_type="HC0")


def row(name, fit):
    ci = np.exp(fit.conf_int().loc["peak"].values)
    return dict(model=name, multiplier=np.exp(fit.params["peak"]),
                ci_lo=ci[0], ci_hi=ci[1])


print(pd.DataFrame([row("Poisson (naive SE)", pois),
                    row("Poisson (robust SE)", pois_rob),
                    row("NegBin", nb)]).round(3))
# If phi ~ 1: keep Poisson, report it. If phi >> 1: report the robust/NB CI instead.

# ---- 1c. Is a binary flag enough? Event study by week-of-year -----------------
wk = daily.txns.groupby([daily.index.year, daily.index.isocalendar().week.values]).mean()
base = daily.txns[daily.peak == 0].groupby(daily.index[daily.peak == 0].year).mean()
rel = wk.unstack(0).div(base, axis=1)           # weekly txns / that year's normal-day mean
fig, ax = plt.subplots(figsize=(12, 4))
rel.loc[36:52].dropna(axis=1, how="all").plot(ax=ax, marker="o")
ax.axvspan(44, 52, alpha=0.1, color="grey")     # approx Nov 1 - Dec 30
ax.axhline(1, color="k", lw=0.8)
ax.set_title("Weekly transactions relative to normal-day mean (shaded = assumed peak)")
ax.set_ylabel("ratio"); ax.set_xlabel("ISO week")
plt.tight_layout(); plt.savefig("04_event_study.png", dpi=150)
# Read it: sharp step at wk ~44 and ~52 -> binary flag is fine.
# Gradual ramp before / decay after -> add ramp terms (Step 4).

g = daily.groupby("peak")[["txns", "revenue", "units_per_order",
                           "avg_discount", "aov"]].mean()
print(g.round(3))
print((g.loc[1] / g.loc[0]).round(2))

from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb, warnings
warnings.filterwarnings("ignore")

daily["t"] = (daily.index - daily.index[0]).days / 365.25
daily["dow"] = daily.index.dayofweek
daily["doy"] = daily.index.dayofyear

# Each fold trains only on data BEFORE the test window
folds = {
    "Peak 2024": ("2024-11-01", "2024-12-30"),
    "Peak 2025": ("2025-11-01", "2025-12-30"),
    "Normal Jan-May 2026": ("2026-01-01", "2026-05-31"),
}

def feats(d):
    X = pd.DataFrame({"const": 1.0, "peak": d.peak, "t": d.t}, index=d.index)
    return X.join(pd.get_dummies(d.dow, prefix="d", drop_first=True).astype(float))

def mase_scale(train):   # error of "same day last year" on training data
    y = train.revenue.values
    return np.mean(np.abs(y[365:] - y[:-365]))

rows, cover, saved, allP = [], [], {}, {}
for name, (a, b) in folds.items():
    train, test = daily[daily.index < a], daily.loc[a:b]
    y, sc = test.revenue, mase_scale(train)
    P = {}
    P["Seasonal naive"] = pd.Series(
        [daily.loc[t - pd.DateOffset(years=1), "revenue"] for t in test.index], index=test.index)
    rm = train.groupby("peak").revenue.mean()
    P["Regime mean (peak/normal)"] = test.peak.map(rm)
    ols = sm.OLS(train.revenue, feats(train)).fit()
    P["Regression (peak+trend+weekday)"] = ols.predict(
        feats(test).reindex(columns=feats(train).columns, fill_value=0))
    Xtr, Xte = train[["peak"]].assign(const=1.0), test[["peak"]].assign(const=1.0)
    sx = SARIMAX(train.revenue / 1000, exog=Xtr, order=(1, 0, 1)).fit(disp=False)
    P["SARIMAX(1,0,1) + peak flag"] = pd.Series(sx.forecast(len(test), exog=Xte).values * 1000, index=test.index)
    cols = ["peak", "t", "dow", "doy"]
    gb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8,
                           min_child_samples=20, verbose=-1, random_state=42).fit(train[cols], train.revenue)
    P["LightGBM (calendar features)"] = pd.Series(gb.predict(test[cols]), index=test.index)
    P["Noise floor (oracle)"] = test.peak.map(test.groupby("peak").revenue.mean())
    allP[name] = (y, P)

    for mdl, p in P.items():
        e = y - p
        rows.append(dict(fold=name, model=mdl, MAE=e.abs().mean(),
                         RMSE=np.sqrt((e**2).mean()), MASE=e.abs().mean() / sc))

    # 90% interval = regime mean + 5th/95th percentile of TRAINING residuals in that regime
    res = train.revenue - train.peak.map(rm)
    q = {k: res[train.peak == k].quantile([.05, .95]).values for k in train.peak.unique()}
    lo = test.peak.map(rm) + test.peak.map(lambda k: q[k][0])
    hi = test.peak.map(rm) + test.peak.map(lambda k: q[k][1])
    cover.append(dict(fold=name, coverage_90=((y >= lo) & (y <= hi)).mean(), avg_width=(hi - lo).mean()))
    saved[name] = (y, test.peak.map(rm), lo, hi)

r = pd.DataFrame(rows)
print(r.pivot(index="model", columns="fold", values="MAE").round(0))
print(r.pivot(index="model", columns="fold", values="MASE").round(2))
print(pd.DataFrame(cover).round(3))
rng2 = np.random.default_rng(0)
for name, (y, P) in allP.items():
    d = (y - P["Regime mean (peak/normal)"]).abs() - (y - P["LightGBM (calendar features)"]).abs()
    bt = [rng2.choice(d.values, len(d)).mean() for _ in range(5000)]
    print(name, "MAE diff (regime - LGBM): %.0f, 95%% CI" % d.mean(), np.percentile(bt, [2.5, 97.5]).round(0))

y, pred, lo, hi = saved["Peak 2025"]
plt.figure(figsize=(12, 4))
plt.plot(y.index, y, label="Actual")
plt.plot(pred.index, pred, label="Forecast (regime mean)")
plt.fill_between(lo.index, lo, hi, alpha=0.25, label="90% interval")
plt.title("Nov-Dec 2025: forecast vs actual daily revenue"); plt.legend()
plt.tight_layout(); plt.savefig("02_forecast_peak2025.png", dpi=150)

from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

res = daily.revenue - daily.groupby("peak").revenue.transform("mean")
print(acorr_ljungbox(res, lags=[7, 14, 28], return_df=True).round(3))

for k, lab in [(0, "normal"), (1, "peak")]:
    rr = res[daily.peak == k]
    print(lab, "skew %.2f" % stats.skew(rr), "| Shapiro p = %.3g" % stats.shapiro(rr).pvalue)

g = daily.groupby("peak").revenue
z = (daily.revenue - g.transform("mean")) / g.transform("std")
flags = z[abs(z) > 3]
print(len(flags), "flagged vs about", round(len(daily) * 0.0027, 1), "expected by chance")
print(flags.round(1))

pk = daily[daily.peak == 1]
print((pk.groupby(pk.index.year).revenue.sum() / 1e6).round(2))   # past 60-day totals

daily_mean = pk.revenue.mean()
rng = np.random.default_rng(42)
boot = [rng.choice(pk.revenue.values, 60, replace=True).sum() for _ in range(10000)]
print("Baseline 60-day total: %.2fM" % (daily_mean * 60 / 1e6))
print("90%% interval (daily noise only): %.2fM - %.2fM" % tuple(np.percentile(boot, [5, 95]) / 1e6))
yt = pk.groupby(pk.index.year).revenue.sum() / 1e6
print("Past peak totals range: %.2fM - %.2fM" % (yt.min(), yt.max()))

md_rows = df.Date.dt.month * 100 + df.Date.dt.day
peak_rows = df[(md_rows >= PEAK_START) & (md_rows <= PEAK_END)]
cur = peak_rows.Discount_Applied.mean()
print("discount vs quantity correlation in peak:",
      round(peak_rows.Discount_Applied.corr(peak_rows.Quantity), 3))

for new in [0.15, 0.075]:
    gain = (1 - new) / (1 - cur) - 1
    print(f"discount {cur:.1%} -> {new:.1%}: revenue per order +{gain:.1%}, "
          f"break-even demand drop {gain / (1 + gain):.1%}")

from scipy.stats import mannwhitneyu

for col in ["Location", "Product_Category"]:
    cnt = df.groupby(["Date", col]).size().unstack(fill_value=0)
    is_pk = peak_flag(cnt.index, PEAK_START, PEAK_END).astype(bool)
    out = []
    for c in cnt.columns:
        a_, b_ = cnt.loc[is_pk, c], cnt.loc[~is_pk, c]
        out.append(dict(category=c, peak_vs_normal=a_.mean() / b_.mean(),
                        p=mannwhitneyu(a_, b_).pvalue))
    out = pd.DataFrame(out)
    out["p_bonferroni"] = (out.p * len(out)).clip(upper=1)
    print(col); print(out.to_string(float_format=lambda x: f"{x:.3g}"))

plt.figure(figsize=(12, 4))
daily.revenue.plot(alpha=0.6)
plt.scatter(flags.index, daily.revenue[flags.index], color="red", zorder=3, label="Flagged (|z|>3)")
plt.legend(); plt.title("Daily revenue with flagged days")
plt.tight_layout(); plt.savefig("03_anomalies.png", dpi=150)

results = pd.DataFrame(rows)
results.pivot(index="model", columns="fold", values="MAE").round(0).to_csv("results_mae.csv")
results.pivot(index="model", columns="fold", values="MASE").round(2).to_csv("results_mase.csv")
pd.DataFrame(cover).round(3).to_csv("interval_coverage.csv", index=False)
