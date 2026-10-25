"""
CodeAlpha Data Science Internship - Task 4: Sales Prediction using Python
Author: Rinae Mathaulula

Predicts product sales from advertising spend on TV, Radio and Newspaper,
and analyses how changing the advertising budget affects sales.
Units: spend in thousands of dollars, sales in thousands of units.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")
CHANNELS = ["TV", "Radio", "Newspaper"]

# 1. Load & clean -----------------------------------------------------------
df = pd.read_csv("Advertising.csv")
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == ""])  # row-index column
print("Shape:", df.shape, "| missing:", int(df.isna().sum().sum()), "| duplicates:", df.duplicated().sum())
print(df.describe().round(2), "\n")

# Outlier check (IQR rule)
for c in df.columns:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    n = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
    print(f"Outliers in {c}: {n}")
print()

# 2. Exploration --------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, ch, col in zip(axes, CHANNELS, ["#1565C0", "#2E7D32", "#EF6C00"]):
    sns.regplot(data=df, x=ch, y="Sales", ax=ax, color=col, scatter_kws={"alpha": .6})
    ax.set_title(f"{ch} spend vs Sales (r = {df[ch].corr(df['Sales']):.2f})")
    ax.set_xlabel(f"{ch} spend ($000s)")
axes[0].set_ylabel("Sales (000s units)")
plt.tight_layout(); plt.savefig("spend_vs_sales.png", dpi=120); plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig("correlation.png", dpi=120); plt.close()
print("Correlation with Sales:\n", df.corr()["Sales"].drop("Sales").round(3), "\n")

# 3. Feature engineering ------------------------------------------------------
df["Total_Spend"] = df[CHANNELS].sum(axis=1)
df["TV_x_Radio"] = df["TV"] * df["Radio"]          # synergy between channels
df["sqrt_TV"] = np.sqrt(df["TV"])                  # diminishing returns on TV

# 4. Model comparison ---------------------------------------------------------
y = df["Sales"]
feature_sets = {
    "Linear (TV, Radio, Newspaper)": CHANNELS,
    "Linear (TV, Radio)": ["TV", "Radio"],
    "Linear + engineered features": ["sqrt_TV", "Radio", "TV_x_Radio"],
}
models = {name: (LinearRegression(), cols) for name, cols in feature_sets.items()}
models["Random Forest"] = (RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE), CHANNELS)
models["Gradient Boosting"] = (GradientBoostingRegressor(random_state=RANDOM_STATE), CHANNELS)

train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE)
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows, fitted = [], {}
for name, (model, cols) in models.items():
    X = df[cols]
    cv_r2 = cross_val_score(model, X.loc[train_idx], y[train_idx], cv=cv, scoring="r2").mean()
    model.fit(X.loc[train_idx], y[train_idx])
    pred = model.predict(X.loc[test_idx])
    rows.append((name, cv_r2, r2_score(y[test_idx], pred), mean_absolute_error(y[test_idx], pred),
                 np.sqrt(mean_squared_error(y[test_idx], pred))))
    fitted[name] = (model, cols)
results = pd.DataFrame(rows, columns=["Model", "CV R2", "Test R2", "Test MAE", "Test RMSE"]).sort_values("Test R2", ascending=False)
print(results.round(3).to_string(index=False), "\n")
results.round(4).to_csv("model_comparison.csv", index=False)

# 5. Interpret the baseline linear model -------------------------------------
lin, _ = fitted["Linear (TV, Radio, Newspaper)"]
coef = pd.Series(lin.coef_, index=CHANNELS)
print("Linear model coefficients (extra sales units per extra $1 spent):")
for ch, c in coef.items():
    print(f"  {ch:10s}: +{c*1000:.1f} units per $1,000")
print(f"  Intercept : {lin.intercept_:.2f}k units with zero advertising\n")

plt.figure(figsize=(6, 4))
(coef * 1000).plot.bar(color=["#1565C0", "#2E7D32", "#EF6C00"])
plt.title("Extra units sold per extra $1,000 spent"); plt.ylabel("Units"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("channel_effect.png", dpi=120); plt.close()

# 6. Best model: actual vs predicted ----------------------------------------
best_name = results.iloc[0]["Model"]
best, best_cols = fitted[best_name]
pred = best.predict(df.loc[test_idx, best_cols])
plt.figure(figsize=(6, 6))
plt.scatter(y[test_idx], pred, color="#1565C0", alpha=.8)
lims = [y.min() - 1, y.max() + 1]; plt.plot(lims, lims, "--", color="grey")
plt.xlabel("Actual sales (000s units)"); plt.ylabel("Predicted sales (000s units)")
plt.title(f"Actual vs predicted - {best_name}\nTest R² = {r2_score(y[test_idx], pred):.3f}")
plt.tight_layout(); plt.savefig("actual_vs_predicted.png", dpi=120); plt.close()

# 7. What-if: how should a fixed budget be split? ----------------------------
def predict_sales(tv, radio, news=0.0):
    row = pd.DataFrame({"TV": [tv], "Radio": [radio], "Newspaper": [news]})
    row["sqrt_TV"] = np.sqrt(row["TV"]); row["TV_x_Radio"] = row["TV"] * row["Radio"]
    return best.predict(row[best_cols])[0]

budget = 200  # $200k, close to the dataset's average total spend
shares = np.linspace(0.5, 1.0, 51)
split = pd.DataFrame({"TV share": shares,
                      "Sales": [predict_sales(budget * s, budget * (1 - s)) for s in shares]})
split = split[budget * (1 - split["TV share"]) <= df["Radio"].max()]  # stay inside observed range
opt = split.loc[split["Sales"].idxmax()]
print(f"Best split of a ${budget}k budget: {opt['TV share']:.0%} TV / {1-opt['TV share']:.0%} Radio "
      f"-> {opt['Sales']:.1f}k units")
current = df[["TV", "Radio", "Newspaper", "Sales"]].mean()
print(f"Average campaign today: TV {current.TV:.0f}k, Radio {current.Radio:.0f}k, Newspaper {current.Newspaper:.0f}k "
      f"-> {current.Sales:.1f}k units")
print(f"Model prediction for the average campaign: {predict_sales(current.TV, current.Radio, current.Newspaper):.1f}k units\n")

plt.figure(figsize=(8, 4.5))
plt.plot(split["TV share"] * 100, split["Sales"], color="#1565C0", lw=2)
plt.axvline(opt["TV share"] * 100, color="#C62828", ls="--", label=f"Best within data: {opt['TV share']:.0%} TV / ${budget*(1-opt['TV share']):.0f}k Radio\n(Radio capped at max observed ${df['Radio'].max():.0f}k)")
plt.xlabel(f"% of ${budget}k budget on TV (rest on Radio)"); plt.ylabel("Predicted sales (000s units)")
plt.title("Budget split simulation: more Radio keeps paying off up to the data limit"); plt.legend()
plt.tight_layout(); plt.savefig("budget_split.png", dpi=120); plt.close()

# Effect of +10% on each channel from the average campaign
base = predict_sales(current.TV, current.Radio, current.Newspaper)
print("Effect of adding $10k to one channel (from the average campaign):")
for ch in CHANNELS:
    c = current.copy(); c[ch] += 10
    print(f"  +$10k {ch:10s}: {predict_sales(c.TV, c.Radio, c.Newspaper) - base:+.2f}k units")
