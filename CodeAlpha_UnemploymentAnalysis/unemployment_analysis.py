"""
CodeAlpha Data Science Internship - Task 2: Unemployment Analysis with Python
Author: Rinae Mathaulula

Cleans, explores and visualises India's unemployment data (CMIE, May 2019 - Oct 2020)
to measure the impact of Covid-19 and identify regional and urban/rural patterns.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates

sns.set_theme(style="whitegrid")

# 1. Load & clean -----------------------------------------------------------
def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    df["Region"] = df["Region"].str.strip()
    df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%d-%m-%Y")
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()  # month-end -> month label
    return df

ts = load("Unemployment in India.csv").rename(columns={
    "Estimated Unemployment Rate (%)": "Unemployment", "Estimated Employed": "Employed",
    "Estimated Labour Participation Rate (%)": "Participation"})
ts["Area"] = ts["Area"].str.strip()

geo = load("Unemployment_Rate_upto_11_2020.csv")
geo.columns = ["State", "Date", "Frequency", "Unemployment", "Employed", "Participation", "Zone", "Longitude", "Latitude"]

print("Dataset 1 (Rural/Urban, May 2019 - Jun 2020):", ts.shape, "| missing:", int(ts.isna().sum().sum()))
print("Dataset 2 (State + zone, Jan - Oct 2020):", geo.shape, "| missing:", int(geo.isna().sum().sum()))
print("Duplicates:", ts.duplicated().sum(), geo.duplicated().sum(), "\n")
print(ts[["Unemployment", "Employed", "Participation"]].describe().round(2), "\n")

# Covid phase labels (national lockdown began 25 March 2020)
def phase(d):
    if d < pd.Timestamp("2020-03-01"): return "Pre-Covid"
    if d <= pd.Timestamp("2020-06-30"): return "Lockdown (Mar-Jun 2020)"
    return "Unlock (Jul-Oct 2020)"
geo["Phase"] = geo["Date"].map(phase)
ts["Phase"] = ts["Date"].map(phase)

# 2. National trend, rural vs urban ----------------------------------------
monthly = ts.groupby(["Date", "Area"])["Unemployment"].mean().unstack()
ax = monthly.plot(marker="o", figsize=(10, 5), color=["#2E7D32", "#C62828"])
ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-10"), color="grey", alpha=0.15, label="Lockdown")
ax.set_title("Average unemployment rate across Indian states: Rural vs Urban")
ax.set_ylabel("Unemployment rate (%)"); ax.set_xlabel(""); ax.legend()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); ax.xaxis.set_minor_locator(mdates.MonthLocator())
plt.tight_layout(); plt.savefig("trend_rural_urban.png", dpi=120); plt.close()

# 3. Covid impact: before vs during vs after -------------------------------
national = geo.groupby("Date")[["Unemployment", "Participation"]].mean()
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(national.index, national["Unemployment"], marker="o", color="#C62828", label="Unemployment rate")
ax1.set_ylabel("Unemployment rate (%)", color="#C62828")
ax2 = ax1.twinx()
ax2.plot(national.index, national["Participation"], marker="s", color="#1565C0", label="Labour participation")
ax2.set_ylabel("Labour participation rate (%)", color="#1565C0"); ax2.grid(False)
ax1.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-10"), color="grey", alpha=0.15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax1.set_title("Covid-19 shock: unemployment spiked as labour participation fell (2020)")
plt.tight_layout(); plt.savefig("covid_impact.png", dpi=120); plt.close()

phase_order = ["Pre-Covid", "Lockdown (Mar-Jun 2020)", "Unlock (Jul-Oct 2020)"]
phase_summary = geo.groupby("Phase")[["Unemployment", "Participation"]].mean().reindex(phase_order).round(2)
print("Average by Covid phase (2020 state data):\n", phase_summary, "\n")

# 4. Which states were hit hardest? ----------------------------------------
pre = geo[geo["Date"] < "2020-03-01"].groupby("State")["Unemployment"].mean()
peak = geo[(geo["Date"] >= "2020-04-01") & (geo["Date"] <= "2020-05-31")].groupby("State")["Unemployment"].mean()
impact = (peak - pre).dropna().sort_values()
plt.figure(figsize=(8, 9))
impact.plot.barh(color=["#C62828" if v > 0 else "#2E7D32" for v in impact])
plt.title("Rise in unemployment during Apr-May 2020 lockdown\n(percentage points vs Jan-Feb 2020)")
plt.xlabel("Change in unemployment rate (pp)")
plt.tight_layout(); plt.savefig("state_covid_impact.png", dpi=120); plt.close()
print("Hardest hit states (pp increase):\n", impact.sort_values(ascending=False).head(5).round(1), "\n")

# 5. Zone comparison & state heatmap ---------------------------------------
plt.figure(figsize=(9, 5))
sns.boxplot(data=geo, x="Zone", y="Unemployment", hue="Phase", hue_order=phase_order,
            palette=["#90CAF9", "#EF5350", "#A5D6A7"])
plt.title("Unemployment by zone and Covid phase"); plt.ylabel("Unemployment rate (%)")
plt.tight_layout(); plt.savefig("zone_boxplot.png", dpi=120); plt.close()

heat = geo.pivot_table(index="State", columns=geo["Date"].dt.strftime("%b"), values="Unemployment")
heat = heat[["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]]
plt.figure(figsize=(10, 10))
sns.heatmap(heat, cmap="Reds", annot=True, fmt=".0f", cbar_kws={"label": "Unemployment %"})
plt.title("Monthly unemployment rate by state, 2020"); plt.xlabel("")
plt.tight_layout(); plt.savefig("state_heatmap.png", dpi=120); plt.close()

# 6. Seasonal / monthly pattern (pre-Covid only, to avoid the shock) -------
pre_covid = ts[ts["Date"] < "2020-03-01"]
print("Pre-Covid average by month:\n", pre_covid.groupby(pre_covid["Date"].dt.strftime("%Y-%m"))["Unemployment"].mean().round(2), "\n")
print("Rural vs Urban (whole period):\n", ts.groupby("Area")["Unemployment"].mean().round(2))
print("\nCorrelation (unemployment vs participation):", round(geo["Unemployment"].corr(geo["Participation"]), 3))
