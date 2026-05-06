"""
Asteroid Threat Assessment 2026-2035
=====================================
Type III Original Data Analysis

This script analyzes near-Earth asteroid close approach data to identify
the most threatening asteroids during 2026-2035. It builds a Composite
Threat Score (CTS) from five dimensions: proximity, size, velocity,
PHA designation, and orbital uncertainty.

Data Sources:
    - NASA CNEOS Close Approach Database (2015-2035)
    - JPL Small-Body Database - Near-Earth Asteroids 2025

Pipeline:
    1. Data loading and inspection
    2. Data cleaning and merging
    3. Feature engineering and normalization
    4. Composite Threat Score calculation
    5. Exploratory Data Analysis (EDA)
    6. Statistical hypothesis testing
    7. Model evaluation (sensitivity analysis)

Hypotheses:
    H1: Larger asteroids (lower H magnitude) have longer observational arcs.
    H2: PHAs have a higher mean approach velocity than non-PHAs.
    H3: Approach distance (proximity) is the strongest predictor of CTS.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr,pearsonr, ttest_ind
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#read two data files
df_app=pd.read_csv("asteroid_close_approaches_2015_2035 (1).csv")
df_near=pd.read_csv("near_earth_asteroids_2025 (1).csv",low_memory=False)
# see rows&columns&dtypes
print(df_app.shape)
print(df_near.shape)

print(df_app.head())
print(df_near.head())

print(df_app.dtypes)
print(df_near.dtypes)

#find the number of NAs and percentage
def missing_report(df,name):
    """Print a summary of missing values for each column in a DataFrame.
    Only columns with at least one missing value are shown, sorted by missing percentage in descending order."""
    missing=df.isnull().sum()
    percent_missing=(missing/len(df)*100).round(2)
    report=pd.DataFrame({"missing_numbers": missing, "percent_missing": percent_missing})
    print(report[report["missing_numbers"]>0].sort_values(by="percent_missing",ascending=False))
missing_report(df_app,"asteroid_close_approaches_2015 (1).csv")
missing_report(df_near,"near_earth_asteroids_2025 (1).csv")

print(df_app.describe())
print(df_near.describe())

#change format of datetime and fiter the range to 2026 to 2035
df_app["close_approach_date"]=pd.to_datetime(df_app["close_approach_date"])
mask=(
    (df_app["is_future"]==True) &
    (df_app["close_approach_date"].dt.year>=2026) &(df_app["close_approach_date"].dt.year<=2035)
)
df_future=df_app[mask].copy()
print(df_future.shape)
print(df_future["close_approach_date"].dt.year.value_counts().sort_index())
#merge two tables
#unify the format of common column full_name
df_future["full_name"]=df_future["full_name"].str.strip()
df_near["full_name"]=df_near["full_name"].str.strip()

df=df_future.merge(df_near,on="full_name",how="left")
print(df.shape)
print("successful_match:", df["pha"].notna().sum())
print("unsuccessful_match:", df["pha"].isna().sum())
# deal with missing diameter values by using this nasa formula: diameter(km) = 1329 / sqrt(albedo) * 10^(-H/5)
def estimate_diameter(row):
    """
       Estimate an asteroid's diameter in meters using the NASA H-magnitude formula.

       If a measured diameter already exists in the row, it is returned directly.
       Otherwise, the diameter is calculated from absolute magnitude (H) and albedo
       using the standard NASA formula:

           diameter_km = 1329 / sqrt(albedo) * 10^(-H / 5)

       If albedo is unknown, a default value of 0.14 is used (average for S-type
       asteroids). If H is also missing from both columns, np.nan is returned."""
    if pd.notna(row["diameter_m"]):
        return row["diameter_m"]
    albedo=row["albedo"] if pd.notna(row["albedo"]) else 0.14
    h = row["H"] if pd.notna(row["H"]) else row["absolute_magnitude"]
    if pd.isna(h):
        return np.nan
    diameter_km = 1329 / np.sqrt(albedo) * 10 ** (-h / 5)
    return diameter_km * 1000
df["diameter_m_filled"] = df.apply(estimate_diameter, axis=1)
print("missing_before_filling:", df["diameter_m"].isna().sum())
print("missing_after_filling:", df["diameter_m_filled"].isna().sum())

# fill velocity_infinity_km with median and fill missing pha with false
df["velocity_infinity_km_s"]=df["velocity_infinity_km_s"].fillna(df["velocity_infinity_km_s"].median())
df["pha"]=df["pha"].fillna(False)
print("final df size:", df.shape)
print(df.isnull().sum()[df.isnull().sum()>0])
print(df["pha"].value_counts())
print(df["size_category"].value_counts())
print(df["close_approach_date"].min(),"~",df["close_approach_date"].max())

#feature engineering
#normalization
def norm_inv(series):
    """
       Min-max normalize a series so that smaller values map to higher scores.
       Used for variables where a lower raw value means greater danger
       (e.g. approach distance: closer = more dangerous = score closer to 1).
       Formula:
           score = 1 - (x - min) / (max - min)"""
    #for numbers the smaller the more dangerous
    mn,mx=series.min(),series.max()
    return 1-(series-mn)/(mx-mn+1e-9)
def norm(series):
    """Min-max normalize a series so that larger values map to higher scores.
        Used for variables where a higher raw value means greater danger
        (e.g. velocity: faster = more dangerous = score closer to 1)"""
    #for numbers the bigger the more dangerous
    mn,mx=series.min(),series.max()
    return(series-mn)/(mx-mn+1e-9)
#calculate 5 scores
#score1:proximity(closer=more dangerous, use norm_inv)
df["score_proximity"]=norm_inv(df["dist_lunar"])
#score2: size(lower H= larger object=more dangerous, use norm_inv)
df["score_size"]=norm_inv(df["H"].fillna(df["absolute_magnitude"]))
#score3: velocity(faster=more dangerous)
df["score_velocity"]=norm(df["velocity_km_s"])
#score4:pha flag( 1 if officially designated hazardous, 0 otherwise)
df["score_pha"]=df["pha"].astype(int)
#score5: orbital uncertainty (wider predication interval=high risk)
df["uncertainty"]=(df["distance_max_au"]-df["distance_min_au"]).abs()
df["score_uncertainty"]=norm(df["uncertainty"])

#weighted composite threat score
df["threat_score"]=(0.35 * df["score_proximity"] +
    0.30 * df["score_size"] +
    0.15 * df["score_velocity"] +
    0.10 * df["score_pha"] +
    0.10 * df["score_uncertainty"]
) * 100
#round to 2 decimal places
df["threat_score"]=df["threat_score"].round(2)
#overall distribution of threat scores
print(df["threat_score"].describe())
#top10 most threatening asteroids
top10 = df.nlargest(10, "threat_score")[
    ["full_name", "close_approach_date", "dist_lunar",
     "velocity_km_s", "H", "size_category", "pha", "threat_score"]
].reset_index(drop=True)
print(top10)
# ── RQ2: Which orbital family contributes most high-threat events? ──

# Define high-threat threshold
high_threat = df[df["threat_score"] >= 60]

print("=== Total events by orbital family ===")
print(df["class"].value_counts())

print("\n=== High-threat events (CTS ≥ 60) by orbital family ===")
print(high_threat["class"].value_counts())

print("\n=== High-threat RATE by family (% of that family that scores ≥60) ===")
total_by_class = df["class"].value_counts()
high_by_class  = high_threat["class"].value_counts()
rate = (high_by_class / total_by_class * 100).round(2).sort_values(ascending=False)
print(rate)

print("\n=== Mean CTS by orbital family ===")
print(df.groupby("class")["threat_score"].mean().round(2).sort_values(ascending=False))

# PHA mean score should be significantly higher than non-PHA
print(df.groupby("pha")["threat_score"].mean())

# Mean threat score by size category — larger should score higher
print(df.groupby("size_category")["threat_score"].mean().sort_values(ascending=False))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
sns.histplot(data=df, x="velocity_km_s", hue="pha", bins=40,
             ax=axes[0], palette={True: "#E24B4A", False: "#4A90D9"})
axes[0].set_title("Velocity Distribution: PHA vs Non-PHA")
axes[0].set_xlabel("Velocity (km/s)")
axes[0].set_ylabel("Count")

# Box plot
sns.boxplot(data=df, x="pha", y="velocity_km_s", ax=axes[1],hue="pha",legend=False,
            palette={True: "#E24B4A", False: "#4A90D9"})
axes[1].set_title("Velocity Spread: PHA vs Non-PHA")
axes[1].set_xlabel("PHA Designation")
axes[1].set_ylabel("Velocity (km/s)")
axes[1].set_xticklabels(["Non-PHA", "PHA"])
plt.tight_layout()
plt.savefig("velocity_pha_comparison.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Count of asteroids per size class
size_order = [
    "Tiny (<25m) — Airburst/harmless",
    "Small (25-140m) — Local damage",
    "Medium (140m-1km) — Regional damage",
    "Large (>1 km) — City killer+"
]

sns.countplot(data=df, y="size_category", order=size_order,
              ax=axes[0], color="#4A90D9")
axes[0].set_title("Number of Close Approaches by Size Class")
axes[0].set_xlabel("Count")
axes[0].set_ylabel("")

# Mean threat score per size class
sns.barplot(data=df, y="size_category", x="threat_score",
            order=size_order, ax=axes[1], color="#E24B4A")
axes[1].set_title("Mean Threat Score by Size Class")
axes[1].set_xlabel("Mean Threat Score")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig("size_category_analysis.png", dpi=150)
plt.show()

cols = ["dist_lunar", "velocity_km_s", "H", "moid_au",
        "e", "i", "data_arc_years", "threat_score"]
corr = df[cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5)
plt.title("Correlation Matrix — Key Variables")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()
plt.figure(figsize=(14, 6))

# Color by threat level
colors = df["threat_score"].apply(
    lambda x: "#E24B4A" if x >= 60 else "#EF9F27" if x >= 40 else "#4A90D9"
)
plt.scatter(df["close_approach_date"], df["dist_lunar"],
            c=colors, alpha=0.4, s=15)
plt.axhline(y=1, color="red", linestyle="--", linewidth=1, label="1 Lunar Distance")
plt.title("Close Approach Timeline 2026–2035\n(Red = High Threat ≥60, Orange = Medium ≥40, Blue = Low)")
plt.xlabel("Date")
plt.ylabel("Approach Distance (Lunar Distances)")
plt.legend()
plt.tight_layout()
plt.savefig("approach_timeline.png", dpi=150)
plt.show()

top20 = df.nlargest(20, "threat_score")[
    ["full_name", "close_approach_date", "dist_lunar", "dist_km",
     "velocity_km_s", "H", "diameter_m_filled", "size_category",
     "pha", "moid_au", "threat_score"]
].reset_index(drop=True)
# Add rank column starting from 1
top20.index = top20.index + 1
top20.index.name = "rank"

# Round for readability
top20["dist_lunar"] = top20["dist_lunar"].round(3)
top20["dist_km"] = top20["dist_km"].round(0).astype(int)
top20["velocity_km_s"] = top20["velocity_km_s"].round(2)
top20["diameter_m_filled"] = top20["diameter_m_filled"].round(1)
top20["threat_score"] = top20["threat_score"].round(2)

print(top20)
top20.to_csv("top20_asteroid_threats_2026_2035.csv")
# Save the full merged and scored dataset for future reference
df.to_csv("asteroid_threat_full_dataset.csv", index=False)
print(f"Full dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
# Each figure before plt.show()
plt.savefig("velocity_pha_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig("size_category_analysis.png", dpi=150, bbox_inches="tight")
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.savefig("approach_timeline.png", dpi=150, bbox_inches="tight")

print("=" * 60)
print("ANALYSIS SUMMARY: Asteroid Threat Assessment 2026-2035")
print("=" * 60)

print(f"\nTotal close approach events analyzed: {len(df)}")
print(f"Potentially Hazardous Asteroids (PHA): {df['pha'].sum()} ({df['pha'].mean()*100:.1f}%)")
print(f"\nThreat Score Statistics:")
print(f"  Mean:  {df['threat_score'].mean():.2f}")
print(f"  Max:   {df['threat_score'].max():.2f} ({df.loc[df['threat_score'].idxmax(), 'full_name']})")

print(f"\nTop 3 Most Threatening Asteroids:")
for i, row in top20.head(3).iterrows():
    print(f"  #{i} {row['full_name']}")
    print(f"      Date: {row['close_approach_date'].date()}")
    print(f"      Distance: {row['dist_lunar']:.3f} LD | Velocity: {row['velocity_km_s']} km/s")
    print(f"      Threat Score: {row['threat_score']}")

print(f"\nMean Threat Score — PHA vs Non-PHA:")
print(df.groupby("pha")["threat_score"].mean().round(2))

print(f"\nMean Threat Score by Size Category:")
print(df.groupby("size_category")["threat_score"].mean().round(2).sort_values(ascending=False))

#evaluation
#sensitivity of weights
weight_configs = {
    "Original":        (0.35, 0.30, 0.15, 0.10, 0.10),
    "Proximity-Heavy": (0.50, 0.20, 0.15, 0.10, 0.05),
    "Size-Heavy":      (0.25, 0.40, 0.20, 0.10, 0.05),
    "Velocity-Heavy":  (0.30, 0.25, 0.30, 0.10, 0.05),
    "Equal-Weight":    (0.20, 0.20, 0.20, 0.20, 0.20),
}
# Calculate CTS for each config and store rankings
rankings = {}
for name, (w1, w2, w3, w4, w5) in weight_configs.items():
    df[f"cts_{name}"] = (
        w1 * df["score_proximity"] +
        w2 * df["score_size"] +
        w3 * df["score_velocity"] +
        w4 * df["score_pha"] +
        w5 * df["score_uncertainty"]
    ) * 100
    rankings[name] = df[f"cts_{name}"].rank(ascending=False)

# Spearman correlation between Original and all other configs
print("=== Spearman Rank Correlation vs Original ===")
for name in list(weight_configs.keys())[1:]:
    r, p = spearmanr(rankings["Original"], rankings[name])
    print(f"{name:20s}  r = {r:.4f}  p = {p:.4e}")

# Top 20 overlap comparison
print("\n=== Top 20 Overlap with Original ===")
original_top20 = set(df.nlargest(20, "cts_Original")["full_name"])
for name in list(weight_configs.keys())[1:]:
    col = f"cts_{name}"
    other_top20 = set(df.nlargest(20, col)["full_name"])
    overlap = len(original_top20 & other_top20)
    print(f"{name:20s}  overlap = {overlap}/20 ({overlap/20*100:.0f}%)")

#build linear regression model to test hypotheses
print("=" * 55)
print("H1: Diameter (H magnitude) vs Observational Arc Length")
print("=" * 55)

h1_data = df[["H", "data_arc_years"]].dropna()
r, p = stats.pearsonr(h1_data["H"], h1_data["data_arc_years"])
print(f"Pearson r      = {r:.4f}")
print(f"p-value        = {p:.4e}")
print(f"Interpretation = {'Supported' if p < 0.05 else 'Not Supported'}")
# r is already the effect size for correlation

print("\n" + "=" * 55)
print("H2: PHA vs Non-PHA Approach Velocity")
print("=" * 55)

pha_vel    = df[df["pha"] == True]["velocity_km_s"].dropna()
nonpha_vel = df[df["pha"] == False]["velocity_km_s"].dropna()
#Welch t-test
t_stat, p_val = stats.ttest_ind(pha_vel, nonpha_vel, equal_var=False)

# Cohen's d effect size
pooled_std = np.sqrt((pha_vel.std()**2 + nonpha_vel.std()**2) / 2)
cohens_d = (pha_vel.mean() - nonpha_vel.mean()) / pooled_std

# 95% confidence interval for the mean difference
mean_diff = pha_vel.mean() - nonpha_vel.mean()
se = np.sqrt(pha_vel.std()**2/len(pha_vel) + nonpha_vel.std()**2/len(nonpha_vel))
ci_low  = mean_diff - 1.96 * se
ci_high = mean_diff + 1.96 * se

print(f"PHA mean       = {pha_vel.mean():.2f} km/s")
print(f"Non-PHA mean   = {nonpha_vel.mean():.2f} km/s")
print(f"Mean diff      = {mean_diff:.2f} km/s")
print(f"95% CI         = [{ci_low:.2f}, {ci_high:.2f}]")
print(f"t-statistic    = {t_stat:.4f}")
print(f"p-value        = {p_val:.4e}")
print(f"Cohen's d      = {cohens_d:.4f}  ({'large' if abs(cohens_d)>0.8 else 'medium' if abs(cohens_d)>0.5 else 'small'} effect)")
print(f"Interpretation = {'Supported' if p_val < 0.05 else 'Not Supported'}")

print("\n" + "=" * 55)
print("H3: Proximity = Strongest Predictor of CTS")
print("=" * 55)
# Select predictors and drop rows with any missing values
features = ["score_proximity", "score_size", "score_velocity", "score_uncertainty"]
h3_data  = df[features + ["threat_score"]].dropna()
X = h3_data[features].values
y = h3_data["threat_score"].values

# Standardize so beta coefficients are comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# R-squared
r_squared = model.score(X_scaled, y)

print("Standardized Beta Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:25s}  β = {coef:.4f}")

print(f"\nR² = {r_squared:.4f}")
print(f"Best predictor = {features[np.argmax(np.abs(model.coef_))]}")
print(f"Interpretation = {'Supported' if features[np.argmax(np.abs(model.coef_))] == 'score_proximity' else 'Not Supported'}")

#statistic testing
#print reports
print("\n" + "=" * 55)
print("EVALUATION SUMMARY")
print("=" * 55)

print("\n[Model Robustness]")
print("  Sensitivity analysis across 5 weight configs")
print("  → Report Spearman r values and Top 20 overlap %")

print("\n[Hypothesis Results]")
print(f"  H1 (Size → Arc):  r = {r:.3f},  p = {p:.2e}")
print(f"  H2 (PHA → Velocity):d = {cohens_d:.3f}, p = {p_val:.2e}")
print(f"  H3 (Proximity → CTS): R² = {r_squared:.3f}, β_proximity = {model.coef_[0]:.3f}")
