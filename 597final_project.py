import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_app=pd.read_csv("asteroid_close_approaches_2015_2035 (1).csv")
df_near=pd.read_csv("near_earth_asteroids_2025 (1).csv",low_memory=False)

print(df_app.shape)
print(df_near.shape)

print(df_app.head())
print(df_near.head())

print(df_app.dtypes)
print(df_near.dtypes)

def missing_report(df,name):
    missing=df.isnull().sum()
    percent_missing=(missing/len(df)*100).round(2)
    report=pd.DataFrame({"missing_numbers": missing, "percent_missing": percent_missing})
    print(report[report["missing_numbers"]>0].sort_values(by="percent_missing",ascending=False))
missing_report(df_app,"asteroid_close_approaches_2015 (1).csv")
missing_report(df_near,"near_earth_asteroids_2025 (1).csv")

print(df_app.describe())
print(df_near.describe())

df_app["close_approach_date"]=pd.to_datetime(df_app["close_approach_date"])
mask=(
    (df_app["is_future"]==True) &
    (df_app["close_approach_date"].dt.year>=2026) &(df_app["close_approach_date"].dt.year<=2035)
)
df_future=df_app[mask].copy()
print(df_future.shape)
print(df_future["close_approach_date"].dt.year.value_counts().sort_index())

df_future["full_name"]=df_future["full_name"].str.strip()
df_near["full_name"]=df_near["full_name"].str.strip()

df=df_future.merge(df_near,on="full_name",how="left")
print(df.shape)
print("successful_match:", df["pha"].notna().sum())
print("unsuccessful_match:", df["pha"].isna().sum())

def estimate_diameter(row):
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

df["velocity-infinity_km_s"]=df["velocity_infinity_km_s"].fillna(df["velocity_infinity_km_s"].median())
df["pha"]=df["pha"].fillna(False)
print("final df size:", df.shape)

print(df.isnull().sum()[df.isnull().sum()>0])
print(df["pha"].value_counts())
print(df["size_category"].value_counts())
print(df["close_approach_date"].min(),"~",df["close_approach_date"].max())

#feature engineering
def norm_inv(series):
    #for numbers the smaller the more dangerous
    mn,mx=series.min(),series.max()
    return 1-(series-mn)/(mx-mn+1e-9)
def norm(series):
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
plt.title("Close Approach Timeline 2025–2035\n(Red = High Threat ≥60, Orange = Medium ≥40, Blue = Low)")
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
top20.to_csv("top20_asteroid_threats_2025_2035.csv")
# Save the full merged and scored dataset for future reference
df.to_csv("asteroid_threat_full_dataset.csv", index=False)
print(f"Full dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
# Each figure before plt.show()
plt.savefig("velocity_pha_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig("size_category_analysis.png", dpi=150, bbox_inches="tight")
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.savefig("approach_timeline.png", dpi=150, bbox_inches="tight")

print("=" * 60)
print("ANALYSIS SUMMARY: Asteroid Threat Assessment 2025-2035")
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