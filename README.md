# 2026Spring_projects
Forks from here that were made March-April 2026 are final projects from that semester.

Project Type: Type III Project, an Original Data Analysis 

# Overview:

This project aims to conduct an analysis to find out which near-Earth asteroids pose the greatest threat to Earth during the 2025–2035 period. By building a Composite Threat Score based on multiple asteroid attributes—such as approach proximity, impactor size, flyby velocity, PHA designation, and orbital uncertainty—we can answer the following questions:

# Research questions:
Which asteroids pose the greatest threat during 2025-2035?

Which specific orbital family (Apollo, Aten, or Amor) contributes the most high-threat close approach events during 2025–2035?

Besides PHA designation, which factor (approach distance, velocity, size, or orbital uncertainty) best explains variance in composite threat scores?

# Hypotheses: 

H1: Near-Earth asteroids with larger estimated diameters have significantly longer observational arcs than smaller asteroids.

H2: Potentially Hazardous Asteroids (PHAs) have a higher mean relative velocity at close approach than non-PHAs.

H3 :Approach distance is the strongest predictor of composite threat scores compared to size, velocity, and orbital uncertainty.

# Data:

NASA CNEOS Close Approach Database (2015–2035), including close approach dates, distances, velocities, and orbital uncertainty bounds. https://cneos.jpl.nasa.gov/ca/ Links to an external site.

JPL Small-Body Database — Near-Earth Asteroids 2025, including physical properties (diameter, albedo), orbital elements (eccentricity, inclination, MOID), PHA designation, and observational arc data for 41,281 known near-Earth objects. https://ssd.jpl.nasa.gov/tools/sbdb_query.html Links to an external site.

# Get started:
1. pip install pandas numpy matplotlib seaborn scipy scikit-learn

2. Run the script
The script must be run from inside the Data/ folder — it uses relative paths to load the CSV files, so running it from the repo root will cause a FileNotFoundError.

3. Expected console output
The script prints progress and results as it runs, including:
Dataset shapes and missing value reports
Top 10 and Top 20 asteroid rankings
Hypothesis test results (Pearson r, Welch's t, β coefficients)
Sensitivity analysis (Spearman r and Top-20 overlap across weight configs)

# Acknowledge of AI Usage
1. Suggested scientific and reasonable hypothesis 
2. Explained how many steps and what to do in each step to do to analyze this topic
3. Gave and explained code especially for feature engineering and data visualization and EDA
4. Answered demand of evaluation part
5. Translated English and Chinese, especially for some domain concept and background knowledge 
6. Built docstrings for complex functions
7. Gave arbitrary weighted composite threat score
8. Explained and compared the different usage of Pearson r spearman, Welch t and linear regression model for evaluation part.

#citation
Claude:
https://claude.ai/share/353b52d9-8209-41da-a60c-f0f1e29a56d7
https://claude.ai/share/87c1f0ac-c669-4bac-8122-3533d3c19839


