import pandas as pd

# 1. Load the datasets
locations = pd.read_csv("data/locations.csv")
incidents = pd.read_csv("data/incidents.csv")
complaints = pd.read_csv("data/public_complaints.csv")


# 2. Convert severity text into numeric scores
severity_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4
}

incidents["severity"] = incidents["severity"].map(severity_map)


# 3. Aggregate incidents by location
incident_summary = incidents.groupby("location_id").agg(
    incident_count=("incident_id", "count"),
    avg_severity=("severity", "mean"),
    total_injuries=("injuries", "sum"),
    unresolved_incidents=("resolved", lambda x: (x == 0).sum())
).reset_index()


# 4. Aggregate public complaints by location
complaint_summary = complaints.groupby("location_id").agg(
    complaint_count=("complaint_id", "count")
).reset_index()


# 5. Combine location details with incident data
combined = locations.merge(
    incident_summary,
    on="location_id",
    how="left"
)


# 6. Add complaint data to the combined dataset
combined = combined.merge(
    complaint_summary,
    on="location_id",
    how="left"
)


# 7. Replace missing counts with zero
count_columns = [
    "incident_count",
    "total_injuries",
    "unresolved_incidents",
    "complaint_count"
]

combined[count_columns] = combined[count_columns].fillna(0)


# 8. Display the results
print("Combined dataset shape:", combined.shape)

print("\nCombined dataset columns:")
print(combined.columns.tolist())

print("\nFirst 5 rows of combined data:")
print(combined.head())


# 9. Save the combined dataset
combined.to_csv("data/combined_data.csv", index=False)

print("\nCombined dataset saved successfully!")
print("File location: data/combined_data.csv")