
import pandas as pd

# 1. Load the combined dataset
data = pd.read_csv("data/combined_data.csv")


# 2. Convert road conditions into numeric scores
road_scores = {
    "Good": 1,
    "Fair": 2,
    "Poor": 3,
    "Critical": 4
}

data["road_score"] = data["road_condition"].map(road_scores)


# 3. Convert pedestrian activity into numeric scores
pedestrian_scores = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

data["pedestrian_score"] = data["pedestrian_activity"].map(
    pedestrian_scores
)


# 4. Normalize numeric columns to a 0–1 scale
def normalize(column):
    minimum = column.min()
    maximum = column.max()

    if maximum == minimum:
        return pd.Series(0.0, index=column.index)

    return (column - minimum) / (maximum - minimum)


data["incident_score"] = normalize(data["incident_count"])
data["severity_score"] = normalize(data["avg_severity"])
data["injury_score"] = normalize(data["total_injuries"])
data["unresolved_score"] = normalize(data["unresolved_incidents"])
data["complaint_score"] = normalize(data["complaint_count"])
data["population_score"] = normalize(
    data["population_density_per_km2"]
)


# 5. Calculate the overall priority score
data["priority_score"] = (
    0.25 * data["incident_score"]
    + 0.20 * data["severity_score"]
    + 0.20 * data["injury_score"]
    + 0.15 * data["unresolved_score"]
    + 0.10 * data["complaint_score"]
    + 0.05 * normalize(data["road_score"])
    + 0.025 * normalize(data["pedestrian_score"])
    + 0.025 * data["population_score"]
)


# 6. Rank locations from highest priority to lowest
data = data.sort_values(
    by="priority_score",
    ascending=False
)

data["priority_rank"] = range(1, len(data) + 1)


# 7. Assign priority categories
data["priority_category"] = pd.cut(
    data["priority_rank"],
    bins=[0, 100, 300, len(data)],
    labels=["High", "Medium", "Low"]
)


# 8. Display the top 10 priority locations
print("\nTOP 10 LOCATIONS TO FIX FIRST")
print(
    data[
        [
            "priority_rank",
            "location_id",
            "zone",
            "road_condition",
            "incident_count",
            "total_injuries",
            "complaint_count",
            "priority_score",
            "priority_category"
        ]
    ].head(10).to_string(index=False)
)


# 9. Save the ranked dataset
data.to_csv("data/priority_ranked_locations.csv", index=False)

print("\nPriority scoring completed successfully!")
print("Saved to: data/priority_ranked_locations.csv")
