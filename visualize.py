
import pandas as pd
import matplotlib.pyplot as plt

# Load ranked locations
data = pd.read_csv("data/priority_ranked_locations.csv")

# Chart 1: Top 10 locations by priority score
top10 = data.head(10).sort_values("priority_score")

plt.figure(figsize=(10, 6))
plt.barh(top10["location_id"], top10["priority_score"])
plt.xlabel("Priority Score")
plt.ylabel("Location ID")
plt.title("Top 10 Locations Requiring Attention")
plt.tight_layout()
plt.savefig("data/top10_priority.png")
plt.show()


# Chart 2: Number of locations in each priority category
category_counts = data["priority_category"].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(
    category_counts.index.astype(str),
    category_counts.values
)
plt.xlabel("Priority Category")
plt.ylabel("Number of Locations")
plt.title("Locations by Priority Category")
plt.tight_layout()
plt.savefig("data/priority_categories.png")
plt.show()


# Chart 3: Incident count vs priority score
plt.figure(figsize=(8, 5))
plt.scatter(data["incident_count"], data["priority_score"])
plt.xlabel("Incident Count")
plt.ylabel("Priority Score")
plt.title("Incident Count vs Priority Score")
plt.tight_layout()
plt.savefig("data/incidents_vs_priority.png")
plt.show()

print("All charts generated successfully!")
print("Check the data folder for the PNG files.")
