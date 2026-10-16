import pandas as pd
import numpy as np

# 1. Load the combined dataset
data = pd.read_csv("data/combined_data.csv")

# 2. Select features for clustering
features = [
    "incident_count",
    "total_injuries",
    "unresolved_incidents",
    "complaint_count",
    "population_density_per_km2"
]

X = data[features].fillna(0).to_numpy(dtype=float)

# 3. Standardize features
means = X.mean(axis=0)
stds = X.std(axis=0)
stds[stds == 0] = 1

X_scaled = (X - means) / stds


# 4. K-Means implementation from scratch
class KMeans:

    def __init__(self, k=3, max_iterations=100, random_state=42):
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)

        # Choose initial centroids from different data points
        initial_indices = rng.choice(
            len(X), size=self.k, replace=False
        )
        self.centroids = X[initial_indices].copy()

        for _ in range(self.max_iterations):

            # Assign each point to its nearest centroid
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :],
                axis=2
            )
            labels = np.argmin(distances, axis=1)

            # Recalculate centroids
            new_centroids = self.centroids.copy()

            for cluster in range(self.k):
                points = X[labels == cluster]

                if len(points) > 0:
                    new_centroids[cluster] = points.mean(axis=0)

            # Stop when centroids stop changing
            if np.allclose(self.centroids, new_centroids):
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

        # Store final cluster assignments
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :],
            axis=2
        )
        self.labels_ = np.argmin(distances, axis=1)

        return self

    def predict(self, X):
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :],
            axis=2
        )
        return np.argmin(distances, axis=1)


# 5. Train K-Means with three clusters
model = KMeans(k=3, max_iterations=100, random_state=42)
model.fit(X_scaled)

# 6. Add cluster labels to the dataset
data["cluster"] = model.labels_

# 7. Display cluster sizes
print("SMART CITY - K-MEANS CLUSTERING")
print("=" * 40)
print("Algorithm: K-Means from scratch")
print("Number of clusters:", model.k)

print("\nLocations in each cluster:")
print(data["cluster"].value_counts().sort_index())

# 8. Summarize the characteristics of each cluster
print("\nAverage characteristics by cluster:")
print(
    data.groupby("cluster")[features]
    .mean()
    .round(2)
)

# 9. Show example locations
print("\nSample clustered locations:")
print(
    data[
        ["location_id", "zone", "incident_count", "total_injuries", "cluster"]
    ].head(10).to_string(index=False)
)

# 10. Save the results
output_path = "data/kmeans_clustered_locations.csv"
data.to_csv(output_path, index=False)

print("\nClustered data saved to:", output_path)
