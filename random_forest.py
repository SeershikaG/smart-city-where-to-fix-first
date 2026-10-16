import pandas as pd
import numpy as np
from collections import Counter

# 1. Load the dataset
data = pd.read_csv("data/combined_data.csv")

# 2. Create target labels
# 1 = above-median incident count
# 0 = median or below-median incident count
threshold = data["incident_count"].median()
y = (data["incident_count"] > threshold).astype(int).to_numpy()

# 3. Select input features
features = [
    "population_density_per_km2",
    "total_injuries",
    "unresolved_incidents",
    "complaint_count",
    "near_school",
    "near_hospital"
]

X = data[features].fillna(0).to_numpy(dtype=float)

# 4. Split into training and testing sets
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))

train_indices = indices[:split]
test_indices = indices[split:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]


# 5. Decision Tree implementation from scratch
class DecisionTree:

    def __init__(self, max_depth=4, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None

    def gini(self, y):
        if len(y) == 0:
            return 0

        counts = Counter(y)
        return 1 - sum(
            (count / len(y)) ** 2
            for count in counts.values()
        )

    def best_split(self, X, y):
        parent_gini = self.gini(y)
        best_gain = 0
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        if self.max_features is None:
            feature_indices = np.arange(n_features)
        else:
            feature_indices = np.random.choice(
                n_features,
                size=min(self.max_features, n_features),
                replace=False
            )

        for feature in feature_indices:
            values = np.unique(X[:, feature])

            for threshold in values[:-1]:
                left = X[:, feature] <= threshold
                right = ~left

                if not left.any() or not right.any():
                    continue

                weighted_gini = (
                    (left.sum() / len(y)) * self.gini(y[left])
                    + (right.sum() / len(y)) * self.gini(y[right])
                )

                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if (
            len(np.unique(y)) == 1
            or depth >= self.max_depth
            or len(y) < 2
        ):
            return Counter(y).most_common(1)[0][0]

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return Counter(y).most_common(1)[0][0]

        left = X[:, feature] <= threshold
        right = ~left

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build_tree(
                X[left], y[left], depth + 1
            ),
            "right": self.build_tree(
                X[right], y[right], depth + 1
            )
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_one(self, row, node):
        if not isinstance(node, dict):
            return node

        if row[node["feature"]] <= node["threshold"]:
            return self.predict_one(row, node["left"])

        return self.predict_one(row, node["right"])

    def predict(self, X):
        return np.array([
            self.predict_one(row, self.tree)
            for row in X
        ])


# 6. Random Forest implementation from scratch
class RandomForest:

    def __init__(self, n_trees=10, max_depth=4):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        # Use a different random bootstrap sample for each tree
        for _ in range(self.n_trees):
            sample_indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True
            )

            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                max_features=max(1, int(np.sqrt(n_features)))
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Collect predictions from every tree
        all_predictions = np.array([
            tree.predict(X) for tree in self.trees
        ])

        # Majority vote for each row
        final_predictions = []

        for i in range(X.shape[0]):
            votes = all_predictions[:, i]
            prediction = Counter(votes).most_common(1)[0][0]
            final_predictions.append(prediction)

        return np.array(final_predictions)


# 7. Train the Random Forest
np.random.seed(42)

model = RandomForest(n_trees=10, max_depth=4)
model.fit(X_train, y_train)

# 8. Make predictions
predictions = model.predict(X_test)

# 9. Evaluate accuracy
accuracy = np.mean(predictions == y_test)

print("SMART CITY - RANDOM FOREST")
print("=" * 40)
print("Algorithm: Random Forest from scratch")
print("Number of trees:", model.n_trees)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Incident count threshold:", threshold)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# 10. Display sample predictions
print("\nSample predictions")
print("-" * 40)

for i in range(min(10, len(y_test))):
    actual = "High" if y_test[i] == 1 else "Low"
    predicted = "High" if predictions[i] == 1 else "Low"

    print(f"Actual: {actual} | Predicted: {predicted}")

# 11. Show prediction summary
print("\nPrediction summary")
print("-" * 40)
print("Actual High:", int(np.sum(y_test == 1)))
print("Actual Low:", int(np.sum(y_test == 0)))
print("Predicted High:", int(np.sum(predictions == 1)))
print("Predicted Low:", int(np.sum(predictions == 0)))
