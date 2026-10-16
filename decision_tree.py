import pandas as pd
import numpy as np
from collections import Counter

# 1. Load the combined dataset
data = pd.read_csv("data/combined_data.csv")

# 2. Create the target variable
# Predict whether a location has above-median incident count
threshold = data["incident_count"].median()

data["target"] = (
    data["incident_count"] > threshold
).astype(int)

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
y = data["target"].to_numpy(dtype=int)

# 4. Split data into training and testing sets
np.random.seed(42)
indices = np.random.permutation(len(X))

split = int(0.8 * len(X))

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]


# 5. Decision Tree algorithm from scratch
class DecisionTree:

    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.tree = None

    # Calculate Gini impurity
    def gini(self, labels):
        if len(labels) == 0:
            return 0

        counts = Counter(labels)

        probabilities = [
            count / len(labels)
            for count in counts.values()
        ]

        return 1 - sum(p ** 2 for p in probabilities)

    # Find the best feature and split threshold
    def best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None

        parent_gini = self.gini(y)

        for feature in range(X.shape[1]):
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

    # Build the decision tree recursively
    def build_tree(self, X, y, depth=0):

        # Stop when labels are identical or depth is reached
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

    # Train the model
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    # Predict one location
    def predict_one(self, row, node):

        if not isinstance(node, dict):
            return node

        if row[node["feature"]] <= node["threshold"]:
            return self.predict_one(row, node["left"])

        return self.predict_one(row, node["right"])

    # Predict multiple locations
    def predict(self, X):
        return np.array([
            self.predict_one(row, self.tree)
            for row in X
        ])


# 6. Train the model
model = DecisionTree(max_depth=4)
model.fit(X_train, y_train)

# 7. Predict on test data
predictions = model.predict(X_test)

# 8. Evaluate model performance
accuracy = np.mean(predictions == y_test)

print("SMART CITY - DECISION TREE MODEL")
print("=" * 40)
print("Algorithm: Decision Tree from scratch")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Incident count threshold:", threshold)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# 9. Show sample predictions
print("\nSample predictions")
print("-" * 40)

for i in range(min(10, len(y_test))):
    actual = "High" if y_test[i] == 1 else "Low"
    predicted = "High" if predictions[i] == 1 else "Low"

    print(f"Actual: {actual} | Predicted: {predicted}")

# 10. Show prediction counts
print("\nPrediction summary")
print("-" * 40)
print("Actual High:", int(np.sum(y_test == 1)))
print("Actual Low:", int(np.sum(y_test == 0)))
print("Predicted High:", int(np.sum(predictions == 1)))
print("Predicted Low:", int(np.sum(predictions == 0)))
