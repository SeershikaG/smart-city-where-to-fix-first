import pandas as pd
import numpy as np

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

# 5. Standardize features using training data only
means = X_train.mean(axis=0)
stds = X_train.std(axis=0)
stds[stds == 0] = 1

X_train = (X_train - means) / stds
X_test = (X_test - means) / stds

# 6. Logistic Regression implemented from scratch
class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        # Clip values to help avoid numerical overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        samples, features = X.shape

        self.weights = np.zeros(features)
        self.bias = 0.0

        # Learn weights using gradient descent
        for _ in range(self.epochs):
            linear_output = np.dot(X, self.weights) + self.bias
            probabilities = self.sigmoid(linear_output)

            dw = np.dot(X.T, probabilities - y) / samples
            db = np.sum(probabilities - y) / samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


# 7. Train the model
model = LogisticRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

# 8. Predict on test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 9. Evaluate accuracy
accuracy = np.mean(predictions == y_test)

print("SMART CITY - LOGISTIC REGRESSION")
print("=" * 40)
print("Algorithm: Logistic Regression from scratch")
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

    print(
        f"Actual: {actual} | Predicted: {predicted} "
        f"| High-class probability: {probabilities[i]:.3f}"
    )

# 11. Show prediction summary
print("\nPrediction summary")
print("-" * 40)
print("Actual High:", int(np.sum(y_test == 1)))
print("Actual Low:", int(np.sum(y_test == 0)))
print("Predicted High:", int(np.sum(predictions == 1)))
print("Predicted Low:", int(np.sum(predictions == 0)))
