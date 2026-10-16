
import pandas as pd

data = pd.read_csv("data/combined_data.csv")

print("Road conditions:")
print(data["road_condition"].unique())

print("\nNear school values:")
print(data["near_school"].unique())

print("\nNear hospital values:")
print(data["near_hospital"].unique())

print("\nPedestrian activity values:")
print(data["pedestrian_activity"].unique())
