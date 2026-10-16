
import pandas as pd
locations = pd.read_csv("data/locations.csv")
incidents = pd.read_csv("data/incidents.csv")
complaints = pd.read_csv("data/public_complaints.csv")

# Display dataset sizes
print("LOCATIONS:", locations.shape)
print("INCIDENTS:", incidents.shape)
print("COMPLAINTS:", complaints.shape)

# Display column names
print("\nLocation columns:")
print(locations.columns.tolist())

print("\nIncident columns:")
print(incidents.columns.tolist())

print("\nComplaint columns:")
print(complaints.columns.tolist())

# Display first 5 rows
print("\nLocations sample:")
print(locations.head())

print("\nIncidents sample:")
print(incidents.head())

print("\nComplaints sample:")
print(complaints.head())

import pandas as pd

# Load the three datasets
locations = pd.read_csv("data/locations.csv")
incidents = pd.read_csv("data/incidents.csv")
complaints = pd.read_csv("data/public_complaints.csv")

# Display dataset sizes
print("LOCATIONS:", locations.shape)
print("INCIDENTS:", incidents.shape)
print("COMPLAINTS:", complaints.shape)

# Display column names
print("\nLocation columns:")
print(locations.columns.tolist())

print("\nIncident columns:")
print(incidents.columns.tolist())

print("\nComplaint columns:")
print(complaints.columns.tolist())

# Display first 5 rows
print("\nLocations sample:")
print(locations.head())

print("\nIncidents sample:")
print(incidents.head())

print("\nComplaints sample:")
print(complaints.head())
