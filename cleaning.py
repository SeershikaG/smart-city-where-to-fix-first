
import pandas as pd

# Load datasets
locations = pd.read_csv("data/locations.csv")
incidents = pd.read_csv("data/incidents.csv")
complaints = pd.read_csv("data/public_complaints.csv")

# 1. Check missing values
print("MISSING VALUES")

print("\nLocations:")
print(locations.isnull().sum())

print("\nIncidents:")
print(incidents.isnull().sum())

print("\nComplaints:")
print(complaints.isnull().sum())


# 2. Check duplicate records
print("\nDUPLICATE ROWS")

print("Locations:", locations.duplicated().sum())
print("Incidents:", incidents.duplicated().sum())
print("Complaints:", complaints.duplicated().sum())


# 3. Check duplicate IDs
print("\nDUPLICATE IDS")

print("Location IDs:", locations["location_id"].duplicated().sum())
print("Incident IDs:", incidents["incident_id"].duplicated().sum())
print("Complaint IDs:", complaints["complaint_id"].duplicated().sum())


# 4. Check location ID matches
location_ids = set(locations["location_id"])

print("\nUNMATCHED LOCATION IDS")

print(
    "Incidents:",
    (~incidents["location_id"].isin(location_ids)).sum()
)

print(
    "Complaints:",
    (~complaints["location_id"].isin(location_ids)).sum()
)
