
# 🏙️ Smart City – Where to Fix First?

## 📌 Project Overview
Smart City – Where to Fix First? is a data-driven dashboard designed to identify locations that may need attention based on recorded incidents, public complaints, and other available indicators.

The project uses Python and Streamlit to present location priority rankings and interactive visualizations to support smarter urban planning.

## 🎯 Objectives
- Identify high-priority locations.
- Analyze incidents and public complaints.
- Search for specific locations using Location ID.
- Visualize priority scores and distributions.
- Explore location data and K-Means clusters.
- Support data-driven city maintenance decisions.

## ✨ Features
- Interactive dashboard
- Priority ranking of locations
- Search by Location ID
- Incident type analysis
- Public complaint analysis
- Waterlogged road illustration
- Charts and summary statistics
- Location-level data lookup
- Optional K-Means clustering

## 🛠️ Technologies Used
- Python
- Pandas
- Streamlit
- Matplotlib
- Machine Learning

## 📂 Project Structure

```text
datathon-project/
├── app.py
├── cleaning.py
├── combine.py
├── priority.py
├── visualize.py
├── decision_tree.py
├── random_forest.py
├── logistic_regression.py
├── kmeans.py
├── data/
└── images/
    └── waterlogged.png
```

## ⚙️ Installation and Setup

### 1. Install Python
Install Python if it is not already installed.

### 2. Install required packages

Open PowerShell in the project directory and run:

```bash
python -m pip install streamlit pandas matplotlib
```

Install any additional libraries required by your other Python scripts.

### 3. Run the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

## 📊 Data Analysis
The project processes location information, incidents, and public complaints to produce combined data and priority rankings. It also supports visual exploration of recorded data and optional K-Means clustering.

## ⚠️ Disclaimer
This project is a prototype for decision support. Priority rankings depend on the available dataset and scoring method. Results should be verified through real-world inspection before decisions are made.

## 👩‍💻 Future Improvements
- Integrate live city data.
- Improve prediction models using validated datasets.
- Add an interactive city map.
- Deploy the dashboard online.
- Improve recommendations for road maintenance.
