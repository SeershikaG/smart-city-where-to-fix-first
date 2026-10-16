
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Smart City | Where to Fix First?",
    page_icon="🏙️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_PATH = BASE_DIR / "images" / "waterlogged.png"


# ==================================================
# PAGE DESIGN
# ==================================================
st.markdown("""
<style>
.stApp {
    background-color: #F3F7FB;
    color: #102A43;
}
.block-container {
    padding-top: 1.5rem;
}
h1, h2, h3 {
    color: #102A43;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #D9E5F0;
    padding: 16px;
    border-radius: 12px;
}
.hero {
    background: linear-gradient(120deg, #102A43, #087E8B);
    padding: 25px;
    border-radius: 16px;
    color: white;
}
.hero h1, .hero p {
    color: white;
}
.priority-card {
    background: white;
    border-left: 6px solid #D92D20;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.warning(f"Unable to read {filename}: {error}")
        return pd.DataFrame()


def find_column(data, possible_names):
    if data.empty:
        return None

    column_lookup = {
        str(column).strip().lower(): column
        for column in data.columns
    }

    for name in possible_names:
        if name.lower() in column_lookup:
            return column_lookup[name.lower()]

    return None


def normalize_id(value):
    """Normalize IDs so values such as 12 and 12.0 can match."""
    if pd.isna(value):
        return ""

    value = str(value).strip()

    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (ValueError, TypeError):
        pass

    return value


def get_location_rows(data, selected_id):
    if data.empty:
        return pd.DataFrame()

    id_column = find_column(
        data,
        [
            "location_id",
            "location id",
            "locationid",
            "loc_id",
            "location"
        ]
    )

    if id_column is None:
        return pd.DataFrame()

    matching_rows = data[
        data[id_column].apply(normalize_id)
        == normalize_id(selected_id)
    ]

    return matching_rows.copy()


def heading(title, subtitle=None):
    st.markdown(f"## {title}")

    if subtitle:
        st.caption(subtitle)


# ==================================================
# LOAD DATASETS
# ==================================================
ranked = load_csv("priority_ranked_locations.csv")
locations = load_csv("locations.csv")
incidents = load_csv("incidents.csv")
complaints = load_csv("public_complaints.csv")
combined = load_csv("combined_data.csv")
clustered = load_csv("kmeans_clustered_locations.csv")

if ranked.empty:
    st.error(
        "priority_ranked_locations.csv was not found or is empty. "
        "Place the file inside the data folder and run your data "
        "preparation scripts first."
    )
    st.stop()

df = ranked.copy()

# Identify important columns.
location_id_col = find_column(
    df, ["location_id", "location id", "locationid", "id"]
)

score_col = find_column(
    df, ["priority_score", "score", "final_score"]
)

rank_col = find_column(
    df, ["rank", "priority_rank", "ranking"]
)

priority_col = find_column(
    df,
    ["priority_category", "priority_level", "priority", "risk_level"]
)

incident_count_col = find_column(
    df, ["incident_count", "total_incidents", "number_of_incidents"]
)

if location_id_col is None:
    st.error("The priority ranking file needs a location ID column.")
    st.stop()

if score_col is None:
    df["priority_score"] = 0
    score_col = "priority_score"

if incident_count_col is None:
    df["incident_count"] = 0
    incident_count_col = "incident_count"

df[score_col] = pd.to_numeric(
    df[score_col], errors="coerce"
).fillna(0)

df[incident_count_col] = pd.to_numeric(
    df[incident_count_col], errors="coerce"
).fillna(0)

if priority_col is None:
    if rank_col is not None:
        ranks = pd.to_numeric(df[rank_col], errors="coerce")

        df["priority_category"] = "Low"
        df.loc[ranks <= 100, "priority_category"] = "High"
        df.loc[ranks.between(101, 300), "priority_category"] = "Medium"
    else:
        median_score = df[score_col].median()
        df["priority_category"] = "Low"
        df.loc[df[score_col] >= median_score, "priority_category"] = "High"

    priority_col = "priority_category"

df[priority_col] = (
    df[priority_col]
    .fillna("Low")
    .astype(str)
    .str.strip()
    .str.title()
)

df[priority_col] = df[priority_col].replace({
    "Critical": "High",
    "Moderate": "Medium"
})

if rank_col is None:
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    rank_col = "rank"
else:
    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
    df = df.sort_values(
        rank_col, ascending=True, na_position="last"
    ).reset_index(drop=True)


# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <h1>🏙️ Smart City: Where to Fix First?</h1>
    <p>
        A data-driven dashboard to identify high-priority locations,
        analyze incidents, and support smarter city planning.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Prototype dashboard | Historical data analysis | "
    "Verify findings through real-world inspections."
)


# ==================================================
# SIDEBAR FILTERS
# ==================================================
st.sidebar.title("🔎 Dashboard Filters")

priorities = ["High", "Medium", "Low"]

selected_priorities = st.sidebar.multiselect(
    "Filter by priority",
    options=priorities,
    default=priorities
)

if not selected_priorities:
    selected_priorities = priorities

search_id = st.sidebar.text_input(
    "Search by Location ID",
    placeholder="Enter Location ID"
)

filtered = df[df[priority_col].isin(selected_priorities)].copy()

if search_id.strip():
    filtered = filtered[
        filtered[location_id_col]
        .apply(normalize_id)
        .str.contains(search_id.strip(), case=False, na=False)
    ]

st.sidebar.caption(
    "Priority and ID filters apply to the location table."
)


# ==================================================
# CITY OVERVIEW
# ==================================================
heading("📊 City Overview")

total_locations = len(df)
high_priority_count = int((df[priority_col] == "High").sum())
total_incidents = int(df[incident_count_col].sum())
average_score = df[score_col].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("📍 Total Locations", f"{total_locations:,}")
col2.metric("🔴 High Priority", f"{high_priority_count:,}")
col3.metric("🚧 Recorded Incidents", f"{total_incidents:,}")
col4.metric("⭐ Average Score", f"{average_score:.2f}")


# ==================================================
# PRIORITY DISTRIBUTION
# ==================================================
heading(
    "Priority Distribution",
    "Number of locations assigned to each priority category."
)

priority_counts = (
    df[priority_col]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
)

chart1, chart2 = st.columns(2)

with chart1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(
        priority_counts.values,
        labels=priority_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Location Priority Breakdown")
    st.pyplot(fig)
    plt.close(fig)

with chart2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(priority_counts.index, priority_counts.values)
    ax.set_title("Locations by Priority")
    ax.set_xlabel("Priority")
    ax.set_ylabel("Number of Locations")
    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# WATERLOGGED ROAD IMAGE
# ==================================================
st.markdown("---")

heading(
    "🚨 High-Priority Road Condition",
    "Illustrative waterlogging example for road inspection."
)

image_col, description_col = st.columns([1.1, 1])

with image_col:
    if IMAGE_PATH.exists():
        st.image(
            str(IMAGE_PATH),
            caption="Waterlogged road — illustrative inspection image",
            use_container_width=True
        )
    else:
        st.warning(
            "Waterlogged image not found. Add your image here:\n\n"
            "images/waterlogged.png"
        )

with description_col:
    st.markdown("""
    <div class="priority-card">
        <h3 style="color:#B42318;">🔴 HIGH PRIORITY EXAMPLE</h3>
        <h3>Waterlogged Road</h3>
        <p>
            Standing water can hide potholes, affect road safety,
            disrupt traffic, and damage road surfaces.
        </p>
        <b>Recommended actions:</b>
        <ul>
            <li>Inspect the affected road.</li>
            <li>Check drainage systems and blocked outlets.</li>
            <li>Record the location and severity of the problem.</li>
            <li>Prioritize repairs after field verification.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.caption(
    "The image is illustrative. It does not automatically change "
    "the priority score or category in your dataset."
)


# ==================================================
# TOP 10 PRIORITY LOCATIONS
# ==================================================
st.markdown("---")

heading(
    "🏆 Top 10 Priority Locations",
    "Locations with the highest priority scores."
)

top10 = df.sort_values(
    score_col, ascending=False
).head(10)

chart_col, table_col = st.columns([1.2, 1])

with chart_col:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        top10[location_id_col].astype(str).iloc[::-1],
        top10[score_col].iloc[::-1]
    )

    ax.set_title("Top 10 Locations by Priority Score")
    ax.set_xlabel("Priority Score")
    ax.set_ylabel("Location ID")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with table_col:
    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# FILTERED LOCATION TABLE
# ==================================================
st.markdown("---")

heading(
    "📍 Location Ranking Table",
    "Use the sidebar to filter locations by ID or priority."
)

st.write(f"Showing **{len(filtered)}** location(s).")

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "⬇️ Download Filtered Locations CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_locations.csv",
    mime="text/csv"
)


# ==================================================
# SEARCH LOCATION AND SHOW ALL RELATED DATA
# ==================================================
st.markdown("---")

heading(
    "🔍 Location Details Lookup",
    "Choose a Location ID to view its ranking and all available related records."
)

location_ids = (
    df[location_id_col]
    .dropna()
    .apply(normalize_id)
    .unique()
    .tolist()
)

location_ids = sorted(location_ids)

if location_ids:
    selected_id = st.selectbox(
        "Select or search for a Location ID",
        options=location_ids,
        index=None,
        placeholder="Choose a Location ID...",
        key="location_details_lookup"
    )

    if selected_id is not None:
        selected_rows = df[
            df[location_id_col].apply(normalize_id) == selected_id
        ]

        if not selected_rows.empty:
            selected_row = selected_rows.iloc[0]

            a, b, c, d = st.columns(4)

            a.metric("Priority Score", f"{selected_row[score_col]:.2f}")

            rank_value = selected_row[rank_col]
            b.metric(
                "Priority Rank",
                str(int(rank_value))
                if pd.notna(rank_value) else "N/A"
            )

            c.metric(
                "Priority Category",
                str(selected_row[priority_col])
            )

            d.metric(
                "Recorded Incidents",
                int(selected_row[incident_count_col])
            )

            st.markdown("### Complete Location Profile")

            st.dataframe(
                selected_row.to_frame(name="Value"),
                use_container_width=True
            )

            # Every dataset that may contain the selected location.
            related_datasets = {
                "Priority Ranking": df,
                "Location Information": locations,
                "Combined Data": combined,
                "Incidents": incidents,
                "Public Complaints": complaints,
                "K-Means Clusters": clustered
            }

            st.markdown("### All Related Records")

            st.caption(
                "Incident and complaint files may contain multiple records "
                "for one location. Select a tab to view matching records."
            )

            tabs = st.tabs(list(related_datasets.keys()))

            for tab, (dataset_name, dataset) in zip(
                tabs, related_datasets.items()
            ):
                with tab:
                    if dataset.empty:
                        st.info(
                            f"{dataset_name} is missing or contains no data."
                        )
                        continue

                    matching_rows = get_location_rows(
                        dataset, selected_id
                    )

                    id_column = find_column(
                        dataset,
                        [
                            "location_id",
                            "location id",
                            "locationid",
                            "loc_id",
                            "location"
                        ]
                    )

                    if id_column is None:
                        st.warning(
                            f"No recognizable Location ID column was found "
                            f"in {dataset_name}."
                        )
                        st.write(
                            "Available columns:",
                            list(dataset.columns)
                        )

                    elif matching_rows.empty:
                        st.info(
                            f"No records for Location ID {selected_id} "
                            f"were found in {dataset_name}."
                        )

                    else:
                        st.success(
                            f"{len(matching_rows)} matching record(s) found."
                        )

                        st.dataframe(
                            matching_rows,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            label=f"Download {dataset_name} records",
                            data=matching_rows.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name=(
                                f"location_{selected_id}_"
                                f"{dataset_name.lower().replace(' ', '_')}.csv"
                            ),
                            mime="text/csv",
                            key=f"download_{dataset_name}_{selected_id}"
                        )

            # Show a saved prediction only if the dataset contains one.
            prediction_col = find_column(
                df,
                [
                    "predicted_incidents",
                    "predicted_incident_count",
                    "prediction",
                    "model_prediction",
                    "predicted_class"
                ]
            )

            if prediction_col is not None:
                st.info(
                    f"Prediction field ({prediction_col}): "
                    f"{selected_row[prediction_col]}"
                )
            else:
                median_incidents = df[incident_count_col].median()
                observed_incidents = selected_row[incident_count_col]

                if observed_incidents >= median_incidents:
                    indicator = "At or above the dataset median."
                else:
                    indicator = "Below the dataset median."

                st.info(
                    f"Historical incident indicator: {indicator} "
                    "This is a comparison with recorded data, not a "
                    "prediction of future incidents."
                )

else:
    st.info("No Location IDs are available in the priority dataset.")


# ==================================================
# INCIDENT TYPE ANALYSIS
# ==================================================
st.markdown("---")

heading(
    "🚧 Incident Type Analysis",
    "Explore the types of incidents recorded in incidents.csv."
)

if not incidents.empty:
    incident_type_col = find_column(
        incidents,
        ["incident_type", "type", "incident category", "category"]
    )

    if incident_type_col is not None:
        incident_type_counts = (
            incidents[incident_type_col]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
        )

        chart_col, data_col = st.columns(2)

        with chart_col:
            fig, ax = plt.subplots(figsize=(8, 4))

            ax.bar(
                incident_type_counts.index,
                incident_type_counts.values
            )

            ax.set_title("Incidents by Type")
            ax.set_xlabel("Incident Type")
            ax.set_ylabel("Count")

            plt.xticks(rotation=35, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with data_col:
            incident_type_table = (
                incident_type_counts
                .rename_axis("Incident Type")
                .reset_index(name="Count")
            )

            st.dataframe(
                incident_type_table,
                use_container_width=True,
                hide_index=True
            )

    else:
        st.warning(
            "No incident_type column was found in incidents.csv."
        )
else:
    st.info("Add incidents.csv to display incident-type analysis.")


# ==================================================
# HIGH VS LOW PRIORITY COMPARISON
# ==================================================
st.markdown("---")

heading(
    "⚖️ High vs Low Priority Comparison",
    "Compare recorded indicators across priority categories."
)

comparison_fields = []

for possible_names in [
    ["incident_count", "total_incidents", "number_of_incidents"],
    ["avg_severity", "average_severity", "severity"],
    ["injuries", "total_injuries", "injury_count"],
    ["unresolved_incidents", "unresolved_count", "unresolved"],
    ["complaint_count", "total_complaints", "number_of_complaints"]
]:
    column = find_column(df, possible_names)

    if column is not None and column not in comparison_fields:
        comparison_fields.append(column)

if comparison_fields:
    comparison_data = df.copy()

    for column in comparison_fields:
        comparison_data[column] = pd.to_numeric(
            comparison_data[column], errors="coerce"
        )

    comparison = comparison_data.groupby(
        priority_col
    )[comparison_fields].mean()

    priority_order = [
        item for item in ["High", "Medium", "Low"]
        if item in comparison.index
    ]

    comparison = comparison.reindex(priority_order)

    st.dataframe(
        comparison.round(2),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    comparison.plot(kind="bar", ax=ax)

    ax.set_title("Average Indicators by Priority")
    ax.set_xlabel("Priority Category")
    ax.set_ylabel("Average Value")

    plt.xticks(rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("No comparison fields were found in the ranking dataset.")


# ==================================================
# K-MEANS CLUSTER INFORMATION
# ==================================================
st.markdown("---")

heading(
    "🧩 Location Clusters",
    "Optional K-Means results, if available."
)

if not clustered.empty:
    cluster_col = find_column(
        clustered,
        ["cluster", "cluster_id", "kmeans_cluster", "label"]
    )

    if cluster_col is not None:
        cluster_counts = (
            clustered[cluster_col]
            .value_counts()
            .sort_index()
        )

        chart_col, table_col = st.columns(2)

        with chart_col:
            fig, ax = plt.subplots(figsize=(6, 4))

            ax.bar(
                cluster_counts.index.astype(str),
                cluster_counts.values
            )

            ax.set_title("Locations per Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Number of Locations")

            st.pyplot(fig)
            plt.close(fig)

        with table_col:
            cluster_table = (
                cluster_counts
                .rename_axis("Cluster")
                .reset_index(name="Location Count")
            )

            st.dataframe(
                cluster_table,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Cluster numbers identify groups found by K-Means. "
            "A cluster number does not automatically indicate risk."
        )

    else:
        st.info(
            "The cluster file exists, but no cluster column was found."
        )

else:
    st.info(
        "Optional: run kmeans.py to generate "
        "kmeans_clustered_locations.csv."
    )


# ==================================================
# FOOTER
# ==================================================
st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:16px; color:#52677E;">
    <b>Smart City – Where to Fix First?</b><br>
    Data analysis for smarter urban planning<br>
    <small>
        Prototype for decision support. Verify findings through field
        inspections before taking real-world action.
    </small>
</div>
""", unsafe_allow_html=True)
