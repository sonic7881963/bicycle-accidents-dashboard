from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src import bike_data


SEVERITY_COLORS = {"Slight": "#3B82F6", "Serious": "#F59E0B", "Fatal": "#DC2626"}
MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIME_BAND_ORDER = ["Morning", "Afternoon", "Evening", "Night", "Unknown"]


@st.cache_data(show_spinner="Loading and preparing bicycle accident data...")
def cached_data():
    return bike_data.load_data()


def format_int(value: int) -> str:
    return f"{value:,}"


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def bar_chart(data: pd.DataFrame, x: str, y: str, title: str, hue: str | None = None, order: list[str] | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    if data.empty:
        ax.text(0.5, 0.5, "No records for this view", ha="center", va="center")
        ax.set_axis_off()
        return fig
    if hue == "Severity":
        sns.barplot(data=data, x=x, y=y, hue=hue, order=order, ax=ax, palette=SEVERITY_COLORS)
    else:
        sns.barplot(data=data, x=x, y=y, order=order, ax=ax, color="#2563EB")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y.replace("_", " "))
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def line_chart(data: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    if data.empty:
        ax1.text(0.5, 0.5, "No annual records for this view", ha="center", va="center")
        ax1.set_axis_off()
        return fig
    ax1.plot(data["Year"], data["Distinct_Accidents"], color="#2563EB", linewidth=2, label="Distinct accidents")
    ax1.set_ylabel("Distinct accidents")
    ax1.set_xlabel("Year")
    ax2 = ax1.twinx()
    ax2.plot(data["Year"], data["Serious_Fatal_Rate"], color="#DC2626", linewidth=2, label="Serious/fatal rate")
    ax2.set_ylabel("Serious/fatal rate")
    ax1.set_title("Annual Accident Volume and Serious/Fatal Share")
    fig.tight_layout()
    return fig


def severity_distribution(df: pd.DataFrame):
    counts = (
        df["Severity"]
        .value_counts()
        .reindex(bike_data.SEVERITY_ORDER)
        .dropna()
        .rename_axis("Severity")
        .reset_index(name="Records")
    )
    return bar_chart(counts, "Severity", "Records", "Severity Distribution")


def month_chart(data: pd.DataFrame):
    return bar_chart(data, "Month_Name", "Records", "Monthly Seasonality", order=MONTH_ORDER)


def day_time_heatmap(df: pd.DataFrame):
    pivot = (
        df.groupby(["Day", "Time_Band"], dropna=False)
        .size()
        .reset_index(name="Records")
        .pivot(index="Day", columns="Time_Band", values="Records")
        .reindex(index=DAY_ORDER, columns=TIME_BAND_ORDER)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.5, annot=True, fmt=".0f", ax=ax)
    ax.set_title("Records by Day and Time Band")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def options_for(df: pd.DataFrame, column: str) -> list[str]:
    return sorted(df[column].dropna().astype(str).unique())


def main():
    st.set_page_config(page_title="Bicycle Accidents Dashboard", layout="wide")
    sns.set_theme(style="whitegrid")

    st.title("Bicycle Accidents in Great Britain, 1979-2018")
    st.caption("Executive Story dashboard for exploratory, descriptive, and prescriptive analysis.")

    df, quality = cached_data()
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())

    with st.sidebar:
        st.header("Filters")
        year_range = st.slider("Year range", year_min, year_max, (year_min, year_max))
        severity_focus = st.radio("Severity focus", ["All", *bike_data.SEVERITY_ORDER], horizontal=True)
        if severity_focus == "All":
            severities = st.multiselect("Severity", bike_data.SEVERITY_ORDER, default=bike_data.SEVERITY_ORDER)
        else:
            severities = [severity_focus]
            st.caption(f"Showing {severity_focus} records.")
        genders = st.multiselect("Gender", options_for(df, "Gender"), default=options_for(df, "Gender"))
        age_groups = st.multiselect("Age group", options_for(df, "Age_Grp"), default=options_for(df, "Age_Grp"))
        weather = st.multiselect("Weather", options_for(df, "Weather_conditions"), default=options_for(df, "Weather_conditions"))
        road_types = st.multiselect("Road type", options_for(df, "Road_type"), default=options_for(df, "Road_type"))
        light_conditions = st.multiselect("Light condition", options_for(df, "Light_conditions"), default=options_for(df, "Light_conditions"))

    filtered = bike_data.apply_filters(
        df,
        year_range=year_range,
        severities=severities,
        genders=genders,
        age_groups=age_groups,
        weather=weather,
        road_types=road_types,
        light_conditions=light_conditions,
    )
    kpis = bike_data.kpi_summary(filtered)

    st.subheader("Overview")
    cols = st.columns(5)
    cols[0].metric("Distinct accidents", format_int(kpis["distinct_accidents"]))
    cols[1].metric("Cyclist records", format_int(kpis["cyclist_records"]))
    cols[2].metric("Total casualties", format_int(kpis["total_casualties"]))
    cols[3].metric("Serious/fatal rate", format_pct(kpis["serious_fatal_rate"]))
    cols[4].metric("Top age group", kpis["top_age_group"])
    st.caption(
        f"Most common road condition: {kpis['top_road_condition']} | "
        f"Most common light condition: {kpis['top_light_condition']}"
    )

    with st.expander("Data quality notes", expanded=False):
        st.write(
            f"Accident rows: {quality.accident_rows:,}. Biker rows: {quality.biker_rows:,}. "
            f"Matched cyclist records used in analysis: {quality.merged_rows:,}. "
            f"Distinct accident IDs after merge: {quality.distinct_accidents:,}."
        )
        st.write(
            f"Duplicate accident keys: {quality.accident_duplicate_keys:,}. Duplicate biker keys: {quality.biker_duplicate_keys:,}. "
            f"Unmatched biker rows excluded from accident-context analysis: {quality.unmatched_biker_rows:,}. "
            f"Missing cells before merge: {quality.missing_cells:,}. Unknown-like categorical cells: {quality.unknown_like_cells:,}."
        )
        st.write(
            "Accident-level metrics count each Accident_Index once. Cyclist-level severity metrics use the merged cyclist records."
        )

    if filtered.empty:
        st.warning("No records match the selected filters.")
        return

    st.subheader("Time Trends")
    annual = bike_data.annual_summary(filtered)
    st.pyplot(line_chart(annual), clear_figure=True)
    monthly = bike_data.monthly_summary(filtered)
    st.pyplot(month_chart(monthly), clear_figure=True)

    st.subheader("Severity and Demographics")
    left, right = st.columns(2)
    with left:
        st.pyplot(severity_distribution(filtered), clear_figure=True)
        gender_counts = bike_data.categorical_summary(filtered, "Gender", top_n=8)
        st.pyplot(bar_chart(gender_counts, "Gender", "Records", "Gender Distribution"), clear_figure=True)
    with right:
        age_rates = bike_data.severity_rate_summary(filtered, "Age_Grp", min_records=max(1, int(len(filtered) * 0.01))).head(10)
        st.pyplot(bar_chart(age_rates, "Age_Grp", "Serious_Fatal_Rate", "Top Age Groups by Serious/Fatal Share"), clear_figure=True)
        st.dataframe(age_rates, use_container_width=True, hide_index=True)

    st.subheader("Conditions and Context")
    road_tab, weather_tab, light_tab, speed_tab, timing_tab = st.tabs(["Road", "Weather", "Light", "Speed", "Day/Time"])
    with road_tab:
        road_rates = bike_data.severity_rate_summary(filtered, "Road_conditions", min_records=max(1, int(len(filtered) * 0.01))).head(8)
        st.pyplot(bar_chart(road_rates, "Road_conditions", "Serious_Fatal_Rate", "Road Conditions by Serious/Fatal Share"), clear_figure=True)
        road_type_rates = bike_data.severity_rate_summary(filtered, "Road_type", min_records=max(1, int(len(filtered) * 0.01))).head(8)
        st.pyplot(bar_chart(road_type_rates, "Road_type", "Serious_Fatal_Rate", "Road Type by Serious/Fatal Share"), clear_figure=True)
    with weather_tab:
        weather_rates = bike_data.severity_rate_summary(filtered, "Weather_conditions", min_records=max(1, int(len(filtered) * 0.01))).head(8)
        st.pyplot(bar_chart(weather_rates, "Weather_conditions", "Serious_Fatal_Rate", "Weather by Serious/Fatal Share"), clear_figure=True)
        st.dataframe(weather_rates, use_container_width=True, hide_index=True)
    with light_tab:
        light_rates = bike_data.severity_rate_summary(filtered, "Light_conditions", min_records=max(1, int(len(filtered) * 0.01))).head(8)
        st.pyplot(bar_chart(light_rates, "Light_conditions", "Serious_Fatal_Rate", "Light Conditions by Serious/Fatal Share"), clear_figure=True)
        st.dataframe(light_rates, use_container_width=True, hide_index=True)
    with speed_tab:
        speed_summary = bike_data.speed_limit_summary(filtered)
        st.dataframe(speed_summary, use_container_width=True, hide_index=True)
    with timing_tab:
        st.pyplot(day_time_heatmap(filtered), clear_figure=True)
        day_time = (
            filtered.groupby(["Day", "Time_Band"], dropna=False)
            .size()
            .reset_index(name="Records")
            .sort_values("Records", ascending=False)
            .head(20)
        )
        st.dataframe(day_time, use_container_width=True, hide_index=True)

    st.subheader("Prescriptive Recommendations")
    for item in bike_data.recommendation_summary(filtered):
        st.write(f"- {item}")


if __name__ == "__main__":
    main()
