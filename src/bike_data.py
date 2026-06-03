from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCIDENTS_PATH = PROJECT_ROOT / "archive" / "Accidents.csv"
BIKERS_PATH = PROJECT_ROOT / "archive" / "Bikers.csv"

SEVERITY_ORDER = ["Slight", "Serious", "Fatal"]
SERIOUS_FATAL = ["Serious", "Fatal"]
UNKNOWN_VALUES = {"Unknown", "Missing Data", "", "nan", "NaN", "None"}


@dataclass(frozen=True)
class DataQualitySummary:
    accident_rows: int
    biker_rows: int
    merged_rows: int
    distinct_accidents: int
    accident_duplicate_keys: int
    biker_duplicate_keys: int
    unmatched_biker_rows: int
    missing_cells: int
    unknown_like_cells: int


def standardize_unknowns(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.isin(UNKNOWN_VALUES), "Unknown").fillna("Unknown")


def make_time_band(hour: float | int | None) -> str:
    if pd.isna(hour):
        return "Unknown"
    hour_int = int(hour)
    if 5 <= hour_int <= 11:
        return "Morning"
    if 12 <= hour_int <= 16:
        return "Afternoon"
    if 17 <= hour_int <= 20:
        return "Evening"
    return "Night"


def preprocess(accidents: pd.DataFrame, bikers: pd.DataFrame) -> tuple[pd.DataFrame, DataQualitySummary]:
    accidents = accidents.copy()
    bikers = bikers.copy()

    categorical_accident_cols = [
        "Road_conditions",
        "Weather_conditions",
        "Day",
        "Road_type",
        "Light_conditions",
    ]
    categorical_biker_cols = ["Gender", "Severity", "Age_Grp"]

    for col in categorical_accident_cols:
        accidents[col] = standardize_unknowns(accidents[col])
    for col in categorical_biker_cols:
        bikers[col] = standardize_unknowns(bikers[col])

    accidents["Date"] = pd.to_datetime(accidents["Date"], errors="coerce")
    accidents["Year"] = accidents["Date"].dt.year.astype("Int64")
    accidents["Month"] = accidents["Date"].dt.month.astype("Int64")
    accidents["Month_Name"] = accidents["Date"].dt.month_name().fillna("Unknown")

    parsed_time = pd.to_datetime(accidents["Time"], format="%H:%M", errors="coerce")
    accidents["Hour"] = parsed_time.dt.hour.astype("Int64")
    accidents["Time_Band"] = accidents["Hour"].map(make_time_band)

    accidents["Speed_limit"] = pd.to_numeric(accidents["Speed_limit"], errors="coerce")
    accidents["Number_of_Vehicles"] = pd.to_numeric(accidents["Number_of_Vehicles"], errors="coerce").fillna(0).astype(int)
    accidents["Number_of_Casualties"] = pd.to_numeric(accidents["Number_of_Casualties"], errors="coerce").fillna(0).astype(int)

    quality = DataQualitySummary(
        accident_rows=len(accidents),
        biker_rows=len(bikers),
        merged_rows=0,
        distinct_accidents=accidents["Accident_Index"].nunique(),
        accident_duplicate_keys=int(accidents["Accident_Index"].duplicated().sum()),
        biker_duplicate_keys=int(bikers["Accident_Index"].duplicated().sum()),
        unmatched_biker_rows=0,
        missing_cells=int(accidents.isna().sum().sum() + bikers.isna().sum().sum()),
        unknown_like_cells=int(
            accidents[categorical_accident_cols].eq("Unknown").sum().sum()
            + bikers[categorical_biker_cols].eq("Unknown").sum().sum()
        ),
    )

    merged_with_indicator = bikers.merge(accidents, on="Accident_Index", how="left", validate="many_to_one", indicator=True)
    unmatched_biker_rows = int((merged_with_indicator["_merge"] == "left_only").sum())
    merged = merged_with_indicator.loc[merged_with_indicator["_merge"] == "both"].drop(columns="_merge").copy()
    quality = DataQualitySummary(
        accident_rows=quality.accident_rows,
        biker_rows=quality.biker_rows,
        merged_rows=len(merged),
        distinct_accidents=merged["Accident_Index"].nunique(),
        accident_duplicate_keys=quality.accident_duplicate_keys,
        biker_duplicate_keys=quality.biker_duplicate_keys,
        unmatched_biker_rows=unmatched_biker_rows,
        missing_cells=quality.missing_cells,
        unknown_like_cells=quality.unknown_like_cells,
    )
    return merged, quality


def load_data(accidents_path: Path = ACCIDENTS_PATH, bikers_path: Path = BIKERS_PATH) -> tuple[pd.DataFrame, DataQualitySummary]:
    accidents = pd.read_csv(accidents_path)
    bikers = pd.read_csv(bikers_path)
    return preprocess(accidents, bikers)


def apply_filters(
    df: pd.DataFrame,
    year_range: tuple[int, int],
    severities: list[str],
    genders: list[str],
    age_groups: list[str],
    weather: list[str],
    road_types: list[str],
    light_conditions: list[str],
) -> pd.DataFrame:
    start_year, end_year = year_range
    mask = df["Year"].between(start_year, end_year, inclusive="both")
    mask &= _apply_in_filter(df, "Severity", severities)
    mask &= _apply_in_filter(df, "Gender", genders)
    mask &= _apply_in_filter(df, "Age_Grp", age_groups)
    mask &= _apply_in_filter(df, "Weather_conditions", weather)
    mask &= _apply_in_filter(df, "Road_type", road_types)
    mask &= _apply_in_filter(df, "Light_conditions", light_conditions)
    return df.loc[mask].copy()


def kpi_summary(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "distinct_accidents": 0,
            "cyclist_records": 0,
            "total_casualties": 0,
            "serious_fatal_rate": 0.0,
            "top_age_group": "No data",
            "top_road_condition": "No data",
            "top_light_condition": "No data",
        }
    accident_level = df.drop_duplicates("Accident_Index")
    serious_fatal_rate = df["Severity"].isin(SERIOUS_FATAL).mean()
    return {
        "distinct_accidents": int(df["Accident_Index"].nunique()),
        "cyclist_records": int(len(df)),
        "total_casualties": int(accident_level["Number_of_Casualties"].sum()),
        "serious_fatal_rate": float(serious_fatal_rate),
        "top_age_group": str(df["Age_Grp"].mode().iloc[0]),
        "top_road_condition": str(accident_level["Road_conditions"].mode().iloc[0]),
        "top_light_condition": str(accident_level["Light_conditions"].mode().iloc[0]),
    }


def annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Year", "Distinct_Accidents", "Cyclist_Records", "Serious_Fatal_Records"])
    grouped = (
        df.assign(Is_Serious_Fatal=df["Severity"].isin(SERIOUS_FATAL).astype(int))
        .groupby("Year", dropna=True)
        .agg(
            Distinct_Accidents=("Accident_Index", "nunique"),
            Cyclist_Records=("Accident_Index", "size"),
            Serious_Fatal_Records=("Is_Serious_Fatal", "sum"),
        )
        .reset_index()
        .sort_values("Year")
    )
    grouped["Serious_Fatal_Rate"] = grouped["Serious_Fatal_Records"] / grouped["Cyclist_Records"]
    return grouped


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Month", "Month_Name", "Records", "Serious_Fatal_Records", "Serious_Fatal_Rate"])
    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    summary = (
        df.assign(Is_Serious_Fatal=df["Severity"].isin(SERIOUS_FATAL).astype(int))
        .groupby("Month", dropna=True)
        .agg(Records=("Accident_Index", "size"), Serious_Fatal_Records=("Is_Serious_Fatal", "sum"))
        .reset_index()
        .sort_values("Month")
    )
    summary["Month_Name"] = summary["Month"].astype(int).map(month_names)
    summary["Serious_Fatal_Rate"] = summary["Serious_Fatal_Records"] / summary["Records"]
    return summary[["Month", "Month_Name", "Records", "Serious_Fatal_Records", "Serious_Fatal_Rate"]]


def categorical_summary(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[column, "Records", "Share"])
    counts = df[column].value_counts(dropna=False).head(top_n).rename_axis(column).reset_index(name="Records")
    counts["Share"] = counts["Records"] / len(df)
    return counts


def severity_rate_summary(df: pd.DataFrame, group_col: str, min_records: int = 100) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, "Records", "Serious_Fatal_Records", "Serious_Fatal_Rate"])
    summary = (
        df.assign(Is_Serious_Fatal=df["Severity"].isin(SERIOUS_FATAL).astype(int))
        .groupby(group_col, dropna=False)
        .agg(Records=("Accident_Index", "size"), Serious_Fatal_Records=("Is_Serious_Fatal", "sum"))
        .reset_index()
    )
    summary["Serious_Fatal_Rate"] = summary["Serious_Fatal_Records"] / summary["Records"]
    return summary.loc[summary["Records"] >= min_records].sort_values(
        ["Serious_Fatal_Rate", "Records"], ascending=[False, False]
    )


def speed_limit_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Speed_limit", "Distinct_Accidents", "Cyclist_Records", "Serious_Fatal_Records", "Serious_Fatal_Rate"])
    summary = (
        df.assign(Is_Serious_Fatal=df["Severity"].isin(SERIOUS_FATAL).astype(int))
        .groupby("Speed_limit", dropna=False)
        .agg(
            Distinct_Accidents=("Accident_Index", "nunique"),
            Cyclist_Records=("Accident_Index", "size"),
            Serious_Fatal_Records=("Is_Serious_Fatal", "sum"),
        )
        .reset_index()
        .sort_values("Speed_limit")
    )
    summary["Serious_Fatal_Rate"] = summary["Serious_Fatal_Records"] / summary["Cyclist_Records"]
    return summary


def recommendation_summary(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No records match the selected filters; broaden the filters to generate recommendations."]

    age_rates = severity_rate_summary(df, "Age_Grp", min_records=max(1, int(len(df) * 0.01)))
    light_rates = severity_rate_summary(df, "Light_conditions", min_records=max(1, int(len(df) * 0.01)))
    road_rates = severity_rate_summary(df, "Road_type", min_records=max(1, int(len(df) * 0.01)))

    recommendations = []
    if not age_rates.empty:
        row = age_rates.iloc[0]
        recommendations.append(
            f"Review rider safety messaging for age group {row['Age_Grp']}, which has "
            f"{row['Records']:,} records and a {row['Serious_Fatal_Rate']:.1%} serious/fatal share in the filtered data."
        )
    if not light_rates.empty:
        row = light_rates.iloc[0]
        recommendations.append(
            f"Consider visibility interventions for {row['Light_conditions']} conditions because this group shows "
            f"a {row['Serious_Fatal_Rate']:.1%} serious/fatal share across {row['Records']:,} records."
        )
    if not road_rates.empty:
        row = road_rates.iloc[0]
        recommendations.append(
            f"Prioritize road-type safety review for {row['Road_type']} locations, where the filtered data shows "
            f"{row['Records']:,} records and a {row['Serious_Fatal_Rate']:.1%} serious/fatal share."
        )
    recommendations.append("Treat these priorities as descriptive signals for safety review, not causal proof.")
    return recommendations


def _apply_in_filter(df: pd.DataFrame, column: str, selected: list[str]) -> pd.Series:
    if not selected:
        return pd.Series(True, index=df.index)
    return df[column].isin(selected)
