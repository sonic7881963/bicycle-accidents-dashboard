import unittest

import pandas as pd

from src import bike_data


class BikeDataTests(unittest.TestCase):
    def setUp(self):
        self.accidents = pd.DataFrame(
            {
                "Accident_Index": ["A1", "A2", "A3", "A4"],
                "Number_of_Vehicles": [2, 1, 2, 3],
                "Number_of_Casualties": [1, 2, 1, 1],
                "Date": ["1979-01-01", "1980-06-15", "1980-06-16", "1981-12-31"],
                "Time": ["06:30", "13:15", "23:45", None],
                "Speed_limit": [30.0, 40.0, 30.0, 60.0],
                "Road_conditions": ["Dry", "Wet", "Missing Data", "Frost"],
                "Weather_conditions": ["Clear", "Rain", "Unknown", "Snow"],
                "Day": ["Monday", "Sunday", "Monday", "Thursday"],
                "Road_type": ["Single carriageway", "Roundabout", "Unknown", "Dual carriageway"],
                "Light_conditions": ["Daylight", "Darkness lights lit", "Darkness no lights", "Unknown"],
            }
        )
        self.bikers = pd.DataFrame(
            {
                "Accident_Index": ["A1", "A2", "A3", "A4"],
                "Gender": ["Male", "Female", "Male", "Female"],
                "Severity": ["Slight", "Serious", "Fatal", "Slight"],
                "Age_Grp": ["16 to 20", "36 to 45", "36 to 45", "66 to 75"],
            }
        )
        self.df, self.quality = bike_data.preprocess(self.accidents, self.bikers)

    def test_preprocess_adds_date_and_time_features(self):
        self.assertEqual(self.df.loc[self.df["Accident_Index"] == "A1", "Year"].iloc[0], 1979)
        self.assertEqual(self.df.loc[self.df["Accident_Index"] == "A2", "Month"].iloc[0], 6)
        self.assertEqual(self.df.loc[self.df["Accident_Index"] == "A1", "Hour"].iloc[0], 6)
        self.assertEqual(self.df.loc[self.df["Accident_Index"] == "A1", "Time_Band"].iloc[0], "Morning")
        self.assertEqual(self.df.loc[self.df["Accident_Index"] == "A3", "Time_Band"].iloc[0], "Night")
        self.assertEqual(self.df.loc[self.df["Accident_Index"] == "A4", "Time_Band"].iloc[0], "Unknown")

    def test_preprocess_standardizes_unknown_values(self):
        row = self.df.loc[self.df["Accident_Index"] == "A3"].iloc[0]
        self.assertEqual(row["Road_conditions"], "Unknown")
        self.assertEqual(row["Weather_conditions"], "Unknown")
        self.assertEqual(row["Road_type"], "Unknown")

    def test_quality_summary_counts_rows_and_keys(self):
        self.assertEqual(self.quality.accident_rows, 4)
        self.assertEqual(self.quality.biker_rows, 4)
        self.assertEqual(self.quality.merged_rows, 4)
        self.assertEqual(self.quality.distinct_accidents, 4)
        self.assertEqual(self.quality.accident_duplicate_keys, 0)
        self.assertEqual(self.quality.biker_duplicate_keys, 0)
        self.assertEqual(self.quality.unmatched_biker_rows, 0)
        self.assertGreaterEqual(self.quality.unknown_like_cells, 3)

    def test_preprocess_drops_unmatched_biker_rows_from_analysis(self):
        extra_biker = pd.DataFrame(
            {
                "Accident_Index": ["A5"],
                "Gender": ["Male"],
                "Severity": ["Fatal"],
                "Age_Grp": ["26 to 35"],
            }
        )
        df, quality = bike_data.preprocess(self.accidents, pd.concat([self.bikers, extra_biker], ignore_index=True))

        self.assertEqual(quality.biker_rows, 5)
        self.assertEqual(quality.unmatched_biker_rows, 1)
        self.assertEqual(quality.merged_rows, 4)
        self.assertNotIn("A5", set(df["Accident_Index"]))

    def test_apply_filters_limits_rows(self):
        filtered = bike_data.apply_filters(
            self.df,
            year_range=(1980, 1980),
            severities=["Serious", "Fatal"],
            genders=["Female", "Male"],
            age_groups=["36 to 45"],
            weather=["Rain", "Unknown"],
            road_types=["Roundabout", "Unknown"],
            light_conditions=["Darkness lights lit", "Darkness no lights"],
        )
        self.assertEqual(set(filtered["Accident_Index"]), {"A2", "A3"})

    def test_kpi_summary_uses_distinct_accidents_and_cyclist_rows(self):
        summary = bike_data.kpi_summary(self.df)
        self.assertEqual(summary["distinct_accidents"], 4)
        self.assertEqual(summary["cyclist_records"], 4)
        self.assertEqual(summary["total_casualties"], 5)
        self.assertEqual(summary["serious_fatal_rate"], 0.5)
        self.assertEqual(summary["top_age_group"], "36 to 45")

    def test_annual_summary_counts_accidents_once(self):
        summary = bike_data.annual_summary(self.df)
        row_1980 = summary.loc[summary["Year"] == 1980].iloc[0]
        self.assertEqual(row_1980["Distinct_Accidents"], 2)
        self.assertEqual(row_1980["Cyclist_Records"], 2)
        self.assertEqual(row_1980["Serious_Fatal_Records"], 2)

    def test_monthly_summary_orders_months(self):
        summary = bike_data.monthly_summary(self.df)
        self.assertEqual(summary.iloc[0]["Month"], 1)
        self.assertEqual(summary.iloc[0]["Month_Name"], "January")
        self.assertEqual(summary.loc[summary["Month"] == 6, "Records"].iloc[0], 2)

    def test_severity_rate_summary_includes_counts_and_rates(self):
        summary = bike_data.severity_rate_summary(self.df, "Age_Grp", min_records=1)
        row = summary.loc[summary["Age_Grp"] == "36 to 45"].iloc[0]
        self.assertEqual(row["Records"], 2)
        self.assertEqual(row["Serious_Fatal_Records"], 2)
        self.assertEqual(row["Serious_Fatal_Rate"], 1.0)

    def test_speed_limit_summary_counts_distinct_accidents(self):
        summary = bike_data.speed_limit_summary(self.df)
        row = summary.loc[summary["Speed_limit"] == 30.0].iloc[0]
        self.assertEqual(row["Distinct_Accidents"], 2)
        self.assertEqual(row["Cyclist_Records"], 2)

    def test_recommendations_are_grounded_in_filtered_data(self):
        recommendations = bike_data.recommendation_summary(self.df)
        joined = " ".join(recommendations)
        self.assertIn("36 to 45", joined)
        self.assertIn("serious/fatal", joined.lower())


if __name__ == "__main__":
    unittest.main()
