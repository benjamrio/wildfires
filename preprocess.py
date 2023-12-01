import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import argparse


def load_data(data_path):
    """Load the dataframes from the preprocessed files (non statistical engineering)

    Arguments:
        data_path {str} -- data folder
    """
    incidents = pd.read_csv(os.path.join(data_path, "incidents.csv"), index_col=0)
    print(f"Incidents shape: {incidents.shape}")
    reports = pd.read_csv(os.path.join(data_path, "reports.csv"))
    print(f"Reports shape: {reports.shape}")
    resources = pd.read_csv(os.path.join(data_path, "resources.csv"), index_col=0)
    print(f"Resources shape: {resources.shape}")
    incidents.merge(reports, left_on="INCIDENT_IDENTIFIER", right_on="INC_IDENTIFIER")
    merged = incidents.merge(
        reports, left_on="INCIDENT_IDENTIFIER", right_on="INC_IDENTIFIER"
    )
    #! Important: mean and sum lead to very different results for pivot table ie not unique
    resources = pd.read_csv(os.path.join(input_path, "resources.csv")).rename(
        columns={"report_id": "INC209R_IDENTIFIER"}
    )

    result = merged.merge(resources, on="INC209R_IDENTIFIER")
    return result


def create_features(df):
    """Create feature for each day of a fire
    1. Lag/rolling variables, anything that takes into account the past
    2. Target variables
    Arguments:
        df {DataFrame} -- merged dataframe: incidents, reports, resources. A row is a report.
        features {List[str]} -- List of features to use for the task.
        target {str} -- target to predict
    """

    # RENAME COLUMNS
    col_map = {
        "INCIDENT_IDENTIFIER": "fire_id",
        "INC209R_IDENTIFIER": "report_id",
        "year_y": "year",
        "CURR_INCIDENT_AREA": "incident_area",
        "INC209RU_IDENTIFIER": "resource_id",
        "INC209R_IDENTIFIER": "report_id",
        "RESOURCE_QUANTITY": "quantity",
        "RESOURCE_PERSONNEL": "personnel",
        "CURR_INCIDENT_AREA": "area",
    }
    df = df.copy().rename(columns=col_map)
    # Time window select
    df = df[df["year"] > 2013]
    df = df.sort_values(by=["fire_id", "mean_report_date"])
    df["date"] = pd.to_datetime(df["mean_report_date"])
    # Time features
    df["report_number"] = df.groupby("fire_id").cumcount() + 1
    df["prev_area"] = df.groupby("fire_id").shift(1)["area"]
    df["next_area"] = df.groupby("fire_id").shift(-1)["area"]
    df["prev_date"] = df.groupby("fire_id").shift(1)["date"]
    df["next_date"] = df.groupby("fire_id").shift(-1)["date"]
    df["prev_date_diff"] = (df["date"] - df["prev_date"]).dt.total_seconds() / (
        24 * 3600
    )
    df["next_date_diff"] = (df["next_date"] - df["date"]).dt.total_seconds() / (
        24 * 3600
    )
    df["prev_area_diff"] = df["area"] - df["prev_area"]
    df["next_area_diff"] = df["next_area"] - df["area"]
    df["prev_derivate"] = df["prev_area_diff"] / df["prev_date_diff"]
    df["next_derivate"] = df["next_area_diff"] / df["next_date_diff"]
    df["will_grow"] = df["next_area_diff"] > 0
    df["time_to_first_report"] = (
        df["date"] - df.groupby("fire_id")["date"].transform("min")
    ).dt.total_seconds() / (24 * 3600)
    return df


if __name__ == "__main__":
    input_path = "data/cleaned"
    output_path = "data/preprocessed"
    df = load_data(input_path)
    print(f"Initial shape: {df.shape}")
    full_df = create_features(df)
    print(f"Post preprocessing shape: {full_df.shape}")
    full_df.to_csv(os.path.join(output_path, "full_dataset.csv"))
