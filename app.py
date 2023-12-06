import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import datetime
import numpy as np
import plotly.express as px

df = pd.read_csv("data/preprocessed/full_dataset.csv", index_col=0, low_memory=False)
df["total_personnel"] = df.iloc[:, 58:128].sum(axis=1)
treatment_features_personnel = df.columns[58:128]
treatment_features_qty = df.columns[128:-16]

# Get the current time
current_time = datetime.datetime.now()

# Print the current time
print("Current Time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
print(f"{df.shape[1]} features\n{df.fire_id.nunique()} fires\n{df.shape[0]} reports")


def filter_data(log_area_range, reports_range, cause):
    min_area, max_area = np.power(10, log_area_range)
    min_reports, max_reports = reports_range
    subset_df = df.copy()

    # Cause filter
    if "all" not in cause:
        causes = [int(c) for c in cause]
        subset_df = subset_df[subset_df.cause_id.notna()]
        subset_df = subset_df[subset_df["cause_id"].astype(int).isin(causes)]

    # Fire size filter
    fire_sizes = subset_df.groupby("fire_id").size()
    long_fires = fire_sizes[(fire_sizes >= min_reports) & (fire_sizes <= max_reports)]
    subset_df = subset_df[subset_df.fire_id.isin(long_fires.index)]

    # Fire area filter
    fire_max_area = subset_df.groupby("fire_id")["area"].max()
    large_fires = fire_max_area[
        (fire_max_area >= min_area) & (fire_max_area <= max_area)
    ]
    subset_df = subset_df[subset_df.fire_id.isin(large_fires.index)]

    return subset_df


def create_figure_for_fire(fire_df):
    fig = sp.make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=fire_df["time_to_first_report"],
            y=fire_df["total_personnel"],
            mode="markers+lines",
            name="Treatment Features",
            marker=dict(size=5, opacity=0.2, line=dict(width=2, color="DarkSlateGrey")),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=fire_df["time_to_first_report"],
            y=fire_df["area"],
            mode="markers+lines",
            name="Area",
            marker=dict(size=5, opacity=0.2, line=dict(width=2, color="DarkSlateGrey")),
            line=dict(width=3, color="red", dash="dot"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Total Personnel and Area over Time",
        xaxis_title="Time to First Report",
        template="plotly_white",
    )

    fig.update_yaxes(title_text="Treatment Features", secondary_y=False)
    fig.update_yaxes(title_text="Area", secondary_y=True)

    return fig


def create_treatment_figure_for_fire(fire_df):
    fig = sp.make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=fire_df["time_to_first_report"],
            y=fire_df["area"],
            mode="lines",
            name="Area",
            line=dict(width=3, color="red", dash="dot"),
        ),
        secondary_y=True,
    )
    for feature in fire_df.columns[1:-2]:
        fig.add_trace(
            go.Scatter(
                x=fire_df["time_to_first_report"],
                y=fire_df[feature],
                mode="lines",
                name=feature,
            ),
            secondary_y=False,
        )

    fig.update_layout(
        title="Resource quantity and Area over Time",
        xaxis_title="Time to First Report",
        template="plotly_white",
    )

    fig.update_yaxes(title_text="Treatment Features", secondary_y=False)
    fig.update_yaxes(title_text="Area", secondary_y=True)

    return fig


# ---------------------------------------------------------------------------- #
#                                    FILTER                                    #
# ---------------------------------------------------------------------------- #
st.title("Fire vs suppression")
col1, col2, col3 = st.columns(3)
with col1:
    cause = st.multiselect("Select Cause", ["all", "1", "2", "3", "4"], default="all")

with col2:
    log_area_range = st.slider("Log Area Range", 0.0, 7.0, (5.0, 6.0))

with col3:
    reports_range = st.slider("Number of Reports Range", 1, 150, (10, 100))

filtered_df = filter_data(log_area_range, reports_range, cause)

col1, col2 = st.columns(2)
n = filtered_df.fire_id.nunique()
fire_indices = list(range(n))

with col1:
    st.text(
        f"Number of fires: {len(fire_indices)} \nNumber of reports: {len(filtered_df)}"
    )
with col2:
    selected_fire_index = st.number_input(
        "Select Fire Index",
        min_value=min(fire_indices),
        max_value=max(fire_indices),
        value=0,
    )
fire_id = filtered_df.fire_id.unique()[selected_fire_index]
df_fire = filtered_df[filtered_df.fire_id == fire_id]


# ---------------------------------------------------------------------------- #
#                                     PLOTS                                    #
# ---------------------------------------------------------------------------- #

fig = create_figure_for_fire(df_fire)
st.plotly_chart(fig)


sum_pers = df_fire[treatment_features_personnel].sum()
sum_pers = sum_pers[sum_pers > 0]


col1, col2 = st.columns(2)
fig = px.pie(
    sum_pers,
    values=sum_pers.values,
    names=sum_pers.index,
    title="Total personnel by resource",
)
with col1:
    st.plotly_chart(fig)


sum_qty = df_fire[treatment_features_qty].sum()
sum_qty = sum_qty[sum_qty > 0]

fig = px.pie(
    sum_qty,
    values=sum_qty.values,
    names=sum_qty.index,
    title="Total quantity by resource ",
)
with col2:
    st.plotly_chart(fig)

fig = create_treatment_figure_for_fire(
    df_fire.loc[:, list(sum_qty.index) + ["area", "time_to_first_report"]]
)
st.plotly_chart(fig)


# ---------------------------------------------------------------------------- #
#                                      DF                                      #
# ---------------------------------------------------------------------------- #
st.write(
    df_fire[
        [
            "fire_id",
            "cause_id",
            "area",
            "total_personnel",
            "date",
            "time_to_first_report",
        ]
        + sum_pers.index.tolist()
        + sum_qty.index.tolist()
    ]
)
