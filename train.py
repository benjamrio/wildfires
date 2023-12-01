import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GroupKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV
import pickle
import os
from sklearn.preprocessing import OneHotEncoder

REPORT_ID = "report_id"
GROUP_ID = "fire_id"


def create_task_dataset(df, id_cols, cat_features, num_features, target):
    """Feature and row selection, encoding

    Arguments:
        df {DataFrame} -- merged dataframe: incidents, reports, resources. A row is a report.
        cat_features {List[str]} -- List of cat_features to use for the task.
        target {str} -- target to predict
    """

    assert target in df.columns, f"Target {target} not in dataframe"
    assert all(
        [col in df.columns for col in cat_features]
    ), f"cat_features {cat_features} not in dataframe"
    assert all(
        [col in df.columns for col in id_cols]
    ), f"ID columns {id_cols} not in dataframe"
    assert len(cat_features) > 0, "You need to select at least one feature"

    # Select cat_features
    df = df[id_cols + cat_features + num_features + [target]]

    # Drop rows with missing values for target
    df = df.dropna(subset=[target])

    # Find categorical columns
    print(f"Identified {len(cat_features)} categorical columns: \n{cat_features}")

    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df[cat_features])
    encoded_df = pd.DataFrame(
        encoded_data, columns=encoder.get_feature_names_out(cat_features)
    )
    df = df.drop(columns=cat_features).reset_index(drop=True)
    encoded_df = pd.concat([df, encoded_df], axis=1)
    return encoded_df, encoder


def split_train_test(
    task_df, test_size, group_id, random_state=42, save_datasets=False
):
    """Split data into train and test sets using groups (group_id)

    Arguments:
        task_df {DataFrame} --
        group_id {str} -- group identifier
        random_state {int} -- random state
        target {str} -- target feature
        save_datasets {bool} -- whether to save train and test sets (useful for later evals, tests)

    Returns:
        DataFrame -- train set
        DataFrame -- test set
    """
    assert group_id in task_df.columns, f"Group id {group_id} not in dataframe"
    assert (
        test_size > 0 and test_size < 1
    ), f"Test size {test_size} must be between 0 and 1"

    df = task_df.sample(frac=1, random_state=random_state)
    unique_ids = df[GROUP_ID].unique()
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=test_size, random_state=random_state
    )
    train_df = df[df[GROUP_ID].isin(train_ids)]
    test_df = df[df[GROUP_ID].isin(test_ids)]
    if save_datasets:
        train_df.to_csv("data/preprocessed/train.csv", index=False)
        test_df.to_csv("data/preprocessed/test.csv", index=False)
    return train_df, test_df


def split_X_y(df, id_cols, target):
    """Split data into X and y

    Arguments:
        df {DataFrame} -- dataset
        id_cols {List[str]} -- identifiers columns (not features, not target)
        target {str} -- target feature

    Returns:
        DataFrame -- X
        DataFrame -- y
    """
    assert target in df.columns, f"{target} not in df.columns"
    assert all([col in df.columns for col in id_cols]), f"Not all id_cols in df.columns"

    X = df.drop(columns=[target] + id_cols)
    y = df[target]
    return X, y


def cv_train(
    train_df,
    test_df,
    n_splits,
    models,
    scoring_metric,
    random_state,
    id_cols,
    group_id,
    cv,
    save_best_estimators=False,
):
    """Compute the score of each model

    Arguments:
        X {DataFrame} -- features
        y {Series} -- target
        n_splits {int} -- number of cv splits
        models {Dict} -- dict of models

    Returns:
        DataFrame -- scores
    """
    searches = {}
    X, y = split_X_y(train_df, id_cols, target)
    gkf = GroupKFold(n_splits=n_splits)
    for model_name, model in models.items():
        print(f"Running {cv} for {model.__class__.__name__}")
        if cv == "RandomizedSearchCV":
            search = RandomizedSearchCV(
                model,
                hp_grid[model_name],
                scoring=scoring_metric,
                cv=gkf.split(X, y, groups=train_df[group_id]),
                refit=True,
                random_state=random_state,
                n_jobs=-1,
            )
        elif cv == "GridSearchCV":
            search = GridSearchCV(
                model,
                hp_grid[model_name],
                scoring=scoring_metric,
                cv=gkf.split(X, y, groups=train_df[group_id]),
                refit=True,
                n_jobs=-1,
            )
        elif cv == "BayesSearchCV":
            if model_name == "LinearRegression":
                search = LinearRegression(n_jobs=-1)
            else:
                search = BayesSearchCV(
                    model,
                    hp_grid[model_name],
                    scoring=scoring_metric,
                    cv=gkf.split(X, y, groups=train_df[group_id]),
                    refit=True,
                    n_jobs=-1,
                )
        search.fit(X, y)
        searches[model_name] = search

        if save_best_estimators:
            folder_name = f"models/{scoring_metric}/{cv}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name, exist_ok=True)
            with open(os.path.join(folder_name, f"{model_name}.pkl"), "wb") as f:
                pickle.dump(search.best_estimator_, f)
    return searches


def get_info_from_searches(searches, train_df, test_df, id_cols, target):
    """Transform searches to a dict of details

    Arguments:
        searches {Dict} -- dict of searches

    Returns:
        Dict -- dict of details
    """
    searches_details = {}
    # Evaluate on test set
    X_train, y_train = split_X_y(train_df, id_cols, target)
    X_test, y_test = split_X_y(test_df, id_cols, target)
    for model in searches.keys():
        print(model)
        model_details = {}
        estimator = searches[model].best_estimator_
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        model_details["test_rmse"] = mean_squared_error(y_test, y_pred, squared=False)
        model_details["test_mae"] = mean_absolute_error(y_test, y_pred)
        model_details["best_score"] = searches[model].best_score_
        model_details["best_params"] = searches[model].best_params_
        model_details["std_test_score"] = (
            searches[model].cv_results_["std_test_score"].mean()
        )
        model_details["mean_test_score"] = (
            searches[model].cv_results_["mean_test_score"].mean()
        )
        searches_details[model] = model_details

    searches_details["baseline"] = {}
    searches_details["baseline"]["test_rmse"] = mean_squared_error(
        y_test, np.zeros(len(y_test)), squared=False
    )
    searches_details["baseline"]["test_mae"] = mean_absolute_error(
        y_test, np.zeros(len(y_test))
    )
    return searches_details


def save_info_from_searches(
    searches,
    filename,
    metric,
    optimizer,
    train_df,
    test_df,
    id_cols,
    target,
    overwrite=True,
):
    """Save searches details to a json file

    Arguments:
        searches {Dict} -- dict of searches
        filename {str} -- filename
    """
    infos = get_info_from_searches(
        searches,
        train_df,
        test_df,
        id_cols,
        target,
    )
    if not os.path.exists(filename):
        with open(filename, "w") as jsonfile:
            json.dump({}, jsonfile)
    with open(filename, "r") as jsonfile:
        data = json.load(jsonfile)
    if metric in data.keys():
        if optimizer not in data[metric].keys() or overwrite:
            data[metric][optimizer] = infos
        else:
            print(f"Searches for {optimizer} already in {metric}.json")
    else:
        data[metric] = {optimizer: infos}

    with open(filename, "w") as jsonfile:
        json.dump(data, jsonfile)


METRICS = ["neg_mean_absolute_error", "neg_root_mean_squared_error"]  # r2

OPTIMIZERS = [
    "RandomizedSearchCV"
]  # "BayesSearchCV"]  # "GridSearchCV", #"RandomizedSearchCV",

MODELS = {
    "regression": {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "XGB": xgb.XGBRegressor(),
        #'RandomForest': RandomForestRegressor(),
        #'SVR': SVR()
    },
    "classification": {
        "LogisticRegression": LogisticRegression(),
        "XGB": xgb.XGBClassifier(),
        "RandomForest": RandomForestClassifier(),
        "SVC": SVC(),
    },
}

HP_GRID = {
    "LinearRegression": {},
    "Ridge": {"alpha": [0.1, 1, 10], "max_iter": [1000, 2000, 3000]},
    "Lasso": {"alpha": [0.1, 1, 10], "max_iter": [1000, 2000, 3000]},
    "XGB": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.5, 1],
    },
    #'RandomForest': {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'criterion': ["squared_error"]},
    #'SVR': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], "gamma": ["scale", "auto", 0.01, 0.1, 1, 10], 'C': [0.1, 1, 10]}
}

if __name__ == "__main__":
    cat_features = [
        "cause_id",
        "month",
        "year",
        "STATUS",
    ]
    num_features = ["report_number", "incident_area", "previous_diff_incident_area"]
    target = "diff_incident_area"
    id_cols = [REPORT_ID, GROUP_ID]
    metrics = METRICS
    optimizers = OPTIMIZERS
    models = MODELS
    hp_grid = HP_GRID
    random_state = 42
    save_train_test_datasets = False
    save_best_estimators = False
    test_size = 0.2

    # load
    df = pd.read_csv("data/preprocessed/prep_nan.csv")

    task_df, encoder = create_task_dataset(
        df, id_cols, cat_features, num_features, target
    )
    non_id_cols = [col for col in task_df.columns if col not in id_cols]

    # split
    train_df, test_df = split_train_test(
        task_df.dropna(),
        test_size=test_size,
        group_id=GROUP_ID,
        random_state=random_state,
        save_datasets=save_train_test_datasets,
    )
    # train_df = pd.read_csv("data/preprocessed/train.csv")
    # test_df = pd.read_csv("data/preprocessed/test.csv")
    X_test, y_test = split_X_y(test_df, id_cols, target)

    # train
    for metric in metrics:
        print(f"Training for {metric}")
        hp_filename = f"models/{metric}/hp_searches.json"
        for optimizer in optimizers:
            print(f"Optimizer: {optimizer}")
            searches = cv_train(
                train_df,
                test_df,
                n_splits=5,
                models=models["regression"],
                scoring_metric=metric,
                random_state=random_state,
                id_cols=id_cols,
                group_id=GROUP_ID,
                cv=optimizer,
                save_best_estimators=False,
            )

            save_info_from_searches(
                searches,
                hp_filename,
                metric,
                optimizer,
                train_df,
                test_df,
                id_cols,
                target,
                overwrite=True,
            )
