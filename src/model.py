"""Sales forecasting model: predict next week's units sold per product."""

import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.features import MODEL_FEATURES

MIN_TRAINING_ROWS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200


def train_and_predict(
    training: pd.DataFrame, latest_week: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """Train a RandomForestRegressor on weekly sales and forecast the upcoming week.

    The model learns the mapping (this week's features -> next week's units sold),
    is evaluated on a chronologically held-out test split, and is then applied to
    the most recent week of data to produce next week's forecast.

    Returns:
        metrics: evaluation metrics (MAE, RMSE, R²) plus training metadata.
        predictions: latest-week rows with predicted units and revenue,
            sorted best-sellers first.
    """
    if len(training) < MIN_TRAINING_ROWS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough sales history to train a model. "
                "Provide at least a few weeks of data across products."
            ),
        )

    # Chronological split: hold out the most recent weeks for evaluation.
    # A random split would leak future weeks into training and overstate
    # accuracy, since this is a forecasting problem.
    training = training.sort_values("date")
    split_index = int(len(training) * (1 - TEST_SIZE))
    train_set, test_set = training.iloc[:split_index], training.iloc[split_index:]

    X_train, y_train = train_set[MODEL_FEATURES], train_set["next_week_units_sold"]
    X_test, y_test = test_set[MODEL_FEATURES], test_set["next_week_units_sold"]

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_test, y_pred)), 2),
        "r2": round(float(r2_score(y_test, y_pred)), 3),
        "training_rows": len(train_set),
        "test_rows": len(test_set),
        "model_name": "RandomForestRegressor",
    }

    # Refit on the full history so the final forecast uses every week of data.
    model.fit(training[MODEL_FEATURES], training["next_week_units_sold"])

    predictions = latest_week.copy()
    predictions["predicted_next_week_units"] = model.predict(predictions[MODEL_FEATURES]).round(1)
    predictions["predicted_next_week_revenue"] = (
        predictions["predicted_next_week_units"] * predictions["price"]
    ).round(2)
    predictions = predictions.sort_values("predicted_next_week_units", ascending=False)

    return metrics, predictions
