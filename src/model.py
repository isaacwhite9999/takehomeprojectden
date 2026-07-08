"""Train a regression model to predict next week's units sold per product."""

import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.features import MODEL_FEATURES


def train_and_predict(training: pd.DataFrame, latest_week: pd.DataFrame):
    """Train a RandomForest on weekly data and predict the upcoming week.

    Returns (metrics_dict, predictions_df).
    """
    if len(training) < 10:
        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough sales history to train a model. "
                "Provide at least a few weeks of data across products."
            ),
        )

    X = training[MODEL_FEATURES]
    y = training["next_week_units_sold"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "r2": round(float(r2_score(y_test, y_pred)), 3),
        "training_rows": len(training),
        "model_name": "RandomForestRegressor",
    }

    predictions = latest_week.copy()
    predictions["predicted_next_week_units"] = model.predict(predictions[MODEL_FEATURES]).round(1)
    predictions["predicted_next_week_revenue"] = (
        predictions["predicted_next_week_units"] * predictions["price"]
    ).round(2)
    predictions = predictions.sort_values("predicted_next_week_units", ascending=False)

    return metrics, predictions
