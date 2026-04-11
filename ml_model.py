"""
ml_model.py
===========
This file trains three machine learning models to predict lead time.

Models used:
  1. Linear Regression   - simple baseline (like drawing a straight line through data)
  2. Random Forest       - many decision trees working together (more accurate)
  3. Gradient Boosting   - builds trees one-by-one, each fixing the last one's mistakes

For beginners:
  - "Training" a model means showing it your historical data so it learns patterns.
  - "Predicting" means using the trained model to guess values for new inputs.
  - "Evaluation" means checking how accurate the predictions are.
"""

import pandas as pd
import numpy as np

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score


def encode_features(df):
    """
    Machine learning models only understand numbers, not text.
    This function converts text columns (like "Atlantic", "First Class")
    into numbers using LabelEncoder.

    For example:
      "Atlantic"  → 0
      "Gulf"      → 1
      "Interior"  → 2
      "Pacific"   → 3

    Parameters:
        df : the cleaned DataFrame from data_loader.py

    Returns:
        df_enc   : DataFrame with new numeric columns added
        encoders : dictionary of encoders (needed to reverse the conversion later)
    """
    df_enc = df.copy()

    encoders = {}
    for col in ["Region", "Ship Mode", "Factory", "Product Name"]:
        le = LabelEncoder()
        df_enc[col + "_enc"] = le.fit_transform(df_enc[col])
        encoders[col] = le

    return df_enc, encoders


def train_models(df):
    """
    Splits the data into training and test sets, then trains all three models.

    Why split the data?
      - We train on 80% of the data (the model "studies" this).
      - We test on the remaining 20% (data the model has NEVER seen).
      - This tells us how well the model generalises to new orders.

    Parameters:
        df : the cleaned DataFrame from data_loader.py

    Returns:
        results  : dictionary with model names, trained models, and their scores
        encoders : the label encoders (needed for prediction later)
        features : list of column names used as inputs to the model
    """
    # Step 1: Encode text columns to numbers
    df_enc, encoders = encode_features(df)

    # Step 2: Choose which columns to use as model inputs (features)
    features = [
        "Region_enc",       # which region the customer is in
        "Ship Mode_enc",    # how the order is shipped
        "Factory_enc",      # which factory made the product
        "Product Name_enc", # which product
        "Distance_Miles",   # how far the factory is from the customer
        "Sales",            # order value
        "Units",            # number of units ordered
        "Cost",             # manufacturing cost
    ]

    # Step 3: Separate features (X) from the target we want to predict (y)
    X = df_enc[features]
    y = df_enc["Lead Time"]

    # Step 4: Split into training (80%) and test (20%) sets
    # random_state=42 means the split is the same every time (reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 5: Define the three models
    model_defs = {
        "Linear Regression":   LinearRegression(),
        "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in model_defs.items():
        # Train the model (it "learns" from X_train, y_train)
        model.fit(X_train, y_train)

        # Make predictions on the test set (data the model has NOT seen)
        y_pred = model.predict(X_test)

        # Evaluate the model using three metrics:
        mae  = mean_absolute_error(y_test, y_pred)   # Average error in days
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Penalises large errors more
        r2   = r2_score(y_test, y_pred)              # 1.0 = perfect, 0 = no better than average

        results[name] = {
            "model":  model,
            "MAE":    round(mae,  2),
            "RMSE":   round(rmse, 2),
            "R2":     round(r2,   4),
        }

        print(f"{name:25s} | MAE: {mae:.1f} days | RMSE: {rmse:.1f} | R²: {r2:.4f}")

    return results, encoders, features


def predict_lead_time(model, encoders, features, product, factory, region, ship_mode,
                       distance, sales=50, units=2, cost=10):
    """
    Uses a trained model to predict the lead time for a single scenario.

    Parameters:
        model     : a trained model object (e.g. the Random Forest)
        encoders  : the dictionary of label encoders from train_models()
        features  : the list of feature column names
        product   : product name (text)
        factory   : factory name (text)
        region    : region name (text)
        ship_mode : shipping method (text)
        distance  : distance in miles (number)
        sales, units, cost : order details (numbers)

    Returns:
        Predicted lead time in days (a number)
    """
    # Encode each text value into a number using the same encoder used during training
    region_enc   = encoders["Region"].transform([region])[0]
    ship_enc     = encoders["Ship Mode"].transform([ship_mode])[0]
    factory_enc  = encoders["Factory"].transform([factory])[0]
    product_enc  = encoders["Product Name"].transform([product])[0]

    # Build a one-row DataFrame matching the training feature order
    row = pd.DataFrame([[region_enc, ship_enc, factory_enc, product_enc,
                          distance, sales, units, cost]], columns=features)

    prediction = model.predict(row)[0]
    return round(prediction, 1)
