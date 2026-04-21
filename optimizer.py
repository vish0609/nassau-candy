import numpy as np
import pandas as pd
from data_loader import FACTORY_COORDS, REGION_COORDS, haversine_distance, PRODUCT_FACTORY
from ml_model    import predict_lead_time


def simulate_factory_options(df, model, encoders, features, product, region, ship_mode):
   
    # Get average order characteristics for this product (for realistic predictions)
    product_rows = df[df["Product Name"] == product]
    avg_sales = product_rows["Sales"].mean()
    avg_units = product_rows["Units"].mean()
    avg_cost  = product_rows["Cost"].mean()
    current_factory = PRODUCT_FACTORY.get(product, "Unknown")

    # Decide which regions to simulate over
    regions = list(REGION_COORDS.keys()) if region == "All Regions" else [region]

    results = []
    for factory_name, factory_coords in FACTORY_COORDS.items():
        # Check this factory is known to the encoder (it was in training data)
        if factory_name not in encoders["Factory"].classes_:
            continue

        lead_times = []
        distances  = []
        for reg in regions:
            dist = haversine_distance(
                factory_coords["lat"], factory_coords["lng"],
                REGION_COORDS[reg]["lat"], REGION_COORDS[reg]["lng"]
            )
            lt = predict_lead_time(
                model, encoders, features,
                product, factory_name, reg, ship_mode,
                dist, avg_sales, avg_units, avg_cost
            )
            lead_times.append(lt)
            distances.append(dist)

        results.append({
            "Factory":        factory_name,
            "Avg Lead Time":  round(np.mean(lead_times), 1),
            "Avg Distance":   round(np.mean(distances),  1),
            "Avg Profit":     round(product_rows["Gross Profit"].mean(), 2),
            "Is Current":     factory_name == current_factory,
        })

    # Sort by lead time (best = shortest first)
    results.sort(key=lambda x: x["Avg Lead Time"])
    return results


def generate_recommendations(df, model, encoders, features, top_n=10):
    """
    Loops through every product and finds the best alternative factory
    (the one that reduces average lead time the most vs. the current factory).

    Parameters:
        df      : cleaned data
        model   : trained ML model
        encoders, features : from ml_model.py
        top_n   : how many recommendations to return (default: 10)

    Returns:
        recommendations : list of dicts, sorted by % improvement (best first)
    """
    ship_mode = "Standard Class"  # Use standard class as the default for comparison
    recommendations = []

    for product, current_factory in PRODUCT_FACTORY.items():
        # Skip products not in the encoder (weren't in training data)
        if product not in encoders["Product Name"].classes_:
            continue
        if current_factory not in encoders["Factory"].classes_:
            continue

        # Simulate all factory options for this product across all regions
        options = simulate_factory_options(
            df, model, encoders, features, product, "All Regions", ship_mode
        )

        # Current factory's predicted lead time
        current_option = next((o for o in options if o["Is Current"]), None)
        if current_option is None:
            continue
        current_lt = current_option["Avg Lead Time"]

        # Find the best alternative factory (not the current one)
        alternatives = [o for o in options if not o["Is Current"]]
        if not alternatives:
            continue
        best_alt = alternatives[0]  # Already sorted by lead time, so first = best

        # Calculate improvement
        days_saved = current_lt - best_alt["Avg Lead Time"]
        pct_saved  = (days_saved / current_lt * 100) if current_lt > 0 else 0

        if days_saved > 0:
            avg_profit = df[df["Product Name"] == product]["Gross Profit"].mean()
            risk_level = (
                "High"   if avg_profit < 5  else
                "Medium" if avg_profit < 15 else
                "Low"
            )
            recommendations.append({
                "Product":             product,
                "Division":            df[df["Product Name"] == product]["Division"].iloc[0],
                "Current Factory":     current_factory,
                "Recommended Factory": best_alt["Factory"],
                "Current LT (days)":   current_lt,
                "New LT (days)":       best_alt["Avg Lead Time"],
                "Days Saved":          round(days_saved, 1),
                "Improvement (%)":     round(pct_saved,  1),
                "Avg Profit ($)":      round(avg_profit, 2),
                "Risk Level":          risk_level,
            })

    # Sort by improvement percentage (best improvements first)
    recommendations.sort(key=lambda x: x["Improvement (%)"], reverse=True)
    return recommendations[:top_n]


def get_risk_summary(recommendations):
    """
    Counts how many recommendations fall into each risk category.

    Returns:
        dict with counts for High, Medium, Low risk
    """
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for r in recommendations:
        counts[r["Risk Level"]] += 1
    return counts
