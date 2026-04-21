import pandas as pd
import numpy as np

# ── Factory information ──────────────────────────────────────────────────────
# Each factory has GPS coordinates (latitude, longitude).
# We use these to calculate distances from factory to customer regions.

FACTORY_COORDS = {
    "Lot's O' Nuts":    {"lat": 32.881893, "lng": -111.768036},
    "Wicked Choccy's":  {"lat": 32.076176, "lng":  -81.088371},
    "Sugar Shack":      {"lat": 48.119140, "lng":  -96.181150},
    "Secret Factory":   {"lat": 41.446333, "lng":  -90.565487},
    "The Other Factory":{"lat": 35.117500, "lng":  -89.971107},
}

# Region centre-points (approximate centres of each US region)
REGION_COORDS = {
    "Interior": {"lat": 39.50, "lng":  -98.35},
    "Atlantic":  {"lat": 37.00, "lng":  -76.00},
    "Gulf":      {"lat": 29.50, "lng":  -90.00},
    "Pacific":   {"lat": 37.70, "lng": -122.40},
}

# Which factory makes each product (given in the project document)
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}


def haversine_distance(lat1, lng1, lat2, lng2):
    """
    Calculate the straight-line distance (in miles) between two GPS points.
    Uses the Haversine formula — standard for calculating distances on Earth.
    
    Parameters:
        lat1, lng1 : GPS coordinates of the first location
        lat2, lng2 : GPS coordinates of the second location
    
    Returns:
        Distance in miles (a number)
    """
    R = 3958.8  # Earth's radius in miles
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi   = np.radians(lat2 - lat1)
    dlambda = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_data(filepath):
    """
    Loads the Nassau Candy CSV file and adds useful columns.
    
    Parameters:
        filepath : the path to your CSV file (e.g. "data/Nassau_Candy_Distributor.csv")
    
    Returns:
        df : a cleaned pandas DataFrame ready for analysis and modelling
    """
    # ── 1. Read the CSV ──────────────────────────────────────────────────────
    df = pd.read_csv(filepath)

    # ── 2. Parse dates (format is DD-MM-YYYY in this dataset) ───────────────
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True)

    # ── 3. Calculate Lead Time (how many days from order to shipment) ────────
    df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # ── 4. Add factory column based on product name ──────────────────────────
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)

    # ── 5. Calculate distance from factory to customer region ────────────────
    def get_distance(row):
        factory = row["Factory"]
        region  = row["Region"]
        # If we don't recognise the factory or region, return NaN (missing)
        if factory not in FACTORY_COORDS or region not in REGION_COORDS:
            return np.nan
        f = FACTORY_COORDS[factory]
        r = REGION_COORDS[region]
        return haversine_distance(f["lat"], f["lng"], r["lat"], r["lng"])

    df["Distance_Miles"] = df.apply(get_distance, axis=1)

    # ── 6. Remove rows where lead time is missing or negative ────────────────
    df = df.dropna(subset=["Lead Time", "Factory", "Distance_Miles"])
    df = df[df["Lead Time"] >= 0]

    return df


def get_summary_stats(df):
    """
    Returns a dictionary of key summary statistics for the dashboard.
    """
    return {
        "total_orders":    len(df),
        "avg_lead_time":   round(df["Lead Time"].mean(), 1),
        "total_products":  df["Product Name"].nunique(),
        "total_factories": df["Factory"].nunique(),
        "avg_profit":      round(df["Gross Profit"].mean(), 2),
        "total_sales":     round(df["Sales"].sum(), 2),
    }
