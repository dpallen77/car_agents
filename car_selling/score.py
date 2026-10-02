from pathlib import Path

import pandas as pd

# Load only the columns we need (much faster than all ~400)
cols = ["make", "model", "model_year", "odometer_km",
        "listing_price", "country", "vehicle_condition"]
CSV_PATH = Path(__file__).resolve().parent.parent / "cars_dataset.csv"
cars = pd.read_csv(CSV_PATH, usecols=cols)

# Drop rows missing anything we need
cars = cars.dropna()

# Convert km to miles so it matches how US listings are described
cars["mileage"] = cars["odometer_km"] * 0.621371

def comp_stats(make, model, year, mileage, country=None):
    comps = cars[
        (cars["make"] == make)
        & (cars["model"] == model)
        & (cars["model_year"].between(year - 1, year + 1))
        & (cars["mileage"].between(mileage - 15000, mileage + 15000))
    ]
    if country:
        comps = comps[comps["country"] == country]
    if len(comps) < 5:
        return {"n": len(comps)}
    prices = comps["listing_price"]
    median = prices.median()
    q1, q3 = prices.quantile(0.25), prices.quantile(0.75)
    spread = (q3 - q1) / median
    confidence = "high" if spread < 0.25 else "medium" if spread < 0.5 else "low"
    return {"n": len(comps), "median": median, "q1": q1, "q3": q3,
            "confidence": confidence}

def score_deal(make, model, year, mileage, asking_price, country=None):
    s = comp_stats(make, model, year, mileage, country)
    if s["n"] < 5:
        return f"Only {s['n']} comparable cars found, not enough to judge this price."
    diff = asking_price - s["median"]
    verdict = "above" if diff > 0 else "below"
    return (f"{s['n']} comps, median ${s['median']:,.0f} "
            f"(middle half of prices: ${s['q1']:,.0f} to ${s['q3']:,.0f}). "
            f"Asking ${asking_price:,} is ${abs(diff):,.0f} {verdict} median. "
            f"Confidence: {s['confidence']}.")

# Try a real combination from your data
if __name__ == "__main__":
    print(score_deal("Kia", "Sportage", 2024, 15000, 16000))