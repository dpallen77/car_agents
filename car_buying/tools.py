from pathlib import Path
import pandas as pd
from car_selling.score import comp_stats

CSV_PATH = Path(__file__).resolve().parent.parent / "cars_dataset.csv"

CONDITION_ORDER = ["Poor", "Fair", "Good", "Very Good", "Excellent"]

COLS = ["listing_id", "make", "model", "model_year", "odometer_km",
        "listing_price", "listing_status", "body_type", "fuel_type",
        "country", "vehicle_condition", "accident_count"]

cars = pd.read_csv(CSV_PATH, usecols=COLS)
# Only drop rows missing the fields we filter or sort on
cars = cars.dropna(subset=["make", "model", "model_year",
                           "odometer_km", "listing_price"])
cars["mileage"] = (cars["odometer_km"] * 0.621371).round()


def search_listings(make=None, model=None, body_type=None,
                    max_price=None, max_mileage=None,
                    min_year=None, sort_by="Price", limit=10, 
                    country=None):
    results = cars[cars["listing_status"] == "Available"]
    if make:
        results = results[results["make"] == make]
    if model:
        results = results[results["model"] == model]
    if body_type:
        results = results[results["body_type"] == body_type]
    if max_price:
        results = results[results["listing_price"] <= max_price]
    if max_mileage:
        results = results[results["mileage"] <= max_mileage]
    if min_year:
        results = results[results["model_year"] >= min_year]
    if country:
        results = results[results["country"] == country]

    sort_options = {
        "price": ("listing_price", True),    # cheapest first
        "mileage": ("mileage", True),        # lowest mileage first
        "year": ("model_year", False),       # newest first
    }
    column, ascending = sort_options.get(sort_by, ("listing_price", True))

    total = len(results)
    results = results.sort_values(column, ascending=ascending).head(limit)
    return {
        "total_matches": total,
        "showing": len(results),
        "listings": results[["listing_id", "make", "model", "model_year",
                             "mileage", "listing_price", "body_type",
                             "vehicle_condition", "accident_count"]].to_dict("records"),
    }

def _available(make=None, model=None, body_type=None,
               max_price=None, max_mileage=None, min_year=None,
               min_condition=None, max_accidents=None):
    results = cars[cars["listing_status"] == "Available"]
    if make:
        results = results[results["make"] == make]
    if model:
        results = results[results["model"] == model]
    if body_type:
        results = results[results["body_type"] == body_type]
    if max_price:
        results = results[results["listing_price"] <= max_price]
    if max_mileage:
        results = results[results["mileage"] <= max_mileage]
    if min_year:
        results = results[results["model_year"] >= min_year]
    if min_condition in CONDITION_ORDER:
        allowed = CONDITION_ORDER[CONDITION_ORDER.index(min_condition):]
        results = results[results["vehicle_condition"].isin(allowed)]
    if max_accidents is not None:
        results = results[results["accident_count"] <= max_accidents]
    return results


def rank_by_value(make=None, model=None, body_type=None, max_price=None,
                  max_mileage=None, min_year=None, min_condition=None,
                  max_accidents=None,top_n=10, sample_size=500):
    results = _available(make, model, body_type, max_price, max_mileage, min_year, min_condition, max_accidents)
    total = len(results)
    if total == 0:
        return {"total_matches": 0, "message": "No listings match these filters."}

    # Random sample, so we're not only looking at the cheapest cars
    sample = results.sample(min(total, sample_size), random_state=42)

    rows = []
    for _, car in sample.iterrows():
        s = comp_stats(car["make"], car["model"],
                       int(car["model_year"]), car["mileage"])
        if s["n"] < 10:   # skip cars without enough comps
            continue
        pct_below = (s["median"] - car["listing_price"]) / s["median"] * 100
        rows.append({
            "listing_id": car["listing_id"],
            "make": car["make"],
            "model": car["model"],
            "model_year": int(car["model_year"]),
            "mileage": int(car["mileage"]),
            "listing_price": round(float(car["listing_price"])),
            "median_price": round(float(s["median"])),
            "pct_below_median": round(float(pct_below), 1),
            "comps": s["n"],
            "confidence": s["confidence"],
            "condition": car["vehicle_condition"],
            "accidents": None if pd.isna(car["accident_count"]) else int(car["accident_count"]),
            "flag": "unusually far below market; verify" if pct_below > 40 else None,
        })

    rows.sort(key=lambda r: r["pct_below_median"], reverse=True)
    return {"total_matches": total, "sampled": len(sample),
            "scored": len(rows), "top": rows[:top_n]}

if __name__ == "__main__":
    print(cars["listing_status"].value_counts())
    print(cars["body_type"].value_counts())
    print(search_listings(body_type="SUV", max_price=20000,
                          max_mileage=60000, min_year=2018,
                          country="USA"))
    print(cars.groupby("country")["listing_price"].median().sort_values())