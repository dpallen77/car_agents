from car_selling.extract import extract_listing
from car_selling.score import score_deal

listing_text = """
2023 Kia Sportage EX AWD - $24,500 OBO
Selling my Sportage, only 18k miles. Bought it new but moving abroad
so need gone ASAP. Small fender bender last year, fixed at a body shop,
no CarFax handy. Cash only, come see it this weekend. Runs great!!
Text me, no lowballers.
"""

listing = extract_listing(listing_text)

required = ["make", "model", "year", "mileage", "price"]
missing = [key for key in required if listing.get(key) is None]

if missing:
    print("Can't score this listing, missing:", missing)
else:
    verdict = score_deal(
        listing["make"],
        listing["model"],
        listing["year"],
        listing["mileage"],
        listing["price"],
    )
    print(listing["summary"])
    print("Price check:", verdict)
    print("Red flags:")
    for flag in listing["red_flags"]:
        print(" -", flag)