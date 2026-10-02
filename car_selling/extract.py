import json
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You extract structured data from used car listings.
Respond with ONLY a JSON object, no explanation and no markdown fences.
Use exactly these keys:
- make (string)
- model (string)
- year (integer)
- mileage (integer, in miles)
- price (integer, in US dollars)
- title_status (string: "clean", "salvage", "rebuilt", or "unknown")
- owners (integer or null)
- accident_history (string describing it, or null)
- red_flags (list of short strings; empty list if none)
- summary (one sentence)
Rules for red_flags:
- Only list concerns a used-car buyer should verify or be cautious about.
- Look for: missing vehicle history report, cash-only or unusual payment
  demands, pressure tactics or artificial urgency, vague accident
  descriptions, unverified mileage claims, salvage/rebuilt titles,
  prices far below market, and refusal to allow inspections.
- Seller urgency is a pressure tactic, not a sign the buyer has less
  leverage. Never describe it as reducing the buyer's negotiating power.
- Quote the seller's facts faithfully. Do not upgrade "fender bender"
  to "accident" or add details not in the listing.
- Only use explicitly stated values. Do not infer owners, title status,
  or anything else that isn't written in the listing.
If a value is not stated in the listing, use null. Never guess."""

def extract_listing(text):
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = message.content[0].text
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

if __name__ == "__main__":
    listing_text = """
    2023 Kia Sportage EX AWD - $24,500 OBO
    Selling my Sportage, only 18k miles. Bought it new but moving abroad
    so need gone ASAP. Small fender bender last year, fixed at a body shop,
    no CarFax handy. Cash only, come see it this weekend. Runs great!!
    Text me, no lowballers.
    """
    result = extract_listing(listing_text)
    print(json.dumps(result, indent=2))