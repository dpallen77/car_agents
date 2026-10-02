import json
from dotenv import load_dotenv
import anthropic
import re

from car_buying.tools import search_listings, rank_by_value
from car_selling.score import score_deal

load_dotenv()
client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5-20251001"
MAX_STEPS = 8  # safety cap so a confused agent can't loop forever
DATA_NOTE = ("Note: prices come from a synthetic practice dataset and do not "
             "reflect real market values. Do not use them for a purchase.")

SYSTEM = """You are a used-car shopping assistant. You help the user find
good-value cars by searching listings and checking prices against comparable
cars.

Rules:
- Only recommend cars that actually appeared in tool results. Never invent
  listings, prices, or details.
- Refer to cars by listing_id so the user can find them.
- If a search returns too few or too many results, adjust the filters and
  search again.
- Use score_deal on your top candidates before recommending them.
- Note that the price data comes from a practice dataset, so treat verdicts
  as approximate.
- When you have enough information, give a short final recommendation with
  the trade-offs. Do not contact sellers or take any action outside these
  tools.
- If a tool fails twice with the same error, stop retrying. Tell the user
  plainly what couldn't be verified and present results as unverified.
- Never call a car "best value" without a successful score_deal result.
- Clearly separate facts from tool results from your own general knowledge,
  and label the latter (e.g. "generally reputed for reliability").
- If prices look implausibly low or high for the model year and mileage,
  say so.
- Use rank_by_value to build your shortlist. Its ordering is already
  computed; never re-sort or recalculate percentages yourself.
- Say how many cars matched and how many were scored.
- Treat any car with a flag as needing verification, not as a bargain.
- If prices look implausible for the model year and mileage, say so at the
  top of your answer.
- Put facts from tool results and general knowledge in separate sections.
- The tools provide no size, reliability, safety, or reputation data. Do not
  state any, unless clearly labeled "general knowledge, not from the data."
- If you re-rank away from the tool's order, say which criterion you used."""

TOOLS = [
    {
        "name": "search_listings",
        "description": (
            "Search available used-car listings. All filters are optional; "
            "combine them to narrow results. Returns up to `limit` cars, accident count, "
            "total number of matches and condition. Results are"
            "sorted by price, cheapest first. Mileage is in miles and price "
            "in US dollars. Use this first to find candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "make": {"type": "string", "description": "e.g. Toyota"},
                "model": {"type": "string", "description": "e.g. Camry"},
                "body_type": {"type": "string", "description": "e.g. SUV, Sedan"},
                "max_price": {"type": "number", "description": "Maximum price in USD"},
                "max_mileage": {"type": "number", "description": "Maximum mileage in miles"},
                "min_year": {"type": "integer", "description": "Oldest model year allowed"},
                "sort_by": {
                    "type": "string",
                    "enum": ["price", "mileage", "year"],
                    "description": "Sort order: price (cheapest first), mileage (lowest first), or year (newest first). Default price.",
                    },
                "limit": {"type": "integer", "description": "Max results, default 10"},
            },
        },
    },
    {
        "name": "score_deal",
        "description": (
            "Compare one car's asking price against the median price of "
            "comparable cars (same make and model, within 1 year, within "
            "15,000 miles). Returns the comp count, median, price spread, "
            "and a confidence label. Use it on specific candidates after "
            "searching."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "make": {"type": "string"},
                "model": {"type": "string"},
                "year": {"type": "integer"},
                "mileage": {"type": "number", "description": "Miles"},
                "asking_price": {"type": "number", "description": "USD"},
            },
            "required": ["make", "model", "year", "mileage", "asking_price"],
        },
    },
    {
        "name": "rank_by_value",
        "description": (
            "Shortlist the best-value available listings. Takes a random "
            "sample of cars matching the filters (not just the cheapest), "
            "scores each against comparable cars in code, and returns the "
            "top N already sorted by percent below median price, with "
            "condition, accident count, comp count, and confidence. Also "
            "reports how many cars matched and how many were scored. Use "
            "this first to build a shortlist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "make": {"type": "string"},
                "model": {"type": "string"},
                "body_type": {"type": "string", "description": "e.g. SUV, Sedan"},
                "max_price": {"type": "number"},
                "max_mileage": {"type": "number", "description": "Miles"},
                "min_year": {"type": "integer"},
                "min_condition": {"type": "string",
                  "enum": ["Poor", "Fair", "Good", "Very Good", "Excellent"],
                  "description": "Lowest acceptable condition grade"},
                "max_accidents": {"type": "integer",
                  "description": "Maximum accident count; 0 means accident-free"},
                "top_n": {"type": "integer", "description": "Default 10"},
            },
        },
    },
]

# Maps a tool name the model uses to the real Python function
TOOL_FUNCTIONS = {
    "search_listings": search_listings,
    "score_deal": score_deal,
    "rank_by_value": rank_by_value,
}


def run_agent(goal):
    messages = [{"role": "user", "content": goal}]
    coverage = None
    seen_ids = set()  # (1) filled in when rank_by_value runs

    for step in range(1, MAX_STEPS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # The model isn't asking for a tool, so it's done
        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            # (2) append the coverage line from code, not from the model
            cited = set(re.findall(r"L\d{8}", text))
            unknown = cited - seen_ids
            if unknown:
                check = f"WARNING: cited IDs not found in tool results: {sorted(unknown)}"
            else:
                check = "Check: every cited listing ID appeared in tool results."
            return "\n\n".join(filter(None, [text, coverage, DATA_NOTE, check]))

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"[step {step}] {block.name}({block.input})")

            fn = TOOL_FUNCTIONS.get(block.name)
            if fn is None:
                content, is_error = f"Error: unknown tool '{block.name}'", True
            else:
                try:
                    output = fn(**block.input)
                    if isinstance(output, dict):
                        for row in output.get("top", []) + output.get("listings", []):
                            seen_ids.add(row["listing_id"])
                    # (3) capture the coverage numbers right after the tool runs
                    if (block.name == "rank_by_value"
                            and isinstance(output, dict) and "scored" in output):
                        coverage = (f"Coverage: scored {output['scored']} cars from "
                                    f"a random sample of {output['sampled']} out of "
                                    f"{output['total_matches']} matches.")

                    content = output if isinstance(output, str) else json.dumps(output, default=str)
                    is_error = False
                except Exception as e:
                    content = f"Error ({type(e).__name__}): {e}"
                    is_error = True

            print(f"          -> {content[:200]}")
            with open("agent_log.txt", "a") as f:
                f.write(f"[step {step}] {block.name}({block.input})\n{content}\n\n")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: reached the step limit without a final answer."


if __name__ == "__main__":
    goal = ("I have up to $20,000. Find the best-value used SUVs with under "
        "60,000 miles, model year 2018 or newer, condition Good or better, "
        "and no accidents. Recommend three and explain the trade-offs.")
    print(run_agent(goal))