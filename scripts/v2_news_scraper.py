import time
import re
import html
from urllib.parse import quote
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ============================================================
# V2 EXPANDED NEWS SCRAPER
# Goal:
#   - Collect more Bangladesh-focused stray animal news
#   - Keep headline + RSS description/snippet
#   - Preserve old raw data
#   - Save a NEW expanded raw dataset
#
# Output:
#   data/raw_news_v2_expanded.csv
# ============================================================

OUTPUT_FILE = "data/raw_news_v2_expanded.csv"
OLD_RAW_FILE = "data/raw_news.csv"

Path("data").mkdir(exist_ok=True)

# Bangladesh-focused Google News RSS
RSS_BASE = "https://news.google.com/rss/search?q={query}&hl=en&gl=BD&ceid=BD:en"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    )
}

# ------------------------------------------------------------
# 1. Search queries
# ------------------------------------------------------------

CORE_QUERIES = [
    # General stray animal welfare
    '"stray dog" Bangladesh',
    '"stray dogs" Bangladesh',
    '"street dog" Bangladesh',
    '"street dogs" Bangladesh',
    '"free roaming dog" Bangladesh',
    '"free-roaming dog" Bangladesh',
    '"stray cat" Bangladesh',
    '"street cat" Bangladesh',
    '"animal welfare" Bangladesh',
    '"animal cruelty" Bangladesh',
    '"animal abuse" Bangladesh',

    # Abuse / killing / poisoning
    '"dog killed" Bangladesh',
    '"dogs killed" Bangladesh',
    '"cat killed" Bangladesh',
    '"dog poisoning" Bangladesh',
    '"dogs poisoned" Bangladesh',
    '"animal poisoning" Bangladesh',
    '"dog abuse" Bangladesh',
    '"dog cruelty" Bangladesh',
    '"cat abuse" Bangladesh',

    # Human-animal conflict
    '"dog bite" Bangladesh',
    '"dog attack" Bangladesh',
    '"stray dog attack" Bangladesh',
    '"rabies dog" Bangladesh',
    '"rabies stray dog" Bangladesh',
    '"animal bite" Bangladesh',

    # Rescue / treatment
    '"dog rescue" Bangladesh',
    '"cat rescue" Bangladesh',
    '"stray animal rescue" Bangladesh',
    '"animal shelter" Bangladesh',
    '"street animal treatment" Bangladesh',

    # Vaccination / sterilization / population management
    '"dog vaccination" Bangladesh',
    '"rabies vaccination dog" Bangladesh',
    '"stray dog vaccination" Bangladesh',
    '"dog sterilization" Bangladesh',
    '"stray dog sterilization" Bangladesh',
    '"dog population control" Bangladesh',

    # Policy / relocation / conflict
    '"stray dog relocation" Bangladesh',
    '"dog relocation" Bangladesh',
    '"animal rights" Bangladesh dog',
    '"stray dog protest" Bangladesh',
]

# Major Bangladesh locations to improve geographic coverage
LOCATIONS = [
    "Dhaka",
    "Chattogram",
    "Chittagong",
    "Gazipur",
    "Narayanganj",
    "Cumilla",
    "Rajshahi",
    "Khulna",
    "Sylhet",
    "Barishal",
    "Rangpur",
    "Mymensingh",
    "Cox's Bazar",
]

LOCATION_TEMPLATES = [
    '"stray dog" {location}',
    '"dog bite" {location}',
    '"dog attack" {location}',
    '"dog killed" {location}',
    '"dog rescue" {location}',
    '"rabies" dog {location}',
]

queries = list(CORE_QUERIES)

for location in LOCATIONS:
    for template in LOCATION_TEMPLATES:
        queries.append(template.format(location=location))

# remove duplicate query strings while keeping order
queries = list(dict.fromkeys(queries))

# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------

def clean_html_text(value):
    """Convert RSS HTML/snippet into plain text."""
    if value is None:
        return ""
    value = html.unescape(str(value))
    value = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def get_source(item):
    source_tag = item.find("source")
    if source_tag:
        return source_tag.get_text(" ", strip=True)
    return ""

def fetch_query(query):
    encoded = quote(query)
    url = RSS_BASE.format(query=encoded)

    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] {query} -> {e}")
        return []

    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")

    rows = []
    for item in items:
        title = item.title.get_text(" ", strip=True) if item.title else ""
        link = item.link.get_text(" ", strip=True) if item.link else ""
        pub_date = item.pubDate.get_text(" ", strip=True) if item.pubDate else ""
        description = clean_html_text(
            item.description.decode_contents() if item.description else ""
        )
        source = get_source(item)

        rows.append(
            {
                "keyword": query,
                "title": title,
                "description": description,
                "source": source,
                "link": link,
                "date": pub_date,
            }
        )

    return rows

# ------------------------------------------------------------
# 3. Scrape
# ------------------------------------------------------------

all_rows = []

print("===== V2 EXPANDED NEWS SCRAPING STARTED =====")
print("Total search queries:", len(queries))

for i, query in enumerate(queries, start=1):
    rows = fetch_query(query)
    all_rows.extend(rows)

    print(f"[{i}/{len(queries)}] {query} -> {len(rows)} items")

    # Small delay to avoid hammering the endpoint
    time.sleep(0.6)

new_df = pd.DataFrame(all_rows)

if new_df.empty:
    raise RuntimeError(
        "No new records were collected. Check internet connection or RSS access."
    )

print("\nNew scraped rows before deduplication:", len(new_df))

# ------------------------------------------------------------
# 4. Merge old V1 raw data safely
# ------------------------------------------------------------

if Path(OLD_RAW_FILE).exists():
    old_df = pd.read_csv(OLD_RAW_FILE)

    # Add description to old rows if it does not exist
    if "description" not in old_df.columns:
        old_df["description"] = ""

    # Ensure expected columns
    expected = ["keyword", "title", "description", "source", "link", "date"]

    for col in expected:
        if col not in old_df.columns:
            old_df[col] = ""

    old_df = old_df[expected]
    new_df = new_df[expected]

    combined = pd.concat([old_df, new_df], ignore_index=True)
else:
    combined = new_df.copy()

# ------------------------------------------------------------
# 5. Basic cleaning + deduplication
# ------------------------------------------------------------

for col in ["keyword", "title", "description", "source", "link", "date"]:
    combined[col] = combined[col].fillna("").astype(str).str.strip()

combined = combined[
    ~combined["title"].str.lower().isin(["", "nan", "no title"])
].copy()

# First remove duplicate links; then duplicate titles
combined = combined.drop_duplicates(subset=["link"], keep="first")
combined = combined.drop_duplicates(subset=["title"], keep="first")

combined["parsed_date"] = pd.to_datetime(
    combined["date"], errors="coerce", utc=True
)

# Useful combined text for later NLP
combined["text_for_nlp"] = (
    combined["title"].fillna("")
    + " "
    + combined["description"].fillna("")
).str.replace(r"\s+", " ", regex=True).str.strip()

# Stable sort
combined = combined.sort_values(
    by=["parsed_date", "title"],
    ascending=[False, True],
    na_position="last"
).reset_index(drop=True)

combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 6. Summary
# ------------------------------------------------------------

print("\n===== V2 EXPANDED SCRAPING COMPLETE =====")
print("Final unique rows:", len(combined))
print("Rows with usable RSS description:", (combined["description"].str.len() > 0).sum())
print("Rows with parsed date:", combined["parsed_date"].notna().sum())
print("Unique sources:", combined["source"].nunique())
print("Saved:", OUTPUT_FILE)

print("\nIMPORTANT:")
print("Do NOT overwrite data/raw_news.csv.")
print("The expanded dataset is saved separately as raw_news_v2_expanded.csv.")
