import pandas as pd
import re
from pathlib import Path

INPUT_FILE = "data/v2_master_dataset.csv"
OUTPUT_FILE = "data/v2_label_validation.csv"
Path("data").mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)
required = ["article_id", "title", "source"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

def has(text, words):
    text = str(text).lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words)

def suggest_label(title):
    t = str(title).lower()
    if has(t, ["poison", "poisoned", "poisoning", "toxic"]): return "Poisoning"
    if has(t, ["abuse", "abused", "cruelty", "torture", "tortured", "beaten"]): return "Abuse"
    if has(t, ["bite", "bitten", "attack", "attacked", "aggressive", "injured", "injury"]): return "Human-Animal Conflict"
    if has(t, ["rabies", "disease", "infection", "infected", "vaccination", "vaccine", "vaccinated"]): return "Disease"
    if has(t, ["rescue", "rescued", "saved", "adoption", "adopted", "shelter"]): return "Rescue"
    if has(t, ["kill", "killed", "killing", "dead", "death"]): return "Killing"
    if has(t, ["relocate", "relocation", "protest", "conflict", "rights"]): return "Conflict"
    return "Other"

out = df[["article_id", "title", "source"]].copy()
out["suggested_label"] = out["title"].apply(suggest_label)
out["needs_review"] = (out["suggested_label"] == "Other").map({True:"YES", False:"NO"})
out["final_label"] = ""
out["review_note"] = ""
out["_priority"] = out["needs_review"].map({"YES":0, "NO":1})
out = out.sort_values(["_priority","article_id"]).drop(columns="_priority")
out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("===== LABEL VALIDATION FILE CREATED =====")
print("Total records:", len(out))
print(out["suggested_label"].value_counts())
print("Priority manual-review records:", (out["needs_review"]=="YES").sum())
print("Saved:", OUTPUT_FILE)
print("Allowed final_label values: Abuse | Killing | Poisoning | Disease | Rescue | Human-Animal Conflict | Conflict | Other")
