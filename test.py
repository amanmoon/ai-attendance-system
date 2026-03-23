import pandas as pd
from rapidfuzz import process, fuzz

# Load files
real_df = pd.read_csv("real.csv")
test_df = pd.read_csv("final_attendance.csv")

# Rename columns for clarity
real_df.columns = ["Name", "Real_Status"]
test_df.columns = ["Name", "Test_Status"]

# -------- Step 1: Normalize names --------
def normalize(name):
    words = str(name).lower().split()
    words.sort()
    return " ".join(words)

real_df["norm"] = real_df["Name"].apply(normalize)
test_df["norm"] = test_df["Name"].apply(normalize)

# -------- Step 2: Fuzzy match --------
real_names = real_df["norm"].tolist()

def match_name(name):
    match, score, _ = process.extractOne(name, real_names, scorer=fuzz.token_sort_ratio)
    return match if score > 70 else None   # threshold adjustable

test_df["matched_norm"] = test_df["norm"].apply(match_name)

# -------- Step 3: Merge --------
merged = pd.merge(
    real_df,
    test_df,
    left_on="norm",
    right_on="matched_norm",
    how="left"
)

# -------- Step 4: Final output --------
result = merged[["Name_x", "Real_Status", "Test_Status"]]
result.columns = ["Name", "Real_File", "Test_File"]

# Fill missing test entries (student not in test file)
result["Test_File"] = result["Test_File"].fillna("Not Found")

# Save
result.to_csv("comparison.csv", index=False)