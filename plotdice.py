import csv
from collections import defaultdict
import matplotlib.pyplot as plt

# --- paths ---
dice_csv = "/home/erna/3Fixedwithaug25epochs/dice_scores.csv"
output_png = "Fixed_dice_plot_withaug3.png"

# --- read & detect column ---
with open(dice_csv, newline="") as f:
    reader = csv.DictReader(f)
    cols = [c.strip() for c in (reader.fieldnames or [])]

    # pick the first dice-like column present
    dice_col = next((c for c in ("Dice", "Dice_Region", "Dice_Global") if c in cols), None)
    if dice_col is None:
        raise KeyError(f"No dice column found. Columns present: {cols}")

    # collect scores (average per Case if multiple rows)
    per_case_scores = defaultdict(list)
    for row in reader:
        case = (row.get("Case") or "").strip()
        val_str = (row.get(dice_col) or "").strip()
        if not case or not val_str:
            continue
        try:
            per_case_scores[case].append(float(val_str))
        except ValueError:
            pass  # skip non-numeric

# --- aggregate (mean per case) ---
pairs = []
for case, vals in per_case_scores.items():
    if vals:
        pairs.append((case, sum(vals) / len(vals)))

# --- sort by score (desc) ---
pairs.sort(key=lambda x: x[1], reverse=True)
cases_sorted = [p[0] for p in pairs]
scores_sorted = [p[1] for p in pairs]

# overall mean (of per-case means)
mean_dice = sum(scores_sorted) / len(scores_sorted) if scores_sorted else 0.0

# --- plot ---
plt.figure(figsize=(14, 5.5))
plt.bar(range(len(scores_sorted)), scores_sorted, color="#87CEEB", edgecolor="white", linewidth=0.5)
plt.axhline(mean_dice, color="green", linestyle="--", linewidth=2, label=f"Mean Dice: {mean_dice:.3f}")

# x-axis as case index (sparse ticks)
n = len(scores_sorted)
if n > 0:
    step = max(1, n // 15)  # ~15 ticks max
    idx_ticks = list(range(0, n, step))
    plt.xticks(idx_ticks, [str(i) for i in idx_ticks], rotation=0)
else:
    plt.xticks([])

plt.ylim(0, 1)
plt.ylabel("Dice Score")
plt.xlabel("Case Index")
plt.title("Dice Scores per Case")
plt.grid(axis="y", linestyle=":", alpha=0.35)
plt.legend(loc="upper right", frameon=False)
plt.tight_layout()
plt.savefig(output_png, dpi=150)
print(f"📈 Saved plot: {output_png}  |  Cases: {len(scores_sorted)}  |  Using column: {dice_col}")
