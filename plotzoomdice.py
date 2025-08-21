import csv
import matplotlib.pyplot as plt

dice_csv = "path_to.csv"
output_png = "25epochsFixeddice_plot_withaug_ROI3.png"

# Read CSV
cases = []
scores = []

with open(dice_csv, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            dice = float(row["Dice_Region"])
            scores.append(dice)
            cases.append(row["Case"])
        except ValueError:
            continue

# Sort by score (keeping cases in sync)
sorted_pairs = sorted(zip(scores, cases), reverse=True)
sorted_scores, sorted_cases = zip(*sorted_pairs)

# Mean
mean_dice = sum(sorted_scores) / len(sorted_scores)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_scores)), sorted_scores, color='skyblue', edgecolor='black')
plt.axhline(mean_dice, color='green', linestyle='--', linewidth=2, label=f"Mean ROI Dice: {mean_dice:.3f}")
plt.xticks(range(len(sorted_scores)), sorted_cases, rotation=45, ha="right", fontsize=8)
plt.xlabel("Case")
plt.ylabel("Dice Score")
plt.title("Dice Scores from Dice_Region (11 cases)")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.close()

print(f"📈 Saved plot: {output_png}")
