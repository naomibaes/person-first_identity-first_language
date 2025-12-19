import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- config ----------
UNWEIGHTED_CSV = "output_GoogleBooks/identity_first_proportion_overall_unweighted.csv"
WEIGHTED_CSV   = "output_GoogleBooks/identity_first_proportion_overall_weighted.csv"

OUTPUT_DIR = Path("output_GoogleBooks/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUTPUT_DIR / "IF_overall_weighted_vs_unweighted_panels.png"

# ---------- load ----------
df_u = pd.read_csv(UNWEIGHTED_CSV)
df_w = pd.read_csv(WEIGHTED_CSV)

df_u["Year"] = df_u["Year"].astype(int)
df_w["Year"] = df_w["Year"].astype(int)

df_u = df_u.sort_values(["number", "Year"])
df_w = df_w.sort_values(["number", "Year"])

# ---------- plot ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# LEFT: weighted (singular+plural)
ax = axes[0]
for number, sub in df_w.groupby("number"):
    ax.plot(sub["Year"], sub["IF_proportion"], label=number.capitalize(), linestyle="-")
ax.set_title("Weighted overall")
ax.set_xlabel("Year")
ax.set_ylabel("IF index (IF / (IF + PF))")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend(title="Number", frameon=False)

# RIGHT: unweighted (singular+plural)
ax = axes[1]
for number, sub in df_u.groupby("number"):
    ax.plot(sub["Year"], sub["IF_proportion"], label=number.capitalize(), linestyle="-")
ax.set_title("Unweighted overall")
ax.set_xlabel("Year")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend(title="Number", frameon=False)

plt.suptitle("Overall identity-first proportion over time (Google Books)", y=1.02)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print(f"Saved overall plot to: {OUT_PATH.resolve()}")
