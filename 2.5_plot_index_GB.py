import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- config ----------
INPUT_CSV = "output_GoogleBooks/identity_first_proportion_by_target.csv"
OUTPUT_DIR = Path("output_GoogleBooks/plots") / "IF_proportion_by_target"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- load data ----------
df = pd.read_csv(INPUT_CSV)

# Ensure correct dtypes
df["Year"] = df["Year"].astype(int)

# Sort for cleaner plotting
df = df.sort_values(["target", "Year", "number"])

# ---------- plotting per target ----------
for target, sub in df.groupby("target"):
    sub = sub.sort_values(["Year", "number"])  # safety
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot a separate line for singular vs plural
    for number, sub_num in sub.groupby("number"):
        ax.plot(
            sub_num["Year"],
            sub_num["IF_proportion"],
            marker="o",
            linestyle="-",
            label=number.capitalize()
        )

    # Labels and title
    title = f"IF proportion over time: {target}"
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("IF index (IF / (IF + PF))")
    ax.set_ylim(0, 1)

    ax.legend(title="Number")
    ax.grid(True, alpha=0.3)

    # Ensure integer year ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Safe filename
    safe_target = (
        str(target)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("|", "_")
    )
    out_path = OUTPUT_DIR / f"IF_proportion_{safe_target}.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

print(f"Saved plots to: {OUTPUT_DIR.resolve()}")
