import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

summary_path = "pca_summary.csv"
df = pd.read_csv(summary_path)

pa_counts = df["pa_k_95"].value_counts().sort_index()
k80_counts = df["k_80"].value_counts().sort_index()

all_k = sorted(set(pa_counts.index).union(set(k80_counts.index)))
pa = np.array([pa_counts.get(k, 0) for k in all_k])
k80 = np.array([k80_counts.get(k, 0) for k in all_k])

x = np.array(all_k, dtype=float)
bar_w = 0.35

plt.figure(figsize=(6, 4.5))
plt.bar(x - bar_w / 2, pa, width=bar_w, label="Parallel Analysis")
plt.bar(x + bar_w / 2, k80, width=bar_w, label="80% variance threshold")

plt.xticks(all_k, [str(k) for k in all_k])

plt.xlabel("Selected number of components (k)")
plt.ylabel("Number of ROIs")
plt.title(f"PCA dimension selection across {len(df)} ROIs")
plt.grid(True, axis="y")

ax = plt.gca()
ymax = int(np.ceil(max(pa.max(), k80.max())))
ax.set_yticks(np.arange(0, ymax + 1, 2))

if 3 in all_k:
    ax.axvline(3, linestyle="--")
    ax.text(3 + 0.05, ymax * 0.95, "k=3", va="top")

plt.legend()
plt.tight_layout()
plt.savefig("sum/pca_k_distribution.png", dpi=300)

print("PA mode:", df["pa_k_95"].mode().tolist(), "| k80 mode:", df["k_80"].mode().tolist())