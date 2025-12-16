import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("throughput_log.csv", header=None, names=["y", "x", "value"])

# Force numeric (important given your earlier error)
df = df.apply(pd.to_numeric, errors="coerce").dropna()

heatmap = df.pivot(index="y", columns="x", values="value")

fig, ax = plt.subplots()

im = ax.imshow(heatmap.values, origin="lower", aspect="auto")
fig.colorbar(im, ax=ax, label="Throughput")

# Axis ticksa
ax.set_xticks(range(len(heatmap.columns)))
ax.set_yticks(range(len(heatmap.index)))
ax.set_xticklabels(heatmap.columns)
ax.set_yticklabels(heatmap.index)

ax.set_xlabel("Number of workers")
ax.set_ylabel("Batch size")
ax.set_title("Throughput scores with varying batch size and number of workers")

# ---- Cell annotations ----
for i in range(heatmap.shape[0]):       # rows (Y)
    for j in range(heatmap.shape[1]):   # cols (X)
        val = heatmap.iloc[i, j]
        ax.text(
            j, i,
            f"{val:.1f}",               # format here
            ha="center",
            va="center",
            color="black" if val > heatmap.values.mean() else "white",
            fontsize=9
        )

plt.show()
