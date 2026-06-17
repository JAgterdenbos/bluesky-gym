import matplotlib.pyplot as plt
import numpy as np

configs = ["No-HER", "HER", "No-HER, hdg", "HER, hdg"]
models = ["ET", "RF", "KNN", "HistGB", "MLP", "GB"]

R2 = {
    "ET":     [0.9995, 0.9994, 0.9994, 0.9996],
    "RF":     [0.9994, 0.9994, 0.9993, 0.9996],
    "KNN":    [0.9991, 0.9990, 0.9988, 0.9992],
    "HistGB": [0.9982, 0.9982, 0.9978, 0.9983],
    "MLP":    [0.9974, 0.9972, 0.9971, 0.9976],
    "GB":     [0.9955, 0.9955, 0.9953, 0.9958],
}
MAE = {
    "ET":     [0.9054, 0.9808, 0.7003, 0.5421],
    "RF":     [0.9563, 1.0158, 0.7743, 0.6075],
    "KNN":    [1.3579, 1.3928, 1.3006, 1.0579],
    "HistGB": [2.0782, 2.0843, 2.1625, 1.8009],
    "MLP":    [2.4687, 2.4638, 2.5257, 2.0018],
    "GB":     [3.7346, 3.7465, 3.7901, 3.5142],
}
RMSE = {
    "ET":     [1.9368, 2.0448, 2.1028, 1.6896],
    "RF":     [2.0413, 2.0830, 2.2724, 1.7852],
    "KNN":    [2.6207, 2.6824, 2.9667, 2.4622],
    "HistGB": [3.7043, 3.7041, 4.1086, 3.5250],
    "MLP":    [4.3887, 4.5810, 4.6572, 4.1615],
    "GB":     [5.7607, 5.7953, 5.9530, 5.4723],
}
sMAPE = {
    "ET":     [0.9736, 1.1001, 0.8471, 0.6510],
    "RF":     [1.0095, 1.1260, 0.9022, 0.7016],
    "KNN":    [1.4608, 1.5542, 1.4724, 1.1833],
    "HistGB": [2.1443, 2.2113, 2.2699, 1.8928],
    "MLP":    [2.8811, 2.7845, 2.8841, 2.2224],
    "GB":     [4.0091, 4.0038, 4.1023, 3.7072],
}

metrics = [
    ("$R^2$", R2, False),
    ("MAE (km)", MAE, False),
    ("RMSE (km)", RMSE, False),
    ("sMAPE (%)", sMAPE, False),
]

# Okabe-Ito Colorblind-Friendly Palette (Optimized for distinctness across all vision types)
# Blue, Orange, Bluish Green, Reddish Purple
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

x = np.arange(len(models))
width = 0.18  # Crisp bar width configuration

# Thesis/Publication-ready RC parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.titleweight": "bold",
})

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

for ax, (label, data, _) in zip(axes.flat, metrics):
    for i, cfg in enumerate(configs):
        vals = [data[m][i] for m in models]
        ax.bar(x + (i - 1.5) * width, vals, width,
               label=cfg, color=colors[i], edgecolor="none")
    
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title(label, pad=10)
    ax.grid(axis="y", linestyle=":", color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)
    
    if label == "$R^2$":
        ax.set_ylim(0.995, 1.0002)  # Buffer added to prevent upper spine clipping

# Extract legend details
handles, labels = axes[0, 0].get_legend_handles_labels()

# Clean up dynamic spacing
plt.tight_layout()

# Affix legend and super title tightly above the grid layout
fig.legend(handles, labels, loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 0.94), frameon=False, fontsize=11)

fig.suptitle("DTG sampler cross-validation: polar $\\hat{d}_\\text{s}$, top 6 regressors",
             y=1.00, fontsize=13.5, fontweight="bold")

# Micro-adjust spatial headroom for the legend and title area
plt.subplots_adjust(top=0.86, hspace=0.32, wspace=0.22)

# Save with publication-grade 300 DPI resolution
plt.savefig("dtg_regressor_comparison.png", dpi=300, bbox_inches="tight")