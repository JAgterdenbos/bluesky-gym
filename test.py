import matplotlib.pyplot as plt
import numpy as np

# Set style for a clean, academic look with white background
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#111111'
plt.rcParams['axes.labelcolor'] = '#111111'
plt.rcParams['xtick.color'] = '#111111'
plt.rcParams['ytick.color'] = '#111111'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Generate inverted U-curve data
x = np.linspace(0, 10, 100)
y = -0.25 * (x - 5)**2 + 7

fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)

# Plot the trade-off curve
ax.plot(x, y, color='#1d3557', linewidth=2.5)

# Highlight the optimal trade-off point
optimal_x = 5
optimal_y = 7
ax.plot(optimal_x, optimal_y, marker='*', color='#1d3557', markersize=14, zorder=5)
ax.text(optimal_x, optimal_y + 0.4, 'Optimal\nTrade-off Point', 
        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1d3557')

# Annotate Left Side (Low Entropy)
ax.text(1.1, 4.9, 'Inefficient\n(Rigid Routes)', ha='center', va='center', fontsize=8, color='#333333')
ax.text(2.0, 1.8, 'Low Entropy $\\rightarrow$\nRigid Noise Corridors', ha='center', va='center', fontsize=8.5)

# Annotate Right Side (High Entropy)
ax.text(9.1, 5.0, 'Stochastic\n(Wandering/Excessive\nEmissions)', ha='center', va='center', fontsize=8, color='#333333')
ax.text(8.0, 1.8, 'High Entropy $\\rightarrow$\nInefficient Wandering', ha='center', va='center', fontsize=8.5)


# Style Axes Labels
ax.set_xlabel('Route Dispersion\n(Entropy H)', fontsize=10, fontweight='bold', labelpad=8)
ax.set_ylabel('Policy Effectiveness\n(e.g., Noise Abatement)', fontsize=10, fontweight='bold', labelpad=8)
ax.set_title('Policy Optimisation Curve:\nEntropy vs. Effectiveness', fontsize=12, fontweight='bold', pad=15, color='#1d3557')

# Customize ticks
ax.set_xticks([0, 10])
ax.set_xticklabels(['Low', 'High'], fontsize=9)
ax.set_yticks([]) # Qualitative scale, so y-ticks can remain hidden or we can add low/high

# Ensure all 4 axes borders (spines) are visible as requested
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('#111111')

# Adjust limits
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(0, 9)

plt.tight_layout()

# Save the figure with a solid white background
plt.savefig('entropy_tradeoff_curve.png', bbox_inches='tight', facecolor='white', transparent=False)
print("Successfully generated image")



"""
20260504_140314 -> No-HER (with hdg) 
20260417_185011   -> No-HER (without hdg) 
20260504_173205 -> HER (with hdg) 
20260417_114027 -> HER (without hdg)


No-HER (without hdg) -> HER (without hdg)
No-HER (with hdg) -> HER (with hdg)
No-HER (without hdg) -> No-HER (with hdg)
HER (without hdg) -> HER (with hdg)


uv run path_planning/rta/testing/spatial_visitation_analysis.py --data_a path_planning/rta/data/spatial/no_HER/rta_data_deterministic.parquet --data_b path_planning/rta/data/spatial/HER/rta_data_deterministic.parquet --output experiments/results/spatial/visitation/No-HER_vs_HER_without_hdg --label_a "No-HER (without hdg)" --label_b "HER (without hdg)"

uv run path_planning/rta/testing/spatial_visitation_analysis.py --data_a path_planning/rta/data/spatial/no_HER_hdg/rta_data_deterministic.parquet --data_b path_planning/rta/data/spatial/HER/hdg_rta_data_deterministic.parquet --output experiments/results/spatial/visitation/No-HER_vs_HER_with_hdg --label_a "No-HER (with hdg)" --label_b "HER (with hdg)"

uv run path_planning/rta/testing/spatial_visitation_analysis.py --data_a path_planning/rta/data/spatial/no_HER/rta_data_deterministic.parquet --data_b path_planning/rta/data/spatial/no_HER_hdg/rta_data_deterministic.parquet --output experiments/results/spatial/visitation/No-HER_vs_No-HER-hdg --label_a "No-HER (without hdg)" --label_b "No-HER (with hdg)"

uv run path_planning/rta/testing/spatial_visitation_analysis.py --data_a path_planning/rta/data/spatial/HER/rta_data_deterministic.parquet --data_b path_planning/rta/data/spatial/HER/hdg_rta_data_deterministic.parquet --output experiments/results/spatial/visitation/HER_vs_HER-hdg --label_a "HER (without hdg)" --label_b "HER (with hdg)"
"""

"""
20260609_131943 -> No-HER (x, y, t)
20260610_092116 -> HER (x, y, t)
20260603_035602 -> No-HER (x, y, t, cos_phi, sin_phi)
20260603_092624 -> HER (x, y, t, cos_phi, sin_phi)

No-HER (without hdg) -> HER (without hdg)
No-HER (with hdg) -> HER (with hdg)
No-HER (without hdg) -> No-HER (with hdg)
HER (without hdg) -> HER (with hdg)

uv run path_planning/rta/testing/spatial_visitation_analysis.py --data_a path_planning/rta/data/temporal/no_HER/350k_training_rta_data.parquet --data_b path_planning/rta/data/temporal/HER/350k_training_rta_data.parquet --output experiments/results/temporal/visitation/No-HER_vs_HER_without_hdg --label_a "No-HER (without hdg)" --label_b "HER (without hdg)"

uv run path_planning/rta/testing/spatial_visitation_analysis.py --data_a path_planning/rta/data/temporal/no_HER_hdg/350k_training_rta_data.parquet --data_b path_planning/rta/data/temporal/HER_hdg/350k_training_rta_data.parquet --output experiments/results/temporal/visitation/No-HER_vs_HER_with_hdg --label_a "No-HER (with hdg)" --label_b "HER (with hdg)"

uv run path_planning/rta/testing/spatial_visitation_analysis.py --data_a path_planning/rta/data/temporal/no_HER/350k_training_rta_data.parquet --data_b path_planning/rta/data/temporal/no_HER_hdg/350k_training_rta_data.parquet --output experiments/results/temporal/visitation/No-HER_vs_No-HER-hdg --label_a "No-HER (without hdg)" --label_b "No-HER (with hdg)"

uv run path_planning/rta/testing/spatial_visitation_analysis.py --data_a path_planning/rta/data/temporal/HER/350k_training_rta_data.parquet --data_b path_planning/rta/data/temporal/HER_hdg/350k_training_rta_data.parquet --output experiments/results/temporal/visitation/HER_vs_HER-hdg --label_a "HER (without hdg)" --label_b "HER (with hdg)"
"""

"""
uv run path_planning/rta/testing/spatial_visitation_analysis.py --data_a path_planning/rta/data/temporal/no_HER/350k_training_rta_data.parquet --data_b path_planning/rta/data/spatial/no_HER/rta_data_deterministic.parquet --label_a "Spatio-Temporal" --label_b "Spatial" --output experiments/results/temporal/main/visitation_analysis
"""