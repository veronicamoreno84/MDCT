 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Import functions from Depth.py
from Depth import compute_distance_matrix, compute_depth, select_modes_by_dedup


# 1. Distance function (1D Euclidean)
def euclidean_distance_1d(a, b):
    return abs(a - b)


# --- Data Generation (1D Mixture) ---
np.random.seed(42)
n = 1500
n1, n2 = int(0.33 * n), int(0.33 * n)
n3 = n - n1 - n2

x1 = np.random.normal(-10, 1.5, n1)
x2 = np.random.normal(0, 3, n2)
x3 = np.random.normal(10, 3, n3)
X = np.concatenate([x1, x2, x3])

# Theoretical centers/modes of the components (-5 for Normal, 0 for Gamma(1,3), 7 for Normal)
true_centers = np.array([-10.0, 0.0, 10.0])

# --- Calculations independent of Beta ---
D = compute_distance_matrix(X, euclidean_distance_1d)

# Density Estimation (KDE)
kde_func = gaussian_kde(X)
density = kde_func(X)
true_densities = kde_func(true_centers)

cant_modas = 3
merge_thresh = 3.0  # Minimum separation distance for 1D modes

# KDE modes (fixed across all subplots)
kde_mode_idxs = select_modes_by_dedup(density, D, merge_thresh=merge_thresh)[
    :cant_modas
]

# Array of 4 Beta values to compare
betas = [0.01, 0.1, 0.2, 0.4]

# --- 2x2 Grid Plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, beta in enumerate(betas):
    ax = axes[idx]

    # Calculate Local Depth for current Beta
    depths = compute_depth(X, dist_fn=euclidean_distance_1d, beta=beta)

    # Calculate Depth Modes for current Beta
    depth_mode_idxs = select_modes_by_dedup(
        depths, D, merge_thresh=merge_thresh
    )[:cant_modas]

    # Sample Scatter Plot (X vs Density colored by Local Depth)
    scatter = ax.scatter(
        X,
        density,
        c=depths,
        cmap="plasma",
        s=15,
        alpha=0.6,
        edgecolors="none",
        label="Sample",
    )

    # Colorbar for Local Depth
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Local Depth $D(X)$", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # 1. True Centers (Black Crosses)
    ax.scatter(
        true_centers,
        true_densities,
        color="black",
        marker="X",
        s=140,
        edgecolor="white",
        linewidth=1.2,
        label="True Centers",
        zorder=8,
    )

    # 2. KDE Modes (Uniform Cyan Triangles)
    ax.scatter(
        X[kde_mode_idxs],
        density[kde_mode_idxs],
        color="cyan",
        marker="^",
        s=130,
        edgecolor="black",
        linewidth=1.2,
        label="KDE Modes",
        zorder=6,
    )

    # 3. Depth Modes (Uniform Gold Stars)
    ax.scatter(
        X[depth_mode_idxs],
        density[depth_mode_idxs],
        color="gold",
        marker="*",
        s=220,
        edgecolor="black",
        linewidth=1.2,
        label="Depth Modes",
        zorder=7,
    )

    # Subplot Configuration
    ax.set_title(rf"$\beta = {beta}$", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Sample $X$", fontsize=12)
    ax.set_ylabel(r"Density Estimation $\hat{f}(X)$", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Subplot Legend
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

# Global Figure Title
fig.suptitle(
    r"Mode Comparison across Different $\beta$ Values (1D Mixture)",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout()
plt.show()

