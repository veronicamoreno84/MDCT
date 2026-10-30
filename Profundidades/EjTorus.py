
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Import functions from Depth.py
from Depth import compute_distance_matrix, compute_depth, select_modes_by_dedup


# 1. Geodesic distance on the torus S1 x S1
def torus_distance(p1, p2):
    d_phi = abs(p1[0] - p2[0])
    d_phi = min(d_phi, 2 * np.pi - d_phi)

    d_psi = abs(p1[1] - p2[1])
    d_psi = min(d_psi, 2 * np.pi - d_psi)

    return np.sqrt(d_phi**2 + d_psi**2)


# --- Data Generation in S1 x S1 ---
np.random.seed(42)
n_per_cluster = 150

# True cluster centers
true_centers = np.array([[-2.8, -2.8], [1.5, 1.5], [-1.0, 2.0]])

c1 = np.random.normal(loc=true_centers[0], scale=0.4, size=(n_per_cluster, 2))
c2 = np.random.normal(loc=true_centers[1], scale=0.6, size=(n_per_cluster, 2))
c3 = np.random.normal(loc=true_centers[2], scale=0.3, size=(n_per_cluster, 2))

X = np.vstack([c1, c2, c3])
X = (X + np.pi) % (2 * np.pi) - np.pi

# --- Calculations independent of Beta ---
D = compute_distance_matrix(X, torus_distance)

kde_func = gaussian_kde(X.T)
density = kde_func(X.T)

cant_modas = 3
merge_thresh = 0.8

# KDE modes (fixed across all subplots as density does not depend on beta)
kde_mode_idxs = select_modes_by_dedup(density, D, merge_thresh=merge_thresh)[
    :cant_modas
]

# Array of 4 Beta values to compare
betas = [0.01, 0.1, 0.2, 0.4]

# --- 2x2 Grid Plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
axes = axes.flatten()

for idx, beta in enumerate(betas):
    ax = axes[idx]

    # Calculate Local Depth for current Beta
    depths = compute_depth(X, torus_distance, beta=beta)

    # Calculate Depth Modes for current Beta
    depth_mode_idxs = select_modes_by_dedup(
        depths, D, merge_thresh=merge_thresh
    )[:cant_modas]

    # Sample Scatter Plot
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=depths,
        cmap="plasma",
        s=20,
        alpha=0.5,
        edgecolors="none",
        label="Sample",
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Local Depth", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # 1. True Centers (Black Crosses)
    ax.scatter(
        true_centers[:, 0],
        true_centers[:, 1],
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
        X[kde_mode_idxs, 0],
        X[kde_mode_idxs, 1],
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
        X[depth_mode_idxs, 0],
        X[depth_mode_idxs, 1],
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
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r"$\Phi$ (rad)", fontsize=12)
    ax.set_ylabel(r"$\Psi$ (rad)", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Subplot Legend
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

# Global Figure Title
fig.suptitle(
    r"Mode Comparison across Different $\beta$ Values on $S^1 \times S^1$",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout()
plt.show()
