import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from MDDT_fast import fit_tree, predict, ToroidalModeRandomForest

np.random.seed(42)

# --- 1. Generación de Datos (2 Modas Desalineadas, Clases Balanceadas 50/50) ---
centros = np.array([
    [-1.8, -0.8],   # Cluster 1 (Cuadrante III)
    [ 1.2,  1.6]    # Cluster 2 (Cuadrante I)
])

n_por_moda = 250
c1 = np.vstack([np.random.normal(loc=c, scale=0.4, size=(n_por_moda, 2)) for c in centros])

def distancia_toro(X, centros):
    dists = []
    for c in centros:
        dx = np.abs(X[:, 0] - c[0])
        dx = np.minimum(dx, 2*np.pi - dx)
        dy = np.abs(X[:, 1] - c[1])
        dy = np.minimum(dy, 2*np.pi - dy)
        dists.append(np.sqrt(dx**2 + dy**2))
    return np.min(np.column_stack(dists), axis=1)

c2_base = np.random.uniform(-np.pi, np.pi, size=(1200, 2))
mask = distancia_toro(c2_base, centros) > 0.8
c2_filtrado = c2_base[mask][:500]

X_ang = np.vstack([c1, c2_filtrado])
y = np.concatenate([np.zeros(len(c1)), np.ones(len(c2_filtrado))])
X_ang = (X_ang + np.pi) % (2 * np.pi) - np.pi

# Embedding a R^4
X_embed = np.column_stack([
    np.cos(X_ang[:, 0]), np.sin(X_ang[:, 0]),
    np.cos(X_ang[:, 1]), np.sin(X_ang[:, 1])
])

# --- 2. 5-Fold Cross Validation (Evaluando Únicamente Accuracy) ---
n_est = 25
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_accs = []
custom_accs = []

for train_idx, val_idx in skf.split(X_ang, y):
    X_tr_emb, X_va_emb = X_embed[train_idx], X_embed[val_idx]
    X_tr_ang, X_va_ang = X_ang[train_idx], X_ang[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]

    # Modelo 1: Random Forest Estándar (R^4)
    rf_fold = RandomForestClassifier(n_estimators=n_est, max_depth=3, min_samples_split=5, random_state=42)
    rf_fold.fit(X_tr_emb, y_tr)
    pred_rf = rf_fold.predict(X_va_emb)
    rf_accs.append(accuracy_score(y_va, pred_rf))

    # Modelo 2: Toroidal Mode RF Custom (S^1 x S^1)
    rf_toro_fold = ToroidalModeRandomForest(n_estimators=n_est, max_depth=3, min_samples_split=5, modes=5, mode_est='Kernel')
    rf_toro_fold.fit(X_tr_ang, y_tr)
    pred_custom = rf_toro_fold.RF_predic(X_va_ang)
    custom_accs.append(accuracy_score(y_va, pred_custom))

# --- 3. Tabla Resumen Unificada de Accuracy (Mean ± Std) ---
tabla_accuracy = pd.DataFrame({
    'Method': ['Random Forest ($\mathbb{R}^4$)', 'MDCT with Bagging ($S^1 \times S^1$)'],
    'Accuracy (Mean ± Std)': [
        f"{np.mean(rf_accs):.4f} ± {np.std(rf_accs):.4f}",
        f"{np.mean(custom_accs):.4f} ± {np.std(custom_accs):.4f}"
    ]
}).set_index('Method')

print("="*65)
print(" 5-FOLD CROSS VALIDATION: COMPARATIVA DE ACCURACY")
print("="*65)
print(tabla_accuracy.to_string())
print("="*65 + "\n")

from sklearn.metrics import accuracy_score

# --- 4. Reentrenamiento con el 100% de los Datos y Cálculo de Accuracy ---
rf_full = RandomForestClassifier(
    n_estimators=n_est, max_depth=3, min_samples_split=5, random_state=42
)
rf_full.fit(X_embed, y)
y_pred_rf = rf_full.predict(X_embed)
acc_rf_full = accuracy_score(y, y_pred_rf)

rf_toro_full = ToroidalModeRandomForest(
    n_estimators=n_est,
    max_depth=3,
    min_samples_split=5,
    modes=5,
    mode_est='Kernel',
)
rf_toro_full.fit(X_ang, y)
y_pred_custom = rf_toro_full.RF_predic(X_ang)
acc_custom_full = accuracy_score(y, y_pred_custom)

print(f"Accuracy RF Estándar (Full Dataset): {acc_rf_full:.2%}")
print(f"Accuracy Toroidal RF (Full Dataset): {acc_custom_full:.2%}")

# --- 5. Generación de Grilla para Evaluación de Fronteras ---
grid_steps = 200
t_grid = np.linspace(-np.pi, np.pi, grid_steps)
xx, yy = np.meshgrid(t_grid, t_grid)

X_grid_embed = np.column_stack([
    np.cos(xx.ravel()),
    np.sin(xx.ravel()),
    np.cos(yy.ravel()),
    np.sin(yy.ravel()),
])
Z_rf = rf_full.predict(X_grid_embed).reshape(xx.shape)

X_grid_ang = np.column_stack([xx.ravel(), yy.ravel()])
Z_custom = rf_toro_full.RF_predic(X_grid_ang).reshape(xx.shape)

# --- 6. Graficado Único (1 Fila x 3 Subplots) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

colores = ['crimson', 'royalblue']
ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
labels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
nombres_clases = ['Class 0', 'Class 1']

# Subplot 1: Muestra de datos completos
for i in range(2):
  m = y == i
  axes[0].scatter(
      X_ang[m, 0],
      X_ang[m, 1],
      color=colores[i],
      label=nombres_clases[i],
      s=20,
      alpha=0.8,
      edgecolors='k',
      linewidths=0.2,
  )
axes[0].set_title('Sample', fontsize=16, fontweight='bold')

# Subplot 2: RF Estándar ajustado
axes[1].contourf(
    xx, yy, Z_rf, levels=[-0.5, 0.5, 1.5], colors=colores, alpha=0.3
)
for i in range(2):
  m = y == i
  axes[1].scatter(
      X_ang[m, 0],
      X_ang[m, 1],
      color=colores[i],
      label=nombres_clases[i],
      s=20,
      alpha=0.8,
      edgecolors='k',
      linewidths=0.2,
  )
axes[1].set_title(
    f'Random Forest (Full Fitted)\nAccuracy: {acc_rf_full:.2%}',
    fontsize=16,
    fontweight='bold',
)

# Subplot 3: Toroidal Mode RF ajustado
axes[2].contourf(
    xx, yy, Z_custom, levels=[-0.5, 0.5, 1.5], colors=colores, alpha=0.3
)
for i in range(2):
  m = y == i
  axes[2].scatter(
      X_ang[m, 0],
      X_ang[m, 1],
      color=colores[i],
      label=nombres_clases[i],
      s=20,
      alpha=0.8,
      edgecolors='k',
      linewidths=0.2,
  )
axes[2].set_title(
    f'MDCT with Bagging (Full Fitted)\nAccuracy: {acc_custom_full:.2%}',
    fontsize=16,
    fontweight='bold',
)

# Formato visual unificado
for ax in axes:
  ax.set_xticks(ticks)
  ax.set_xticklabels(labels)
  ax.set_yticks(ticks)
  ax.set_yticklabels(labels)
  ax.set_xlim(-np.pi, np.pi)
  ax.set_ylim(-np.pi, np.pi)
  ax.set_xlabel(r'$\phi$', fontsize=11)
  ax.set_ylabel(r'$\psi$', fontsize=11)
  ax.set_aspect('equal')
  ax.grid(True, linestyle='--', alpha=0.5)
  ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()
