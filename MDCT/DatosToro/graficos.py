from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_PALETTE = {
    'H': '#E41A1C',  # Rojo (Hélices)
    'E': '#377EB8',  # Azul (Láminas Beta)
    'C': '#4DAF4A'   # Verde (Coils / Bucles)
}


def Mapa(grid_phi, grid_psi, grid_preds, method, data=None, palette=None):
    if palette is None:
        palette = DEFAULT_PALETTE

    class_to_int = {"H": 0, "E": 1, "C": 2}

    # Conversión vectorizada de clases a valores numéricos
    grid_array = np.array(grid_preds)
    grid_numeric = np.vectorize(class_to_int.get)(grid_array).reshape(grid_phi.shape)

    # Colormap para el fondo
    cmap_bg = ListedColormap([palette["H"], palette["E"], palette["C"]])

    plt.figure(figsize=(9, 8))

    # Regiones de decisión
    plt.pcolormesh(
        grid_phi,
        grid_psi,
        grid_numeric,
        cmap=cmap_bg,
        alpha=0.25,
        shading="nearest",
    )

    # Parches de leyenda para regiones
    legend_handles = [
        mpatches.Patch(color=palette[cls], alpha=0.3, label=f"Región {cls}")
        for cls in ["H", "E", "C"]
    ]

    # Overlay de datos reales
    if data is not None:
        for dssp_type, color in palette.items():
            subset = data[data["q3"] == dssp_type]
            scatter_handle = plt.scatter(
                subset["phi"],
                subset["psi"],
                c=color,
                s=12,
                alpha=0.6,
                edgecolors="none",
                label=f"Datos {dssp_type}",
            )
            legend_handles.append(scatter_handle)

    # Formato del toro [-π, π] y ejes
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)

    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    
    # --- TAMAÑO DE FUENTE EN LAS MARCAS DE LOS EJES (TICKS) ---
    plt.xticks(ticks, labels, fontsize=13)
    plt.yticks(ticks, labels, fontsize=13)

    plt.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    plt.axvline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)

    # --- TAMAÑO DE FUENTE EN ETIQUETAS DE LOS EJES (LABELS) ---
    plt.xlabel(r"$\phi$ (phi)", fontsize=16, labelpad=8)
    plt.ylabel(r"$\psi$ (psi)", fontsize=16, labelpad=8)

    # --- TAMAÑO DE FUENTE Y FORMATO EN EL TÍTULO ---
    plt.title(f"Ramachandran Map with {method} in the Torus", fontsize=18, fontweight="bold", pad=12)

    # Ajuste del tamaño de la leyenda
    plt.legend(
        handles=legend_handles,
        title="DSSP Class",
        title_fontsize=16,
        fontsize=16,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def graficar_datos_toro(datos):
    #  Dibujar puntos por categoría q3
    for dssp_type, color in DEFAULT_PALETTE.items():
        subset = datos[datos['q3'] == dssp_type]
        plt.scatter(
            subset['phi'], 
            subset['psi'], 
            c=color, 
            label=f"{dssp_type} ({len(subset)})", 
            s=12, 
            alpha=0.5, 
            edgecolors='none'
        )

    # Formato del toro [-π, π]
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)

    # Etiquetas en formato de radianes
    ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    labels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
    plt.xticks(ticks, labels, fontsize=11)
    plt.yticks(ticks, labels, fontsize=11)

    # Estética y referencias
    plt.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
    plt.xlabel(r'$\phi$ (phi)', fontsize=12)
    plt.ylabel(r'$\psi$ (psi)', fontsize=12)
    plt.title('Ramachandran Plot: DSSP Distribution on the Torus', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.3)

    # --- Leyenda FUERA del gráfico ---
    plt.legend(
        title='DSSP structure', 
        bbox_to_anchor=(1.02, 1), # Posiciona el cuadro a la derecha (x=1.02, y=1)
        loc='upper left',          # Ancla la esquina superior izquierda de la leyenda
        borderaxespad=0
    )


    # Guardar/Mostrar asegurando que la leyenda exterior no se corte
    plt.tight_layout()
    plt.show()
