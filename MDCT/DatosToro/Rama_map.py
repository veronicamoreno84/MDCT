import pandas as pd
from MDDT_fast import fit_tree,predict
import numpy as np
from sklearn.model_selection import StratifiedKFold,ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from graficos import Mapa,graficar_datos_toro
from sklearn.metrics import  accuracy_score,f1_score

# Leo el set de datos
df = pd.read_csv('dataset_aminoacidos_dssp.csv')
print(df.info())
print("--- Frecuencia de aminoácidos ---")
print(df['res_name'].value_counts(dropna=False))

#  Limpieza de espacios y normalización a mayúsculas
df['res_name_clean'] = df['res_name'].astype(str).str.strip().str.upper()
aminoacidos_excluir = ['GLY', 'PRO', 'G', 'P']

df_general = df[~df['res_name_clean'].isin(aminoacidos_excluir)].copy()

# Columnas críticas
cols_backbone = ['dssp', 'phi', 'psi']

# Elimina la fila si falta cualquiera de estas 3 columnas
datos_limpios = df_general.dropna(subset=cols_backbone)

# Convertir las columnas phi y psi a radianes directamente
datos_limpios[['phi', 'psi']] = np.radians(datos_limpios[['phi', 'psi']])

X = datos_limpios[['phi', 'psi']]  

print(X.info())

y=datos_limpios['dssp']
print(y)
np.unique(y)

# 1. Diccionario de mapeo de 8 estados a Q3 (H: Hélice, E: Lámina, C: Coil)
q3_map = {
    'H': 'H', 'G': 'H', 'I': 'H',  # Hélices (Alfa, 3-10, Pi)
    'E': 'E', 'B': 'E',            # Láminas y puentes Beta
    'T': 'C', 'S': 'C', ' ': 'C',  # Coils, Giros, Curvaturas y Bucles
    'C': 'C', '-': 'C', 'NA': 'C'
}


#  Crear la columna 'q3' (limpia espacios invisibles y asigna 'C' a los nulos)
datos_limpios['q3'] = datos_limpios['dssp'].astype(str).str.strip().map(q3_map).fillna('C')


# Filtrar nulos en las columnas requeridas
datos_filtrados = datos_limpios.dropna(subset=['phi', 'psi', 'q3'])

#  Tomar 1000 muestras por cada clase de q3 para graficar
np.random.seed(42)
datos_reducidos = datos_filtrados.groupby('q3', group_keys=False).sample(n=1000, random_state=42)

#  Verificar el conteo de la nueva columna
print(datos_reducidos['q3'].value_counts())

graficar_datos_toro(datos_reducidos)
 
# Clasificador

# Entrenar el árbol 
X = datos_reducidos[['phi', 'psi']].values
y = datos_reducidos['q3'].values
np.unique(y)
# Helix ($\alpha$-hélice)
#  Sheet (Lámina $\beta$)
#  Loop (Bucle / Rizo / Giro):

    
# 1. Preparar matrices (asegurando que sean arreglos NumPy para indexación)
X_data = np.array(X) if isinstance(X, pd.DataFrame) else X
y_data = np.array(y) if isinstance(y, pd.Series) else y


# 2. Preparar los datos en 2D (radianes)
X_2D = np.array(X) if isinstance(X, pd.DataFrame) else X
y_data = np.array(y) if isinstance(y, pd.Series) else y

# 3. Transformación trigonométrica continua a 4D
# Entrada: X_2D con columnas [phi_rad, psi_rad]
X_4D = np.column_stack([
    np.sin(X_2D[:, 0]),  # sin(phi)
    np.cos(X_2D[:, 0]),  # cos(phi)
    np.sin(X_2D[:, 1]),  # sin(psi)
    np.cos(X_2D[:, 1])   # cos(psi)
])


### Cross-valido en una grilla de hiperparámetros

# 1. Grilla de hiperparámetros
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'modes': [5]
}

grid = ParameterGrid(param_grid)

# 4. Configurar 5-Fold CV
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

resultados = []

print(f"=== Iniciando Evaluación Comparativa ({len(grid)} comb. x {k}-Fold CV) ===\n")

for i, params in enumerate(grid, 1):
    # Vectores para métricas de tu árbol custom (PLS)
    acc_custom_folds = []
    f1_custom_folds = []
    
    # Vectores para métricas del árbol tradicional (CART 4D)
    acc_cart_folds = []
    f1_cart_folds = []
    
    for train_idx, test_idx in skf.split(X_2D, y_data):
        # Data splits para espacio 2D
        X_train_2D, X_test_2D = X_2D[train_idx], X_2D[test_idx]
        # Data splits para espacio 4D
        X_train_4D, X_test_4D = X_4D[train_idx], X_4D[test_idx]
        
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        
        # MODELO CUSTOM 
        arbol_custom = fit_tree(
            X_train_2D, y_train,
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            modes=params['modes']
        )
        y_pred_custom = predict(arbol_custom, X_test_2D)
        
        acc_custom_folds.append(accuracy_score(y_test, y_pred_custom))
        f1_custom_folds.append(f1_score(y_test, y_pred_custom, average='macro'))
        
        # --- B) MODELO TRADICIONAL (CART con Scikit-Learn en 4D) ---
        cart_4D = DecisionTreeClassifier(
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            random_state=42
        )
        cart_4D.fit(X_train_4D, y_train)
        y_pred_cart = cart_4D.predict(X_test_4D)
        
        acc_cart_folds.append(accuracy_score(y_test, y_pred_cart))
        f1_cart_folds.append(f1_score(y_test, y_pred_cart, average='macro'))
    
    # Guardar métricas comparativas
    resultados.append({
        'max_depth': params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'modes': params['modes'],
        
        # Métricas Custom (PLS)
        'custom_acc_mean': np.mean(acc_custom_folds),
        'custom_acc_std': np.std(acc_custom_folds, ddof=1),
        'custom_acc_var': np.var(acc_custom_folds, ddof=1),
        'custom_f1_mean': np.mean(f1_custom_folds),
        
        # Métricas CART 4D
        'cart_acc_mean': np.mean(acc_cart_folds),
        'cart_acc_std': np.std(acc_cart_folds, ddof=1),
        'cart_acc_var': np.var(acc_cart_folds, ddof=1),
        'cart_f1_mean': np.mean(f1_cart_folds)
    })
    
    print(f"Comb {i}/{len(grid)} | depth={params['max_depth']}, min_split={params['min_samples_split']}, modes={params['modes']} -> "
          f"Custom Acc: {np.mean(acc_custom_folds):.4f} (±{np.std(acc_custom_folds, ddof=1):.4f}) | "
          f"CART 4D Acc: {np.mean(acc_cart_folds):.4f} (±{np.std(acc_cart_folds, ddof=1):.4f})")

# Convertir a DataFrame
df_res = pd.DataFrame(resultados)

# Ordenar por el mejor resultado de tu árbol custom
df_res = df_res.sort_values(by='custom_acc_mean', ascending=False).reset_index(drop=True)
print(df_res)
df_res.to_csv('Result_18Sep.csv')

# Imprimir el mejor hiperparámetro comparado
best = df_res.iloc[0]
print("\n" + "="*75)
print("=== RESUMEN DE LA MEJOR CONFIGURACIÓN ENCONTRADA ===")
print("="*75)
print(f"Parámetros: max_depth={int(best['max_depth'])}, min_samples_split={int(best['min_samples_split'])}, modes={int(best['modes'])}\n")
print(f"Custom PLS Tree  (2D) -> Acc: {best['custom_acc_mean']:.4f} (±{best['custom_acc_std']:.4f}) | Var: {best['custom_acc_var']:.6f} | F1: {best['custom_f1_mean']:.4f}")
print(f"Standard CART    (4D) -> Acc: {best['cart_acc_mean']:.4f} (±{best['cart_acc_std']:.4f}) | Var: {best['cart_acc_var']:.6f} | F1: {best['cart_f1_mean']:.4f}")
print("="*75)

## ahora fiteo un arbol con todos los datos juntos, y hago el mapa
## FITEO UN MDCT
arbol_general = fit_tree(
    X, y, 
    max_depth=3, 
    min_samples_split=5, 
    modes=5
)

# Predicción para datos  en una grilla
#  Generar grilla uniforme 2D sobre todo el toro [-π, π]
resolution = 250
phi_lin = np.linspace(-np.pi, np.pi, resolution)
psi_lin = np.linspace(-np.pi, np.pi, resolution)
grid_phi, grid_psi = np.meshgrid(phi_lin, psi_lin)

# Convertir la grilla en matriz de puntos (N x 2)
X_grid = np.column_stack([grid_phi.ravel(), grid_psi.ravel()])

#  Predecir clase con el árbol para toda la grilla
# (Asegúrate de haber entrenado 'arbol_general' previamente)
grid_preds = predict(arbol_general, X_grid)

method = 'MDCT'
Mapa(grid_phi, grid_psi, grid_preds, method, data=None)

method = 'CART'

# --- B) MODELO TRADICIONAL (CART con Scikit-Learn en 4D) ---
cart_4D = DecisionTreeClassifier(
    max_depth=3, 
    min_samples_split=5,
    random_state=42
)
cart_4D.fit(X_4D, y)


# Transformación trigonométrica de los 62.500 puntos a 4D

# Puntos aplanados de toda la grilla (62.500 puntos)
phi_flat = grid_phi.ravel()
psi_flat = grid_psi.ravel()

X_grid_4D = np.column_stack([
    np.sin(phi_flat),  # sin(phi) para los 62.500 puntos
    np.cos(phi_flat),  # cos(phi)
    np.sin(psi_flat),  # sin(psi)
    np.cos(psi_flat),  # cos(psi)
])

pred_grid_cart = cart_4D.predict(X_grid_4D)

# Verificación de forma
print("Forma del resultado de CART 4D:", pred_grid_cart.shape)  # Debe imprimir: (62500,)

Mapa(grid_phi, grid_psi, pred_grid_cart, method="CART 4D", data=datos_reducidos)





