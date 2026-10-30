import pandas as pd
from MDDT_fast import ToroidalModeRandomForest
import numpy as np
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import cross_val_score
from imblearn.ensemble import BalancedRandomForestClassifier
from graficos import Mapa
from sklearn.model_selection import StratifiedKFold


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

# Definir la paleta global unificada
PALETA_Q3 = {
    'H': '#E41A1C',  # Rojo (Hélices)
    'E': '#377EB8',  # Azul (Láminas Beta)
    'C': '#4DAF4A'   # Verde (Coils / Bucles)
}

#  Crear la columna 'q3' (limpia espacios invisibles y asigna 'C' a los nulos)
datos_limpios['q3'] = datos_limpios['dssp'].astype(str).str.strip().map(q3_map).fillna('C')

#  Verificar el conteo de la nueva columna
print(datos_limpios['q3'].value_counts())

# Filtrar nulos en las columnas requeridas
datos_filtrados = datos_limpios.dropna(subset=['phi', 'psi', 'q3'])


## Ajusto un random forest 4D con TODOS los datos originales
X = datos_filtrados[['phi', 'psi']].values
y = datos_filtrados['q3'].values
np.unique(y)
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

# voy a equilibrar las clases
sampling_strategy = {"H": 1000, "E": 1000, "C": 1000}

model = BalancedRandomForestClassifier(
    n_estimators=25,
    sampling_strategy=sampling_strategy,
    replacement=True,
    max_depth=3,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
#model = RandomForestClassifier(n_estimators=100,max_samples=3000, max_depth=3, min_samples_split=5, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X_4D, y, cv=cv, scoring="accuracy")

mean_acc = scores.mean() # 0.7877
std_acc = scores.std() # 0.0011

# Reporte 
print(f"Accuracy por fold: {scores}")
print(f"Accuracy promedio: {mean_acc:.4f}")
print(f"Desviación estándar: {std_acc:.4f}")
print(f"\nResultado final: {mean_acc:.2%} ± {std_acc:.2%}")



model.fit(X_4D,y)

# hago grilla y predigo
resolution = 250
phi_lin = np.linspace(-np.pi, np.pi, resolution)
psi_lin = np.linspace(-np.pi, np.pi, resolution)
grid_phi, grid_psi = np.meshgrid(phi_lin, psi_lin)

# 2. Aplanar las matrices 2D a vectores 1D de 62,500 elementos
phi_flat = grid_phi.ravel()
psi_flat = grid_psi.ravel()

# 3. Transformación trigonométrica continua a 4D -> Matriz de (62500, 4)
X_grid = np.column_stack(
    [
        np.sin(phi_flat),  # sin(phi)
        np.cos(phi_flat),  # cos(phi)
        np.sin(psi_flat),  # sin(psi)
        np.cos(psi_flat),  # cos(psi)
    ]
)

pred_grid = model.predict(X_grid).reshape(resolution, resolution)
print(pred_grid)
print(pred_grid.shape)

method = 'Random Forest 4D'
Mapa(grid_phi,grid_psi,pred_grid,method,data=None)

## voy a ajustar RANDOM FOREST con mi método

## Tengo que cross validar el modelo
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_custom_folds = []
for train_idx, test_idx in skf.split(X_2D, y):
    # Data splits para espacio 2D
    X_train_2D, X_test_2D = X_2D[train_idx], X_2D[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Random Forest custom
    rf_toro = ToroidalModeRandomForest(n_estimators=25, samples_per_class=1000, 
        max_depth=3, min_samples_split=5, modes=5,mode_est = 'Depth')
    rf_toro.fit(X_train_2D, y_train)
    y_pred_custom = rf_toro.RF_predic(X_test_2D)
    
    acc_custom_folds.append(accuracy_score(y_test, y_pred_custom))


acc_custom_folds
mean_acc = np.mean(acc_custom_folds)
std_acc = np.std(acc_custom_folds)

# Reporte 

print(f"Accuracy promedio: {mean_acc:.4f}") # 0.7880
print(f"Desviación estándar: {std_acc:.4f}") # 0.0010
print(f"\nResultado final: {mean_acc:.2%} ± {std_acc:.2%}")

## VOY a fitear UN arbol
rf_toro = ToroidalModeRandomForest(n_estimators=25, samples_per_class=1000, 
    max_depth=3, min_samples_split=5, modes=5,mode_est = 'Depth')
rf_toro.fit(X_2D, y)

#  Aplanar las matrices 2D a vectores 1D de 62,500 elementos
phi_flat = grid_phi.ravel()
psi_flat = grid_psi.ravel()

# Transformación-> Matriz 
X_grid = np.column_stack(
    [phi_flat, psi_flat])
pred_grid = rf_toro.RF_predic(X_grid).reshape(resolution, resolution)
print(pred_grid)
print(pred_grid.shape)

method = 'Bagging MDCT'
Mapa(grid_phi,grid_psi,pred_grid,method,data=None)