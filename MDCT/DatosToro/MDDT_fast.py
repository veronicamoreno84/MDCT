import numpy as np


#  Matriz de distancias genérica
def compute_distance_matrix(X, dist_fn):
    n = len(X)
    distances = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            d_val = dist_fn(X[i], X[j])
            distances[i, j] = d_val
            distances[j, i] = d_val
    return distances


#  Función de Profundidad Local
def compute_depth(n, dis, beta=0.1):
    #dis = distance_matrix  

    selector = int(np.trunc(n * beta))
    selector = max(1, min(selector, n - 1))

    depths = np.zeros(n)
    for i in range(n):
        lamb = np.sort(dis[i, :])[selector]
        depths[i] = 1.0 / (1.0 + lamb)

    return depths

def make_bandwidth_grid(D,
                        n_grid=25,
                        mode='median',
                        scale_factors=(0.25, 4.0),
                        percentiles=(25, 75),
                        explicit=None):
    """
    Genera una rejilla de valores h a partir de la matriz de distancias D.

    Args:
        D: array (n,n) pairwise distances (simétrica, diagonal 0).
        n_grid: número de puntos en la rejilla (log-espaciada).
        mode: 'median'|'mean'|'std'|'percentile' indica h0 de referencia.
        scale_factors: (min_factor, max_factor) multiplicadores respecto a h0 para el rango.
        percentiles: tuple usado si mode=='percentile' -> h0 = median of D in [p_low,p_high] range.
        explicit: si no es None, usar esa lista/array explícito de h (ignora resto).
    Returns:
        hs: array shape (n_grid,) con valores positivos ordenados crecientes.
    """
    if explicit is not None:
        hs = np.array(explicit, dtype=float)
        hs = np.sort(hs)
        return hs

    # obtener valores únicos de la mitad superior (i<j) para robustez y velocidad
    triu = D[np.triu_indices_from(D, k=1)]
    triu = triu[~np.isnan(triu)]
    if triu.size == 0:
        raise ValueError("D no contiene distancias válidas (triángulo superior vacío).")

    # referencia h0
    if mode == 'median':
        h0 = np.median(triu)
    elif mode == 'mean':
        h0 = np.mean(triu)
    elif mode == 'std':
        h0 = np.std(triu)
    elif mode == 'percentile':
        p_low, p_high = percentiles
        # usar mediana entre percentiles p_low..p_high
        mask = (triu >= np.percentile(triu, p_low)) & (triu <= np.percentile(triu, p_high))
        if mask.sum() == 0:
            h0 = np.median(triu)
        else:
            h0 = np.median(triu[mask])
    else:
        raise ValueError("mode must be one of 'median','mean','std','percentile'")

    # si h0==0 (ej. datos idénticos) expandir por pequeño valor
    if h0 <= 0 or np.isclose(h0, 0.0):
        h0 = np.mean(triu) + 1e-8
        if h0 <= 0:
            h0 = 1.0

    # rango multiplicativo y rejilla log-espaciada
    min_factor, max_factor = scale_factors
    h_min = h0 * float(min_factor)
    h_max = h0 * float(max_factor)
    # evitar h_min<=0
    h_min = max(h_min, 1e-12)

    hs = np.logspace(np.log10(h_min), np.log10(h_max), num=n_grid)
    return hs

def thresh_by_percentile(D, candidates, p=10):
    pairs = []
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            pairs.append(D[candidates[i], candidates[j]])
    pairs = np.array(pairs)
    return np.percentile(pairs, p)

# --- seleccionar máximos discretos y eliminar cercanos ---
def select_modes_by_dedup(densities, D, top_k=None, merge_thresh=0.1):
    """
    densities: (n,)
    D: pairwise distance matrix (n,n)
    top_k: si no None, limitar a primeros top_k candidatos antes de dedupe
    merge_thresh: distancia umbral para considerar dos máximos equivalentes (en la escala de D)
    Retorna lista de índices de modos retenidos (ordenados por densidad decreciente).
    """
    idx_sorted = np.argsort(densities)[::-1]
    if top_k is not None:
        idx_sorted = idx_sorted[:top_k]
    kept = []
    if merge_thresh=='percentil10':
        merge_thresh=thresh_by_percentile(D, idx_sorted, p=10)
    for idx in idx_sorted:
        too_close = False
        for k in kept:
            if D[idx, k] <= merge_thresh:
                too_close = True
                break
        if not too_close:
            kept.append(idx)
    return kept

def gini(y):
        if len(y) == 0:
            return 0.0
        vals, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs**2)

def majority_class(y):
        if len(y) == 0:
            return None
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

def distancias_toro(X, period=2 * np.pi):
    """
    Calcula la matriz de distancias en el toro (N x N) para todo el dataset X.
    """
    X_arr = np.asarray(X)
    
    # Diferencia por pares usando broadcasting: forma (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = np.abs(X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :])
    
    # Condición de frontera periódica en el toro
    diff = np.minimum(diff, period - diff)
    
    # Distancia geodésica euclídea sobre el toro (N, N)
    return np.sqrt(np.sum(diff**2, axis=2))

# --- Kernels ---
def gaussian_kernel(u):
    return np.exp(-0.5 * (u**2))

class Nodo:
    def __init__(self,  mode=None, threshold=None, left=None, right=None, value=None):
        self.mode = mode
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # clase si es hoja


def _fast_gini_splits(left_counts, right_counts, n_left, n_right):
    """Calcula Gini para múltiples cortes en una sola operación vectorial."""
    p_left = left_counts / n_left[:, None]
    p_right = right_counts / n_right[:, None]
    
    gini_left = 1.0 - np.sum(p_left**2, axis=1)
    gini_right = 1.0 - np.sum(p_right**2, axis=1)
    
    return (n_left * gini_left + n_right * gini_right) / (n_left + n_right)

def best_split_fast(y_node, Dist_node, modes=3, merge_thresh='percentil10', kernel_func=gaussian_kernel,mode_est = 'Kernel'):
    n = len(y_node)
    classes, y_encoded = np.unique(y_node, return_inverse=True)
    num_classes = len(classes)
    
    best_loc_idx = None
    best_threshold = None
    best_impurity = float("inf")
    if mode_est == 'Kernel':
        hs = make_bandwidth_grid(Dist_node)
        for h in hs:
            # Sumas del kernel sobre la submatriz local
            densities = kernel_func(Dist_node / h).sum(axis=1)
            
            mode_idxs = select_modes_by_dedup(densities, Dist_node, merge_thresh=merge_thresh)[:modes]

            for loc_idx in mode_idxs:
                loc_idx = int(loc_idx)
                distancias = Dist_node[loc_idx]
    
                # Vectorización con Sumas Acumuladas
                sort_order = np.argsort(distancias)
                d_sorted = distancias[sort_order]
                y_sorted = y_encoded[sort_order]
    
                cum_counts = np.cumsum(np.eye(num_classes)[y_sorted], axis=0)
                total_counts = cum_counts[-1]
    
                split_mask = d_sorted[:-1] != d_sorted[1:]
                valid_indices = np.where(split_mask)[0]
    
                if len(valid_indices) == 0:
                    continue
    
                if len(valid_indices) > 200:
                    pos = np.linspace(0, len(valid_indices) - 1, 200, dtype=int)
                    valid_indices = valid_indices[pos]
    
                n_left = valid_indices + 1
                n_right = n - n_left
    
                left_counts = cum_counts[valid_indices]
                right_counts = total_counts - left_counts
    
                impurities = _fast_gini_splits(left_counts, right_counts, n_left, n_right)
    
                min_imp_idx = np.argmin(impurities)
                min_imp = impurities[min_imp_idx]
    
                if min_imp < best_impurity:
                    best_impurity = min_imp
                    best_loc_idx = loc_idx
                    split_i = valid_indices[min_imp_idx]
                    best_threshold = (d_sorted[split_i] + d_sorted[split_i + 1]) / 2.0
    if mode_est == 'Depth':
        
        betas = np.linspace(0.05,0.5,num=10)
        for beta in betas:
            profs = compute_depth(n, Dist_node, beta=beta)
            
            mode_idxs = select_modes_by_dedup(profs, Dist_node, merge_thresh=merge_thresh)[:modes]

            for loc_idx in mode_idxs:
                loc_idx = int(loc_idx)
                distancias = Dist_node[loc_idx]
    
                # Vectorización con Sumas Acumuladas
                sort_order = np.argsort(distancias)
                d_sorted = distancias[sort_order]
                y_sorted = y_encoded[sort_order]
    
                cum_counts = np.cumsum(np.eye(num_classes)[y_sorted], axis=0)
                total_counts = cum_counts[-1]
    
                split_mask = d_sorted[:-1] != d_sorted[1:]
                valid_indices = np.where(split_mask)[0]
    
                if len(valid_indices) == 0:
                    continue
    
                if len(valid_indices) > 200:
                    pos = np.linspace(0, len(valid_indices) - 1, 200, dtype=int)
                    valid_indices = valid_indices[pos]
    
                n_left = valid_indices + 1
                n_right = n - n_left
    
                left_counts = cum_counts[valid_indices]
                right_counts = total_counts - left_counts
    
                impurities = _fast_gini_splits(left_counts, right_counts, n_left, n_right)
    
                min_imp_idx = np.argmin(impurities)
                min_imp = impurities[min_imp_idx]
    
                if min_imp < best_impurity:
                    best_impurity = min_imp
                    best_loc_idx = loc_idx
                    split_i = valid_indices[min_imp_idx]
                    best_threshold = (d_sorted[split_i] + d_sorted[split_i + 1]) / 2.0

    return best_loc_idx, best_threshold

def fit_tree(X, y, max_depth=5, min_samples_split=2, modes=3, merge_thresh='percentil10',mode_est = 'Depth'):
    """Punto de entrada principal: calcula la matriz global una sola vez."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    
    # 1. Matriz global de distancias N x N (Calculada UNA sola vez)
    Dist_global = distancias_toro(X_arr)
    
    # 2. Iniciar recursión con todos los índices [0, 1, ..., N-1]
    initial_indices = np.arange(len(y_arr))
    
    return _build_tree_recursive(
        X_arr, y_arr, initial_indices, Dist_global, 
        depth=0, max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        modes=modes, merge_thresh=merge_thresh,
        mode_est = mode_est
    )


def _build_tree_recursive(X, y, indices, Dist_global, depth, max_depth, min_samples_split, modes, merge_thresh,mode_est):
    y_node = y[indices]
    
    # Criterio de parada
    if len(indices) < min_samples_split or depth >= max_depth or len(np.unique(y_node)) == 1:
        return Nodo(value=majority_class(y_node))

    # 3. Slicing O(1) de la submatriz de distancias para el nodo actual
    Dist_node = Dist_global[np.ix_(indices, indices)]

    # Evaluar divisiones pasando la submatriz precalculada
    best_loc_idx, threshold = best_split_fast(
        y_node=y_node, 
        Dist_node=Dist_node, 
        modes=modes, 
        merge_thresh=merge_thresh,
        mode_est = mode_est
    )

    if best_loc_idx is None:
        return Nodo(value=majority_class(y_node))

    # 4. Las distancias al modo están en la fila correspondiente de Dist_node
    distancias = Dist_node[best_loc_idx]
    
    left_mask = distancias <= threshold
    right_mask = ~left_mask

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return Nodo(value=majority_class(y_node))

    # Dividir índices sin duplicar memoria de las características
    left_indices = indices[left_mask]
    right_indices = indices[right_mask]

    # Mapear el índice local del modo al índice global para guardar sus coordenadas reales
    global_mode_idx = indices[best_loc_idx]
    mode_coordinates = X[global_mode_idx]

    # Llamadas recursivas
    left_child = _build_tree_recursive(
        X, y, left_indices, Dist_global, 
        depth + 1, max_depth, min_samples_split, modes, merge_thresh,mode_est = mode_est
    )
    right_child = _build_tree_recursive(
        X, y, right_indices, Dist_global, 
        depth + 1, max_depth, min_samples_split, modes, merge_thresh,mode_est = mode_est
    )

    return Nodo(mode=mode_coordinates, threshold=threshold, left=left_child, right=right_child)


def distances_to_mode(X, mode, period=2 * np.pi):
    """
    Calcula la distancia geodésica en el toro de N puntos a un punto de referencia (modo).
    
    Parámetros:
    -----------
    X : array-like de forma (N, d)
        Puntos (ej. ángulos phi, psi).
    mode : array-like de forma (d,)
        Coordenadas del modo.
    period : float, por defecto 2*pi (radianes). Usar 360.0 si los datos están en grados.
    
    Retorna:
    --------
    distancias : np.ndarray de forma (N,)
    """
    X_arr = np.asarray(X)
    mode_arr = np.asarray(mode)
    
    # 1. Diferencia absoluta por dimensión (N, d)
    diff = np.abs(X_arr - mode_arr)
    
    # 2. Condición periódica del toro: min(|Δ|, T - |Δ|)
    diff = np.minimum(diff, period - diff)
    
    # 3. Distancia euclídea geodésica en la superficie del toro
    return np.sqrt(np.sum(diff**2, axis=1))

def predict(root_node, X_new, dist_func_to_mode=distances_to_mode):
    """
    Predice la clase para un conjunto de datos X_new navegando el árbol por lotes.
    
    Parámetros:
    -----------
    root_node : Nodo
        La raíz del árbol entrenado.
    X_new : DataFrame o np.ndarray
        Puntos a clasificar (coordenadas angulares).
    dist_func_to_mode : función
        Función que calcula la distancia de un conjunto de puntos a un único punto/modo.
    """
    X_arr = np.asarray(X_new)
    n_samples = len(X_arr)
    
    # Vector donde guardaremos las predicciones finales
    predictions = np.empty(n_samples, dtype=object)
    
    def _traverse(node, indices):
        if len(indices) == 0:
            return
            
        # Si es un nodo hoja, asignamos la clase a todas las muestras que llegaron aquí
        if node.value is not None:
            predictions[indices] = node.value
            return
        
        # 1. Distancia solo para el subconjunto de puntos que llegó a este nodo
        distancias = dist_func_to_mode(X_arr[indices], node.mode)
        
        # 2. Separar índices según el umbral del nodo
        left_mask = distancias <= node.threshold
        right_mask = ~left_mask
        
        # 3. Descender recursivamente en las ramas
        if node.left is not None:
            _traverse(node.left, indices[left_mask])
        if node.right is not None:
            _traverse(node.right, indices[right_mask])

    # Iniciar el recorrido con todos los índices de X_new [0, 1, ..., N-1]
    _traverse(root_node, np.arange(n_samples))
    
    return predictions

class ToroidalModeRandomForest:
    def __init__(self, n_estimators=50, samples_per_class=None, max_depth=4, min_samples_split=10, modes=3,mode_est = 'Depth'):
        self.n_estimators = n_estimators
        self.samples_per_class = samples_per_class
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.modes = modes
        self.trees = []
        self.classes_ = None
        self.mode_est = mode_est

    def fit(self, X, y):
        self.trees = []
        self.classes_ = np.unique(y)


        for _ in range(self.n_estimators):
            # 1. Bootstrap: Muestreo aleatorio con reemplazo
            
            if self.samples_per_class == None:
                boot_idx = np.random.choice(len(X), size=len(X), replace=True)
            
            else:
                boot_idx_list = []
                for c in np.unique(y):
                    
                    # 1. Obtener los índices de los datos que pertenecen a la clase 'c'
                    idx_class = np.where(y == c)[0]
                
                    # 2. Tomar sample_per_class muestras aleatorias CON reemplazo de esa clase
                    sampled_idx = np.random.choice(idx_class, size=self.samples_per_class, replace=True)
                
                    # 3. Guardar los índices elegidos
                    boot_idx_list.append(sampled_idx)
                
                # Unir todos los índices en un solo vector (tendrá tamaño 3,000)
                boot_idx = np.concatenate(boot_idx_list)
                
            # Mezclar para que no queden ordenados por clase
            np.random.shuffle(boot_idx)
            X_boot, y_boot = X[boot_idx], y[boot_idx]
            
            print(_)

            # 2. Entrenar árbol individual sobre la muestra bootstrap
            tree = fit_tree(
                X_boot, y_boot,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                modes=self.modes, mode_est = self.mode_est
            )
            self.trees.append(tree)

    def RF_predic(self, X):
        # 3. Recopilar predicciones de cada árbol -> forma (n_estimators, N_puntos)
        all_preds = np.array([predict(tree, X) for tree in self.trees])
        
        # 4. Votación mayoritaria por columna (muestra)
        final_preds = []
        for i in range(len(X)):
            votes = all_preds[:, i]
            vals, counts = np.unique(votes, return_counts=True)
            final_preds.append(vals[np.argmax(counts)])

        return np.array(final_preds)

'''
## EJEMPLO DE USO:
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
datos = pd.read_csv('sample_1000_mdtraj_t2.csv')
print(datos.head())

print(datos.info())

X = datos[['phi', 'psi']]  

print(X.head())

y=datos['dssp']
np.unique(y)

tree= fit_tree(X, y, max_depth=5, min_samples_split=2, modes=3, merge_thresh='percentil10')
# Predicción del conjunto
y_pred = predict(tree,X)
acc_custom = accuracy_score(y, y_pred)
print("=== Evaluación arbol clasificacion custom con distancia PLS ===")
print("Accuracy:", accuracy_score(y, y_pred))
report=classification_report(y, y_pred)
print(classification_report(y, y_pred))


## CON TRAIN Y TEST
from sklearn.model_selection import train_test_split
## DIVIDIMOS EN TRAIN Y TEST
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

tree = fit_tree(X_tr, y_tr, max_depth=5, min_samples_split=2, modes=5,merge_thresh='percentil10')
y_pred_custom = predict(tree,X_te)
np.unique(y_pred_custom)
acc_custom = accuracy_score(y_te, y_pred_custom)
print("=== Evaluación arbol clasificacion custom con distancia PLS ===")
print("Accuracy:", accuracy_score(y_te, y_pred_custom))
report=classification_report(y_te, y_pred_custom)
print(classification_report(y_te, y_pred_custom))

'''