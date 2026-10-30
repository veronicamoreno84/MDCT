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
def compute_depth(X, dist_fn, beta=0.1):
    n = len(X)
    dis = compute_distance_matrix(X, dist_fn)

    selector = int(np.trunc(n * beta))
    selector = max(1, min(selector, n - 1))

    depths = np.zeros(n)
    for i in range(n):
        lamb = np.sort(dis[i, :])[selector]
        depths[i] = 1.0 / (1.0 + lamb)

    return depths

#  Desduplicación de modas por distancia mínima en el toro
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