import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


class Nodo:
    def __init__(self, feature=None, mode=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.mode = mode
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # clase si es hoja

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

def best_split(X, y, modes=3):
        best_feature = None
        best_mode = None
        best_threshold = None
        best_impurity = float("inf")
    
        n, m = X.shape
    
        for feature in range(m):
            x = X[:, feature]
            # KDE para encontrar modos
            try:
                kde = gaussian_kde(x)
                xs = np.linspace(x.min() - 1, x.max() + 1, 2000)
                ys = kde(xs)
                peaks_idx, _ = find_peaks(ys, height=0)
                kde_peaks = xs[peaks_idx]
                peak_heights = ys[peaks_idx]
                order = np.argsort(peak_heights)[::-1]
                kde_peaks_sorted = kde_peaks[order][:modes]
            except Exception:
                # fallback: usar k-means-like split en los percentiles si KDE falla
                kde_peaks_sorted = np.unique(np.percentile(x, [25, 50, 75]))[:modes]
    
            for p in kde_peaks_sorted:
                distancias = np.abs(x - p)
                # limitar umbrales a un número razonable para velocidad (unique puede ser grande)
                unique_d = np.unique(distancias)
                if unique_d.size > 200:
                    # tomar percentiles de distancias en vez de todas
                    thresholds = np.unique(np.percentile(distancias, np.linspace(0, 100, 201)))
                else:
                    thresholds = unique_d
    
                for t in thresholds:
                    left_idx = distancias <= t
                    right_idx = distancias > t
    
                    if left_idx.sum() == 0 or right_idx.sum() == 0:
                        continue
    
                    imp_left = gini(y[left_idx])
                    imp_right = gini(y[right_idx])
    
                    impurity = (left_idx.sum() * imp_left + right_idx.sum() * imp_right) / n
    
                    if impurity < best_impurity:
                        best_impurity = impurity
                        best_feature = feature
                        best_mode = p
                        best_threshold = t
    
        return best_feature, best_mode, best_threshold

def build_tree(X, y, depth=0, max_depth=5, min_samples_split=2, modes=3):
        # criterio de parada
        if len(y) < min_samples_split or depth >= max_depth or len(np.unique(y)) == 1:
            return Nodo(value=majority_class(y))

        feature, mode, threshold = best_split(X, y, modes)
    
        if feature is None:
            return Nodo(value=majority_class(y))
    
        distancias = np.abs(X[:, feature] - mode)
        left_idx = distancias <= threshold
        right_idx = distancias > threshold
    
        # Si alguna rama queda vacía, hacer hoja
        if left_idx.sum() == 0 or right_idx.sum() == 0:
            return Nodo(value=majority_class(y))
    
        left = build_tree(X[left_idx], y[left_idx], depth + 1, max_depth, min_samples_split, modes)
        right = build_tree(X[right_idx], y[right_idx], depth + 1, max_depth, min_samples_split, modes)
    
        return Nodo(feature, mode, threshold, left, right)

def predict_one(x, nodo):
        if nodo.value is not None:
            return nodo.value
        if np.abs(x[nodo.feature] - nodo.mode) <= nodo.threshold:
            return predict_one(x, nodo.left)
        else:
            return predict_one(x, nodo.right)

def predict(X, tree):
        return np.array([predict_one(x, tree) for x in X])




# usa build_tree(X,y,...) y predict(X,tree) que ya tienes

def leaf_id_for_sample(x, nodo):
        if nodo.value is not None:
            return id(nodo)
        if np.abs(x[nodo.feature] - nodo.mode) <= nodo.threshold:
            return leaf_id_for_sample(x, nodo.left)
        else:
            return leaf_id_for_sample(x, nodo.right)
    
def leaf_class_probs(tree, X_train, y_train, classes):
    counts = {}
    for xi, yi in zip(X_train, y_train):
        lid = leaf_id_for_sample(xi, tree)
        if lid not in counts:
            counts[lid] = {c:0 for c in classes}
        counts[lid][yi] += 1
    probs_map = {}
    for lid, d in counts.items():
        tot = sum(d.values())
        probs_map[lid] = np.array([d[c]/tot for c in classes])
    return probs_map

def predict_proba_from_leafmap(tree, X, probs_map, classes):
    P = np.zeros((X.shape[0], len(classes)))
    for i, xi in enumerate(X):
        lid = leaf_id_for_sample(xi, tree)
        P[i] = probs_map.get(lid, np.ones(len(classes))/len(classes))
    return P

def compare_on_train(X, y, max_depth=4, min_samples_split=5, modes=3):
    classes = np.unique(y)

    # custom tree (sin modificar)
    tree_c = build_tree(X, y, max_depth=max_depth, min_samples_split=min_samples_split, modes=modes)
    y_pred_c = predict(X, tree_c)
    probs_map = leaf_class_probs(tree_c, X, y, classes)
    probs_c = predict_proba_from_leafmap(tree_c, X, probs_map, classes)

    # sklearn tree con mismos hiperparámetros
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=0)
    clf.fit(X, y)
    y_pred_s = clf.predict(X)
    probs_s = clf.predict_proba(X)

    # métricas sobre TRAIN
    y_bin = label_binarize(y, classes=classes)
    auc_c = roc_auc_score(y_bin, probs_c, multi_class='ovr', average='weighted')
    auc_s = roc_auc_score(y_bin, probs_s, multi_class='ovr', average='weighted')

    return {
        'custom_acc': float(accuracy_score(y, y_pred_c)),
        'sk_acc': float(accuracy_score(y, y_pred_s)),
        'custom_auc': float(auc_c),
        'sk_auc': float(auc_s)
    }


# --- división train/test y evaluación ---
def train_test_compare(X, y, test_size=0.3, random_state=0, max_depth=4, min_samples_split=5, modes=3):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    classes = np.unique(y_tr)

    # custom tree (sin tocar tu código)
    tree_c = build_tree(X_tr, y_tr, max_depth=max_depth, min_samples_split=min_samples_split, modes=modes)
    y_pred_c = predict(X_te, tree_c)
    probs_map = leaf_class_probs(tree_c, X_tr, y_tr, classes)
    probs_c = predict_proba_from_leafmap(tree_c, X_te, probs_map, classes)

    # sklearn tree con mismos hiperparámetros
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=0)
    clf.fit(X_tr, y_tr)
    y_pred_s = clf.predict(X_te)
    probs_s = clf.predict_proba(X_te)
    
    print("classes:", classes)
    print("y_te unique:", np.unique(y_te, return_counts=True))
    print("probs_c shape:", probs_c.shape)
    print("probs_s shape:", probs_s.shape)

        # métricas
    if len(classes) == 2:
        # binary: use 1D y_true and probability of positive class (classes[1])
        auc_c = roc_auc_score(y_te, probs_c[:, 1])
        auc_s = roc_auc_score(y_te, probs_s[:, 1])
    else:
        # multiclass
        y_te_bin = label_binarize(y_te, classes=classes)
        auc_c = roc_auc_score(y_te_bin, probs_c, multi_class='ovr', average='weighted')
        auc_s = roc_auc_score(y_te_bin, probs_s, multi_class='ovr', average='weighted')


    return {
        'custom_acc': float(accuracy_score(y_te, y_pred_c)),
        'sk_acc': float(accuracy_score(y_te, y_pred_s)),
        'custom_auc': float(auc_c),
        'sk_auc': float(auc_s),
        'confusion_custom': confusion_matrix(y_te, y_pred_c),
        'confusion_sklearn': confusion_matrix(y_te, y_pred_s)
    }







