from itertools import product
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize

# Importar funciones de tu implementación personalizada
from Clasification_tree import build_tree, predict, leaf_class_probs, predict_proba_from_leafmap

def prepare_dataset(X, y):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    df = X.copy()
    
    # 1. Eliminar columnas completamente vacías
    df = df.dropna(axis=1, how='all')
    
    # 2. Imputación de numéricos con mediana
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
    # 3. Imputación de categóricos con la moda
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'missing'
            df[col] = df[col].fillna(mode_val)
            
    # 4. One-hot encoding para categóricos
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
        
    # 5. Etiquetado de la variable objetivo y
    if y is None:
        raise ValueError("y no puede ser None")
        
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_arr = y.to_numpy().ravel()
    else:
        y_arr = np.ravel(y)
        
    if y_arr.dtype.kind in 'OUb' or str(y_arr.dtype) == 'category':
        le = LabelEncoder()
        y_arr = le.fit_transform(y_arr)
    else:
        y_arr = y_arr.astype(int)
        
    return df.values.astype(np.float64), y_arr


def calculate_roc_auc(y_true, y_probs, classes):
    """Calcula ROC-AUC manejando casos binarios y multiclase."""
    if len(classes) == 2:
        return float(roc_auc_score(y_true, y_probs[:, 1]))
    else:
        y_true_bin = label_binarize(y_true, classes=classes)
        return float(roc_auc_score(y_true_bin, y_probs, multi_class='ovr', average='weighted'))


def evaluate_5fold_cv(name, X, y, param_grid, n_splits=5, random_state=0):
    Xp, yp = prepare_dataset(X, y)
    classes = np.unique(yp)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # ---------------------------------------------------------
    # 1. Búsqueda de parámetros para Árbol Personalizado (Custom)
    # ---------------------------------------------------------
    best_custom_auc_mean = -1.0
    best_custom_metrics = {}
    best_custom_params = {}

    custom_combinations = list(product(
        param_grid['max_depth'], 
        param_grid['min_samples_split'], 
        param_grid['modes']
    ))

    for max_depth, min_samples_split, modes in custom_combinations:
        auc_fold_scores = []
        acc_fold_scores = []

        for train_idx, val_idx in skf.split(Xp, yp):
            X_tr, X_val = Xp[train_idx], Xp[val_idx]
            y_tr, y_val = yp[train_idx], yp[val_idx]

            tree_c = build_tree(X_tr, y_tr, max_depth=max_depth, min_samples_split=min_samples_split, modes=modes)
            y_pred_c = predict(X_val, tree_c)
            probs_map = leaf_class_probs(tree_c, X_tr, y_tr, classes)
            probs_c = predict_proba_from_leafmap(tree_c, X_val, probs_map, classes)

            auc_fold_scores.append(calculate_roc_auc(y_val, probs_c, classes))
            acc_fold_scores.append(accuracy_score(y_val, y_pred_c))

        mean_auc = np.mean(auc_fold_scores)
        if mean_auc > best_custom_auc_mean:
            best_custom_auc_mean = mean_auc
            best_custom_params = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'modes': modes}
            best_custom_metrics = {
                'custom_auc_mean': mean_auc,
                'custom_auc_std': np.std(auc_fold_scores),
                'custom_acc_mean': np.mean(acc_fold_scores),
                'custom_acc_std': np.std(acc_fold_scores)
            }

    # ---------------------------------------------------------
    # 2. Búsqueda de parámetros para Scikit-Learn
    # ---------------------------------------------------------
    best_sk_auc_mean = -1.0
    best_sk_metrics = {}
    best_sk_params = {}

    sk_combinations = list(product(
        param_grid['max_depth'], 
        param_grid['min_samples_split']
    ))

    for max_depth, min_samples_split in sk_combinations:
        auc_fold_scores = []
        acc_fold_scores = []

        for train_idx, val_idx in skf.split(Xp, yp):
            X_tr, X_val = Xp[train_idx], Xp[val_idx]
            y_tr, y_val = yp[train_idx], yp[val_idx]

            clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
            clf.fit(X_tr, y_tr)
            y_pred_s = clf.predict(X_val)
            probs_s = clf.predict_proba(X_val)

            auc_fold_scores.append(calculate_roc_auc(y_val, probs_s, classes))
            acc_fold_scores.append(accuracy_score(y_val, y_pred_s))

        mean_auc = np.mean(auc_fold_scores)
        if mean_auc > best_sk_auc_mean:
            best_sk_auc_mean = mean_auc
            best_sk_params = {'max_depth': max_depth, 'min_samples_split': min_samples_split}
            best_sk_metrics = {
                'sk_auc_mean': mean_auc,
                'sk_auc_std': np.std(auc_fold_scores),
                'sk_acc_mean': np.mean(acc_fold_scores),
                'sk_acc_std': np.std(acc_fold_scores)
            }

    # Unificar resultados del dataset
    res = {
        'dataset': name,
        'n_samples': Xp.shape[0],
        'n_features': Xp.shape[1],
        'custom_best_params': str(best_custom_params),
        'custom_auc_mean': best_custom_metrics['custom_auc_mean'],
        'custom_auc_std': best_custom_metrics['custom_auc_std'],
        'sk_best_params': str(best_sk_params),
        'sk_auc_mean': best_sk_metrics['sk_auc_mean'],
        'sk_auc_std': best_sk_metrics['sk_auc_std'],
        'custom_acc_mean': best_custom_metrics['custom_acc_mean'],
        'custom_acc_std': best_custom_metrics['custom_acc_std'],
        'sk_acc_mean': best_sk_metrics['sk_acc_mean'],
        'sk_acc_std': best_sk_metrics['sk_acc_std'],
    }
    return res

# ---------------------------------------------------------
# Definición de la grilla de parámetros
# ---------------------------------------------------------
param_grid = {
    'max_depth': [3, 4, 6, 8],
    'min_samples_split': [2, 5, 10],
    'modes': [1, 2, 3]
}

results = []

# 1) Breast Cancer
print("Procesando Breast Cancer...")
bc = load_breast_cancer()
results.append(evaluate_5fold_cv('breast_cancer', bc.data, bc.target, param_grid))

# 2) Pima Indians Diabetes
print("Procesando Pima Diabetes...")
pima = None
try:
    pima = fetch_openml('diabetes', version=1, as_frame=True, parser='auto')
except Exception:
    try:
        pima = fetch_openml(data_id=37, as_frame=True, parser='auto')
    except Exception as e:
        print(f"Error cargando Pima: {e}")

if pima is not None:
    results.append(evaluate_5fold_cv('pima', pima.data, pima.target, param_grid))

# 3) German Credit
print("Procesando German Credit...")
credit = None
try:
    credit = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
except Exception:
    try:
        credit = fetch_openml(data_id=31, as_frame=True, parser='auto')
    except Exception as e:
        print(f"Error cargando German Credit: {e}")

if credit is not None:
    results.append(evaluate_5fold_cv('credit_german', credit.data, credit.target, param_grid))

# Guardar y mostrar resultados
df_res = pd.DataFrame(results)
df_res.to_csv('cv_comparison_results.csv', index=False)

print("\n--- Resultados de Selección de Parámetros con 5-Fold CV ---")
print(df_res.to_string(index=False))
