from typing import Tuple


import matplotlib.pyplot as plt

from matplotlib.figure import Figure

from sklearn.datasets import load_iris

from all_for_prepare import *
from eight_val import *


def pca(X: Matrix, k: int) -> Tuple[Matrix, float]:
    Xc = center_data(X)
    C = covariance_matrix(Xc)
    vals = find_eigenvalues(C)
    vecs = find_eigenvectors(C, vals)
    idx = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)[:k]
    V = Matrix(X.rows, k, [[vec.values[r][0] for vec in [vecs[i] for i in idx]] for r in range(X.cols)])
    Xp = Xc*V
    ratio = explained_variance_ratio(vals, k)
    return Xp, ratio

def auto_select_k(eigenvalues: List[float], threshold: float = 0.95) -> int:
    """
    Вход:
    eigenvalues: список собственных значений
    threshold: порог объяснённой дисперсии
    Выход: оптимальное число главных компонент k
    """
    vals=sorted(eigenvalues, reverse=True)
    total=sum(vals)
    sumas=0
    for idx, v in enumerate(vals, start=1):
        sumas+=v
        if sumas / total >= threshold:
            return idx
    return len(vals)

def apply_pca_to_dataset(dataset_name: str, k: int) -> Tuple['Matrix', float]:
    """
    Вход:
    dataset_name: название датасета
    k: число главных компонент
    Выход: кортеж (проекция данных, качество модели)
    """
    name = dataset_name.lower()
    if name == 'iris':
        data = load_iris()
        X_np = data.data
    else:
        raise ValueError("Приходите позже. Ваш датасет будет обработан через n часов")
    X = Matrix(len(X_np), len(X_np[0]), X_np.tolist())
    X_proj, explained_ratio, *_ = pca(X, k)
    return X_proj, explained_ratio

# Визуализация проекции в 2D
def plot_pca_projection(X_proj: Matrix) -> Figure:
    data = X_proj.values
    xs = [row[0] for row in data]
    ys = [row[1] for row in data]
    fig, ax = plt.subplots()  
    ax.scatter(xs, ys)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Projection')
    return fig

def pca_qr(X: Matrix, k: int) -> Tuple[Matrix, float]:
    Xc = center_data(X)
    Cov = covariance_matrix(Xc)
    raw_vals, Qtot = Cov.qr_eigen(max_iter=1000, tol=1e-12)
    pairs = sorted([(val, [Qtot.values[r][i] for r in range(Cov.rows)]) for i, val in enumerate(raw_vals)], key=lambda x: x[0], reverse=True)
    top_vals, top_vecs = zip(*pairs[:k])
    W = Matrix(Cov.rows,k,  [[vec[row] for vec in top_vecs] for row in range(Cov.rows)])
    Xp = Xc*W
    ratio = sum(top_vals) / sum(raw_vals) if sum(raw_vals) else 1.0
    return Xp, ratio