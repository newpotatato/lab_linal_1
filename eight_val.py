from all_for_prepare import *

def determinant(mat: 'Matrix') -> float:
        """Вычисляет определитель матрицы методом Гаусса."""
        n = mat.rows
        det = 1.0
        aug = Matrix(n, n, [row.copy() for row in mat.values])
        
        for i in range(n):
            max_row = i
            for k in range(i, n):
                if abs(aug.get(k, i)) > abs(aug.get(max_row, i)):
                    max_row = k
            if max_row != i:
                aug.swap_rows(i, max_row)
                det *= -1
                
            pivot = aug.get(i, i)
            if abs(pivot) < 1e-12:
                return 0.0
            det *= pivot
            
            for k in range(i+1, n):
                factor = aug.get(k, i) / pivot
                for j in range(i, n):
                    aug.set(k, j, aug.get(k, j) - factor * aug.get(i, j))
        return det


def det_diff(C: Matrix, lmbd: float) -> float:
    """
    Вход:
    C: матрица ковариаций (m×m)
    lmbd: собственное значение
    Выход: определитель разности матриц""" 

    n = C.rows
    I = Matrix(n,n)
    for i in range(n):
            I.set(i,i,lmbd)
    return determinant(C - I)

def find_eigenvalues(C: Matrix, tol: float = 1e-6) -> List[float]:
    """
    Вход:
    C: матрица ковариаций (m×m)
    tol: допустимая погрешность
    Выход:
    """
    n = C.rows
    r = max(
        sum(abs(C.values[i][j]) for j in range(n) if j != i) for i in range(n)
    )
    d_min = min(C.values[i][i] for i in range(n))
    d_max = max(C.values[i][i] for i in range(n))
    lower = d_min - r
    upper = d_max + r


    intervals = 10000
    xs = [lower + i * (upper - lower) / intervals for i in range(intervals + 1)]
    roots: List[float] = []

    for i in range(intervals):
        a, b = xs[i], xs[i + 1]
        fa, fb = det_diff(C,a), det_diff(C,b)
        if fa * fb <= 0:
            left, right = a, b
            while right - left > tol:
                mid = (left + right) / 2
                fm = det_diff(C,mid)
                if fa * fm <= 0:
                    right, fb = mid, fm
                else:
                    left, fa = mid, fm
            roots.append((left + right) / 2)

    return sorted({round(val, 5) for val in roots})

def find_eigenvectors(C: Matrix, eigenvalues: List[float]) -> List[Matrix]:
    """
    Вход:
    C: матрица ковариаций (m×m)
    eigenvalues: список собственных значений
    Выход: список собственных векторов (каждый вектор - объект Matrix)
    """
    n = C.rows
    I = Matrix(n,n)
    for i in range(n):
            I.set(i,i,1)
    eigenvectors = []
    for eigenvalue in eigenvalues:
        M = C - I * eigenvalue
        nuls = Matrix(n,1)
        eigenvector = M.solve_homogeneous(tol=1e-12, round_decimals=6)
        eigenvectors.append(eigenvector)
    return eigenvectors

def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    """
    Вход:
    eigenvalues: список собственных значений
    k: число компонент
    Выход: доля объяснённой дисперсии
    """

    total = sum(eigenvalues)
    top = sum(sorted(eigenvalues, reverse=True)[:k])
    return top / total