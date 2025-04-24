
from typing import List, Tuple
import math

class Matrix:
    def __init__(self, rows: int, cols: int, values: list = None):
        self.rows = rows
        self.cols = cols
        if values:
            if len(values) != rows or any(len(row) != cols for row in values):
                raise ValueError("Некорректные размеры данных для матрицы.")
            self.values = values
        else:
            # Инициализация нулевой матрицы
            self.values = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def get(self, row: int, col: int) -> float:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise IndexError("Выход за границы матрицы.")
        return self.values[row][col]

    def set(self, row: int, col: int, value: float):
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise IndexError("Выход за границы матрицы.")
        self.values[row][col] = value

    def swap_rows(self, row1: int, row2: int):
        if row1 < 0 or row1 >= self.rows or row2 < 0 or row2 >= self.rows:
            raise IndexError("Некорректные номера строк.")
        self.values[row1], self.values[row2] = self.values[row2], self.values[row1]

    def T(self):
        matrix_T = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                matrix_T.set(j,i,self.get(i,j))
        return matrix_T
    
    def __mul__(self, other):
        # Умножение матрицы на число
        if isinstance(other, (int, float)):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.set(i, j, self.get(i, j) * other)
            return result

        # Умножение матрицы на матрицу
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Количество столбцов первой матрицы не совпадает с количеством строк второй.")
            result = Matrix(self.rows, other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    sum_val = 0.0
                    for k in range(self.cols):
                        sum_val += self.get(i, k) * other.get(k, j)
                    result.set(i, j, sum_val)
            return result

        else:
            raise TypeError("Неподдерживаемый тип данных(")

    def __rmul__(self, other):
        # Умножение числа на матрицу (коммутативность)
        return self.__mul__(other)
    
    
    def qr_decomposition(self) -> Tuple['Matrix','Matrix']:
        A = Matrix(self.rows, self.cols, self.values)
        m, n = self.rows, self.cols
        Q = [[0.0]*n for _ in range(m)]
        R = [[0.0]*n for _ in range(n)]
        for j in range(n):
            v = [A.values[i][j] for i in range(m)]
            for i in range(j):
                qi = [Q[k][i] for k in range(m)]
                R[i][j] = sum(A.values[k][j]*qi[k] for k in range(m))
                for k in range(m): v[k] -= R[i][j]*qi[k]
            norm_v = math.sqrt(sum(val*val for val in v))
            R[j][j] = norm_v
            for i in range(m): Q[i][j] = v[i]/norm_v
        return Matrix(len(Q), len(Q[0]), Q), Matrix(len(R), len(R[0]), R)

    def qr_eigen(self, max_iter: int=1000, tol: float=1e-20) -> Tuple[List[float],'Matrix']:
        A_k = Matrix(self.rows, self.cols, self.values)
        Q_total = Matrix(self.rows, self.rows)
        for i in range(self.rows):
            Q_total.set(i, i, 1.0)
        for _ in range(max_iter):
            Q, R = A_k.qr_decomposition()
            A_k = R*Q
            Q_total = Q_total*Q
            off = sum(A_k.values[i][j]**2 for i in range(self.rows) for j in range(self.cols) if i!=j)
            if math.sqrt(off) < tol: break
        return [A_k.values[i][i] for i in range(self.rows)], Q_total




    def __repr__(self):
        return " \n ".join([" ".join(map(str, row)) for row in self.values])


def gauss_solver(A: 'Matrix', b: 'Matrix') -> List['Matrix']:
    """
    Вход:
    A: матрица коэффициентов (n×n). Используется класс Matrix из предыдущей
    ,→ лабораторной работы
    b: вектор правых частей (n×1)
    Выход:
    list[Matrix]: список базисных векторов решения системы
    Raises:
    ValueError: если система несовместна
    """
    n = A.rows
    if n != A.cols:
        raise ValueError("Матрица A должна быть квадратной.")
    if b.rows != n or b.cols != 1:
        raise ValueError("Вектор b должен быть размерности n x 1.")

    # Создаем расширенную матрицу [A | b]
    add_matrix = Matrix(n, n + 1)
    for i in range(n):
        for j in range(n):
            add_matrix.set(i, j, A.get(i, j))
        add_matrix.set(i, n, b.get(i, 0))

    # Прямой ход метода Гаусса
    for i in range(n):
        # Выбор ведущего элемента
        max_row = i
        max_val = abs(add_matrix.get(i, i))
        for k in range(i + 1, n):
            current_val = abs(add_matrix.get(k, i))
            if current_val > max_val:
                max_val, max_row = current_val, k
        if max_row != i:
            add_matrix.swap_rows(i, max_row)

        # Проверка на вырожденность
        pivot = add_matrix.get(i, i)
        if abs(pivot) < 1e-10:
            continue

        # Нормализация строки
        for j in range(i, n + 1):
            add_matrix.set(i, j, add_matrix.get(i, j) / pivot)

        # Обнуление элементов ниже
        for k in range(i + 1, n):
            factor = add_matrix.get(k, i)
            for j in range(i, n + 1):
                add_matrix.set(k, j, add_matrix.get(k, j) - factor * add_matrix.get(i, j))

    # Проверка на несовместность
    for i in range(n):
        if all(abs(add_matrix.get(i, j)) < 1e-10 for j in range(n)) and abs(add_matrix.get(i, n)) >= 1e-10:
            raise ValueError("Система несовместна.")

    # Обратный ход
    for i in reversed(range(n)):
        pivot_col = -1
        for j in range(n):
            if abs(add_matrix.get(i, j)) >= 1e-10:
                pivot_col = j
                break
        if pivot_col == -1:
            continue

        for k in range(i):
            factor = add_matrix.get(k, pivot_col)
            for j in range(pivot_col, n + 1):
                add_matrix.set(k, j, add_matrix.get(k, j) - factor * add_matrix.get(i, j))

    # Определение базисных и свободных переменных
    lead_cols = []
    for i in range(n):
        for j in range(n):
            if abs(add_matrix.get(i, j)) >= 1e-10:
                lead_cols.append(j)
                break

    rank = len(lead_cols)
    free_vars = [j for j in range(n) if j not in lead_cols]

    # Единственное решение
    if rank == n:
        solution = Matrix(n, 1)
        for i in range(n):
            solution.set(i, 0, add_matrix.get(i, n))
        return [solution]

    # Фундаментальная система решений
    solutions = []
    for var in free_vars:
        vec = Matrix(n, 1)
        vec.set(var, 0, 1.0)
        for i in reversed(range(rank)):
            row = i
            col = lead_cols[i]
            sum_val = 0.0
            for j in range(col + 1, n):
                sum_val += add_matrix.get(row, j) * vec.get(j, 0)
            val = (add_matrix.get(row, n) - sum_val) / add_matrix.get(row, col)
            vec.set(col, 0, val)
        solutions.append(vec)

    return solutions

def center_data(X: 'Matrix') -> 'Matrix':
    """
    Вход: матрица данных X (n×m)
    Выход: центрированная матрица X_centered (n×m)
    """
    # Средние (вектор 1*m)
    center_matrix = Matrix(1,X.cols)
    for i in range(X.cols):
        sum = 0
        for j in range(X.rows):
            sum+=X.get(j,i)
        center_matrix.set(0,i,sum/X.rows)
    

    # Получение усреднённой матрицы
    X_center=Matrix(X.rows, X.cols)
    for i in range(X.rows):
        for j in range(X.cols):
            X_center.set(i,j, X.get(i,j)-center_matrix.get(0,j))

    return X_center

def covariance_matrix(X_centered: 'Matrix') -> 'Matrix':
    """
    Вход: центрированная матрица X_centered (n×m)
    Выход: матрица ковариаций C (m×m)
    """
    alpha = 1/(X_centered.rows-1)
    result_matrix = alpha * (X_centered.T()* X_centered)
    return result_matrix
        
def get_covariance_matrix(X: 'Matrix') -> 'Matrix':
    """
    Вход: матрица данных X (n×m)
    Выход: матрица ковариаций C (m×m)
    """
    X_centered = center_data(X)
    return covariance_matrix(X_centered)

def handle_missing_values(X: Matrix) -> Matrix:
    """
    Вход: матрица данных X (n×m) с возможными NaN
    Выход: матрица данных X_filled (n×m) без NaN
    """
    n, m = X.rows, X.cols
    sums = [0.0] * m
    counts = [0] * m
    for i in range(n):
        for j in range(m):
            v = X.values[i][j]
            if v == v:
                sums[j] += v
                counts[j] += 1
    
    means = [(sums[j] / counts[j]) if counts[j] > 0 else 0.0 for j in range(m)]
    filled = []
    for i in range(n):
        row = []
        for j in range(m):
            v = X.values[i][j]
            row.append(v if v == v else means[j])
        filled.append(row)
    return Matrix(n,m,filled)