import math
import numpy as np


def read_data_from_file(filename):
    matrix_a = []
    vector_b = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            values = [float(x) for x in parts]
            matrix_a.append(values[:-1])
            vector_b.append(values[-1])
    n = len(matrix_a)
    for row in matrix_a:
        if len(row) != n:
            raise ValueError("Ошибка: Матрица коэффициентов должна быть квадратной!")

    return matrix_a, vector_b, n


def check_diagonal_dominance(A, n):
    is_dominant = True
    print("\nПроверка диагонального преобладания")
    for i in range(n):
        diag = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        print(f"Строка {i+1}: |{A[i][i]}| = {diag:.2f} против суммы остальных {row_sum:.2f}")
        if diag <= row_sum:
            is_dominant = False

    if is_dominant:
        print("Диагональное преобладание строго выполняется. Методы гарантированно сойдутся!\n")
    else:
        print("ВНИМАНИЕ: Строгое диагональное преобладание НЕ выполняется. Сходимость не гарантирована.\n")
    return is_dominant

# Метод простых итераций (Якоби)
def simple_iteration(A, b, n, eps=1e-5, max_iter=1000):
    x_old = [0.0] * n
    x_new = [0.0] * n

    for iteration in range(1, max_iter + 1):
        for i in range(n):
            s = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        norm_diff = max(abs(x_new[i] - x_old[i]) for i in range(n))

        if norm_diff < eps:
            return x_new, iteration

        x_old = list(x_new)

    return x_new, max_iter

# Метод Зейделя
def gauss_seidel(A, b, n, eps=1e-5, max_iter=1000):
    x = [0.0] * n

    for iteration in range(1, max_iter + 1):
        max_diff = 0.0
        for i in range(n):
            old_val = x[i]
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i][i]

            diff = abs(x[i] - old_val)
            if diff > max_diff:
                max_diff = diff

        if max_diff < eps:
            return x, iteration

    return x, max_iter


def main():
    filename = "data.txt"
    eps = 1e-4

    A, b, n = read_data_from_file(filename)

    print(f"Размерность считанной системы: {n}x{n}")
    print(f"Заданная точность ε = {eps}")

    check_diagonal_dominance(A, b_len := n)

    x_jacobi, iters_jacobi = simple_iteration(A, b, n, eps=eps)

    x_seidel, iters_seidel = gauss_seidel(A, b, n, eps=eps)

    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    x_ref = np.linalg.solve(A_np, b_np)

    print("РЕЗУЛЬТАТЫ РАСЧЕТОВ")

    print(f"Эталонное решение (NumPy):")
    print("  " + ", ".join([f"x{i+1} = {val:.6f}" for i, val in enumerate(x_ref)]))

    print(f"\n1. Метод простых итераций:")
    print("  " + ", ".join([f"x{i+1} = {val:.6f}" for i, val in enumerate(x_jacobi)]))
    print(f"Количество итераций: {iters_jacobi}")

    print(f"\n2. Метод Зейделя:")
    print("  " + ", ".join([f"x{i+1} = {val:.6f}" for i, val in enumerate(x_seidel)]))
    print(f"Количество итераций: {iters_seidel}")

    print("\nСРАВНЕНИЕ СХОДИМОСТИ")
    print(f"Метод простых итераций сошелся за: {iters_jacobi} итераций.")
    print(f"Метод Зейделя сошелся за: {iters_seidel} итераций.")

    if iters_seidel < iters_jacobi:
        speedup = (iters_jacobi - iters_seidel) / iters_jacobi * 100
        print(f"Метод Зейделя сошелся быстрее на {speedup:.1f}%.")


if __name__ == "__main__":
    main()
