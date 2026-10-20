def read_data(filename="data.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    matrix_data = [list(map(float, line.split())) for line in lines]

    b = matrix_data[-1]
    A = matrix_data[:-1]
    n = len(A)

    for row in A:
        if len(row) != n:
            raise ValueError("Матрица A должна быть квадратной")
    if len(b) != n:
        raise ValueError("Размерность вектора b не совпадает с размерностью A")

    return A, b, n


def print_matrix(M, name="Matrix"):
    print(f"\n {name}")
    for row in M:
        print(" ".join(f"{val:10.4f}" for val in row))


def print_vector(v, name="Vector"):
    print(f"\n {name}")
    print(" ".join(f"{val:10.4f}" for val in v))


def matrix_multiply(A, B):
    n = len(A)
    m = len(B[0])
    k_len = len(B)

    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(k_len):
                C[i][j] += A[i][k] * B[k][j]
    return C


def create_identity(n):
    I = [[0.0] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I


def lup_decomposition(A, n):
    U = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    P = create_identity(n)

    for i in range(n):
        L[i][i] = 1.0

    swaps_count = 0

    for k in range(n):
        max_val = 0.0
        pivot_row = k
        for i in range(k, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                pivot_row = i

        if max_val < 1e-12:
            raise ValueError("Матрица вырождена. LU-разложение невозможно.")

        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            P[k], P[pivot_row] = P[pivot_row], P[k]

            for j in range(k):
                L[k][j], L[pivot_row][j] = L[pivot_row][j], L[k][j]

            swaps_count += 1

        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]

    return L, U, P, swaps_count


def solve_lu(L, U, P, b, n):
    Pb = [0.0] * n
    for i in range(n):
        Pb[i] = sum(P[i][j] * b[j] for j in range(n))

    # прямой ход
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = Pb[i] - s

    # обратный ход
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]

    return x


def compute_determinant(U, swaps_count, n):
    det = 1.0
    for i in range(n):
        det *= U[i][i]

    if swaps_count % 2 != 0:
        det = -det

    return det


def compute_inverse_matrix(L, U, P, n):
    I = create_identity(n)
    A_inv = [[0.0] * n for _ in range(n)]

    for k in range(n):
        e_k = [I[i][k] for i in range(n)]
        x_k = solve_lu(L, U, P, e_k, n)

        for i in range(n):
            A_inv[i][k] = x_k[i]

    return A_inv


def main():
    try:
        A, b, n = read_data("data.txt")
    except Exception as e:
        print(f"Ошибка чтения данных: {e}")
        return

    print(f"Загружена система размерности {n}x{n}")
    print_matrix(A, "Исходная матрица A")
    print_vector(b, "Вектор b")

    # Выполнение LUP-факторизации
    L, U, P, swaps_count = lup_decomposition(A, n)

    print_matrix(L, "1. Матрица L (Нижнетреугольная)")
    print_matrix(U, "1. Матрица U (Верхнетреугольная)")

    LU = matrix_multiply(L, U)
    print_matrix(LU, "2. Результат умножения L * U")

    x = solve_lu(L, U, P, b, n)
    print_vector(x, "3. Решение системы x (A * x = b)")

    A_inv = compute_inverse_matrix(L, U, P, n)
    print_matrix(A_inv, "4. Обратная матрица A^(-1)")

    det = compute_determinant(U, swaps_count, n)
    print("\n5. Определитель матрицы A")
    print(f"det(A) = {det:.4f}")

    A_Ainv = matrix_multiply(A, A_inv)
    print_matrix(A_Ainv, "6. Проверка A * A^(-1) (Единичная матрица E)")

    # Проверка L * U = P * A
    PA = matrix_multiply(P, A)
    print_matrix(PA, "7. Проверка P * A (Сравнить с п.2 L * U)")


if __name__ == "__main__":
    main()
