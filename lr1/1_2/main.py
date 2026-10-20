def load_tridiagonal_from_file(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    n = int(lines[0])
    b = list(map(float, lines[1].split()))
    a_raw = list(map(float, lines[2].split()))
    c_raw = list(map(float, lines[3].split()))
    d = list(map(float, lines[4].split()))

    if (
        len(b) != n
        or len(d) != n
        or len(a_raw) != n - 1
        or len(c_raw) != n - 1
    ):
        raise ValueError("Ошибка: количество элементов диагоналей не совпадает с размерностью n")

    a = [0.0] + a_raw
    c = c_raw + [0.0]

    return n, a, b, c, d


def check_diagonally_dominant(n: int, a: list, b: list, c: list):
    print("Проверка достаточных условий устойчивости")
    is_strict = False
    for i in range(n):
        diag = abs(b[i])
        sub = abs(a[i]) if i > 0 else 0.0
        sup = abs(c[i]) if i < n - 1 else 0.0
        row_sum = sub + sup

        if diag < row_sum:
            print(
                f"В строке {i+1} условие диагонального преобладания НЕ выполняется"
                f"|{b[i]}| < {row_sum}"
            )
            return False
        if diag > row_sum:
            is_strict = True

    if is_strict:
        print("Матрица удовлетворяет условию диагонального преобладания.")
        print("Метод прогонки гарантированно устойчив и корректен.\n")
    return True


def tridiagonal_solve(n: int, a: list, b: list, c: list, d: list):
    P = [0.0] * n
    Q = [0.0] * n

    # прямой ход
    print("ПРЯМОЙ ХОД (Вычисление P_i, Q_i)")
    # первое уравнение
    denominator = b[0]
    if abs(denominator) < 1e-15:
        raise ZeroDivisionError("Деление на 0: b[0] равен 0. Метод прогонки неприменим.")

    P[0] = -c[0] / denominator
    Q[0] = d[0] / denominator
    print(f"i = 1: P_1 = {P[0]:10.6f}, Q_1 = {Q[0]:10.6f}")

    # промежуточные уравнения
    for i in range(1, n - 1):
        denominator = b[i] + a[i] * P[i - 1]
        if abs(denominator) < 1e-15:
            raise ZeroDivisionError(f"Деление на 0 в строке {i+1}. Метод прогонки неприменим.")

        P[i] = -c[i] / denominator
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denominator
        print(f"i = {i+1}: P_{i+1} = {P[i]:10.6f}, Q_{i+1} = {Q[i]:10.6f}")

    # обратный ход
    print("\nОБРАТНЫЙ ХОД (Поиск неизвестных)")
    x = [0.0] * n

    # нахождение x_n
    denominator_n = b[n - 1] + a[n - 1] * P[n - 2]
    if abs(denominator_n) < 1e-15:
        raise ZeroDivisionError("Деление на 0 при вычислении x_n. Метод неприменим.")

    x[n - 1] = (d[n - 1] - a[n - 1] * Q[n - 2]) / denominator_n
    print(f"x_{n} = {x[n-1]:10.6f}")

    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
        print(f"x_{i+1} = {x[i]:10.6f}")

    return x, P, Q


if __name__ == "__main__":
    filename = "data.txt"

    n, a, b, c, d = load_tridiagonal_from_file(filename)

    check_diagonally_dominant(n, a, b, c)

    x_solution, P_coeffs, Q_coeffs = tridiagonal_solve(n, a, b, c, d)

    print("Итоговое решение системы уравнений:")
    for i in range(n):
        print(f"x[{i+1}] = {x_solution[i]:12.8f}")

    # проверка невязки r = Ax - d
    print("\nПроверка точности (вектор невязки r = Ax - d) ")
    for i in range(n):
        val = b[i] * x_solution[i]
        if i > 0:
            val += a[i] * x_solution[i - 1]
        if i < n - 1:
            val += c[i] * x_solution[i + 1]
        r = val - d[i]
        print(f"Строка {i+1}: невязка = {r:+.2e}")
