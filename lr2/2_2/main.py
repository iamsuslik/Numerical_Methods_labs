import numpy as np
import matplotlib.pyplot as plt


def read_input(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    a = float(lines[0])
    eps = float(lines[1])
    x1_0 = float(lines[2])
    x2_0 = float(lines[3])
    return a, eps, np.array([x1_0, x2_0], dtype=float)


def F(X, a):
    x1, x2 = X
    f1 = x1**2 + x2**2 - a**2
    f2 = x1 - np.exp(x2) + a
    return np.array([f1, f2])


def jacobian(X, a):
    x1, x2 = X
    J = np.array([
        [2 * x1, 2 * x2],
        [1.0, -np.exp(x2)]
    ])
    return J

def solve_linear_system_2x2(J, F_val):
    j11, j12 = J[0][0], J[0][1]
    j21, j22 = J[1][0], J[1][1]

    b1 = -F_val[0]
    b2 = -F_val[1]

    det = j11 * j22 - j12 * j21

    if abs(det) < 1e-12:
        raise ZeroDivisionError("Определитель матрицы Якоби равен нулю (вырождена)")

    det_x1 = b1 * j22 - j12 * b2
    det_x2 = j11 * b2 - b1 * j21

    delta_x1 = det_x1 / det
    delta_x2 = det_x2 / det

    return np.array([delta_x1, delta_x2])


def newton_method(X0, a, eps, max_iter=100):
    X = np.copy(X0)
    errors = []

    for k in range(max_iter):
        Fx = F(X, a)
        Jx = jacobian(X, a)

        try:
            delta_X = solve_linear_system_2x2(Jx, Fx)
        except ZeroDivisionError as e:
            print(f"Ошибка: {e}")
            break

        X_next = X + delta_X

        error = np.max(np.abs(X_next - X))
        errors.append(error)

        if error < eps:
            return X_next, errors, k + 1
            
        X = X_next

    return X, errors, max_iter


def phi(X, a):
    x1, x2 = X
    new_x1 = np.sqrt(max(0.0, a**2 - x2**2))
    new_x2 = np.log(x1 + a)
    return np.array([new_x1, new_x2])

def simple_iteration_method(X0, a, eps, max_iter=1000):
    X = np.copy(X0)
    errors = []

    for k in range(max_iter):
        X_next = phi(X, a)

        error = np.max(np.abs(X_next - X))
        errors.append(error)

        if error < eps:
            return X_next, errors, k + 1

        X = X_next

    return X, errors, max_iter


if __name__ == "__main__":
    try:
        a, eps, X0 = read_input("input.txt")
    except FileNotFoundError:
        print("Файл input.txt не найден")
        exit()

    print(f"Параметр a = {a}")
    print(f"Требуемая точность eps = {eps}")
    print(f"Начальное приближение X0 = {X0}\n---")

    root_newton, errors_newton, iters_newton = newton_method(X0, a, eps)
    print(f"МЕТОД НЬЮТОНА")
    print(f"Найдено решение: {root_newton}")
    print(f"Количество итераций: {iters_newton}")
    print(f"Финальная погрешность: {errors_newton[-1] if errors_newton else 'N/A'}")

    root_mpi, errors_mpi, iters_mpi = simple_iteration_method(X0, a, eps)
    print(f"\nМЕТОД ПРОСТОЙ ИТЕРАЦИИ")
    print(f"Найденное решение: {root_mpi}")
    print(f"Количество итераций: {iters_mpi}")
    print(f"Финальная погрешность: {errors_mpi[-1] if errors_mpi else 'N/A'}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors_newton) + 1), errors_newton, 'ro-', label='Метод Ньютона', linewidth=2)
    plt.plot(range(1, len(errors_mpi) + 1), errors_mpi, 'bs-', label='Метод простой итерации', linewidth=2)
    
    plt.yscale('log')
    plt.xlabel('Номер итерации (k)', fontsize=12)
    plt.ylabel('Погрешность (макс. модуль разности)', fontsize=12)
    plt.title('Зависимость погрешности от количества итераций', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=12)
    plt.show()
