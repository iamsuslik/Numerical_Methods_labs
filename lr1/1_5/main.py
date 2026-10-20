import math
import numpy as np


def mat_mul(A, B):
    n = len(A)
    m = len(B[0])
    k_len = len(B)
    res = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(k_len):
                s += A[i, k] * B[k, j]
            res[i, j] = s
    return res


def transpose(A):
    n, m = A.shape
    res = np.zeros((m, n))
    for i in range(n):
        for j in range(m):
            res[j, i] = A[i, j]
    return res


def vector_norm(v):
    return math.sqrt(sum(x**2 for x in v))


def qr_decomposition(A):
    n = len(A)
    R = A.copy().astype(float)
    Q = np.eye(n)

    for k in range(n - 1):
        x = R[k:n, k]
        norm_x = vector_norm(x)

        if norm_x == 0:
            continue

        v = np.zeros(len(x))

        sign = 1.0 if x[0] >= 0 else -1.0
        v[0] = x[0] + sign * norm_x
        for i in range(1, len(x)):
            v[i] = x[i]

        norm_v = vector_norm(v)
        if norm_v == 0:
            continue

        u = v / norm_v

        sub_R = R[k:n, k:]

        u_col = u.reshape(-1, 1)
        u_t_sub_R = mat_mul(transpose(u_col), sub_R)
        R[k:n, k:] -= 2.0 * mat_mul(u_col, u_t_sub_R)

        H_k = np.eye(n)
        H_k[k:n, k:n] -= 2.0 * mat_mul(u_col, transpose(u_col))

        Q = mat_mul(Q, H_k)

    return Q, R


def solve_block_2x2(b11, b12, b21, b22):
    trace = b11 + b22
    det = b11 * b22 - b12 * b21
    discriminant = trace**2 - 4 * det

    if discriminant >= 0:
        l1 = (trace + math.sqrt(discriminant)) / 2.0
        l2 = (trace - math.sqrt(discriminant)) / 2.0
        return l1, l2
    else:
        real_part = trace / 2.0
        imag_part = math.sqrt(-discriminant) / 2.0
        return complex(real_part, imag_part), complex(real_part, -imag_part)


def get_eigenvalues(Ak, eps):
    n = len(Ak)
    eigenvalues = []
    i = 0
    while i < n:
        if i == n - 1:
            eigenvalues.append(Ak[i, i])
            i += 1
        else:
            if abs(Ak[i + 1, i]) < eps:
                eigenvalues.append(Ak[i, i])
                i += 1
            else:
                l1, l2 = solve_block_2x2(
                    Ak[i, i], Ak[i, i + 1], Ak[i + 1, i], Ak[i + 1, i + 1]
                )
                eigenvalues.append(l1)
                eigenvalues.append(l2)
                i += 2
    return eigenvalues


def qr_algorithm(A, eps=0.01, max_iter=2000):
    Ak = A.copy().astype(float)
    n = len(Ak)

    Q0, R0 = qr_decomposition(Ak)

    prev_eigs = None

    for iteration in range(max_iter):
        Q, R = qr_decomposition(Ak)
        Ak = mat_mul(R, Q)

        curr_eigs = get_eigenvalues(Ak, eps)

        if prev_eigs is not None and len(prev_eigs) == len(curr_eigs):
            diff = max(
                abs(c - p)
                for c, p in zip(
                    sorted(curr_eigs, key=lambda x: (x.real, abs(x.imag))),
                    sorted(prev_eigs, key=lambda x: (x.real, abs(x.imag))),
                )
            )
            if diff < eps:
                return curr_eigs, Q0, R0, iteration + 1

        prev_eigs = curr_eigs

    return curr_eigs, Q0, R0, max_iter


def load_matrix_from_file(filename):
    matrix = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                row = [float(x) for x in line.split()]
                matrix.append(row)
    return np.array(matrix, dtype=float)


def print_matrix(mat, name="Matrix"):
    print(f"{name}:")
    for row in mat:
        print("  [" + ", ".join(f"{val + 0.0:>8.4f}" for val in row) + "]")


def run_experiment(filename, title, eps=0.01):
    print(f"  {title.upper()}")

    A = load_matrix_from_file(filename)

    print_matrix(A, "1. Исходная матрица A")

    eigenvalues, Q0, R0, iters = qr_algorithm(A, eps=eps)

    print_matrix(Q0, "\n2. Матрица Q (первая итерация)")
    print_matrix(R0, "Матрица R (первая итерация)")

    QR = mat_mul(Q0, R0)
    print_matrix(QR, "\n3. Результат умножения Q * R (проверка разложения)")

    print(f"\n4. Найденные собственные значения (точность eps={eps}, шагов: {iters}):")
    for idx, val in enumerate(eigenvalues, 1):
        if isinstance(val, complex):
            sign = "+" if val.imag >= 0 else "-"
            print(f" λ_{idx} = {val.real:.4f} {sign} {abs(val.imag):.4f}i")
        else:
            print(f" λ_{idx} = {val:.4f}")

    np_eigs = np.linalg.eigvals(A)
    print("\n[Проверка с np.linalg.eigvals]:")
    for idx, val in enumerate(np_eigs, 1):
        if np.iscomplex(val):
            sign = "+" if val.imag >= 0 else "-"
            print(f"   λ_{idx} = {val.real:.4f} {sign} {abs(val.imag):.4f}i")
        else:
            print(f"   λ_{idx} = {val.real:.4f}")


if __name__ == "__main__":
    run_experiment("matrix_real.txt", "Вариант из задания (матрица 3x3)", eps=0.01)

    run_experiment("matrix_complex.txt", "Собственный пример (комплексные корни)", eps=0.01)
