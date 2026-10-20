import math

def read_data(filename="input.txt"):
    with open(filename, 'r') as f:
        lines = f.readlines()

    epsilon = float(lines[0].strip())

    A = []
    for line in lines[1:]:
        if line.strip():
            row = [float(x) for x in line.split()]
            A.append(row)
            
    return epsilon, A

def create_identity_matrix(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def matrix_multiply(A, B):
    n_A = len(A)
    m_A = len(A[0])
    n_B = len(B)
    m_B = len(B[0])
    
    if m_A != n_B:
        raise ValueError("Матрицы нельзя перемножить: несовпадение размерностей.")
        
    result = [[0.0] * m_B for _ in range(n_A)]
    for i in range(n_A):
        for j in range(m_B):
            for k in range(m_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def transpose(A):
    n = len(A)
    m = len(A[0])
    return [[A[j][i] for j in range(n)] for i in range(m)]

def print_matrix(name, matrix):
    print(f"{name}:")
    for row in matrix:
        formatted_row = [f"{elem:8.4f}" for elem in row]
        print(" [" + ", ".join(formatted_row) + "]")
    print()

def find_max_off_diagonal(A):
    n = len(A)
    max_val = 0.0
    max_i = 0
    max_j = 1

    for i in range(n - 1):
        for j in range(i + 1, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                max_i = i
                max_j = j
                
    return max_i, max_j, max_val

def jacobi_method(A, epsilon):
    n = len(A)

    V = create_identity_matrix(n)

    A_curr = [[A[i][j] for j in range(n)] for i in range(n)]

    iteration = 0
    while True:
        i, j, max_val = find_max_off_diagonal(A_curr)

        if max_val < epsilon:
            break

        if A_curr[i][i] == A_curr[j][j]:
            phi = math.pi / 4.0 
        else:
            phi = 0.5 * math.atan((2.0 * A_curr[i][j]) / (A_curr[i][i] - A_curr[j][j]))
            
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        U = create_identity_matrix(n)
        U[i][i] = cos_phi
        U[j][j] = cos_phi
        U[i][j] = -sin_phi
        U[j][i] = sin_phi

        U_T = transpose(U)
        A_temp = matrix_multiply(U_T, A_curr)
        A_curr = matrix_multiply(A_temp, U)

        V = matrix_multiply(V, U)
        
        iteration += 1

    print(f"\nАлгоритм сошелся за {iteration} итераций.\n")
    return A_curr, V

# ПРОВЕРКА A * V = V * Lambda 
def verify_result(A_orig, V, Lambda_matrix, epsilon):
    print("ПРОВЕРКА РЕЗУЛЬТАТА: A * V = V * \u039B (Лямбда)")

    left_side = matrix_multiply(A_orig, V)
    right_side = matrix_multiply(V, Lambda_matrix)

    print_matrix("Левая часть (A * V)", left_side)
    print_matrix("Правая часть (V * \u039B)", right_side)

    n = len(A_orig)
    is_correct = True

    tolerance = epsilon 
    
    for i in range(n):
        for j in range(n):
            if abs(left_side[i][j] - right_side[i][j]) > tolerance:
                is_correct = False
                
    if is_correct:
        print(f"Проверка пройдена. Матрицы совпадают с допустимой погрешностью (\u03b5 = {tolerance}).")
    else:
        print("Ошибка. Матрицы не совпадают!")

if __name__ == "__main__":
    try:
        epsilon, A_orig = read_data("data.txt")
        n = len(A_orig)

        print(f"Заданная точность вычислений (\u03b5): {epsilon}")
        print_matrix("Исходная матрица A", A_orig)

        Lambda_matrix, V_matrix = jacobi_method(A_orig, epsilon)

        eigenvalues = [Lambda_matrix[i][i] for i in range(n)]

        print("Найденные собственные значения (\u03bb):")
        for i, val in enumerate(eigenvalues):
            print(f" \u03bb_{i+1} = {val:.4f}")
        print()
        
        print_matrix("Матрица собственных векторов V", V_matrix)

        print("Найденные собственные векторы (h):")
        for j in range(n):
            vector = [V_matrix[i][j] for i in range(n)]
            formatted_vec = [f"{x:8.4f}" for x in vector]
            print(f" h_{j+1} = [{', '.join(formatted_vec)}]^T")
        print()

        Clean_Lambda = create_identity_matrix(n)
        for i in range(n):
            Clean_Lambda[i][i] = eigenvalues[i]
            
        verify_result(A_orig, V_matrix, Clean_Lambda, epsilon)

    except FileNotFoundError:
        print("Ошибка: Файл 'data.txt' не найден. Пожалуйста, создайте его.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
