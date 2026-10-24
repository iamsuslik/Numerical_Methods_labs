import math
import os

def load_input(filename="input.txt"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден! Создайте его рядом со скриптом.")
    
    data = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=")
                data[key.strip()] = float(val.strip())
                
    epsilon = data.get("epsilon", 1e-6)
    x0 = data.get("x0", 0.5)
    max_iter = int(data.get("max_iter", 100))
    return epsilon, x0, max_iter


def f(x):
    return x**3 + x**2 - 2*x - 1

def phi(x):
    val = 1 + 2*x - x**2
    if val >= 0:
        return val ** (1/3)
    else:
        return -((-val) ** (1/3))

def df(x):
    return 3*(x**2) + 2*x - 2


def simple_iteration_method(x0, epsilon, max_iter):
    history = []
    x_current = x0
    
    for i in range(1, max_iter + 1):
        x_next = phi(x_current)
        error = abs(x_next - x_current)
        
        history.append({
            "iteration": i,
            "x": x_next,
            "error": error,
            "f_val": f(x_next)
        })
        
        if error < epsilon:
            return x_next, history
        
        x_current = x_next
        
    raise RuntimeError("МПИ не сошелся за максимальное число итераций.")


def newton_method(x0, epsilon, max_iter):
    history = []
    x_current = x0
    
    for i in range(1, max_iter + 1):
        f_val = f(x_current)
        df_val = df(x_current)
        
        if df_val == 0:
            raise ZeroDivisionError("Производная равна нулю! Метод Ньютона остановлен.")
            
        x_next = x_current - (f_val / df_val)
        error = abs(x_next - x_current)
        
        history.append({
            "iteration": i,
            "x": x_next,
            "error": error,
            "f_val": f_val
        })
        
        if error < epsilon:
            return x_next, history
            
        x_current = x_next
        
    raise RuntimeError("Метод Ньютона не сошелся за максимальное число итераций.")


if __name__ == "__main__":
    try:
        eps, x_start, max_i = load_input("input.txt")

        print(f"Точность вычислений (epsilon): {eps}")
        print(f"Начальное приближение (x0): {x_start}")
        print(f"Максимальное число итераций: {max_i}\n")
        
        root_mpi, history_mpi = simple_iteration_method(x_start, eps, max_i)
        print("Метод простой итерации (МПИ):")
        print(f"Найденный корень: {root_mpi:.7f}")
        print(f"Количество итераций: {len(history_mpi)}")
        
        root_newton, history_newton = newton_method(x_start, eps, max_i)
        print("\nМетод Ньютона:")
        print(f"Найденный корень: {root_newton:.7f}")
        print(f"Количество итераций: {len(history_newton)}")
        
        try:
            import matplotlib.pyplot as plt
            
            iters_mpi = [h["iteration"] for h in history_mpi]
            errors_mpi = [h["error"] for h in history_mpi]
            
            iters_newton = [h["iteration"] for h in history_newton]
            errors_newton = [h["error"] for h in history_newton]
            
            plt.figure(figsize=(10, 6))
            plt.plot(iters_mpi, errors_mpi, marker='o', label='МПИ (Простая итерация)', color='blue')
            plt.plot(iters_newton, errors_newton, marker='s', label='Метод Ньютона', color='red')
            
            plt.yscale('log')
            plt.xlabel('Номер итерации (n)', fontsize=12)
            plt.ylabel('Погрешность (|x_n+1 - x_n|)', fontsize=12)
            plt.title('Зависимость погрешности от количества итераций', fontsize=14)
            plt.grid(True, which="both", ls="--")
            plt.legend(fontsize=12)
            
            plt.savefig("convergence_analysis.png", dpi=300)
            print("\nГрафик зависимости погрешности сохранен в файл 'convergence_analysis.png'.")
            
            plt.show()
            
        except ImportError:
            print("\nБиблиотека matplotlib не найдена.")

    except Exception as e:
        print(f"\n[ОШИБКА]: {e}")
