import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os
from scipy.interpolate import CubicSpline

# 1. Вхідні дані Варіанту 3 [cite: 247-248, 250]
x_nodes = np.array([10000, 20000, 40000, 80000, 160000], dtype=float)
y_nodes = np.array([8, 20, 55, 150, 420], dtype=float)
target_x = 120000

# Еталонна функція для створення гладких дуг похибок
true_func = CubicSpline(x_nodes, y_nodes)


# 2. Функції обчислень [cite: 7, 10, 12, 55]
def get_divided_diff_table(x, y):
    n = len(y)
    table = np.zeros([n, n])
    table[:, 0] = y  # Розділена різниця 0-го порядку [cite: 6, 13]
    for j in range(1, n):
        for i in range(n - j):
            # Рекурентна формула розділених різниць [cite: 10, 12, 15]
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
    return table


def newton_poly(x_n, coef, x_v):
    # Обчислення значення многочлена Ньютона [cite: 54, 55, 159]
    res = coef[0]
    product = 1.0
    for k in range(1, len(coef)):
        product *= (x_v - x_n[k - 1])
        res += coef[k] * product
    return res


def factorial_interp(y_n, x_n, x_v):
    # Обчислення через факторіальні многочлени [cite: 133, 149, 150]
    h = x_n[1] - x_n[0]
    t = (x_v - x_n[0]) / h
    n = len(y_n)
    diffs = np.zeros((n, n))
    diffs[:, 0] = y_n
    for j in range(1, n):
        for i in range(n - j):
            diffs[i, j] = diffs[i + 1, j - 1] - diffs[i, j - 1]
    res = diffs[0, 0]
    t_p, f_p = 1.0, 1.0
    for k in range(1, n):
        t_p *= (t - k + 1)
        f_p *= math.factorial(k)
        res += (diffs[0, k] / f_p) * t_p
    return res


# 3. Виконання розрахунків [cite: 156, 159]
diff_table = get_divided_diff_table(x_nodes, y_nodes)
coefs = diff_table[0, :]  # Коефіцієнти для многочлена Ньютона
pred_newton = newton_poly(x_nodes, coefs, target_x)
pred_fact = factorial_interp(y_nodes, x_nodes, target_x)

# 4. Запис результатів у файл
with open("calculations.txt", "w", encoding="utf-8") as f:
    f.write("РЕЗУЛЬТАТИ ЛАБОРАТОРНОЇ РОБОТИ №2 (ВАРІАНТ 3)\n")
    f.write("=" * 45 + "\n")
    f.write(f"Вузли X: {x_nodes.tolist()}\n")
    f.write(f"Вузли Y: {y_nodes.tolist()}\n\n")
    f.write("ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ:\n")
    # Виводимо тільки верхній трикутник таблиці [cite: 310]
    for i in range(len(x_nodes)):
        row = [f"{diff_table[i, j]:.4e}" for j in range(len(x_nodes) - i)]
        f.write("\t".join(row) + "\n")
    f.write(f"\nПРОГНОЗ ДЛЯ X = {target_x}:\n")
    f.write(f"Метод Ньютона: {pred_newton:.4f} сек\n")
    f.write(f"Факторіальний метод: {pred_fact:.4f} сек\n")

# 5. Візуалізація результатів [cite: 161]
x_plot = np.linspace(min(x_nodes), max(x_nodes), 1000)
y_true = true_func(x_plot)

# Графік 1: Функція та точка прогнозу
plt.figure(1, figsize=(10, 6))
y_newton_plot = [newton_poly(x_nodes, coefs, xi) for xi in x_plot]
plt.plot(x_plot, y_newton_plot, 'b-', label='Поліном Ньютона $N_5(x)$')
plt.scatter(x_nodes, y_nodes, color='red', label='Дані (вузли)')
plt.scatter(target_x, pred_newton, color='green', marker='X', s=200, label=f'Прогноз ({target_x})')
plt.annotate(f'{pred_newton:.2f} s', (target_x, pred_newton), xytext=(0, 15),
             textcoords="offset points", ha='center', fontweight='bold', color='green')
plt.title('Інтерполяційний многочлен Ньютона')
plt.xlabel('Розмір датасету')
plt.ylabel('Час (сек)')
plt.legend()
plt.grid(True, ls='--')

# Графік 2: Аналіз похибок (гладкі дуги) [cite: 259, 301]
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.4)
n_list = [5, 10, 20]
colors = ['tab:blue', 'tab:green', 'tab:red']

for i, (n, col) in enumerate(zip(n_list, colors)):
    x_n = np.linspace(min(x_nodes), max(x_nodes), n)
    y_n = true_func(x_n)
    c_n = get_divided_diff_table(x_n, y_n)[0, :]
    y_interp = np.array([newton_poly(x_n, c_n, xi) for xi in x_plot])

    error = np.abs(y_true - y_interp)
    axs[i].plot(x_plot, error, color=col, lw=2)
    axs[i].set_title(f'Абсолютна похибка для n = {n} вузлів')
    axs[i].set_ylabel('Похибка')
    axs[i].grid(True, ls='--')

plt.show()

print(f"Розрахунок завершено. Файл 'calculations.txt' створено.")
print(f"Передбачення для {target_x}: {pred_newton:.2f} сек")