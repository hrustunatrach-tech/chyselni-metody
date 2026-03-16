import numpy as np
import matplotlib.pyplot as plt
import csv
import os


# --- 1. ПІДГОТОВКА ДАНИХ (Пункт 1) ---
def prepare_data(filename):
    """Створює файл з даними про температуру (табуляція)"""
    # Дані з методички [cite: 101-124]
    months = np.arange(1, 25)
    temps = np.array([-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0,
                      -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3])
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Month', 'Temp'])
        for m, t in zip(months, temps):
            writer.writerow([m, t])


def read_data(filename):
    """Зчитування вхідних даних (Пункт 2)"""
    x, f = [], []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Month']))
            f.append(float(row['Temp']))
    return np.array(x), np.array(f)


# --- 2. МАТЕМАТИЧНИЙ АПАРАТ (Пункт 2) ---
def form_arrays(x, f, m):
    """Формування матриці B та вектора C """
    B = np.zeros((m + 1, m + 1))
    C = np.zeros(m + 1)
    for k in range(m + 1):
        for l in range(m + 1):
            B[k, l] = np.sum(x ** (k + l))
        C[k] = np.sum(f * (x ** k))
    return B, C


def gauss_solve(B, C):
    """Метод Гаусса з вибором головного елемента по стовпцю [cite: 35, 44]"""
    n = len(C)
    A = B.copy().astype(float)
    b = C.copy().astype(float)

    for k in range(n):
        # Вибір головного елемента [cite: 44, 46]
        max_idx = np.argmax(np.abs(A[k:, k])) + k
        A[[k, max_idx]] = A[[max_idx, k]]
        b[[k, max_idx]] = b[[max_idx, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотній хід [cite: 70]
    coeffs = np.zeros(n)
    for i in range(n - 1, -1, -1):
        coeffs[i] = (b[i] - np.dot(A[i, i + 1:], coeffs[i + 1:])) / A[i, i]
    return coeffs


def poly_val(x, coeffs):
    """Обчислення многочлена [cite: 7]"""
    res = np.zeros_like(x, dtype=float)
    for i, a in enumerate(coeffs):
        res += a * (x ** i)
    return res


# --- 3. ОБЧИСЛЕННЯ (Пункт 3) ---
filename = 'temp_data.csv'
prepare_data(filename)
x_data, f_data = read_data(filename)

variances = []
coeffs_storage = []

print(f"{'m':<5} | {'Дисперсія (δ)':<15}")
print("-" * 25)

for m in range(1, 11):
    B, C = form_arrays(x_data, f_data, m)
    coeffs = gauss_solve(B, C)
    f_approx = poly_val(x_data, coeffs)
    # Обчислення дисперсії [cite: 30]
    delta = np.sqrt(np.mean((f_data - f_approx) ** 2))
    variances.append(delta)
    coeffs_storage.append(coeffs)
    print(f"{m:<5} | {delta:<15.4f}")

# Оптимальне m за мінімумом дисперсії
opt_idx = np.argmin(variances)
opt_m = opt_idx + 1
opt_coeffs = coeffs_storage[opt_idx]

print(f"\nОптимальний степінь: m={opt_m}")

# Прогноз на 3 місяці
x_future = np.array([25, 26, 27])
f_future = poly_val(x_future, opt_coeffs)
print(f"Прогноз на 25-27 місяці: {np.round(f_future, 2)}")

# --- 4. ВІЗУАЛІЗАЦІЯ (Пункт 3, 4, 5) ---
plt.figure(figsize=(12, 12))

# Графік 1: Апроксимація та Прогноз
plt.subplot(3, 1, 1)
plt.scatter(x_data, f_data, color='red', label='Фактичні дані')
x_smooth = np.linspace(1, 27, 200)
plt.plot(x_smooth, poly_val(x_smooth, opt_coeffs), 'b-', label=f'Апроксимація (m={opt_m})')
plt.plot(x_future, f_future, 'go--', label='Прогноз')
plt.title('Апроксимація та прогноз температури')
plt.grid(True);
plt.legend()

# Графік 2: Дисперсія
plt.subplot(3, 1, 2)
plt.plot(range(1, 11), variances, 'bo-')
plt.axvline(opt_m, color='green', linestyle='--', label=f'Оптимум m={opt_m}')
plt.title('Залежність дисперсії від степеня m')
plt.xlabel('Степінь m');
plt.ylabel('Дисперсія δ')
plt.grid(True);
plt.legend()

# Графік 3: Похибка (Пункт 4)
error = np.abs(f_data - poly_val(x_data, opt_coeffs))
plt.subplot(3, 1, 3)
plt.bar(x_data, error, color='orange', alpha=0.7, label='|f(x) - φ(x)|')
plt.title('Похибка апроксимації у вузлах')
plt.xlabel('Місяць');
plt.ylabel('Абсолютна похибка')
plt.grid(True);
plt.legend()

plt.tight_layout()
plt.show()