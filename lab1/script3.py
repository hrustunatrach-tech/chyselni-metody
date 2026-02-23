import requests
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. ОТРИМАННЯ ДАНИХ (Пункт 1-2)
# ==========================================

def get_elevation_data():
    """Отримання даних про висоту через Open-Elevation API """
    locations = (
        "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|"
        "48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|"
        "48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
        "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|"
        "48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|"
        "48.160250,24.500106"
    )
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
    try:
        response = requests.get(url, timeout=10)
        return response.json()["results"]
    except Exception as e:
        print(f"API Error: {e}. Використовуємо локальний набір даних.")
        return []


def haversine(lat1, lon1, lat2, lon2):
    """Обчислення відстані за формулою гаверсинуса  """
    R = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ==========================================
# 2. МАТЕМАТИЧНИЙ АПАРАТ (Пункт 6-9)
# ==========================================

def solve_spline_coefficients(x, y):
    """Знаходження коефіцієнтів методом прогонки """
    n = len(x) - 1
    h = np.diff(x)
    alpha, beta, gamma, delta = np.zeros(n + 1), np.ones(n + 1), np.zeros(n + 1), np.zeros(n + 1)

    for i in range(1, n):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    A, B = np.zeros(n + 1), np.zeros(n + 1)
    for i in range(1, n + 1):
        m = alpha[i] * A[i - 1] + beta[i]
        A[i], B[i] = -gamma[i] / m, (delta[i] - alpha[i] * B[i - 1]) / m

    c = np.zeros(n + 1)
    c[n] = B[n]
    for i in range(n - 1, -1, -1):
        c[i] = A[i] * c[i + 1] + B[i]

    a_coeffs = y[:-1]
    d_coeffs = np.diff(c) / (3 * h)
    b_coeffs = (np.diff(y) / h) - (h / 3) * (c[1:] + 2 * c[:-1])
    return a_coeffs, b_coeffs, c[:-1], d_coeffs


def get_spline_value(x_nodes, coeffs, xi):
    """Обчислення значення сплайна S(x) в точці xi  """
    a, b, c, d = coeffs
    idx = np.searchsorted(x_nodes, xi) - 1
    idx = max(0, min(idx, len(a) - 1))
    dx = xi - x_nodes[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3


# ==========================================
# 3. ВИКОНАННЯ ТА ВІЗУАЛІЗАЦІЯ (Пункт 10-12)
# ==========================================

def main():
    results = get_elevation_data()
    if not results: return

    # Еталон (21 вузол)
    lats = [p['latitude'] for p in results]
    lons = [p['longitude'] for p in results]
    elevs = [p['elevation'] for p in results]

    dist = [0.0]
    for i in range(1, len(results)):
        dist.append(dist[-1] + haversine(lats[i - 1], lons[i - 1], lats[i], lons[i]))

    x_ref, y_ref = np.array(dist), np.array(elevs)
    coeffs_ref = solve_spline_coefficients(x_ref, y_ref)

    # 1. Табуляція вузлів [cite: 194, 196]
    print(f"Кількість вузлів: {len(results)}")
    print("\nТабуляція вузлів (Latitude | Longitude | Elevation):")
    for i, p in enumerate(results):
        print(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}")

    # 2. Табуляція відстань/висота
    print("\nТабуляція (Відстань | Висота):")
    for i in range(len(x_ref)):
        print(f"{i:2d} | {x_ref[i]:10.2f} | {y_ref[i]:8.2f}")

    # 3. Аналіз похибок
    test_counts = [10, 15, 20]
    x_smooth = np.linspace(x_ref[0], x_ref[-1], 500)
    y_ref_smooth = np.array([get_spline_value(x_ref, coeffs_ref, xi) for xi in x_smooth])

    plt.figure(1, figsize=(10, 6))  # Графік профілів
    plt.plot(x_smooth, y_ref_smooth, label='21 вузол (еталон)', linewidth=2, color='tab:blue')

    plt.figure(2, figsize=(10, 6))  # Графік похибок

    for count in test_counts:
        indices = np.linspace(0, len(results) - 1, count, dtype=int)
        x_n, y_n = x_ref[indices], y_ref[indices]
        coeffs = solve_spline_coefficients(x_n, y_n)

        y_s = np.array([get_spline_value(x_n, coeffs, xi) for xi in x_smooth])
        errors = np.abs(y_s - y_ref_smooth)

        print(f"\n{count} вузлів")
        print(f"Максимальна похибка: {np.max(errors)}")
        print(f"Середня похибка: {np.mean(errors)}")

        plt.figure(1);
        plt.plot(x_smooth, y_s, label=f'{count} вузлів')
        plt.figure(2);
        plt.plot(x_smooth, errors, label=f'{count} вузлів')

    # Оформлення
    plt.figure(1)
    plt.title("Вплив кількості вузлів");
    plt.legend();
    plt.grid(True)

    plt.figure(2)
    plt.title("Похибка апроксимації");
    plt.legend();
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()