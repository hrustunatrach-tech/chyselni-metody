import requests
import numpy as np
import matplotlib.pyplot as plt


def get_elevation_data():
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
        response = requests.get(url)
        return response.json()["results"]
    except Exception as e:
        print(f"Помилка API: {e}")
        return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def build_cubic_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)

    alpha = np.zeros(n + 1)
    beta = np.ones(n + 1)
    gamma = np.zeros(n + 1)
    delta = np.zeros(n + 1)


    for i in range(1, n):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])


    A = np.zeros(n + 1)
    B = np.zeros(n + 1)
    for i in range(1, n + 1):
        m = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / m
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / m


    c = np.zeros(n + 1)
    c[n] = B[n]
    for i in range(n - 1, -1, -1):
        c[i] = A[i] * c[i + 1] + B[i]


    a_coeffs = y[:-1]
    d_coeffs = np.diff(c) / (3 * h)
    b_coeffs = (np.diff(y) / h) - (h / 3) * (c[1:] + 2 * c[:-1])

    return a_coeffs, b_coeffs, c[:-1], d_coeffs




print("ПУНКТ ОСТАННІЙ")



def main():
    results = get_elevation_data()
    if not results: return


    lats = [p['latitude'] for p in results]
    lons = [p['longitude'] for p in results]
    elevs = [p['elevation'] for p in results]

    dist = [0]
    for i in range(1, len(results)):
        d = haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
        dist.append(dist[-1] + d)

    x_nodes = np.array(dist)
    y_nodes = np.array(elevs)


    a, b, c, d = build_cubic_spline(x_nodes, y_nodes)


    x_smooth = np.linspace(x_nodes[0], x_nodes[-1], 300)
    y_smooth = []
    for xi in x_smooth:

        idx = np.searchsorted(x_nodes, xi) - 1
        idx = max(0, min(idx, len(a) - 1))
        dx = xi - x_nodes[idx]

        val = a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3
        y_smooth.append(val)


    print(f"Загальна відстань: {x_nodes[-1]:.2f} м")
    ascent = sum(max(y_nodes[i] - y_nodes[i - 1], 0) for i in range(1, len(y_nodes)))
    print(f"Сумарний підйом: {ascent:.2f} м")
    print(f"Механічна робота (80кг): {80 * 9.81 * ascent / 1000:.2f} кДж")

    # Візуалізація [cite: 138, 145]

    plt.figure(figsize=(10, 6))
    plt.plot(x_nodes, y_nodes, 'ro', label='GPS вузли (Табуляція)')
    plt.plot(x_smooth, y_smooth, 'b-', label='Кубічний сплайн (Гладкий профіль)')
    plt.fill_between(x_smooth, min(y_smooth) - 10, y_smooth, color='green', alpha=0.1)
    plt.title("Профіль висоти: Станція Заросляк - Гора Говерла")
    plt.xlabel("Кумулятивна відстань (метри)")
    plt.ylabel("Висота над рівнем моря (метри)")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()


if __name__ == "__main__":
    main()