import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


INPUT_FILE = "results/results.csv"
OUTPUT_DIR = "results"


def read_results(filename):
    grouped = defaultdict(list)

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row["mode"]
            kb = float(row["kb"])
            ns = float(row["ns_per_iteration"])
            grouped[mode].append((kb, ns))

    for mode in grouped:
        grouped[mode].sort(key=lambda x: x[0])

    return grouped


def save_plot(mode, points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker=".", linewidth=1)
    plt.xlabel("Размер данных, КБ")
    plt.ylabel("Время одной итерации, нс")
    plt.title(f"Latency test: {mode}")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{mode}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Сохранен график: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Файл не найден: {INPUT_FILE}")

    grouped = read_results(INPUT_FILE)

    for mode, points in grouped.items():
        save_plot(mode, points)


if __name__ == "__main__":
    main()