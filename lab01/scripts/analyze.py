import math
import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = os.path.join("results", "raw_measurements.csv")
OUTPUT_STATS = os.path.join("results", "statistics.csv")


def remove_outliers(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)

    if std == 0 or math.isnan(std):
        return series.copy()

    mask = (series - mean).abs() <= 3 * std
    return series[mask]


def compute_stats(series: pd.Series):
    cleaned = remove_outliers(series)

    count_before = len(series)
    count_after = len(cleaned)
    min_value = cleaned.min()
    mean_value = cleaned.mean()
    std_value = cleaned.std(ddof=0)

    if count_after > 0 and not math.isnan(std_value):
        confidence_interval = 1.96 * std_value / math.sqrt(count_after)
    else:
        confidence_interval = float("nan")

    return {
        "count_before": count_before,
        "count_after": count_after,
        "min": min_value,
        "mean": mean_value,
        "std": std_value,
        "confidence_interval": confidence_interval,
    }


def save_histogram(series: pd.Series, title: str, xlabel: str, filename: str):
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"File not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    methods = {
        "gettick_ms": {
            "title": "Histogram of GetTickCount64 measurements",
            "xlabel": "Time (ms)",
            "file": os.path.join("results", "histogram_gettick.png"),
        },
        "qpc_ms": {
            "title": "Histogram of QueryPerformanceCounter measurements",
            "xlabel": "Time (ms)",
            "file": os.path.join("results", "histogram_qpc.png"),
        },
        "rdtsc_ticks": {
            "title": "Histogram of RDTSC measurements",
            "xlabel": "Ticks",
            "file": os.path.join("results", "histogram_rdtsc.png"),
        },
    }

    stats_rows = []

    for column, meta in methods.items():
        series = df[column].dropna()
        stats = compute_stats(series)

        stats_rows.append({
            "method": column,
            **stats
        })

        save_histogram(
            series=series,
            title=meta["title"],
            xlabel=meta["xlabel"],
            filename=meta["file"]
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUTPUT_STATS, index=False)

    print("Analysis completed.")
    print(f"Statistics saved to {OUTPUT_STATS}")
    print("Histograms saved to results/")


if __name__ == "__main__":
    main()