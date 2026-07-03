import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd


def generate_noisy_points(mu:float=0.0, std_dev:float=0.0) -> List[Tuple[float, float]]:
    if std_dev < 0:
        raise ValueError("std dev must be non-negative")

    points = []
    for x in (0, 10, 20, 30, 40):
        y = x + random.gauss(mu, std_dev)
        points.append((float(x), float(y)))
    return points


def linear_fit(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    if len(points) < 2:
        raise ValueError("need at least two points to fit")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(points)

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in points)
    if ss_xx == 0:
        raise ValueError("cannot fit a line when all x values are identical")

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return slope, intercept, r_squared


def print_linear_fit(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    slope, intercept, r_squared = linear_fit(points)
    print(f"slope={slope:.6f}, intercept={intercept:.6f}, r^2={r_squared:.6f}")
    return slope, intercept, r_squared

def sweep():
    mu_stddev = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
    ]
    excel_file = 'phantom_pack_monte_carlo.xlsx'
    if os.path.exists(excel_file):
        os.remove(excel_file)
    all_results = []
    summaries = []
    for mu, std_dev in mu_stddev:
        results = monte_carlo(mu=mu, std_dev=std_dev)
        print(f"mu {mu}, std_dev {std_dev}")
        print(f"slope:      {results['slope'].mean()} +/- {results['slope'].std()}")
        print(f"intercept:  {results['intercept'].mean()} +/- {results['intercept'].std()}")
        print(f"r^2:        {results['rsquared'].mean()} +/- {results['rsquared'].std()}")
        summary = {
            "mu": mu,
            "std_dev": std_dev,
            "iterations": results['slope'].count(),
            "slope": results['slope'].mean(),
            "slope_stddev": results['slope'].std(),
            "slope_stderr": results['slope'].std()/np.sqrt(results['slope'].count()),
            "intercept": results['intercept'].mean(),
            "intercept_stddev": results['intercept'].std(),
            "intercept_stderr": results['intercept'].std()/np.sqrt(results['intercept'].count()),
            "r^2": results['rsquared'].mean(),
            "r^2_stddev": results['rsquared'].std(),
            "r^2_stderr": results['rsquared'].std()/np.sqrt(results['rsquared'].count()),
        }
        all_results.append((f"mu{mu}_stddev{std_dev}", results))
        summaries.append(summary)

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        sum_df = pd.DataFrame(summaries)
        sum_df.to_excel(writer, sheet_name="summary", index=False)
        for ws_name, df in all_results:
            df.to_excel(writer, sheet_name=ws_name, index=False)


def monte_carlo(mu=0.0, std_dev=0.0, iterations=1000) -> pd.DataFrame:
    results = []
    for _ in range(iterations):
        points = generate_noisy_points(mu=mu, std_dev=std_dev)
        slope, intercept, rsquared = linear_fit(points)
        result = {
            "mu": mu,
            "std_dev": std_dev,
            "0": points[0][1],
            "10": points[1][1],
            "20": points[2][1],
            "30": points[3][1],
            "40": points[4][1],
            "slope": slope,
            "intercept": intercept,
            "rsquared": rsquared,
        }
        results.append(result)
    return pd.DataFrame(results)


def main(std_dev=0.0, mu=0.0) -> None:
    points = generate_noisy_points(mu=mu, std_dev=std_dev)
    print("points:")
    for x, y in points:
        print(f"  x={x:.1f}, y={y:.6f}")
    print_linear_fit(points)


def low_high(offset=5) -> list[tuple[float,float]]:
    return [
        (0,   0-offset),
        (10, 10+offset),
        (20, 20-offset),
        (30, 30+offset),
        (40, 40-offset)
    ]
    

if __name__ == "__main__":
    # mu = 0
    # std_dev = 2
    # results = monte_carlo(mu=mu, std_dev=std_dev)
    # print(f"results: mu {mu}, std_dev {std_dev}")
    # print(f"mean:    {results.mean(axis=0)}")
    # print(f"stddev:  {results.std(axis=0)}")
    # print(f"stderr:  {results.std(axis=0)/len(results)}")

    # points = low_high(5)
    # slope, intercept, rsquared = linear_fit(points)
    # print(f"slope {slope}, intercept {intercept}, r^2 {rsquared}")

    # points = low_high(-3)
    # slope, intercept, rsquared = linear_fit(points)
    # print(f"slope {slope}, intercept {intercept}, r^2 {rsquared}")

    sweep()