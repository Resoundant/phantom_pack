import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import combinations
from time import time

def similar_radius(radii, tol=0.05):
    return max(radii) - min(radii) <= tol * max(radii)

def is_colinear(points, tol):
    pts = np.array(points, dtype=float)
    pts -= pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts)
    return s[1] / s[0] < tol

def is_uniform_spacing(points, tol):
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    dists = [np.linalg.norm(np.array(sorted_pts[i]) - np.array(sorted_pts[i+1])) for i in range(len(points) - 1)]
    mean_d = np.mean(dists)
    return all(abs(d - mean_d) < tol * mean_d for d in dists)

def is_valid_group(group, radius_tol, linear_tol, spacing_tol):
    centers = [(x, y) for (x, y, _) in group]
    radii = [r for (_, _, r) in group]
    return similar_radius(radii, radius_tol) and is_colinear(centers, linear_tol) and is_uniform_spacing(centers, spacing_tol)

def find_valid_circles(circles, radius_tol=0.05, linear_tol=0.05, spacing_tol=0.05):
    results = []
    start = time()
    print("computing")
    for group in combinations(circles, 5):
        if is_valid_group(group, radius_tol, linear_tol, spacing_tol):
            results.append(group)
    end = time()
    print("done in", end - start, "sec")
    return results


# === Test Dataset Generation ===

def plot_circles(circles, title="Circles Visualization"):
    fig, ax = plt.subplots()
    for idx, (x, y, r) in enumerate(circles):
        circle = plt.Circle((x, y), r, fill=False, edgecolor='b')
        ax.add_patch(circle)
        ax.text(x, y, str(idx), fontsize=12, ha='center', va='center', color='r')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.title(title)
    plt.grid(True)
    plt.show()

def generate_test_data():
    data = []

    # Valid aligned group
    base_x, base_y = 10, 20
    spacing = 10
    radius = 3
    
    data.extend([
        (20 + 10.5, 49.5, 3.1),
        (20 + 20, 50, 3),
        (20 + 29.5, 50.5, 3.05),
        (20 + 39.8, 49.5, 2.95),
        (20 + 50.1, 50, 3),
    ])
    
    # Random noise circles
    for _ in range(10):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        r = random.uniform(2, 6)
        data.append((x, y, r))
        
    # for i in range(5):
    #     data.append((base_x + i * spacing, base_y, radius))
        
    return data


# === Run the Test ===

def main():
    test_circles = generate_test_data()
    valid_groups = find_valid_circles(test_circles)

    # Display all circles
    plot_circles(test_circles, "Test Circles with Labels")

    # Print results
    circle_idx_map = {id(c): i for i, c in enumerate(test_circles)}
    valid_group_indices = [[circle_idx_map[id(c)] for c in group] for group in valid_groups]

    if valid_group_indices:
        print("✅ Valid groups of 5 similar, evenly spaced, aligned circles:")
        for group in valid_group_indices:
            print("Group indices:", group)
    else:
        print("❌ No valid groups found.")

if __name__ == "__main__":
    main()