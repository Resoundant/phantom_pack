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

def find_circle_groups(circles, radius_tol=0.2, linear_tol=0.1, spacing_tol=0.1):
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
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    plt.title(title)
    plt.grid(True)
    plt.show()

def generate_test_data():
    data = []
    img_size = 256

    # Valid aligned group
    radius = 7
    spacing = 2.5*radius
    base_x, base_y = int(img_size/2-2*spacing), 60
    radius_range_px = 1
    loc_range_px = 1

    for i in range(5):
        data.extend([
            (base_x + i * spacing + random.uniform(-loc_range_px, loc_range_px),
            base_y + random.uniform(-loc_range_px, loc_range_px), 
            radius + random.uniform(-radius_range_px, radius_range_px)),
        ])

    # Valid non-aligned group
    # data.extend([
    #     (20 + 10.5, 49.5, 3.1),
    #     (20 + 20, 50, 3),
    #     (20 + 29.5, 50.5, 3.05),
    #     (20 + 39.8, 49.5, 2.95),
    #     (20 + 50.1, 50, 3),
    # ])
    
    # Random noise circles
    for _ in range(10):
        x = random.uniform(0, img_size)
        y = random.uniform(0, img_size)
        r = random.uniform(4, 20)
        data.append((x, y, r))
        
    # for i in range(5):
    #     data.append((base_x + i * spacing, base_y, radius))
        
    return data

def randomize_data(data):
    random.shuffle(data)
    return data


# === Run the Test ===

def main():
    test_circles = generate_test_data()
    # randomize_data(test_circles)
    valid_groups = find_circle_groups(test_circles)

    # Print results
    circle_idx_map = {id(c): i for i, c in enumerate(test_circles)}
    valid_group_indices = [[circle_idx_map[id(c)] for c in group] for group in valid_groups]

    if valid_group_indices:
        print("✅ Valid groups of 5 similar, evenly spaced, aligned circles:")
        for group in valid_group_indices:
            print("Group indices:", group)
    else:
        print("❌ No valid groups found.")
        plot_circles(test_circles, "Test Circles with Labels")

    # Display all circles
    # plot_circles(test_circles, "Test Circles with Labels")

if __name__ == "__main__":
    main()