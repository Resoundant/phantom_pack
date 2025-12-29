import cv2
import numpy as np
import plot_utils
from pack_simulators import simulate_phantom_pack
from phantom_pack import find_packs_in_images, composite_statistics


def create_test_image() -> np.ndarray:
    height = 256
    width = 256
    channels = 1 # color channels (1 = grayscale)
    radius = 9 #px
    spacing = 4*radius #px
    x_offset = radius*2 #px
    y_offset = radius*2 #px
    img = np.zeros((height, width , channels), dtype=np.uint8)
    for i in range(7):
        for j in range(7):
            x = int(x_offset + j*spacing)
            y = int(y_offset + i*spacing)
            intensity = int(255/49 * (i*7 + j))
            cv2.circle(img, (x,y), radius, (intensity), -1) # solid circle (thickness = -1) filled with  1
    return img

if __name__ == "__main__":
    fw_series = simulate_phantom_pack()

    # display a few images to be sure
    display_slice = len(fw_series.image_pairs) // 2
    # plot_utils.display_image(fw_series.image_pairs[display_slice].pdff_img, "PDFF, midpoint")
    # plot_utils.display_image(fw_series.image_pairs[display_slice].water_img, "Water, midpoint")

    find_packs_in_images(fw_series)
    fw_series.create_rois(5)
    fw_series.pack_midpoint = fw_series.find_pack_midpoint()
    span_mm = 15
    fw_series.stats_min_loc = fw_series.pack_midpoint - span_mm/2
    fw_series.stats_max_loc   = fw_series.pack_midpoint + span_mm/2
    dict_results = composite_statistics(fw_series, fw_series.stats_min_loc, fw_series.stats_max_loc)

    x=1