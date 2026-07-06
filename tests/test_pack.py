import cv2
import numpy as np
from phantom_pack import plot_utils
from phantom_pack.pack_simulators import simulate_phantom_pack
from phantom_pack.phantom_pack import find_packs_in_images
from phantom_pack.fw import gen_pack_array_image
import phantom_pack.plot_utils as pu


def main():
    results = []
    missing = 0
    for i in range(10):
        results.append(sim_main())
        if results[-1][2] != 50:
            print("missing a slice")
            missing += 1
    print(f"{missing} total runs missing at least one slice")

    # avg_first, avg_last, avg_num_slices = np.mean(results, axis=0)
    # print(
    #     f"Average first slice: {avg_first:.2f}, "
    #     f"last slice: {avg_last:.2f}, "
    #     f"num slices: {avg_num_slices:.2f}"
    # )


def sim_main():
    fw_series = simulate_phantom_pack(
        pdff_variance= 2,
        pdff_noise = 2,
        water_noise = 7  
    )

    # pack_image = gen_pack_array_image(fw_series.image_pairs, max_rows=10)
    # pu.display_cimg(pack_image, "Pack image")

    find_packs_in_images(fw_series)
    fw_series.create_rois(5)

    fw_series.pack_midpoint = fw_series.find_pack_midpoint()
    if fw_series.pack_midpoint == None:
        print("Could not find pack midpoint in test_pack.sim_main()")
        return
    span_mm = 15
    fw_series.stats_min_loc = fw_series.pack_midpoint - span_mm/2
    fw_series.stats_max_loc = fw_series.pack_midpoint + span_mm/2
    fw_series.find_pack_locations()
    fwstats = fw_series.composite_statistics(fw_series.pack_midpoint, span_mm)

    print(f"{fwstats.means}")

    # pack_image_circles = gen_pack_array_image(fw_series.image_pairs, max_rows=8)
    # pu.display_cimg(pack_image_circles, "Pack image w circles")


    # print(  f"First slice: {fw_series.pack_first_slice}/{getattr(fw_series, 'KNOWN_first_slice')}, last slice: {fw_series.pack_last_slice}, count: {fw_series.num_slices_with_circles}/{getattr(fw_series,'KNOWN_num_slices')}")
    return fw_series.pack_first_slice, fw_series.pack_last_slice, fw_series.num_slices_with_circles


if __name__ == "__main__":
    main()
