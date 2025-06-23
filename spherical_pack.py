import pack_simulators as ps
import circle_utils as cu
import masking 

if __name__ == "__main__":
    pdff, water = ps.simulate_sphere(randomize=True)
    # cu.display_image(pdff)
    mask = masking.create_mask_using_clahe(water)
    mask = masking.fill_voids(mask, iterations=10)
    # cu.display_image(mask, normalize=True)
    pdff = pdff * mask
    # cu.display_image(pdff)
    
    circle_radius = int(ps.SPHERE_VIAL_RADIUS_MM/ps.IMG_FOV_MM*ps.IMG_RESOLUTION)
    circles = cu.circle_finder_pdff(pdff, maxRadius=2*circle_radius, minDist=circle_radius)
    circles = cu.sort_topleft_to_bottomright(circles)
    # cu.overlay_circles(pdff, circles)
    rois = cu.circles_to_rois(circles, (circle_radius/2))
    cimg = cu.overlay_circles(pdff, rois)
    cu.display_image(cimg, name="pdff with ROIs")

    water_circles = cu.circle_finder_water(water, minRadius=int(0.8*circle_radius),maxRadius=int(1.25*circle_radius), minDist=3*circle_radius)
    water_circles = cu.sort_topleft_to_bottomright(water_circles)
    cimg = cu.overlay_circles(water, water_circles)
    cu.display_image(cimg, name="water with circles")

    cimg = cu.overlay_circles(pdff, rois)
    cimg = cu.overlay_circles_cimg(cimg, water_circles)
    cu.display_image(cimg, name="pdff with ROIs and water circles")
    circles_match = cu.coords_match_within_tolerance(circles, water_circles, tolerance=4)
    if circles_match:
        print("Circles match")
    else:
        print("Circles don't match")