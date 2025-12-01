import cv2
import numpy as np
from fw import FWSeries, FWImagePair
from copy import deepcopy
import plot_utils

def find_pack_bounding_box(rois:list[list], padding=0) -> tuple[tuple,tuple]:
    """
    Calculate a bounding box that contains all ROIs plus padding.
    Returns tuple of two points, ((x_min,y_min), (x_max,ymax))
    
    rois: iterable of [x, y, r]
    padding: scalar padding applied on all sides
    """
    if not rois:
        return ((0, 0), (0, 0))

    # Support both a flat list of ROIs and a list of lists (per-slice ROIs).
    first_item = rois[0]
    if isinstance(first_item, (list, tuple, np.ndarray)) and first_item and isinstance(first_item[0], (list, tuple, np.ndarray)):
        flat_rois = [roi for group in rois for roi in group]
    else:
        flat_rois = rois

    if not flat_rois:
        return ((0, 0), (0, 0))

    xs = [roi[0] for roi in flat_rois]
    ys = [roi[1] for roi in flat_rois]
    rs = [roi[2] for roi in flat_rois]

    xmin = min(x - r for x, r in zip(xs, rs)) - padding
    xmax = max(x + r for x, r in zip(xs, rs)) + padding
    ymin = min(y - r for y, r in zip(ys, rs)) - padding
    ymax = max(y + r for y, r in zip(ys, rs)) + padding

    xmin = round(xmin)
    xmax = round(xmax)
    ymin = round(ymin)
    ymax = round(ymax)

    return ((xmin, ymin), (xmax, ymax))

    
def create_hepplus_img(fw:FWSeries) -> np.ndarray:
    image_pairs_in_span = fw.img_pairs_in_span(min_loc=fw.stats_min_loc, max_loc=fw.stats_max_loc)
    pack_arr_img = cropped_pack_array(image_pairs_in_span)
    plot_utils.display_cimg(pack_arr_img)
    return pack_arr_img

def cropped_pack_array(img_pairs:list[FWImagePair]) -> np.ndarray:
    cols = 2
    rows = len(img_pairs)

    # find bounding box
    all_circles = [c.circles for c in img_pairs if c.has_circles()]
    if not all_circles:
        raise ValueError("No circles found to build a bounding box")
    bbox = find_pack_bounding_box(all_circles)
    cimg_setup = cv2.cvtColor(np.uint8(img_pairs[0].water_img), cv2.COLOR_GRAY2BGR) 
    img_h, img_w, channels = cimg_setup.shape
    # be sure crop areas don't go outside iamge
    x1 = int(round(max(0,     min(bbox[0][0],bbox[1][0]))))
    y1 = int(round(max(0,     min(bbox[0][1],bbox[1][1]))))
    x2 = int(round(min(img_w, max(bbox[0][0],bbox[1][0]))))
    y2 = int(round(min(img_h, max(bbox[0][1],bbox[1][1]))))
    bbox_w = x2-x1
    bbox_h = y2-y1

    canvas = np.zeros((bbox_h * rows, bbox_w * cols, channels), dtype=np.uint8)
    for i, img_pair in enumerate(img_pairs):
        watr_copy = deepcopy(img_pair.water_img)
        pdff_copy = deepcopy(img_pair.pdff_img)
        cimg_watr = np.uint8(cv2.normalize(watr_copy, None, 0, 255, cv2.NORM_MINMAX))
        cimg_watr = cv2.cvtColor(cimg_watr, cv2.COLOR_GRAY2BGR)
        cimg_pdff = np.uint8(cv2.normalize(pdff_copy, None, 0, 255, cv2.NORM_MINMAX))
        cimg_pdff = cv2.cvtColor(cimg_pdff, cv2.COLOR_GRAY2BGR)
        # draw circles around water vials
        if img_pair.has_circles():
            np_circles = np.uint16(np.around(img_pair.circles))
            for c in np_circles:
                cv2.circle(cimg_watr,(c[0],c[1]),c[2],(0,0,255),1) 
        # draw ROIs in pdff vials
        if img_pair.has_rois():
            np_rois = np.uint16(np.around(img_pair.rois))
            for j, c in enumerate(np_rois):
                cv2.circle(cimg_pdff, (c[0],c[1]),c[2],(255,255,0),1)
        cropped_watr = cimg_watr[y1:y2, x1:x2]
        cropped_pdff = cimg_pdff[y1:y2, x1:x2]
        canvas[i * bbox_h:(i + 1) * bbox_h,      0:bbox_w  ] = cropped_watr
        canvas[i * bbox_h:(i + 1) * bbox_h, bbox_w:bbox_w*2] = cropped_pdff
    return canvas

def plot_results(image_pairs:list[FWImagePair], dest_filepath:str="", display_image=False):
    """Plot PDFF and water images with ROIs using OpenCV."""
    cols = 2
    rows = np.uint32(len(image_pairs))
    # Create a blank canvas to hold the images
    cimg_setup = cv2.cvtColor(np.uint8(image_pairs[0].water_img), cv2.COLOR_GRAY2BGR) #bug: assumes all images same resolution
    height, width, channels = cimg_setup.shape

    canvas = np.zeros((height * rows, width * cols, channels), dtype=np.uint8)
    for i, img_pair in enumerate(image_pairs):
        cimg_water = np.uint8(cv2.normalize(img_pair.water_img, None, 0, 255, cv2.NORM_MINMAX))
        cimg_water = cv2.cvtColor(cimg_water, cv2.COLOR_GRAY2BGR)
        cimg_pdff = np.uint8(cv2.normalize(img_pair.pdff_img, None, 0, 255, cv2.NORM_MINMAX))
        cimg_pdff = cv2.cvtColor(cimg_pdff, cv2.COLOR_GRAY2BGR)
        if img_pair.has_circles():
            np_circles = np.uint16(np.around(img_pair.circles))
            for c in np_circles:
                cv2.circle(cimg_water,(c[0],c[1]),c[2],(0,0,255),1)             # draw the outer circle
        if img_pair.has_rois():
            np_rois = np.uint16(np.around(img_pair.rois))
            # mystats = slice_stats(img_pair)
            for j, c in enumerate(np_rois):
                cv2.circle(cimg_pdff, (c[0],c[1]),c[2],(255,255,0),1)
                # mystr = f"{mystats['pdff_means'][j]:.1f}"
                text_size, _ = cv2.getTextSize(mystr, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                text_w, text_h = text_size
                mypt = (c[0]-3*c[2],c[1]+4*c[2]+text_h) # default/odd, plot below vial
                if (j % 2 == 0): #even, plot above  vial
                    mypt = (c[0]-3*c[2],c[1]-4*c[2])

        #         cv2.rectangle(cimg_pdff, (mypt[0], mypt[1]), (mypt[0] + text_w, mypt[1] - text_h), (0,0,0), -1)
        #         cv2.putText(cimg_pdff, mystr, mypt, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)
        # plot slice location on bottom of pdff image
        # loc_nodecimals = f"{float(img_pair.pdff.get('SliceLocation')):.1f}"
        # loc_str = f"LOC: {loc_nodecimals}"
        # loc_fontscale = 0.6
        # loc_size, _ = cv2.getTextSize(loc_str, cv2.FONT_HERSHEY_SIMPLEX, loc_fontscale, 1)
        # loc_w, loc_h = loc_size
        # loc_pt = (int(width/2 - loc_w/2), 2*loc_h)
        # cv2.rectangle(cimg_pdff, loc_pt, (loc_pt[0] + loc_w, loc_pt[1] - loc_h), (0,0,0), -1)
        # cv2.putText(cimg_pdff, loc_str, loc_pt, cv2.FONT_HERSHEY_SIMPLEX, loc_fontscale, (255,255,0), 1)
        # Place each image on the canvas
        canvas[i * height:(i + 1) * height,     0:width  ] = cimg_water
        canvas[i * height:(i + 1) * height, width:width*2] = cimg_pdff

    # save image
    if (dest_filepath != None):
        cv2.imwrite(dest_filepath, canvas)

    if (display_image):
        cv2.imshow("Images", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def test_bounding_box():
    rois = [
        [
            [10.0, 11.0, 7.0],
            [20.0, 11.0, 7.0],
            [30.0, 11.0, 7.0],
            [40.0, 11.0, 7.0],
            [50.0, 11.0, 7.0],
        ],
        [
            [10.1, 11.1, 7.0],
            [20.1, 11.1, 7.0],
            [30.1, 11.1, 7.0],
            [40.1, 11.1, 7.0],
            [50.1, 11.1, 7.0],
        ]
    ]
    bbox = find_pack_bounding_box(rois, padding=0)
    print(f"LL: {bbox[0][0]},{bbox[0][1]},  UR: {bbox[1][0]},{bbox[1][1]}")


if __name__ == '__main__':
    test_bounding_box()
