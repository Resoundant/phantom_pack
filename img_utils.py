import os
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
    save_img_totemp(fw, pack_arr_img)
    return pack_arr_img

def cropped_pack_array(img_pairs:list[FWImagePair]) -> np.ndarray:
    cols = 3
    rows = len(img_pairs)

    # find bounding box
    all_circles = [c.circles for c in img_pairs if c.has_circles()]
    if not all_circles:
        raise ValueError("No circles found to build a bounding box")
    bbox = find_pack_bounding_box(all_circles, padding=2)
    # get input image dimensions to prepare for crop
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
    watr_col_canvas = np.zeros((bbox_h * rows, bbox_w, channels), dtype=np.uint8)
    pdff_col_canvas = np.zeros((bbox_h * rows, bbox_w, channels), dtype=np.uint8)
    cropped_watr_list = []
    cropped_pdff_list = []
    watr_min = None
    watr_max = None
    pdff_min = None
    pdff_max = None
    for img_pair in img_pairs:
        watr_crop = img_pair.water_img[y1:y2, x1:x2]
        pdff_crop = img_pair.pdff_img[y1:y2, x1:x2]
        cropped_watr_list.append(watr_crop)
        cropped_pdff_list.append(pdff_crop)
        if watr_crop.size:
            w_min = float(np.min(watr_crop))
            w_max = float(np.max(watr_crop))
            watr_min = w_min if watr_min is None else min(watr_min, w_min)
            watr_max = w_max if watr_max is None else max(watr_max, w_max)
        if pdff_crop.size:
            # p_min = float(np.min(pdff_crop))
            # p_max = float(np.max(pdff_crop))
            # pdff_min = p_min if pdff_min is None else min(pdff_min, p_min)
            # pdff_max = p_max if pdff_max is None else max(pdff_max, p_max)
            pdff_min = 0
            pdff_max = 100

    def normalize_to_uint8(img, vmin, vmax):
        if vmin is None or vmax is None or vmax <= vmin:
            return np.zeros_like(img, dtype=np.uint8)
        scale = 255.0 / (vmax - vmin)
        img_f = img.astype(np.float32, copy=False)
        return np.uint8(np.clip((img_f - vmin) * scale, 0, 255))

    for i, img_pair in enumerate(img_pairs):
        cimg_watr = normalize_to_uint8(cropped_watr_list[i], watr_min, watr_max)
        cimg_watr = cv2.cvtColor(cimg_watr, cv2.COLOR_GRAY2BGR)
        cimg_pdff = normalize_to_uint8(cropped_pdff_list[i], pdff_min, pdff_max)
        cimg_pdff = cv2.cvtColor(cimg_pdff, cv2.COLOR_GRAY2BGR)
        # draw circles around water vials
        if img_pair.has_circles():
            np_circles = np.uint16(np.around(img_pair.circles))
            for c in np_circles:
                cv2.circle(cimg_watr,(int(c[0]) - x1,int(c[1]) - y1),c[2],(0,0,255),1) 
        # draw ROIs in pdff vials
        if img_pair.has_rois():
            np_rois = np.uint16(np.around(img_pair.rois))
            for j, c in enumerate(np_rois):
                cv2.circle(cimg_pdff, (int(c[0]) - x1,int(c[1]) - y1),c[2],(255,255,0),1)
        row_text = f"{float(img_pair.location_full):.1f}mm"
        label_box = put_text_in_box(np.zeros((y2-y1, x2-x1, channels), dtype=np.uint8), row_text)
        # row_position = i * bbox_h:(i + 1) * bbox_h
        canvas[i * bbox_h:(i + 1) * bbox_h,        0:bbox_w  ] = cimg_watr
        canvas[i * bbox_h:(i + 1) * bbox_h,   bbox_w:bbox_w*2] = cimg_pdff
        canvas[i * bbox_h:(i + 1) * bbox_h, 2*bbox_w:bbox_w*3] = label_box
    return canvas


def put_text_in_box(img, text, 
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    max_font_scale=5.0, min_font_scale=0.2,
                    color=(255, 255, 255), thickness=1, margin=4):
    """
    Draw text centered in the image, auto-scaling to fit within the image bounds minus margin.
    Modifies img in place and returns it.
    """
    h, w = img.shape[:2]
    box_w, box_h = w - 2 * margin, h - 2 * margin
    if box_w <= 0 or box_h <= 0:
        return img  # image too small

    lo, hi = min_font_scale, max_font_scale
    best_scale = min_font_scale
    while hi - lo > 0.01:
        mid = (lo + hi) / 2.0
        (tw, th), _ = cv2.getTextSize(text, font, mid, thickness)
        if tw <= box_w and th <= box_h:
            best_scale = mid
            lo = mid
        else:
            hi = mid

    (tw, th), baseline = cv2.getTextSize(text, font, best_scale, thickness)
    x = margin # + (box_w - tw) // 2 # left justify
    y = margin + (box_h + th) // 2  # baseline position

    cv2.putText(img, text, (int(x), int(y)), font, best_scale, color, thickness, lineType=cv2.LINE_AA)
    return img


def plot_results(image_pairs:list[FWImagePair], dest_filepath:str="", display_image=False):
    """Plot PDFF and water images with ROIs using OpenCV."""
    cols = 2
    rows = np.uint32(len(image_pairs))
    # Create a blank canvas to hold the images
    cimg_setup = cv2.cvtColor(np.uint8(image_pairs[0].water_img), cv2.COLOR_GRAY2BGR) #bug: assumes all images same resolution
    height, width, channels = cimg_setup.shape

    canvas = np.zeros((height * rows, width * cols, channels), dtype=np.uint8)
    water_stack = np.vstack([img_pair.water_img for img_pair in image_pairs])
    water_stack = np.uint8(cv2.normalize(water_stack, None, 0, 255, cv2.NORM_MINMAX))
    for i, img_pair in enumerate(image_pairs):
        row_start = i * height
        row_end = row_start + height
        cimg_water = water_stack[row_start:row_end, :]
        cimg_water = cv2.cvtColor(cimg_water, cv2.COLOR_GRAY2BGR)
        cimg_pdff = np.uint8(np.clip(img_pair.pdff_img, 0, 255))
        cimg_pdff = cv2.cvtColor(cimg_pdff, cv2.COLOR_GRAY2BGR)
        if img_pair.has_circles():
            np_circles = np.uint16(np.around(img_pair.circles))
            for c in np_circles:
                cv2.circle(cimg_water,(c[0],c[1]),c[2],(0,0,255),1)             # draw the outer circle
        if img_pair.has_rois():
            np_rois = np.uint16(np.around(img_pair.rois))
            mystats = img_pair.slice_stats()
            for j, c in enumerate(np_rois):
                cv2.circle(cimg_pdff, (c[0],c[1]),c[2],(255,255,0),1)
                mystr = f"{mystats['pdff_means'][j]:.1f}"
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
    if (dest_filepath):
        cv2.imwrite(dest_filepath, canvas)

    if (display_image):
        cv2.imshow("Images", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def save_img_totemp(fw:FWSeries, img):
    filepath = os.path.join(r'C:\temp\images', f'phantom_pack_{fw.series_number}.jpg')
    save_img(img, filepath)

def save_img(img, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, img)


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
