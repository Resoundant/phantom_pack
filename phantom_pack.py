import os
import sys
import pydicom
import numpy as np
import cv2
import datetime
from image_labels import identifying_labels
from statistics import mode
from load_dicoms import load_dicoms
from circle_grouping import find_circle_groups
from circle_finder import circle_finder_water
from fw import FWSeries, FWImagePair

from pp_config import PP_CONST
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# debug flags
DEBUG_VERBOSE = True
DEBUG_PLOTS = False
MATCH_TRACE = False
# circle index helpers
CX = 0
CY = 1
CR = 2



TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')




def phantom_pack(
        labeled_dicoms:list[pydicom.Dataset],
        vial_radius = PP_CONST["VIAL_RADIUS_MM"],
        radius_tolerance = PP_CONST["RADIUS_TOLERANCE_MM"], 
        vert_align_tol = PP_CONST["VERT_ALIGN_TOLERANCE_MM"],
        roi_radius = PP_CONST["ROI_RADIUS_MM"],
        span_mm = PP_CONST["ANALYSIS_SPAN_MM"],
    ) -> dict:
    '''
    process all pdff data in directory_path
    return a list of dictionaries containing results for each pdff/water pair
    '''

    if labeled_dicoms == None or len(labeled_dicoms) == 0:
        print("[phantom_pack] ERROR: Input data is empty")
        return {} 

    # prepare output directory
    output_dir = os.path.join(directory_path, PP_CONST["OUTPUT_DIR"])
    os.makedirs(output_dir, exist_ok=True)
    #TODO: make sure directory exist and is writeable

    # find pdff/water pairs; img_packs = [[pair1],[pair2],...]
    fw_series_paired = find_fw_pairs(labeled_dicoms) #todo: rename variables to clarify
    print_paired_summary(fw_series_paired, directory_path)

    # save summary of loaded data: each pdff/water series and description
    log_seriesdata_to_file(output_dir, fw_series_paired)
    # save summary of data loaded but unknown (no matching label)
    log_unknowns_to_file(output_dir, labeled_dicoms)

    # loop over series pairs to find phantom packs
    for fw in fw_series_paired:
        if len(fw.image_pairs) == 0: # no data
            continue
        logger.info("")
        logger.info(f"Processing PDFF series {fw.series_number_pdff} {fw.series_description_pdff}")
        logger.info(f"     with WATER series {fw.series_number_water} {fw.series_description_water}")

        find_packs_in_images(
            fw,
            vial_radius=vial_radius,
            radius_tolerance=radius_tolerance,
            vert_align_tol=vert_align_tol
        )
        fw.sort_data_by_sliceloc()
        fw.create_rois(roi_radius=roi_radius) # put ROIs from all found circles

        # COMPUTE STATISTICS
        pack_midpoint = fw.find_pack_midpoint() #set fw_series.pack_midpoint
        fw.find_pack_locations() # set first and last locations and indeces of pack
        if fw.pack_midpoint is None:
            logger.warning(f"No pack midpoint found for series {fw.series_number_pdff} {fw.series_description_pdff}")
            continue

        try:
            fw.stats_min_loc = fw.pack_midpoint-span_mm/2
            fw.stats_max_loc = fw.pack_midpoint+span_mm/2
        except: 
            logger.warning("ERROR computing min and max slice location")
            return {}
        
        results = fw.compute_and_save_results(span_mm, output_dir)
    return results


# def compute_and_save_results(fw:FWSeries, span_mm, output_dir) -> dict:
#     # this code is a bit rigid in expecting regularly structed data (same number of packs, same pixels in each ROI, etc))
#     # to avoid it crashing the whole works, if something doesn't finish, it will except and move on, saving no data or
#     # partial data

    
#     try:
#         composite_results = composite_statistics(fw, fw.stats_min_loc, fw.stats_max_loc)
#     except:
#         logger.warning(f"ERROR computing comsposite statistics for series {fw.series_number_pdff} {fw.series_description_pdff}")
 
#     cropped_arr = create_hepplus_img(fw)
#     fp = os.path.join(output_dir,f'{fw.series_number}_tiles.png')
#     plot_results(fw.image_pairs, dest_filepath=fp)
#     x=1

#     # DECOUPLE AND RESTORE THIS
#     # try:
#     #     # collect info about dataset
#     #     image_info = get_image_info(fw_serie)
#     #     # save canvas of all pdff water pairs with circles
#     #     array_filepath = os.path.join(output_dir, f"{image_info['PatientName']}_{image_info['SeriesNumber_pdff']}_allimg.png")
#     #     plot_results(fw_serie.image_pairs, dest_filepath=array_filepath, display_image=False)
#     #     # save image of just the selected slices
#     #     array_filepath = os.path.join(output_dir, f"{image_info['PatientName']}_{image_info['SeriesNumber_pdff']}_selected.png")
#     #     image_pairs_in_span = fw_serie.img_pairs_in_span(min_loc=stats_min_loc, max_loc=stats_max_loc)
#     #     plot_results(image_pairs_in_span, dest_filepath=array_filepath, display_image=False)
#     # except:
#     #     logger.warning(f"Error saving plots for series {fw_serie.series_number_pdff} {fw_serie.series_description_pdff}")

#     try:
#         # save plots of slice values
#         plot_slice_values(fw, vert_lines=[fw.stats_min_loc, fw.stats_max_loc], directory_path=output_dir)
#         results = composite_results # RESTORE ME| image_info

#         if results:
#             file_path = os.path.join(output_dir, f"{results['PatientName']}_{results['SeriesNumber_pdff']}.json")
#             with open(file_path, 'w') as file:
#                 json.dump(results, file, indent=4)
#             logger.info(f"  JSON data saved to {file_path}")
#         # if MATCH_TRACE:
#         #     trace_data = []
#         #     for imgs in fw_serie.image_pairs:
#         #         trace_data.append({
#         #             "pdff_trace" : f"Series {imgs.pdff.SeriesNumber}, Instance {imgs.pdff.InstanceNumber}",
#         #             "water_trace" : f"Series {imgs.water.SeriesNumber}, Instance {imgs.water.InstanceNumber}",
#         #             "pdff_filename" : f"Series {imgs.pdff.filename}",
#         #             "water_filename" : f"Series {imgs.water.filename}",
#         #         })
#         #     file_path = os.path.join(directory_path, OUTPUT_DIR, f"{results['PatientName']}_{results['SeriesNumber_pdff']}_trace.json")
#         #     with open(file_path, 'w') as file:
#         #         json.dump(trace_data, file, indent=4)
#         return results
#     except:
#         logger.warning(f"Error computing per-slice for series {fw.series_number_pdff} {fw.series_description_pdff}")
#         return {}

def log_unknowns_to_file(output_dir, all_dicoms):
    unknowns = [x for x in all_dicoms if x.image_label == "unknown"]
    unknown_series_uids = list(set([x.SeriesInstanceUID for x in unknowns]))
    with open(os.path.join(output_dir, f"summary_data_unknown.txt"), 'w') as f:
        for series_uid in unknown_series_uids:
            unk_series = [x for x in unknowns if x.SeriesInstanceUID == series_uid]
            unk_series_nums = list(set([x.SeriesNumber for x in unk_series]))
            unk_series_descs = list(set([x.SeriesDescription for x in unk_series]))
            f.write(f"Unknown series containing {len(unk_series)} images\n")
            f.write(f"Series numbers: {unk_series_nums}\n")
            f.write(f"Series descriptions: {unk_series_descs}\n")
            f.write(f"\n")

def log_seriesdata_to_file(output_dir, fw_series_paired):
    with open(os.path.join(output_dir, f"summary_data_found.txt"), 'w') as f:
        for fw_series in fw_series_paired:
            if len(fw_series.image_pairs) == 0:
                continue
            f.write(f"PDFF  series {fw_series.series_number_pdff},  {fw_series.series_description_pdff}\n")
            f.write(f"WATER series {fw_series.series_number_water}, {fw_series.series_description_water}\n")
            f.write(f"\n")


def label_datasets(datasets:list[pydicom.Dataset]):
    for ds in datasets:
        label_dataset(ds)

def label_dataset(ds:pydicom.Dataset):
    for id in identifying_labels:
        if id["search_for"] in ds.get(id["search_in"]):
            ds.image_label = id["image_label"]
            ds.label_match = id["label_match"]
            return
    # no label matched, set to unknown so the field exists
    ds.image_label = "unknown"
    ds.label_match = "unknown"
    return



# def composite_statistics(fw_series:FWSeries, stats_min_loc, stats_max_loc) -> dict:
#     '''
#     Calculate the composite stats for slices in range (min_loc, max_loc)
#     Output (dict): means:[], stddevs:[], medians:[], mins:[], maxs:[], samples:[]
#     '''

#     num_rois_in_images_list = count_rois_in_images(fw_series)
#     if len(set(num_rois_in_images_list)) != 1:
#         raise ValueError("Inconsistent number of ROIs in series")

#     if not rois_are_aligned(fw_series):
#         do_nothing = True

#     # sort all circles and rois by x-coord
#     # this is necessary so that when cast into an np array, all of the rois across
#     # slices are properly grouped
#     for img_pair in fw_series.image_pairs:
#         if img_pair.has_circles():
#             img_pair.circles = sorted(img_pair.circles, key=lambda x: x[CX])
#         if img_pair.has_rois():
#             img_pair.rois = sorted(img_pair.rois, key=lambda x: x[CX])


#     # to calc mean, create lists made up of all pixels value in rois across all slices in range
#     num_rois_mode = mode(num_rois_in_images_list)
#     masked_values = [[] for _ in range(num_rois_mode)] #create empty list of lists
#     for img_pair in fw_series.image_pairs:
#         if (img_pair.location_full > stats_max_loc) or (img_pair.location_full < stats_min_loc):
#             continue
#         if not img_pair.has_rois():
#             continue
#         for roi_index, roi in enumerate(img_pair.rois):
#             # make a circle mask that can be applied to pdff
#             vals = get_values_in_roi(img_pair.pdff_img, roi)
#             masked_values[roi_index].extend(vals)

#     # calculate stats per ROI list to handle variable pixel counts
#     results_dict = {
#         "means": [],
#         "stddevs": [],
#         "medians": [],
#         "mins": [],
#         "maxs": [],
#         "samples": [],
#     }
#     for roi_vals in masked_values:
#         if not roi_vals:
#             results_dict["means"].append(float("nan"))
#             results_dict["stddevs"].append(float("nan"))
#             results_dict["medians"].append(float("nan"))
#             results_dict["mins"].append(float("nan"))
#             results_dict["maxs"].append(float("nan"))
#             results_dict["samples"].append(0)
#             continue
#         np_vals = np.asarray(roi_vals)
#         results_dict["means"].append(float(np.mean(np_vals)))
#         results_dict["stddevs"].append(float(np.std(np_vals)))
#         results_dict["medians"].append(float(np.median(np_vals)))
#         results_dict["mins"].append(float(np.min(np_vals)))
#         results_dict["maxs"].append(float(np.max(np_vals)))
#         results_dict["samples"].append(int(np_vals.size))
#     results_dict = renormalize_stats(results_dict)
#     return results_dict

def count_rois_in_images(fw_series:FWSeries) -> list:
    #  check that same number of ROIs in all images
    num_rois_in_images = []
    for img_pair in fw_series.image_pairs:
        if img_pair.has_rois():
            num_rois_in_images.append(len(img_pair.rois))
    if len(set(num_rois_in_images)) > 1:
        logger.warning("WARNING: not all images have same number of ROIs! This may cause issues")
    return num_rois_in_images

def rois_are_aligned(fw_series:FWSeries) -> bool:
    """Return True when ROI centers stay within their radius across slices."""
    prev_rois = None
    for img_pair in fw_series.image_pairs:
        if not img_pair.has_rois():
            continue
        current_rois = sorted(img_pair.rois, key=lambda r: r[CX])
        if prev_rois is None: # first loop
            prev_rois = current_rois
            continue
        if len(prev_rois) != len(current_rois):
            logger.warning("WARNING: ROIs not aligned across slices (different ROI counts)")
            return False
        for roi_idx, roi in enumerate(current_rois):
            x_sep = abs(float(prev_rois[roi_idx][CX]) - float(roi[CX]))
            y_sep = abs(float(prev_rois[roi_idx][CY]) - float(roi[CY]))
            allowed_radius = min(prev_rois[roi_idx][CR], roi[CR])
            if (x_sep * x_sep + y_sep * y_sep) > (allowed_radius * allowed_radius):
                logger.warning(f"WARNING: ROIs not aligned across slices at index {roi_idx}")
                return False
        prev_rois = current_rois
    return True


# def renormalize_stats(results_dict:dict) -> dict:
#     # check mean value of the middle vial; it should always be 30% (20-40) if we find it is >101, renormalize by dividing by 100
#     if results_dict["means"][2] < 101:
#         results_dict["renormalized"] = False
#         return results_dict
#     results_dict["renormalized"] = True
#     for key in results_dict.keys():
#         if key == "renormalized": continue
#         if key == "samples": continue
#         results_dict[key] = [x/100 for x in results_dict[key]]
#     return results_dict

# def get_values_in_roi(img:np.ndarray, r) -> list:
#     mask = np.zeros(img.shape, dtype=np.uint8)
#     cv2.circle(mask, (r[CX], r[CY]), r[CR], 1, -1) # solid circle (thickness = -1) filled with  1
#     vals = apply_mask(img, mask)
#     return vals

def create_negative_image(img):
    return 255 - np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))

def find_packs_in_images(
        fw_series:FWSeries,
        vial_radius = PP_CONST["VIAL_RADIUS_MM"],
        radius_tolerance = PP_CONST["RADIUS_TOLERANCE_MM"],
        vert_align_tol = PP_CONST["VERT_ALIGN_TOLERANCE_MM"],
        vial_separation = PP_CONST["VIAL_SEP_MM"],
        vial_sep_tolerance = PP_CONST["VIAL_SEP_TOLERANCE_MM"],
        ):
    '''
        Finds circles in pdff/water image pairs ammends the fw_series.image_pairs to include those circles
    '''
    for ip in fw_series.image_pairs:
        px_size = ip.pixel_spacing
        min_radius, max_radius, min_vail_sep = vial_sizes_in_px(vial_radius, radius_tolerance, px_size)
        water_circles = circles_img_bottom(ip.water_img, min_radius, max_radius, min_vail_sep)
        # display_image_with_circles(ip.water_img, water_circles, name=str(ip.location_full), waitkey=0)
        num_circles_in_pack = 5
        # print(f"LOCATION: {ip.location}")
        if ip.location == 6:
            pause = True
        pack_circles = find_circle_groups(
            water_circles,
            radius = vial_radius/px_size,
            spacing = vial_separation/px_size,
            num_circles_in_group = num_circles_in_pack,
            radius_tol = radius_tolerance/vial_radius,
            linear_tol = vert_align_tol/vial_separation,
            spacing_tol = vial_sep_tolerance/vial_separation)
        if pack_circles == []:
            continue
        # drop trivial first dimension/ hack: keep only first group
        ip.circles = pack_circles[0]
        if ip.circles:
            pause = True

    count_circles = [pair for pair in fw_series.image_pairs if pair.has_circles()]
    logger.info(f"  {len(count_circles)} slices contain phantom pack")
    # if len(count_circles) == 0:
    #     with open("no_packs_found.txt", "w") as f:
    #         f.write(f"Series {water_ds.SeriesNumber}, {water_ds.SeriesDescription}")
    return

def circles_img_bottom(img, min_radius, max_radius, min_vail_sep):
    cropped_img = make_clipped_image(img)
    circles = circle_finder_water(cropped_img, minDist=min_vail_sep, minRadius=min_radius, maxRadius=max_radius)
    return circles

def make_clipped_image(water_img, cutoff_top=0.65) -> np.ndarray:
    ''' Blacks off the top .65 of an image '''
    water_matlike = np.matrix(water_img)
    cutoff = int(np.floor(water_matlike.shape[0]*cutoff_top))
    water_matlike[:cutoff,:] = 0
    normalized_img = np.uint8(cv2.normalize(water_matlike, None, 0, 255, cv2.NORM_MINMAX))
    return normalized_img


def find_fw_pairs(all_dicoms:list[pydicom.Dataset]) -> list[FWSeries]:
    # Here were are matchmaking by enforcing the water image_label is same as pdff "label_match" tag
    # AcquisitionTime and SeriesNumber are used to separate series
    # SliceLocating is used to separate images
    series_found = []
    waters =[x for x in all_dicoms if x.image_label.startswith('water')]
    pdffs = [x for x in all_dicoms if x.image_label.startswith('pdff')]
    # split up pdff's by series number
    series_numbers = list(set([x.SeriesNumber for x in pdffs]))
    for sn in series_numbers:
        # my_img_pack_data = []
        pdffs_in_series = [x for x in pdffs if x.SeriesNumber == sn]
        fw_series = FWSeries(sn)
        for ff in pdffs_in_series:
            fw_series.series_description_pdff = ff.SeriesDescription
            wat_match_label  = [x for x in waters if x.image_label == ff.label_match]
            wat_same_acqtime = [x for x in wat_match_label if acq_time_inrange(ff.AcquisitionTime, x.AcquisitionTime)]
            wat_same_loc     = [x for x in wat_same_acqtime if int(x.SliceLocation) == int(ff.SliceLocation)] # avoid float precision issues by casting to int
            if len(wat_same_loc) == 0:
                logger.warning(f"WARNING: PDFF no water match: {ff.SeriesDescription}, SerNum {ff.SeriesNumber}, AcqTime {ff.AcquisitionTime}, Loc {ff.SliceLocation}")
            if len(wat_same_loc) > 1:
                logger.warning(f"WARNING: PDFF no water match: {ff.SeriesDescription} {ff.SeriesNumber} {ff.AcquisitionTime} {ff.SliceLocation}")
                for w in wat_same_loc:
                    logger.warning(f"  {w.SeriesDescription} {w.SeriesNumber} {w.AcquisitionTime} {w.SliceLocation}")
            if len(wat_same_loc) >= 1:
                fw_series.series_number_water = wat_same_loc[0].SeriesNumber
                fw_series.series_description_water = wat_same_loc[0].SeriesDescription
                px_spacing = wat_same_loc[0].PixelSpacing[0] # todo: check if square and same as pdff
                img_pair = FWImagePair(ff.pixel_array, wat_same_loc[0].pixel_array, px_spacing, ff.SliceLocation)
                fw_series.image_pairs.append(img_pair)
        series_found.append(fw_series)
    return series_found

def vial_sizes_in_px(vial_radius, radius_tolerance, px_size):
    min_radius = int(vial_radius/px_size) - int(radius_tolerance/px_size)
    max_radius = int(vial_radius/px_size) + int(np.ceil(radius_tolerance/px_size))
    min_vail_sep = vial_radius/px_size
    return min_radius,max_radius,min_vail_sep


def sort_circles_by_x_coord(circles:np.ndarray):
    x_locs = []
    for c in circles:
        x_locs.append(c[0])
    x_locs = sorted(x_locs)
    sorted_circles = []
    for loc in x_locs:
        for c in circles:
            if c[0] == loc:
                sorted_circles.append(c)
                break
    return sorted_circles



def print_mean_median_values(pdff_means, pdff_medians):
    mean_str = "Mean values: "
    median_str = "Median values: "
    pdff_means = sorted(pdff_means)
    pdff_medians = sorted(pdff_medians)
    for mn in pdff_means:
        mean_str += f"{mn:.2f}, "
    for md in pdff_medians:
        median_str += f"{md:.2f}, "
    print(mean_str)
    print(median_str)

def find_imagetype(dicoms:list[pydicom.Dataset], contrast:str) -> list[pydicom.Dataset]:
    images = []
    for ds in dicoms:
        if contrast in ds.ImageType:
            images.append(ds)
    return images


def acq_time_inrange(ff_acqtime, wat_acqtime, rng=1):
    return ((int(wat_acqtime) >= int(ff_acqtime)-rng) and (int(wat_acqtime) <= int(ff_acqtime)+rng))



def check_vial_spacing(ph, min_space, max_space) -> bool:
    for i in range(len(ph)-1):
        c0 = ph[i]
        c1 = ph[i+1]
        if ((abs(c0[CX]-c1[CX]) < min_space) or
            (abs(c0[CX]-c1[CX]) > max_space)):
            return False
    return True

def avg_circle_spacing(phantoms):
    avg_spacing = 0
    for i in range(len(phantoms) - 1):
        avg_spacing += phantoms[i][2] - phantoms[i+1][2]
    avg_spacing = avg_spacing/(len(phantoms) - 1)
    return avg_spacing

def avg_circle_radius(phantoms):
    avg = 0
    for i in range(len(phantoms) - 1):
        avg += phantoms[i][0] - phantoms[i+1][0]
    avg = avg/(len(phantoms) - 1)
    return avg

def plot_image(img, name='image', waitkey=1):
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

def plot_circles_list(img, circles, name='image', waitkey=1):
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(circles))
    for c in np_circles:
        cv2.circle(cimg,(c[CX],c[CY]),c[CR],(0,255,0),2)
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

def plot_circles_ndarray(img, circles, name='image', waitkey=1):
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(circles))
    for c in np_circles[0,:]: # don't remembwer what packing needed this slice
        cv2.circle(cimg,(c[CX],c[CY]),c[CR],(0,255,0),2)
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

def plot_selected_image(img_data:dict, dest_filepath:str=None, display_image=False):
    ''' save pdff and water with ROIs on them'''
    # put rois onto pdff and water images
    cimg_water = np.uint8(cv2.normalize(img_data["water"].pixel_array, None, 0, 255, cv2.NORM_MINMAX))
    cimg_water = cv2.cvtColor(cimg_water, cv2.COLOR_GRAY2BGR)
    cimg_pdff = np.uint8(cv2.normalize(img_data["pdff"].pixel_array, None, 0, 255, cv2.NORM_MINMAX))
    cimg_pdff = cv2.cvtColor(cimg_pdff, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(img_data["circles"]))
    for c in np_circles:
        cv2.circle(cimg_pdff, (c[0],c[1]),c[2],(0,255,0),1)
        cv2.circle(cimg_water,(c[0],c[1]),c[2],(0,255,0),1)
    np_rois = np.uint16(np.around(img_data["rois"]))
    for c in np_rois:
        cv2.circle(cimg_pdff, (c[0],c[1]),c[2],(0,0,255),1)
        cv2.circle(cimg_water,(c[0],c[1]),c[2],(0,0,255),1)
    # canvas to have both pdff (left) and water (right) in one image
    height, width, channels = cimg_water.shape
    canvas = np.zeros(( width, 2 * height, channels), dtype=np.uint8)
    canvas[:height, :width] = cimg_pdff
    canvas[:height, width:] = cimg_water

    if (dest_filepath != None):
        cv2.imwrite(dest_filepath, canvas)

    if (display_image):
        cv2.imshow("Selected Image", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def plot_array(img_pack_data:list[dict], dest_filepath:str=None,display_image=False):
    """Plot an array of images using OpenCV."""
    cols = 5
    rows = np.uint8(np.ceil(len(img_pack_data) / cols))
    # Create a blank canvas to hold the images
    # cimg_setup = np.uint8(img_pack_data[0]["water"].pixel_array)
    cimg_setup = cv2.cvtColor(np.uint8(img_pack_data[0]["water"].pixel_array), cv2.COLOR_GRAY2BGR) #bug: assumes all images same resolution
    height, width, channels = cimg_setup.shape

    canvas = np.zeros((height * rows, width * cols, channels), dtype=np.uint8)
    for i, mydict in enumerate(img_pack_data):
        cimg = np.uint8(cv2.normalize(mydict["water"].pixel_array, None, 0, 255, cv2.NORM_MINMAX))
        cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
        if mydict["circles"] is not None:
            np_circles = np.uint16(np.around(mydict["circles"]))
            for c in np_circles:
                cv2.circle(cimg,(c[0],c[1]),c[2],(255,255,0),1)             # draw the outer circle

        # Place each image on the canvas
        row = i // cols
        col = i % cols
        canvas[row * height:(row + 1) * height, col * width:(col + 1) * width] = cimg
        # Display the canvas

    # save image
    if (dest_filepath != None):
        cv2.imwrite(dest_filepath, canvas)

    if (display_image):
        cv2.imshow("Images", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#     # save image
#     if (dest_filepath != None):
#         cv2.imwrite(dest_filepath, canvas)

#     if (display_image):
#         cv2.imshow("Images", canvas)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

def find_closest_value(mylist:list, target_value):
    """Finds the number in a list closest to a given target value."""
    return min(mylist, key=lambda x: abs(x - target_value))

def apply_mask(image, mask) -> list:
    np_img = np.array(image)
    vals = np_img[mask == 1].tolist() # need to be a list? or leave as np.array??
    return vals





def print_paired_summary(fw_series_paired, directory_path):
    if len(fw_series_paired) == 0:
        logger.warning(f"No PDFF/Water data found in {directory_path}")
        return {}
    logger.info(f'========================================')
    logger.info(f'Found {len(fw_series_paired)} pdff/water series')
    logger.info('')

def remove_outliers_mad(image, threshold=3.5):
    ''' introduced this function to try to deal with background white pixels.
        It actually did an admirable job of clipping those, but it also broke
        the ability to detect the vials.  Unused but left for educational purposes.
    '''
    median = np.median(image)
    mad = np.median(np.abs(image - median))
    # MAD to standard deviation approximation (if needed): σ ≈ 1.4826 * MAD
    modified_z_scores = 0.6745 * (image - median) / (mad + 1e-6)
    mask = np.abs(modified_z_scores) < threshold
    # num_outliers = np.sum(mask==False)
    # if num_outliers > 0:
    #     logging.info(f"Removed {num_outliers} outliers from image")
    return np.where(mask, image, 0)  # replace outliers with 0


def load_and_label(directory_path) -> list[pydicom.Dataset]:
    '''
    load all files in directory and apply labels
    '''
    # load dicoms in directory
    logger.info("Loading files...")
    all_dicoms = load_dicoms(directory_path, turbo_mode=True)

    # label each dicom dataset according to identifying_labels
    label_datasets(all_dicoms)

    return all_dicoms


if __name__ == "__main__":
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"Could not find directory {directory_path}")
        exit(1)
    logfile = os.path.join(directory_path, "phantom_pack.log")
    logging.basicConfig(filename=logfile, level=logging.INFO)
    logger.info(f"Processing {directory_path}")
    labeled_dicoms = load_and_label(directory_path)
    results = phantom_pack(labeled_dicoms)

    x=1
    # testimg_path = r'C:\testdata\PhantomPack\PQ024\SER00090\IMG00019.dcm'
    # directory_path = os.path.dirname(testimg_path)4
    # ds = pydicom.dcmread(testimg_path)
    # circles_in_pdff(ds, min_radius=1, max_radius=60, min_sep=1)


