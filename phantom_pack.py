import os
import sys
import pydicom
import numpy as np 
import cv2
import datetime 
import copy
import json
import matplotlib.pyplot as plt
import time
from image_labels import identifying_labels, load_labels
from statistics import mode
import logging

logger = logging.getLogger(__name__)

# debug flags
DEBUG_VERBOSE = False
DEBUG_PLOTS = False
MATCH_TRACE = False
# circle index helpers
CX = 0
CY = 1
CR = 2

# Phantom pack and analysis parameters
OUTPUT_DIR = "phantompack_results"
VIAL_RADIUS_MM = 19/2      # radius of the phantom pack vials
VIAL_SEP_MM = 31           # 20px*1.56mm/px
ROI_RADIUS_MM = 13/2       # radius of the phantom pack ROI
RADIUS_TOLERANCE_MM = 4    # only find circles VIAL_RADIUS +/- RADIUS_TOLERANCE
VERTICAL_ALIGNMENT_TOLERANCE_MM = 7
ANALYSIS_SPAN_MM = 20      # analyze a range of images centered at the midpoint
ANALYSIS_CENTER_MM = None  # center span at a specific location, None to use midpoint
TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
DICOM_TAG_LIST = [
    "PatientName",
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "MagneticFieldStrength",
    "EchoTime",
    "RepetitionTime",
    "EchoTrainLength",
    "FlipAngle",
    "VariableFlipAngleFlag",
    "PulseSequenceName",
    "0019109C",
    "InstitutionName",
    "StationName",
    "AcquisitionDate",
    "AcquisitionTime",
]



def phantom_pack(
        directory_path:str, 
        vial_radius = VIAL_RADIUS_MM,
        radius_tolerance = RADIUS_TOLERANCE_MM,
        vert_align_tol = VERTICAL_ALIGNMENT_TOLERANCE_MM,
        roi_radius = ROI_RADIUS_MM,
        span_mm=ANALYSIS_SPAN_MM,
        center_point = ANALYSIS_CENTER_MM
    ) -> list[dict]:
    # process all pdff data in directory_path
    # return a list of dictionaries containing results for each pdff/water pair

    # prepare output directory
    output_dir = os.path.join(directory_path, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # load dicoms in directory
    print("Loading files...")
    time_load_start = time.perf_counter()
    all_dicoms = load_dicoms(directory_path, turbo_mode=True)
    time_load_end = time.perf_counter()
    print(f"loaded {len(all_dicoms)} dicoms in {time_load_end - time_load_start:.2f} seconds")
    
    # label each dicom according to identifying_labels
    for ds in all_dicoms:
        label_dataset(ds)
    # find pdff/water pairs; img_packs = [[pair1],[pair2],...]
    img_packs = find_fw_pairs(all_dicoms)
    if len(img_packs) == 0:
        print(f"No PDFF/Water data found in {directory_path}")
        return {}
    print(f'========================================')
    print(f'Found {len(img_packs)} pdff/water series')
    print('')

    # save summary of loaded data: each pdff/water series and description
    with open(os.path.join(output_dir, f"found_data_summary.txt"), 'w') as f:
        for img_pack_data in img_packs:
            if len(img_pack_data) == 0:
                continue
            f.write(f"PDFF  series {img_pack_data[0]['pdff'].SeriesNumber}, {img_pack_data[0]['pdff'].SeriesDescription}\n")
            f.write(f"WATER series {img_pack_data[0]['water'].SeriesNumber}, {img_pack_data[0]['water'].SeriesDescription}\n")
            f.write(f"\n")
    # save summary of data loaded but unknown (no matching label)
    unknowns = [x for x in all_dicoms if x.image_label == "unknown"]
    unknown_series_uids = list(set([x.SeriesInstanceUID for x in unknowns]))
    with open(os.path.join(output_dir, f"unknown_data_summary.txt"), 'w') as f:
        for series_uid in unknown_series_uids:
            unk_series = [x for x in unknowns if x.SeriesInstanceUID == series_uid]
            unk_series_nums = list(set([x.SeriesNumber for x in unk_series]))
            unk_series_descs = list(set([x.SeriesDescription for x in unk_series]))
            f.write(f"Unknown series containing {len(unk_series)} images\n")
            f.write(f"Series numbers: {unk_series_nums}\n") 
            f.write(f"Series descriptions: {unk_series_descs}\n")
            f.write(f"\n")

    all_results = []
    for img_pack_data in img_packs:
        if len(img_pack_data) == 0: # no data
            continue
        print("")
        print(f"Processing PDFF series {img_pack_data[0]['pdff'].SeriesNumber} {img_pack_data[0]['pdff'].SeriesDescription}")
        print(f"     with WATER series {img_pack_data[0]['water'].SeriesNumber} {img_pack_data[0]['water'].SeriesDescription}")
        # find circles in images
        # restore this
        img_pack_data = find_packs_in_images(
            img_pack_data,
            vial_radius=vial_radius,
            radius_tolerance=radius_tolerance,
            vert_align_tol=vert_align_tol
        )
        img_pack_data = sort_data_by_sliceloc(img_pack_data)
        create_rois(img_pack_data, roi_radius=roi_radius) # put ROIs from all found circles
        
        sliceslocations_in_middle_span, center_found = find_slices_in_span(
            img_pack_data, 
            span_mm = span_mm,
            center_loc = center_point)
        if len(sliceslocations_in_middle_span) == 0:
            continue
        # composite_results = composite_statistics_fromlist(img_pack_data, sliceslocations_in_middle_span)
        composite_results = composite_statistics(img_pack_data, center_found-span_mm/2, center_found+span_mm/2)

        # collect info about dataset
        image_info = get_image_info(img_pack_data)
        # print(json.dumps(image_info, indent=2))
        # save image of all pdff water pairs
        array_filepath = os.path.join(directory_path, OUTPUT_DIR, f"{image_info['PatientName']}_{image_info['SeriesNumber_pdff']}_allimg.png")
        plot_results(img_pack_data, dest_filepath=array_filepath, display_image=False)
        # save image of just the selected slices
        array_filepath = os.path.join(directory_path, OUTPUT_DIR, f"{image_info['PatientName']}_{image_info['SeriesNumber_pdff']}_selected.png")
        img_pack_data_middlespan = img_data_in_span(img_pack_data, sliceslocations_in_middle_span)
        plot_results(img_pack_data_middlespan, dest_filepath=array_filepath, display_image=False)
        # save plots of slice values
        plot_slice_values(img_pack_data, vert_lines = sliceslocations_in_middle_span, directory_path=os.path.join(directory_path, OUTPUT_DIR))

        results = composite_results | image_info
        if results:
            all_results.append(results)
            file_path = os.path.join(directory_path, OUTPUT_DIR, f"{results['PatientName']}_{results['SeriesNumber_pdff']}.json")
            with open(file_path, 'w') as file:
                json.dump(results, file, indent=4)
            print(f"  JSON data saved to {file_path}")
        if MATCH_TRACE:
            trace_data = []
            for imgs in img_pack_data:
                trace_data.append({
                    "pdff_trace" : f"Series {imgs['pdff'].SeriesNumber}, Instance {imgs['pdff'].InstanceNumber}",
                    "water_trace" : f"Series {imgs['water'].SeriesNumber}, Instance {imgs['water'].InstanceNumber}",
                    "pdff_filename" : f"Series {imgs['pdff'].filename}",
                    "water_filename" : f"Series {imgs['water'].filename}",
                })
            file_path = os.path.join(directory_path, OUTPUT_DIR, f"{results['PatientName']}_{results['SeriesNumber_pdff']}_trace.json")
            with open(file_path, 'w') as file:
                json.dump(trace_data, file, indent=4)
    return all_results

def img_data_in_span(img_pack_data, sliceslocations_in_middle_span):
    sliceslocations_in_middle_span = sorted(sliceslocations_in_middle_span)
    min_loc = np.floor(sliceslocations_in_middle_span[0])
    max_loc = np.ceil(sliceslocations_in_middle_span[-1])
    # img_pack_data_middlespan = [x for x in img_pack_data if x["pdff"].SliceLocation in sliceslocations_in_middle_span]
    img_pack_data_middlespan = [x for x in img_pack_data if (x["pdff"].SliceLocation >= min_loc) & (x["pdff"].SliceLocation <= max_loc)]
    return img_pack_data_middlespan

def calc_mean_and_median(midpoint_image:dict):
    ''' calculates mean and median, and puts a copy of the ROIs in the input dict'''
    pdff = midpoint_image["pdff"]
    water = midpoint_image["water"]
    circles = midpoint_image["circles"]
    rois = create_rois(circles, water.PixelSpacing[0])
    midpoint_image["rois"] = rois
    #apply rois to PDFF
    pdff_means = []
    pdff_medians = []
    for r in rois:
        # make a circle mask that can be applied to pdff
        mask = np.zeros(water.pixel_array.shape, dtype=np.uint8)
        cv2.circle(mask, (r[CX], r[CY]), r[CR], (1), -1) # solid circle (thickness = -1) filled with  1
        # calculate mean and median
        mean_pdff = masked_mean(pdff.pixel_array, mask)
        median_pdff = masked_median(pdff.pixel_array, mask)
        pdff_means.append(mean_pdff)
        pdff_medians.append(median_pdff)
    return pdff_means,pdff_medians


def slice_stats(img:dict) -> dict:
    # Compute the mean, median, and standard dev of a pdff/water pair.
    # img should be a single slice of the img_pack_data,
    # with pdff, water and rois
    # Output: pdff_means:[], pdff_stddevs:[], pdff_medians:[] - lists of mean, stddev, median
    # for the circles
    stats = {}
    # default values are -10
    if "rois" not in img:
        stats["pdff_means"] = [-10]*5 # HACK - fixed for 5 ROIs
        stats["pdff_medians"] = [-10]*5
        stats["pdff_stddevs"] = [0]*5
        return stats
    # apply rois to PDFF
    pdff = img["pdff"]
    rois = img["rois"]
    pdff_means = []
    pdff_medians = []
    pdff_stddevs = []
    for r in rois:
        # make a circle mask that can be applied to pdff
        mask = np.zeros(pdff.pixel_array.shape, dtype=np.uint8)
        cv2.circle(mask, (r[0], r[1]), r[2], (1), -1) # solid circle (thickness = -1) filled with  1
        # calculate mean and median
        mean_pdff   = masked_mean(  pdff.pixel_array, mask)
        median_pdff = masked_median(pdff.pixel_array, mask)
        stddev_pdff = masked_stddev(pdff.pixel_array, mask)
        pdff_means.append(mean_pdff)
        pdff_medians.append(median_pdff)
        pdff_stddevs.append(stddev_pdff)
    stats["pdff_means"] = pdff_means
    stats["pdff_medians"] = pdff_medians
    stats["pdff_stddevs"] = pdff_stddevs
    return stats

def composite_statistics(img_pack_data:list[dict], min_loc, max_loc) -> dict:
    '''
    Calculate the composite stats for slices in range (min_loc, max_loc)
    Output (dict): means:[], stddevs:[], medians:[], mins:[], maxs:[], samples:[]'''

    # quickly check that same number of ROIs in all images
    num_rois_in_images = []
    for img in img_pack_data:
        if "rois" not in img:
            continue
        num_rois_in_images.append(len(img["rois"]))
    if len(set(num_rois_in_images)) > 1:
        print("WARNING: not all images have same number of ROIs: mode is ")
    # create lists of all value in rois across all slices in range
    num_rois_mode = mode(num_rois_in_images)
    masked_values = [[] for _ in range(num_rois_mode)]
    for img in img_pack_data:
        if "rois" not in img:
            continue
        pdff = img["pdff"]
        rois = img["rois"]
        if (pdff.SliceLocation > max_loc) or (pdff.SliceLocation < min_loc): 
            continue
        for roi_index, roi in enumerate(rois):
            # make a circle mask that can be applied to pdff
            vals = get_values_in_roi(pdff, roi)
            masked_values[roi_index].extend(vals)

    # cast into np.array to take mean of each row, where a row contains the values for rois across slices
    np_arr = np.array(masked_values)
    # calculate mean across slices for a given roi
    results_dict = {}
    results_dict['means']    = np.mean(np_arr, axis=1).tolist()
    results_dict['stddevs']  = np.std(np_arr, axis=1).tolist()
    results_dict['medians']  = np.median(np_arr, axis=1).tolist()
    results_dict['mins']     = np.min(np_arr, axis=1).tolist()
    results_dict['maxs']     = np.max(np_arr, axis=1).tolist()
    results_dict['samples']  = [np_arr.shape[1]]*5
    results_dict = renormalize_stats(results_dict)
    return results_dict

def renormalize_stats(results_dict:dict) -> dict:
    # check mean value of the middle vial; it should always be 30% (20-40) if we find it is >101, renormalize by dividing by 100
    if results_dict["means"][2] < 101:
        results_dict["renormalized"] = False
        return results_dict
    results_dict["renormalized"] = True
    for key in results_dict.keys():
        if key == "renormalized": continue
        if key == "samples": continue
        results_dict[key] = [x/100 for x in results_dict[key]]
    return results_dict

def get_values_in_roi(pdff, r) -> list:
    mask = np.zeros(pdff.pixel_array.shape, dtype=np.uint8)
    cv2.circle(mask, (r[CX], r[CY]), r[CR], (1), -1) # solid circle (thickness = -1) filled with  1
    vals = apply_mask(pdff.pixel_array, mask)
    return vals

def composite_statistics_fromlist(img_pack_data:list[dict], sliceslocs_to_analyze:list) -> dict:
    # calculate the composite stats for slices in sliceslocs_to_analyze
    masked_values = None
    # extract the pdff mean and median list
    for img in img_pack_data:
        if "rois" not in img:
            continue
        pdff = img["pdff"]
        rois = img["rois"]
        if pdff.SliceLocation not in sliceslocs_to_analyze: #todo: too specific, just use a range
            continue
        if masked_values == None:
            masked_values = [[] for _ in range(len(rois))]
        roi_index = 0
        for r in rois:
            # make a circle mask that can be applied to pdff
            mask = np.zeros(pdff.pixel_array.shape, dtype=np.uint8)
            cv2.circle(mask, (r[0], r[1]), r[2], (1), -1) # solid circle (thickness = -1) filled with  1
            vals = apply_mask(pdff.pixel_array, mask)
            masked_values[roi_index].extend(vals)
            roi_index += 1
    # caste in np.array to take mean of each row, where a row contains the values for rois across slices
    np_arr = np.array(masked_values)
    # calculate mean across slices for a given roi
    results_dict = {}
    results_dict['means'] = np.mean(np_arr, axis=1).tolist()
    results_dict['medians']  = np.median(np_arr, axis=1).tolist()
    results_dict['mins'] = np.min(np_arr, axis=1).tolist()
    results_dict['maxs'] = np.max(np_arr, axis=1).tolist()
    return results_dict

def create_negative_image(img):
    return 255 - np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))

def make_clipped_image(water_img) -> np.ndarray:
    ''' Blacks off the top .65 of an image '''
    water_matlike = np.matrix(water_img)
    cutoff = int(np.floor(water_matlike.shape[0]*.65))
    water_matlike[:cutoff,:] = 0
    normalized_img = np.uint8(cv2.normalize(water_matlike, None, 0, 255, cv2.NORM_MINMAX))
    return normalized_img

def find_packs_in_images(img_pack_data:list[dict], 
        vial_radius,
        radius_tolerance = RADIUS_TOLERANCE_MM,
        vert_align_tol = VERTICAL_ALIGNMENT_TOLERANCE_MM,
        vial_separation = VIAL_SEP_MM,
        ) -> list[dict]:
    ''' 
        Finds circles in pdff/water image pairs and returns a dictionary of 
        pdff_img, water_img, pack_circles
    '''
    for mydict in img_pack_data:
        water_ds = mydict["water"]
        water_img = water_ds.pixel_array
        px_size = water_ds.PixelSpacing[0]
        min_radius, max_radius, min_vail_sep = vial_sizes_in_px(vial_radius, radius_tolerance, px_size)
        water_circles = circles_in_img(water_img, min_radius, max_radius, min_vail_sep)
        if water_circles is None:
            mydict["circles"] = None
            if DEBUG_PLOTS:
                plot_image(water_img, name=str(water_ds.SliceLocation), waitkey=0)
            continue
        if DEBUG_PLOTS:
            water_img = remove_outliers_mad(water_img, threshold=3.5)
            plot_circles_ndarray(water_img, water_circles, name=str(water_ds.SliceLocation), waitkey=0)
        pack_circles = find_phantom_pack(
            water_circles, 
            num_circles=5,
            row_tolerance_px=vert_align_tol/px_size,
            expected_radius=(vial_radius/px_size, np.ceil(2/px_size)),
            expected_sep_px=(vial_separation/px_size, np.ceil(4/px_size))
        )
        if pack_circles is None:
            mydict["circles"] = None
            continue
        pack_circles = sort_circles_by_x(pack_circles)
        mydict["circles"] = pack_circles
    count_circles = [img for img in img_pack_data if img["circles"] is not None]
    print(f"  {len(count_circles)} slices contain phantom pack")
    # if len(count_circles) == 0:
    #     with open("no_packs_found.txt", "w") as f:
    #         f.write(f"Series {water_ds.SeriesNumber}, {water_ds.SeriesDescription}")
    return img_pack_data

def vial_sizes_in_px(vial_radius, radius_tolerance, px_size):
    min_radius = int(vial_radius/px_size) - int(radius_tolerance/px_size)
    max_radius = int(vial_radius/px_size) + int(np.ceil(radius_tolerance/px_size))
    min_vail_sep = vial_radius/px_size
    return min_radius,max_radius,min_vail_sep

def circles_in_img(img, min_radius, max_radius, min_vail_sep):
    cropped_img = make_clipped_image(img)
    # cropped_img = remove_outliers_mad(cropped_img, threshold=3.5)
    circles = find_circles(cropped_img, minDist=min_vail_sep, minRadius=min_radius, maxRadius=max_radius)
    return circles

def circles_in_ds(water_ds, min_radius, max_radius, min_vail_sep):
    cropped_img = make_clipped_image(water_ds.pixel_array)
    cropped_img = remove_outliers_mad(cropped_img, threshold=3.5)
    circles = find_circles(cropped_img, minDist=min_vail_sep, minRadius=min_radius, maxRadius=max_radius)
    return circles

def find_pack_in_pdff(img_pack_data:list[dict],
        vial_radius,
        radius_tolerance = RADIUS_TOLERANCE_MM,
        vert_align_tol = VERTICAL_ALIGNMENT_TOLERANCE_MM,
        roi_radius = ROI_RADIUS_MM,
        ) -> list[dict]:
    ''' 
        Finds circles in pdff/water image pairs and returns a tuple of 
        pdff_img, water_img, pack_circles
    '''
    for mydict in img_pack_data:
        ds = mydict["pdff"]
        px_size = ds.PixelSpacing[0]
        min_radius = int(vial_radius/px_size) - int(radius_tolerance/px_size)
        max_radius = int(vial_radius/px_size) + int(np.ceil(radius_tolerance/px_size))
        min_vail_sep = vial_radius/px_size
        # mask top portion of water image and find circles in what remains
        my_circles = circles_in_pdff(ds, min_radius, max_radius, min_vail_sep)
        if my_circles is None:
            mydict["circles"] = None
            continue
        # pack_circles = find_phantom_pack(
        #     my_circles, 
        #     num_circles=5,
        #     row_tolerance_px=VERTICAL_ALIGNMENT_TOLERANCE_MM/px_size,
        #     expected_radius=(vial_radius/px_size, np.ceil(2/px_size))
        #     )
        if pack_circles is None:
            mydict["circles"] = None
            continue
        pack_circles = sort_circles_by_x(pack_circles)
        # print(pack_circles) #debug, to be sure circles are sorted left to right
        mydict["circles"] = pack_circles
    count_circles = [img for img in img_pack_data if img["circles"] is not None]
    print(f"  found {len(count_circles)} slices where phantom pack is present")
    return img_pack_data

def circles_in_pdff(ds, min_radius, max_radius, min_sep):
    img_for_hough = create_negative_image(ds.pixel_array)
    img_for_hough = make_clipped_image(img_for_hough)
    my_circles = find_circles(img_for_hough, minDist=min_sep, minRadius=min_radius, maxRadius=max_radius)
    return my_circles
    

def sort_circles_by_x(circles:np.ndarray):
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


def sort_data_by_sliceloc(img_pack_data:list[dict]):
    slicelocs = []
    for mydict in img_pack_data:
        slicelocs.append(mydict["water"].SliceLocation)
    slicelocs = sorted(slicelocs)
    # create new list of images/circle tuples orded by location
    new_img_pack_data = []
    for loc in slicelocs:
        for mydict in img_pack_data:
            if mydict["water"].SliceLocation == loc:
                new_img_pack_data.append(mydict)
                break
    return new_img_pack_data

def find_midpoint_slicelocation(locations:list[dict]):
    locations = sorted(list(set(locations))) # remove duplicates
    midpoint_geo = locations[0] + (locations[-1] - locations[0])/2
    midpoint_loc = find_closest_value(locations, midpoint_geo)
    if midpoint_loc is None:
        print(f"  Could not find midpoint")    
        return None
    print(f"  Of {len(locations)} unique slice locations, min {locations[0]}, max {locations[-1]}, midpoint (closest) {midpoint_loc}")
    return midpoint_loc

# def find_midpoint_slicelocation(img_pack_data:list[dict]):
#     locations = []
#     for phc in img_pack_data:
#         if phc["circles"] is None:
#             continue
#         locations.append(phc["water"].SliceLocation)
#     if locations == []:
#         print("  No phantom packs found in images")
#         return
#     locations = sorted(list(set(locations))) # remove duplicates
#     midpoint_geo = locations[0] + (locations[-1] - locations[0])/2
#     midpoint_loc = find_closest_value(locations, midpoint_geo)
#     print(f"  Of {len(locations)} unique slice locations, min {locations[0]}, max {locations[-1]}, midpoint (closest) {midpoint_loc}")
#     return midpoint_loc

def find_slices_in_span(img_pack_data:list[dict], span_mm=50, center_loc=None) -> list[float]:
    # returns a list of slice locations that are within "center" +/- span/2
    # if center is None, it will default to the midpoint of the phantom pack
    locations = []
    for phc in img_pack_data:
        if phc["circles"] is None:
            continue
        locations.append(phc["water"].SliceLocation)
    if locations == []:
        print("  No phantom packs found in images")
    if center_loc is None:
        center_loc = find_midpoint_slicelocation(locations)
    if center_loc is None:
        return [], None
    locations = sorted(list(set(locations))) # remove duplicates
    locations_in_span = [x for x in locations if (x >= center_loc - span_mm/2 and x <= center_loc + span_mm/2)]
    print(f"  {len(locations_in_span)} images in span of +/-{span_mm/2}mm around {center_loc}")
    return locations_in_span, center_loc

def create_rois(img_pack_data:list[dict], roi_radius):
    ''' Draw ROIs in center of all circles, if present'''
    images_to_analyze = [x for x in img_pack_data if x["circles"]]
    for img in images_to_analyze:
        water = img["water"]
        circles = img["circles"]
        rois = create_rois_from_circles(circles, roi_radius/water.PixelSpacing[0])
        img["rois"] = rois
    return 

def create_rois_from_circles(circles, roi_radius_px):
    rois = copy.deepcopy(circles)
    roi_radius_px = np.uint8(roi_radius_px)
    for i in range(len(rois)):
        rois[i][2] = roi_radius_px
    return rois

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

def load_dicoms(directory_path:str, turbo_mode=False):
    if turbo_mode:
        print("...Fast loading mode enabled")
        return load_dicoms_fromdir_quickly(directory_path)
    print("...Standard loading mode enabled")
    return load_dicoms_fromdir(directory_path)

def load_dicoms_fromdir(directory_path:str) -> list[pydicom.Dataset]:
    dicoms = []
    for path, _, files in os.walk(directory_path):
        dicoms.extend(load_dicoms_fromlist(path, files))
    return dicoms

def load_dicoms_fromlist(path, files:list[str]):
    dicoms = []
    for file in files:
        try:
            fp = os.path.join(path, file)
            ds = pydicom.dcmread(fp)
            # ds.filepath = fp # already in filename
            dicoms.append(ds)
        except:
            continue
    return dicoms

def load_dicoms_fromdir_quickly(directory_path, extensions=None):
    dicoms = []
    # find first and last files, run tests, and load data if necessary
    for path, _, files in os.walk(directory_path):
        if len(files) == 0:
            continue
        #todo: remove files if do not match extension
        files.sort()
        indx = 0
        ds_first = None
        while True: #search for first file
            if abs(indx) >= len(files):
                break
            try:
                fp = os.path.join(path, files[indx])
                ds_first = pydicom.dcmread(fp)
                break
            except:
                indx += 1
                continue
        indx = -1
        ds_last = None
        while (ds_first):  # search for last file if a first file was found
            if abs(indx) >= len(files):
                break
            try:
                fp = os.path.join(path, files[indx])
                ds_last = pydicom.dcmread(fp)
                break
            except:
                indx -= 1
                continue
        #
        if ds_first is None or ds_last is None:
            continue

        # test to see if we need to load this directory
        load_these_files = False
        # different series in one dir: load all
        if (ds_first.SeriesInstanceUID != ds_last.SeriesInstanceUID): 
            load_these_files = True
        if (has_relevant_data(ds_first) or has_relevant_data(ds_last)):
            load_these_files = True
        if load_these_files:
            additional_dicoms=load_dicoms_fromlist(path, files)
            dicoms.extend(additional_dicoms) 
    return dicoms

def has_relevant_data(ds:pydicom.Dataset) -> bool:
    # check if any load_labels match tags in this dataset
    image_type = ds.get("ImageType")
    series_desc = ds.get("SeriesDescription")
    if image_type:
        s1 = set(image_type)
        s2 = set(load_labels['image_type'])
        if len(s1.intersection(s2)) > 0:
            return True
    if series_desc:
        if any(element in series_desc.lower() for element in load_labels['series_description']):
            return True
    return False    

def check_for_duplicates(all_dicoms):
    seen = set()
    for ds in all_dicoms:
        if ds.filename in seen:
            print(f"Duplicate file found: {ds.filename}")
        seen.add(ds.filename)

def find_imagetype(dicoms:list[pydicom.Dataset], contrast:str) -> list[pydicom.Dataset]:
    images = []
    for ds in dicoms:
        if contrast in ds.ImageType:
            images.append(ds)
    return images

def find_fw_pairs(all_dicoms:list[pydicom.Dataset]) -> list[dict]:
    # Here were are matchmaking by enforcing the water image_label is same as pdff "label_match" tag
    # AcquisitionTime and SeriesNumber are used to separate series
    # SliceLocating is used to separate images
    img_packs = []
    waters =[x for x in all_dicoms if x.image_label.startswith('water')] 
    pdffs = [x for x in all_dicoms if x.image_label.startswith('pdff')]
    # split up pdff's by series number
    series_numbers = list(set([x.SeriesNumber for x in pdffs]))
    for sn in series_numbers:
        my_img_pack_data = []
        pdff_series = [x for x in pdffs if x.SeriesNumber == sn]
        for ff in pdff_series:
            if ff.image_label.startswith('pdff_fam_bh_offline'):
                pause=True
            if ff.image_label.startswith('pdff_fam_fb_offline'):
                pause=True
            wat_match_label  = [x for x in waters if x.image_label == ff.label_match]
            wat_same_acqtime = [x for x in wat_match_label if acq_time_inrange(ff.AcquisitionTime, x.AcquisitionTime)]
            # wat_same_loc     = [x for x in wat_same_acqtime if x.SliceLocation == ff.SliceLocation]
            wat_same_loc     = [x for x in wat_same_acqtime if int(x.SliceLocation) == int(ff.SliceLocation)]
            if len(wat_same_loc) == 0:
                logger.warning(f"WARNING: PDFF no water match: {ff.SeriesDescription}, SerNum {ff.SeriesNumber}, AcqTime {ff.AcquisitionTime}, Loc {ff.SliceLocation}")
            if len(wat_same_loc) > 1:
                logger.warning(f"WARNING: PDFF no water match: {ff.SeriesDescription} {ff.SeriesNumber} {ff.AcquisitionTime} {ff.SliceLocation}")
                for w in wat_same_loc:
                    logger.warning(f"  {w.SeriesDescription} {w.SeriesNumber} {w.AcquisitionTime} {w.SliceLocation}")
            if len(wat_same_loc) >= 1:
                my_img_pack_data.append({
                    "pdff": ff,
                    "water": wat_same_loc[0],
                    "loc": ff.SliceLocation
                })
        img_packs.append(my_img_pack_data)
    return img_packs

def acq_time_inrange(ff_acqtime, wat_acqtime, rng=1):
    return ((int(wat_acqtime) >= int(ff_acqtime)-rng) and (int(wat_acqtime) <= int(ff_acqtime)+rng))

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


def find_circles(img, minDist=0.01, param1=300, param2=10, minRadius=2, maxRadius=20):
    # # docstring of HoughCircles: 
    # # HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # HOUGH_GRADIENT_ALT is supposed to be more accurate but it doesn't find any circles
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist, param1=param1, param2=0.9, minRadius=minRadius, maxRadius=maxRadius)
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist=0.01, param1=300, param2=0.99)
    return circles

def find_phantom_pack(
        circles_in, 
        num_circles:int=5, row_tolerance_px=5, 
        expected_radius:tuple[float,float]=None, 
        expected_sep_px:tuple[float,float]=None
        ):
    """
    Finds the  group of 5 circles that lie roughly in a horizontal line.

    Parameters:
        circles (list of lists): A list where each element is [x_center, y_center, radius].
        min_num_circles (int): The minimum number of circles in a row keep.
        row_tolerance: The maximum difference in y-coordinates to consider as vertically aligned (in a row).
        expected_radius (tuple[float,float]): A tuple where the first element is the expected radius and the second element is the tolerance.
            discard circles that are not in expected_radius +/- tolerance
        expected_sep_px (tuple[float,float]): the (expected separation, tolerance) between centers of phantom 
    
    Returns:
        list: The group of circles in the horizontal line.
    """
    phantoms = []
    if (circles_in.shape[1] < num_circles):
        return None
    # Sort circles by y-coordinate to facilitate grouping
    circles = np.uint16(np.around(circles_in))
    circles = circles[0,:]
    circles = sorted(circles, key=lambda c: c[1])
    
    # Find all groups of circles that have similar y-coordinates
    # bug: this will duplicate rows, leaving off first entry in each subsequent row
    horizontal_groups = []
    for i in range(len(circles)):
        group = [circles[i]]
        for j in range(i + 1, len(circles)):
            if abs(int(circles[j][1]) - int(circles[i][1])) <= row_tolerance_px:
                group.append(circles[j])
        horizontal_groups.append(group)

    # only keep horizontal groups with at least 5 circles
    phantoms = [g for g in horizontal_groups if len(g) >= num_circles]
    if phantoms == []:
        return None

    # only keep circles in expected_radius +/- px_tolerance
    if expected_radius != None:
        similar_radius = []
        for row in phantoms:
            # phantoms = [g for g in row if (abs(float(c[2]) - expected_radius[0]) <= expected_radius[1] for c in g)]
            g = []
            for c in row:
                radius = abs(float(c[2]) - expected_radius[0]) 
                if radius <= expected_radius[1]:
                    g.append(c)
            similar_radius.append(g)
        phantoms = similar_radius

    # keep only simarly-radius groups that meet minimum num circles
    phantoms = [g for g in phantoms if len(g) >= num_circles]
    if phantoms == []:
        return None
    if len(phantoms) > 1:
        print(f"HACK - {len(phantoms)} found, keeping only first phantom in list")
        phantoms = phantoms[0] # hack: keep only first group in nested list
    if len(phantoms) == 1:
        print("Note - keeping only first phantom in list")
        phantoms = phantoms[0] # hack: keep only first group in nested list
    

    # only keep circles in expected_sep +/- px_tolerance
    if expected_sep_px != None:
        remove_indeces = []
        phantoms = sort_circles_by_x(phantoms)
        for j in range(len(phantoms)-1):
            phantom_sep = float(phantoms[j+1][0]) - float(phantoms[j][0])
            if (abs(phantom_sep - expected_sep_px[0]) > expected_sep_px[1]):
                print(f"found cirlce outlier")
                remove_indeces.append(j)
                # plot_image(water_img, name=str(water_ds.SliceLocation), waitkey=0)
        sorted(remove_indeces, reverse=True)
        for i in remove_indeces:
            phantoms.pop(i)

    return phantoms

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

def plot_results(img_pack_data:list[dict], dest_filepath:str=None, display_image=False):
    """Plot PDFF and water images with ROIs using OpenCV."""
    cols = 2
    rows = np.uint32(len(img_pack_data))
    # Create a blank canvas to hold the images
    cimg_setup = cv2.cvtColor(np.uint8(img_pack_data[0]["water"].pixel_array), cv2.COLOR_GRAY2BGR) #bug: assumes all images same resolution
    height, width, channels = cimg_setup.shape

    canvas = np.zeros((height * rows, width * cols, channels), dtype=np.uint8)
    for i, mydict in enumerate(img_pack_data):
        cimg_water = np.uint8(cv2.normalize(mydict["water"].pixel_array, None, 0, 255, cv2.NORM_MINMAX))
        cimg_water = cv2.cvtColor(cimg_water, cv2.COLOR_GRAY2BGR)
        cimg_pdff = np.uint8(cv2.normalize(mydict["pdff"].pixel_array, None, 0, 255, cv2.NORM_MINMAX))
        cimg_pdff = cv2.cvtColor(cimg_pdff, cv2.COLOR_GRAY2BGR)
        if mydict["circles"] is not None:
            np_circles = np.uint16(np.around(mydict["circles"]))
            for c in np_circles:
                cv2.circle(cimg_water,(c[0],c[1]),c[2],(0,0,255),1)             # draw the outer circle
        if "rois" in mydict:
            np_rois = np.uint16(np.around(mydict["rois"]))
            mystats = slice_stats(mydict)
            for j, c in enumerate(np_rois):
                cv2.circle(cimg_pdff, (c[0],c[1]),c[2],(255,255,0),1)
                mystr = f"{mystats['pdff_means'][j]:.1f}"
                text_size, _ = cv2.getTextSize(mystr, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                text_w, text_h = text_size
                mypt = (c[0]-3*c[2],c[1]+4*c[2]+text_h) # default/odd, plot below vial
                if (j % 2 == 0): #even, plot above  vial 
                    mypt = (c[0]-3*c[2],c[1]-4*c[2])
                
                cv2.rectangle(cimg_pdff, (mypt[0], mypt[1]), (mypt[0] + text_w, mypt[1] - text_h), (0,0,0), -1)
                cv2.putText(cimg_pdff, mystr, mypt, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)
        # plot slice location on bottom of pdff image
        loc_nodecimals = f"{float(mydict['pdff'].get('SliceLocation')):.1f}"
        loc_str = f"LOC: {loc_nodecimals}"
        loc_fontscale = 0.6
        loc_size, _ = cv2.getTextSize(loc_str, cv2.FONT_HERSHEY_SIMPLEX, loc_fontscale, 1)
        loc_w, loc_h = loc_size
        loc_pt = (int(width/2 - loc_w/2), 2*loc_h)
        cv2.rectangle(cimg_pdff, loc_pt, (loc_pt[0] + loc_w, loc_pt[1] - loc_h), (0,0,0), -1)
        cv2.putText(cimg_pdff, loc_str, loc_pt, cv2.FONT_HERSHEY_SIMPLEX, loc_fontscale, (255,255,0), 1)
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

def find_closest_value(mylist:list, target_value):
    """Finds the number in a list closest to a given target value."""
    return min(mylist, key=lambda x: abs(x - target_value))

def apply_mask(image, mask) -> list:
    np_img = np.array(image)
    vals = np_img[mask == 1].tolist() # need to be a list? or leave as np.array??
    return vals

def masked_mean(image, mask) -> float:
    vals = apply_mask(image, mask)
    if len(vals) == 0:
        return None
    return sum(vals) / len(vals)

def masked_median(image, mask) -> float:
    vals = apply_mask(image, mask)
    if len(vals)  == 0:
        return None
    if len(vals) % 2 == 0:
        return (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
    return  vals[len(vals) // 2 ] 

def masked_stddev(image, mask) -> float:
    vals = apply_mask(image, mask)
    if len(vals)  == 0:
        return None
    return  np.std(vals)

def get_image_info(img_pack_data:list[dict]) -> dict:
    info = {}
    pdff = img_pack_data[0]["pdff"]
    water = img_pack_data[0]["water"]
    for tag in DICOM_TAG_LIST:
        info[tag] = str(pdff.get(tag))
    info["SeriesDescription_pdff"] = pdff.get("SeriesDescription")
    info["SeriesDescription_water"] = water.get("SeriesDescription")
    info["SeriesNumber_pdff"] = pdff.get("SeriesNumber")
    info["SeriesNumber_water"] = water.get("SeriesNumber")
    return info

def plot_slice_values(img_pack_data:list[dict], vert_lines=[], directory_path=''):
    # each entry in this will will be the 5 vials  in a slice
    pdff_means = []
    pdff_medians = []
    pdff_stddevs = []
    slice_locations = []
    pdff_series_number = img_pack_data[0]["pdff"].SeriesNumber
    for img in img_pack_data:
        mystats = slice_stats(img)
        pdff_means.append(mystats['pdff_means'])
        pdff_medians.append(mystats['pdff_medians'])
        pdff_stddevs.append(mystats['pdff_stddevs'])
        slice_locations.append(img["pdff"].SliceLocation)
    #convert to np.array and transpose, so that each row is one roi loc across slices
    data_means = np.array(pdff_means).transpose()
    data_medians = np.array(pdff_means).transpose()
    data_stddevs = np.array(pdff_stddevs).transpose()


    for i in range(data_means.shape[0]):
        plt.errorbar(slice_locations, data_means[i],  yerr=data_stddevs[i], fmt='-o', label=f"Mean {i}")
    if len(vert_lines) > 0:
        plt.axvline(x=vert_lines[0],  color='r', linestyle='--', linewidth=1) # vertical lines at edge of selected slices
        plt.axvline(x=vert_lines[-1], color='r', linestyle='--', linewidth=1)
    plt.title("Mean +/- StdDev across slices for each ROI")
    plt.xlabel("Slice Location")
    plt.ylabel("Mean PDFF")
    plt.legend()
    plt.grid(True)
    # plt.show()
    img_filepath = os.path.join(directory_path, f"{img_pack_data[0]['pdff'].PatientName}_{pdff_series_number}_mean_stddev.png")
    plt.savefig(img_filepath)
    plt.close()

    for i in range(data_means.shape[0]):
        plt.plot(slice_locations, data_medians[i], '-x', label=f"Median {i}")
    if len(vert_lines) > 0: 
        plt.axvline(x=vert_lines[0],  color='r', linestyle='--', linewidth=1) # vertical lines at edge of selected slices
        plt.axvline(x=vert_lines[-1], color='r', linestyle='--', linewidth=1)
    plt.title("Meadian PDFF across slices for each ROI")
    plt.xlabel("Slice Location")
    plt.ylabel("Median PDFF")
    plt.legend()
    plt.grid(True)
    # plt.show()
    img_filepath = os.path.join(directory_path, f"{img_pack_data[0]['pdff'].PatientName}_{pdff_series_number}_median.png")
    plt.savefig(img_filepath)
    plt.close()
    

def remove_outliers_mad(image, threshold=3.5):
    median = np.median(image)
    mad = np.median(np.abs(image - median))
    # MAD to standard deviation approximation (if needed): σ ≈ 1.4826 * MAD
    modified_z_scores = 0.6745 * (image - median) / (mad + 1e-6)
    mask = np.abs(modified_z_scores) < threshold
    # num_outliers = np.sum(mask==False)
    # if num_outliers > 0:
    #     logging.info(f"Removed {num_outliers} outliers from image")
    return np.where(mask, image, 0)  # replace outliers with 0 


if __name__ == "__main__":
    directory_path = sys.argv[1]
    logfile = os.path.join(directory_path, "phantom_pack.log")
    logging.basicConfig(filename=logfile, level=logging.INFO)
    print(f"Processing {directory_path}")
    results = phantom_pack(directory_path)
    

    # testimg_path = r'C:\testdata\PhantomPack\PQ024\SER00090\IMG00019.dcm'
    # directory_path = os.path.dirname(testimg_path)
    # ds = pydicom.dcmread(testimg_path)
    # circles_in_pdff(ds, min_radius=1, max_radius=60, min_sep=1)


