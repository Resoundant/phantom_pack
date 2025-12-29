import copy
import numpy as np
import cv2
import pydicom
import os
import json
# from img_utils import plot_results
from pp_config import DICOM_TAG_LIST

import logging
logger = logging.getLogger(__name__)


# circle index helpers
CX = 0
CY = 1
CR = 2



class FWSeries:
    def __init__(self, series_number:int):
        self.series_number = series_number
        self.series_number_pdff = series_number
        self.series_number_water = -1
        self.series_description_pdff = ""
        self.series_description_water = ""
        self.image_pairs:list[FWImagePair] = []
        self.pack_midpoint:float = -999.9
        self.stats_min_loc = -999.9
        self.stats_max_loc = -999.9
        self.pdff_metadata:pydicom.Dataset|None = None
        self.water_metadata:pydicom.Dataset|None = None

    
    def find_pack_midpoint(self) -> float:
        midpoint = find_midpoint([x.location_full for x in self.image_pairs if x.has_circles()])
        if midpoint is None:
            self.pack_midpoint = -999.9
        else:
            self.pack_midpoint = midpoint
        return self.pack_midpoint


    def number_of_slices_in_span(self, span_mm: float, center_loc: float | None ) -> int:
        if center_loc is None:
            return 0
        min_loc = center_loc - span_mm / 2
        max_loc = center_loc + span_mm / 2
        slices_in_span = [x.location_full for x in self.image_pairs if (min_loc <= x.location_full <= max_loc)]
        return len(slices_in_span)


    def create_rois(self, roi_radius):
        ''' Draw ROIs in center of all circles, if present'''
        # pairs_to_analyze = [x for x in fw_series.image_pairs if x.has_circles()]
        # for img_pair in pairs_to_analyze:
        for img_pair in self.image_pairs:
            if not img_pair.has_circles():
                continue
            roi_rad_px = roi_radius/img_pair.pixel_spacing
            img_pair.rois = create_rois_from_circles(img_pair.circles, roi_rad_px)
        return 


    def find_pack_locations(self):
        all_locs = [x.location_full for x in self.image_pairs]
        pack_locs = [x.location_full for x in self.image_pairs if x.has_circles()]
        if len(pack_locs) == 0:
            self.pack_first_slice_loc = None
            self.pack_last_slice_loc = None
            self.pack_first_slice = None
            self.pack_last_slice = None
            return
        self.pack_first_slice_loc = min(pack_locs)
        self.pack_last_slice_loc = max(pack_locs)
        self.pack_first_slice = all_locs.index(self.pack_first_slice_loc)
        self.pack_last_slice = all_locs.index(self.pack_last_slice_loc)


    def sort_data_by_sliceloc(self):
        ''' Re-orders the fw_sereies.image_paris list by slice location, ascending) '''
        self.image_pairs = sorted(self.image_pairs, key=lambda x: x.location)


    def img_pairs_in_span(self, min_loc:float, max_loc:float):
        images_in_span = [x for x in self.image_pairs if min_loc <= x.location_full <= max_loc]
        images_in_span = sorted(images_in_span, key=lambda x: x.location_full)
        return images_in_span


    def get_image_info(self) -> dict:
        info = extract_dicom_tags(self.pdff_metadata)
        if self.pdff_metadata is not None:
            info["SeriesDescription_pdff"] = self.pdff_metadata.get("SeriesDescription")
            info["SeriesNumber_pdff"] = self.pdff_metadata.get("SeriesNumber")
        if self.water_metadata is not None:
            info["SeriesDescription_water"] = self.water_metadata.get("SeriesDescription")
            info["SeriesNumber_water"] = self.water_metadata.get("SeriesNumber")
        return info


    def metadata(self) -> dict:
        return extract_dicom_tags(self.pdff_metadata)

    def compute_stats_and_metadata(self, span_mm):
        stats_min_loc = self.pack_midpoint - span_mm / 2
        stats_max_loc = self.pack_midpoint + span_mm / 2
        composite_results = self.composite_statistics(stats_min_loc, stats_max_loc)
        image_info = self.metadata()
        return stats_min_loc, stats_max_loc, composite_results, image_info

    def save_plot_images(self, output_dir, image_info, stats_min_loc, stats_max_loc):
        patient_name = image_info.get("PatientName", "unknown")
        series_number = image_info.get("SeriesNumber_pdff", "unknown")
        array_filepath = os.path.join(output_dir, f"{patient_name}_{series_number}_allimg.png")
        # TODO restore
        # plot_results(fw_serie.image_pairs, dest_filepath=array_filepath, display_image=False)
        array_filepath = os.path.join(output_dir, f"{patient_name}_{series_number}_selected.png")
        image_pairs_in_span = self.img_pairs_in_span(min_loc=stats_min_loc, max_loc=stats_max_loc)
        # plot_results(image_pairs_in_span, dest_filepath=array_filepath, display_image=False)

    def save_results_json(self, output_dir, results):
        if results:
            patient_name = results.get("PatientName", "unknown")
            series_number = results.get("SeriesNumber_pdff", "unknown")
            file_path = os.path.join(output_dir, f"{patient_name}_{series_number}.json")
            with open(file_path, 'w') as file:
                json.dump(results, file, indent=4)
            logger.info(f"  JSON data saved to {file_path}")

    def compute_and_save_results(self, span_mm, output_dir) -> dict:
        # this code is a bit rigid in expecting regularly structed data (same number of packs, same pixels in each ROI, etc))
        # to avoid it crashing the whole works, if something doesn't finish, it will except and move on, saving no data or
        # partial data
        stats_min_loc = 0
        stats_max_loc = 0
        composite_results = {}
        image_info = {}
        try:
            stats_min_loc, stats_max_loc, composite_results, image_info = self.compute_stats_and_metadata(span_mm)
        except:
            logger.warning(f"Error computing comsposite statistics for series {self.series_number_pdff} {self.series_description_pdff}")

        try:
            self.save_plot_images(output_dir, image_info, stats_min_loc, stats_max_loc)
        except:
            logger.warning(f"Error saving plots for series {self.series_number_pdff} {self.series_description_pdff}")

        try:
            # save plots of slice values
            # plot_slice_values(fw_serie, vert_lines=[stats_min_loc, stats_max_loc], directory_path=output_dir)
            results = composite_results | image_info
            self.save_results_json(output_dir, results)
            return results
        except:
            logger.warning(f"Error computing per-slice for series {self.series_number_pdff} {self.series_description_pdff}")
            return {}

    def composite_statistics(self, stats_min_loc, stats_max_loc) -> dict:
        '''
        Calculate the composite stats for slices in range (min_loc, max_loc)
        Output (dict): means:[], stddevs:[], medians:[], mins:[], maxs:[], samples:[]
        '''
        # sort all circles and rois by x-coord so ROIs are grouped consistently
        # for img_pair in self.image_pairs:
            # if img_pair.has_circles():
            #     img_pair.circles = sorted(img_pair.circles, key=lambda x: x[CX])
            # if img_pair.has_rois():
            #     img_pair.rois = sorted(img_pair.rois, key=lambda x: x[CX])

        # quickly check that same number of ROIs in all images
        num_rois_in_images = []
        for img_pair in self.image_pairs:
            if img_pair.has_rois():
                num_rois_in_images.append(len(img_pair.rois))
        if len(set(num_rois_in_images)) > 1:
            logger.warning("WARNING: not all images have same number of ROIs! This may cause issues")

        # check that all of the rois are aligned across slices by radius overlap
        for i in range(len(self.image_pairs) - 1):
            if self.image_pairs[i].has_rois() and self.image_pairs[i+1].has_rois():
                for j in range(len(self.image_pairs[i].rois)):
                    x_sep = abs(float(self.image_pairs[i].rois[j][CX]) - float(self.image_pairs[i+1].rois[j][CX]))
                    y_sep = abs(float(self.image_pairs[i].rois[j][CY]) - float(self.image_pairs[i+1].rois[j][CY]))
                    if x_sep > self.image_pairs[i].rois[j][CR]:
                        logger.warning("WARNING: ROIs not aligned across slices!")
                    if y_sep > self.image_pairs[i].rois[j][CR]:
                        logger.warning("WARNING: ROIs not aligned across slices!")

        if not num_rois_in_images:
            return {
                "means": [],
                "stddevs": [],
                "medians": [],
                "mins": [],
                "maxs": [],
                "samples": [],
            }

        # to calc mean, create lists made up of all pixels value in rois across all slices in range
        roi_counts = {}
        for count in num_rois_in_images:
            roi_counts[count] = roi_counts.get(count, 0) + 1
        num_rois_mode = max(roi_counts, key=roi_counts.get)
        masked_values = [[] for _ in range(num_rois_mode)]
        for img_pair in self.image_pairs:
            if (img_pair.location_full > stats_max_loc) or (img_pair.location_full < stats_min_loc):
                continue
            if not img_pair.has_rois():
                continue
            for roi_index, roi in enumerate(img_pair.rois):
                if roi_index >= num_rois_mode:
                    break
                mask = np.zeros(img_pair.pdff_img.shape, dtype=np.uint8)
                cv2.circle(mask, (int(roi[CX]), int(roi[CY])), int(roi[CR]), 1, -1)
                vals = img_pair.pdff_img[mask == 1]
                masked_values[roi_index].extend(vals.tolist())

        results_dict = {
            "means": [],
            "stddevs": [],
            "medians": [],
            "mins": [],
            "maxs": [],
            "samples": [],
        }
        for roi_vals in masked_values:
            if not roi_vals:
                results_dict["means"].append(float("nan"))
                results_dict["stddevs"].append(float("nan"))
                results_dict["medians"].append(float("nan"))
                results_dict["mins"].append(float("nan"))
                results_dict["maxs"].append(float("nan"))
                results_dict["samples"].append(0)
                continue
            np_vals = np.asarray(roi_vals)
            results_dict["means"].append(float(np.mean(np_vals)))
            results_dict["stddevs"].append(float(np.std(np_vals)))
            results_dict["medians"].append(float(np.median(np_vals)))
            results_dict["mins"].append(float(np.min(np_vals)))
            results_dict["maxs"].append(float(np.max(np_vals)))
            results_dict["samples"].append(int(np_vals.size))
        if len(results_dict["means"]) > 2:
            results_dict = renormalize_stats(results_dict)
        return results_dict


class FWImagePair:
    '''
    This class contains a pair of pdff and water images, as would be
    found in a DICOM series at the same location.

    It is pure numeric, no DICOM metadata is allowed (for test and simulation)
    '''
    def __init__(self, pdff_img:np.ndarray, water_img:np.ndarray, pixel_spacing:float, location_full:float=-999.0):
        # self.pdff = pdff
        # self.water = water
        self.pdff_img = pdff_img
        self.water_img = water_img
        self.circles = np.array([])
        self.rois = np.array([])
        self.location_full:float = location_full
        self.location:int = int(location_full)
        self.pixel_spacing = pixel_spacing

    # def set_pixel_spacing(self):
    #     self.pixel_spacing = self.pdff.PixelSpacing[0]
    #     if self.pdff.PixelSpacing[0] != self.water.PixelSpacing[0]:
    #         logger.warning(f"Pixel spacing mismatch! {self.pdff.SeriesDescription}")

    def has_circles(self):
        return len(self.circles) != 0
    
    def has_rois(self):
        return len(self.rois) != 0
    
    def slice_stats(self) -> dict:
        # Compute the mean, median, and standard dev of a pdff/water pair.
        # img should be a single slice of the img_pack_data, with pdff, water and rois
        # Output: pdff_means:[], pdff_stddevs:[], pdff_medians:[] - lists of mean, stddev, median
        # for the circles
        stats = {}
        # default values are -10
        if self.has_rois() is False:
            roi_count = len(self.rois)
            stats["pdff_means"] = [-10] * roi_count
            stats["pdff_medians"] = [-10] * roi_count
            stats["pdff_stddevs"] = [0] * roi_count
            return stats
        # apply rois to PDFF
        pdff_means = []
        pdff_medians = []
        pdff_stddevs = []
        for r in self.rois:
            # make a circle mask that can be applied to pdff
            mask = np.zeros(self.pdff_img.shape, dtype=np.uint8)
            cv2.circle(mask, (r[0], r[1]), r[2], color=1, thickness=-1) # solid circle (thickness = -1) filled with  1
            # calculate mean and median
            mean_pdff   = masked_mean(self.pdff_img, mask)
            median_pdff = masked_median(self.pdff_img, mask)
            stddev_pdff = masked_stddev(self.pdff_img, mask)
            pdff_means.append(mean_pdff)
            pdff_medians.append(median_pdff)
            pdff_stddevs.append(stddev_pdff)
        stats["pdff_means"] = pdff_means
        stats["pdff_medians"] = pdff_medians
        stats["pdff_stddevs"] = pdff_stddevs
        return stats
            
def apply_mask(image, mask) -> list:
    np_img = np.array(image)
    vals = np_img[mask == 1].tolist() # need to be a list? or leave as np.array??
    return vals

def masked_mean(image, mask) -> float|None:
    vals = apply_mask(image, mask)
    if len(vals) == 0:
        return None
    return sum(vals) / len(vals)

def masked_median(image, mask) -> float|None:
    vals = apply_mask(image, mask)
    if len(vals)  == 0:
        return None
    if len(vals) % 2 == 0:
        return (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
    return  vals[len(vals) // 2 ]

def masked_stddev(image, mask) -> float|None:
    vals = apply_mask(image, mask)
    if len(vals)  == 0:
        return None
    return  (np.std(vals))

''' 
required dicom tags
SeriesNumber
SeriesDescription
AcquisitionTime
SliceLocation
'''
def check_required_tags(ds:pydicom.Dataset) -> bool:
    if ds.SeriesNumber and ds.SeriesDescription and ds.AcquisitionTime and ds.SliceLocation:
        return True
    return False

def check_required_tags_all(dicoms:list[pydicom.Dataset]) -> bool:
    for ds in dicoms:
        if not check_required_tags(ds):
            return False
    return True

def find_fw_pairs(all_dicoms:list[pydicom.Dataset]) -> list[FWSeries]:
    # Here were are matchmaking by enforcing the water image_label is same as pdff "label_match" tag
    # AcquisitionTime and SeriesNumber are used to separate series
    # SliceLocating is used to separate images

    if not check_required_tags_all(all_dicoms):
        raise Exception("Not all DICOMs contain required tags (SeriesNumber, SeriesDescription, AcquisitionTime, SliceLocation)")

    # find and match series
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
                logger.warning(f"WARNING: PDFF has no water match: ")
                logger.warning(f"{ff.SeriesDescription}, SeriesNumber {ff.SeriesNumber}, AcqTime {ff.AcquisitionTime}, Loc {ff.SliceLocation}")
            if len(wat_same_loc) > 1:
                logger.warning(f"WARNING: PDFF has multiple water match: ")
                logger.warning(f"{ff.SeriesDescription}, SeriesNumber {ff.SeriesNumber}, AcqTime {ff.AcquisitionTime}, Loc {ff.SliceLocation}")
                logger.warning(f"matching these water images; using the first match:")
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

def acq_time_inrange(ff_acqtime, wat_acqtime, rng=1):
    return ((int(wat_acqtime) >= int(ff_acqtime)-rng) and (int(wat_acqtime) <= int(ff_acqtime)+rng))

def find_midpoint(locations: list) -> float | None:
    """Finds the midpoint of a list of values."""
    if not locations:
        return None
    locations = sorted(list(set(locations)))  # remove duplicates
    midpoint = (locations[0] + locations[-1]) / 2
    return midpoint

def create_rois_from_circles(circles:np.ndarray, roi_radius_px) -> np.ndarray:
    rois = copy.deepcopy(circles)
    rois = np.uint16(np.around(rois)) # controversial: this rounds off circles that may have fractional (x,y) centers
    roi_radius_px = np.uint16(roi_radius_px)
    # for i in range(rois.shape[0]):
    #     rois[i][2] = roi_radius_px
    for roi in rois:
        roi[2] = roi_radius_px
    return rois

def vial_sizes_in_px(vial_radius:float, radius_tolerance:float, px_size:float) -> tuple[float, float, float]:
    min_radius = float(vial_radius/px_size) - float(radius_tolerance/px_size)
    max_radius = float(vial_radius/px_size) + float(np.ceil(radius_tolerance/px_size))
    min_vail_sep = vial_radius/px_size
    return min_radius, max_radius, min_vail_sep

def extract_dicom_tags(dataset:pydicom.Dataset|None) -> dict:
    info = {}
    if dataset is None:
        for tag in DICOM_TAG_LIST:
            info[tag] = ""
        return info
    for tag in DICOM_TAG_LIST:
        info[tag] = str(dataset.get(tag))
    return info


def get_values_in_roi(img:np.ndarray, r) -> list:
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(mask, (r[CX], r[CY]), r[CR], 1, -1) # solid circle (thickness = -1) filled with  1
    vals = apply_mask(img, mask)
    return vals

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


# def plot_circles_ndarray(img, circles, name='image', waitkey=1):
#     cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
#     cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
#     np_circles = np.uint16(np.around(circles))
#     for c in np_circles[0,:]: # don't remembwer what packing needed this slice
#         cv2.circle(cimg,(c[CX],c[CY]),c[CR],(0,255,0),2)
#     cv2.imshow(name, cimg)
#     cv2.waitKey(waitkey)
#     cv2.destroyAllWindows()
