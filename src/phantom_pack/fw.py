from __future__ import annotations

import copy
import numpy as np
import cv2
import pydicom
import os
import json
from pathlib import Path
from .pp_config import DICOM_TAG_LIST
import matplotlib.pyplot as plt


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
    

    def plot_composite_means(self, composite_results, output_dir):
        if composite_results is None:
            return
        means = composite_results.means if isinstance(composite_results, FWStats) else composite_results.get("means", [])
        means = np.asarray(means, dtype=float)
        if means.size == 0:
            return
        os.makedirs(output_dir, exist_ok=True)
        x_vals = np.arange(means.size)
        plt.figure(figsize=(6, 3))
        plt.plot(x_vals, means, marker='o', linewidth=1)
        plt.title(f"Composite Means {self.series_number_pdff}")
        plt.xlabel("ROI Index")
        plt.ylabel("Mean")
        plt.grid(True, alpha=0.3)
        patient_name = "unknown"
        if self.pdff_metadata is not None:
            patient_name = self.pdff_metadata.get("PatientName", "unknown")
        filename = f"{patient_name}_{self.series_number_pdff}_composite_means.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()


    def plot_roi_values_linear(self, composite_results, output_dir):
        if composite_results is None:
            return
        values = composite_results.values if isinstance(composite_results, FWStats) else composite_results.get("values", [])
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return
        os.makedirs(output_dir, exist_ok=True)
        x_vals = np.arange(values.size)
        plt.figure(figsize=(6, 3))
        plt.plot(x_vals, values, marker='o', linewidth=1)
        plt.title(f"Composite Means {self.series_number_pdff}")
        plt.xlabel("ROI Index")
        plt.ylabel("Mean")
        plt.grid(True, alpha=0.3)
        patient_name = "unknown"
        if self.pdff_metadata is not None:
            patient_name = self.pdff_metadata.get("PatientName", "unknown")
        filename = f"{patient_name}_{self.series_number_pdff}_roi_values_lin.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
    

    def save_plot_images(self, output_dir, image_info, stats_min_loc, stats_max_loc):
        patient_name = image_info.get("PatientName", "unknown")
        series_number = self.series_number
        # TODO restore
        # plot_results(fw_serie.image_pairs, dest_filepath=array_filepath, display_image=False)
        array_filepath = os.path.join(output_dir, f"{patient_name}_{series_number}_selected.png")
        image_pairs_in_span = self.img_pairs_in_span(min_loc=stats_min_loc, max_loc=stats_max_loc)
        # plot_results(image_pairs_in_span, dest_filepath=array_filepath, display_image=False)
        hp_img_fp = Path(output_dir) / f'{series_number}_hepplus_img.png'
        hepplus_img = self.create_save_hepplus_img_array(filepath=hp_img_fp)


    def create_save_hepplus_img_array(self, filepath=None) -> np.ndarray:
        image_pairs_in_span = self.img_pairs_in_span(min_loc=self.stats_min_loc, max_loc=self.stats_max_loc)
        pack_arr_img = pack_array(image_pairs_in_span)
        # plot_utils.display_cimg(pack_arr_img)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, pack_arr_img)
        return pack_arr_img


    def save_results_json(self, output_dir, results):
        if results:
            results_dict = results.to_dict() if isinstance(results, FWStats) else results
            patient_name = results_dict.get("PatientName", "unknown")
            series_number = results_dict.get("SeriesNumber_pdff", "unknown")
            file_path = os.path.join(output_dir, f"{patient_name}_{series_number}.json")
            with open(file_path, 'w') as file:
                json.dump(results_dict, file, indent=4)
            logger.info(f"  JSON data saved to {file_path}")

    def slice_stats(self):
        for fw in self.image_pairs:
            fw.slice_stats()


    def compute_and_save_results(self, span_mm, output_dir) -> dict:
        # this code is a bit rigid in expecting regularly structed data (same number of packs, same pixels in each ROI, etc))
        # to avoid it crashing the whole works, if something doesn't finish, it will except and move on, saving no data or
        # partial data
        stats_min_loc = self.pack_midpoint - span_mm / 2
        stats_max_loc = self.pack_midpoint + span_mm / 2 
        image_info = {}

        # slice stats
        try:
            self.slice_stats()
        except:
            logger.warning(f"Error computing per-slices statistics for series {self.series_number_pdff} {self.series_description_pdff}")

        # composite stats
        try:
            composite_results = self.composite_statistics(self.pack_midpoint, span_mm, max_slices=3)
            self.plot_composite_means(composite_results, output_dir)
        except:
            logger.warning(f"Error computing comsposite statistics for series {self.series_number_pdff} {self.series_description_pdff}")

        # images and plots
        try:
            self.save_plot_images(output_dir, image_info, stats_min_loc, stats_max_loc)
        except:
            logger.warning(f"Error saving plots for series {self.series_number_pdff} {self.series_description_pdff}")

        # json results
        try:
            self.save_results_json(output_dir, composite_results)
            return composite_results
        except:
            logger.warning(f"Error computing per-slice for series {self.series_number_pdff} {self.series_description_pdff}")
            return {}

    def composite_statistics(self, center_mm, range_mm, max_slices:int = 0):
        '''
        Calculate the composite stats for slices in range (min_loc, max_loc)
        
        '''
        # sort all circles and rois by x-coord so ROIs are grouped consistently
        for fw in self.image_pairs:
            fw.sort_circles_by_x_coord()

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
                        logger.warning("WARNING: ROIs not aligned across slices (horizontally)!")
                    if y_sep > self.image_pairs[i].rois[j][CR]:
                        logger.warning("WARNING: ROIs not aligned across slices (vertically)!")

        # limit min and max location to include (at most) max_slices
        stats_min_loc = center_mm - range_mm // 2
        stats_max_loc = center_mm + range_mm // 2
        slices_in_range = sorted(
            (
                img_pair for img_pair in self.image_pairs
                if stats_min_loc <= img_pair.location_full <= stats_max_loc
            ),
            key=lambda img_pair: img_pair.location_full
        )
        if (max_slices > 0) and (len(slices_in_range) > max_slices):
            slices_in_range = list(slices_in_range)
            slices_in_range.sort(key=lambda img_pair: abs(img_pair.location_full - center_mm))
            slices_in_range = slices_in_range[:max_slices]
            # stats_min_loc = slices_in_range[0]
            # stats_max_loc = slices_in_range[-1]

        # to calc mean, create lists made up of all pixels value in rois across all slices in range
        roi_counts = np.bincount(num_rois_in_images)
        num_rois_mode = int(np.argmax(roi_counts))
        masked_values = [[] for _ in range(num_rois_mode)]
        for img_pair in slices_in_range:
            if not img_pair.has_rois():
                continue
            img_pair.slice_stats() # this is already done but re-do should be ok
            if img_pair.pdff_stats == None:
                    continue
            for indx, samples in enumerate(img_pair.pdff_stats.samples):
                masked_values[indx].extend(samples.tolist())
        # todo: do something with these masked_values
        return 


class FWStats:
    def __init__(self, image:np.ndarray, rois:np.ndarray):
        n = len(rois)
        self.num_samples = np.zeros(n, dtype=int)
        self.samples = np.empty(n, dtype=object)
        self.samples[:] = [[] for _ in range(n)]
        self.roi_stats:list[RoiStats | None] = [None] * n
        self.means   = np.zeros(n, dtype=float)
        self.stddevs = np.zeros(n, dtype=float)
        self.medians = np.zeros(n, dtype=float)
        self.iqrs    = np.empty(n, dtype=object)
        self.mins    = np.zeros(n, dtype=float)
        self.maxs    = np.zeros(n, dtype=float)
        self.renormalized = False
        self.renormalized_divisor = 1

        for i, roi in enumerate(rois):
            roi_stats               = RoiStats(image, roi)
            self.roi_stats[i]       = roi_stats
            self.num_samples[i]     = roi_stats.num_samples
            self.samples[i]         = roi_stats.samples
            self.means[i]           = roi_stats.mean
            self.stddevs[i]         = roi_stats.stddev
            self.medians[i]         = roi_stats.median
            self.iqrs[i]            = roi_stats.iqr
            self.mins[i]            = roi_stats.min
            self.maxs[i]            = roi_stats.max

    @classmethod
    def from_values(cls, values_by_roi:list[list[float]]) -> FWStats:
        obj = cls.__new__(cls)
        for i, roi_vals in enumerate(values_by_roi):
            if not roi_vals:
                continue
            np_vals = np.asarray(roi_vals)
            obj.means[i]        = float(np.mean(np_vals))
            obj.stddevs[i]      = float(np.std(np_vals))
            obj.medians[i]      = float(np.median(np_vals))
            obj.mins[i]         = float(np.min(np_vals))
            obj.maxs[i]         = float(np.max(np_vals))
            obj.samples[i]      = np_vals
            obj.num_samples[i]  = int(np_vals.size)
        return obj

    def to_dict(self) -> dict:
        data = {
            "means": self.means.tolist(),
            "stddevs": self.stddevs.tolist(),
            "medians": self.medians.tolist(),
            "mins": self.mins.tolist(),
            "maxs": self.maxs.tolist(),
            "num_samples": [int(x) for x in self.num_samples],
            "values": self.samples.tolist(),
            "renormalized": self.renormalized,
        }
        return data
    
    def renormalize(self):
        means = self.means
        self.renormalized = bool(np.count_nonzero(means > 100) > (means.size / 2))
        self.renormalized_divisor = 1
        if self.renormalized:
            divisors = [1, 10, 100, 1000]
            max_mean = float(np.max(means))
            for d in divisors:
                if max_mean < d * 120:
                    self.renormalized_divisor = d
                    break
            self.means   = self.means / self.renormalized_divisor
            self.stddev = self.medians / self.renormalized_divisor
            self.medians = self.medians / self.renormalized_divisor
            self.iqrs = self.iqrs / self.renormalized_divisor
            self.mins = self.medians / self.renormalized_divisor
            self.maxs = self.medians / self.renormalized_divisor
            


class RoiStats:
    def __init__(self, image:np.ndarray, roi:np.ndarray):
        self.roi = roi
        self.num_samples    = 0
        self.samples        = np.zeros
        self.mean           = np.nan
        self.stddev         = np.nan
        self.median         = np.nan    
        self.iqr            = np.nan
        self.min            = np.nan
        self.max            = np.nan

        # if not image:
        #     return
        # if not roi:
        #     return

        center_x = roi[0]
        center_y = roi[1]
        radius   = roi[2]
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        vals = image[mask]
        if vals.size > 0:
            self.samples        = vals
            self.num_samples    = len(vals)
            self.mean           = vals.mean()
            self.stddev         = vals.std()
            self.median         = np.median(vals)
            self.iqr            = np.percentile(vals, (25, 75))
            self.min            = np.min(vals)
            self.max            = np.max(vals)


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
        self.pdff_stats:FWStats|None = None

    def has_circles(self):
        return len(self.circles) != 0
    
    def has_rois(self):
        return len(self.rois) != 0
    
    def slice_stats(self):
        '''
        Compute the mean, median, and standard dev of a pdff pair.
        Output: self.pdff_stats computed from rois
        '''
        # default values are -10
        if not self.has_rois():
            self.pdff_stats = None
            return
        self.sort_circles_by_x_coord()
        self.pdff_stats = FWStats(self.pdff_img, self.rois)
        pause = True
    
    def sort_circles_by_x_coord(self):
        if self.has_circles():
            self.circles = sorted(self.circles, key=lambda c: c[0]) # circles is a list
        if self.has_rois():
            self.rois = self.rois[np.argsort(self.rois[:, 0])] #rois is a ndarray
        # sorted(self.rois   , key=lambda c: c[0]) # AVOID will collapse ndarray



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


# def plot_circles_ndarray(img, circles, name='image', waitkey=1):
#     cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
#     cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
#     np_circles = np.uint16(np.around(circles))
#     for c in np_circles[0,:]: # don't remembwer what packing needed this slice
#         cv2.circle(cimg,(c[CX],c[CY]),c[CR],(0,255,0),2)
#     cv2.imshow(name, cimg)
#     cv2.waitKey(waitkey)
#     cv2.destroyAllWindows()


def pack_array(img_pairs:list[FWImagePair]) -> np.ndarray:
    '''
    Create an image that has water/pdff pairs (left to right)
    and up to three rows (at which point it adds another column)
    '''
    if len(img_pairs) < 1:
        return
    channels = 3
    arr_rows = min(len(img_pairs), 3)
    arr_cols = (len(img_pairs) + arr_rows - 1) // arr_rows
    cimg_setup = cv2.cvtColor(np.uint8(img_pairs[0].water_img), cv2.COLOR_GRAY2BGR) 
    img_h, img_w, channels = cimg_setup.shape

    pair_w = img_w * 2
    canvas = np.zeros((img_h * arr_rows, pair_w * arr_cols, channels), dtype=np.uint8)

    # prepare normalzation
    watr_min, watr_max, pdff_min, pdff_max = normalization_values(img_pairs)

    def normalize_to_uint8(img, vmin, vmax):
        if vmin is None or vmax is None or vmax <= vmin:
            return np.zeros_like(img, dtype=np.uint8)
        scale = 255.0 / (vmax - vmin)
        img_f = img.astype(np.float32, copy=False)
        return np.uint8(np.clip((img_f - vmin) * scale, 0, 255))

    for i, img_pair in enumerate(img_pairs):
        cimg_watr = normalize_to_uint8(img_pair.water_img, watr_min, watr_max)
        cimg_watr = cv2.cvtColor(cimg_watr, cv2.COLOR_GRAY2BGR)
        cimg_pdff = normalize_to_uint8(img_pair.pdff_img, pdff_min, pdff_max)
        cimg_pdff = cv2.cvtColor(cimg_pdff, cv2.COLOR_GRAY2BGR)
        # draw circles around water vials
        if img_pair.has_circles():
            np_circles = np.uint16(np.around(img_pair.circles))
            for c in np_circles:
                cv2.circle(cimg_watr,(int(c[0]),int(c[1])),c[2],(0,0,255),1)
        # draw ROIs in pdff vials
        if img_pair.has_rois():
            np_rois = np.uint16(np.around(img_pair.rois))
            for j, c in enumerate(np_rois):
                cv2.circle(cimg_pdff, (int(c[0]),int(c[1])),c[2],(255,255,0),1)

        row = i % arr_rows
        col = i // arr_rows
        y0 = row * img_h
        x0 = col * pair_w
        canvas[y0:y0 + img_h,           x0:x0 + img_w]  = cimg_watr
        canvas[y0:y0 + img_h,   img_w + x0:x0 + pair_w] = cimg_pdff
    return canvas

def normalization_values(img_pairs):
    watr_min = None
    watr_max = None
    pdff_min = None
    pdff_max = None
    for img_pair in img_pairs:
        if img_pair.water_img.size:
            w_min = float(np.min(img_pair.water_img))
            w_max = float(np.max(img_pair.water_img))
            watr_min = w_min if watr_min is None else min(watr_min, w_min)
            watr_max = w_max if watr_max is None else max(watr_max, w_max)
        if img_pair.pdff_img.size:
            p_min = float(np.min(img_pair.pdff_img))
            p_max = float(np.max(img_pair.pdff_img))
            pdff_min = p_min if pdff_min is None else min(pdff_min, p_min)
            pdff_max = p_max if pdff_max is None else max(pdff_max, p_max)
    return watr_min,watr_max,pdff_min,pdff_max
