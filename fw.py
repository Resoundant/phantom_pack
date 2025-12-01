import pydicom
import copy
import numpy as np
import cv2
from circle_finder import circle_finder_water, circles_to_rois
import logging
logger = logging.getLogger(__name__)

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

    def compute_and_save_results(self, span_mm, output_dir, fw_serie) -> dict:
        # this code is a bit rigid in expecting regularly structed data (same number of packs, same pixels in each ROI, etc))
        # to avoid it crashing the whole works, if something doesn't finish, it will except and move on, saving no data or
        # partial data
        stats_min_loc = 0
        stats_max_loc = 0
        try:
            stats_min_loc = fw_serie.pack_midpoint-span_mm/2
            stats_max_loc = fw_serie.pack_midpoint+span_mm/2
            composite_results = self.composite_statistics(stats_min_loc, stats_max_loc)
        except:
            logger.warning(f"Error computing comsposite statistics for series {fw_serie.series_number_pdff} {fw_serie.series_description_pdff}")

        try:
            # collect info about dataset
            image_info = get_image_info(fw_serie)
            # save canvas of all pdff water pairs with circles
            array_filepath = os.path.join(output_dir, f"{image_info['PatientName']}_{image_info['SeriesNumber_pdff']}_allimg.png")
            plot_results(fw_serie.image_pairs, dest_filepath=array_filepath, display_image=False)
            # save image of just the selected slices
            array_filepath = os.path.join(output_dir, f"{image_info['PatientName']}_{image_info['SeriesNumber_pdff']}_selected.png")
            image_pairs_in_span = self.img_pairs_in_span(min_loc=stats_min_loc, max_loc=stats_max_loc)
            plot_results(image_pairs_in_span, dest_filepath=array_filepath, display_image=False)
        except:
            logger.warning(f"Error saving plots for series {fw_serie.series_number_pdff} {fw_serie.series_description_pdff}")

        try:
            # save plots of slice values
            plot_slice_values(fw_serie, vert_lines=[stats_min_loc, stats_max_loc], directory_path=output_dir)
            results = composite_results | image_info

            if results:
                file_path = os.path.join(output_dir, f"{results['PatientName']}_{results['SeriesNumber_pdff']}.json")
                with open(file_path, 'w') as file:
                    json.dump(results, file, indent=4)
                logger.info(f"  JSON data saved to {file_path}")
            if MATCH_TRACE:
                trace_data = []
                for imgs in fw_serie.image_pairs:
                    trace_data.append({
                        "pdff_trace" : f"Series {imgs.pdff.SeriesNumber}, Instance {imgs.pdff.InstanceNumber}",
                        "water_trace" : f"Series {imgs.water.SeriesNumber}, Instance {imgs.water.InstanceNumber}",
                        "pdff_filename" : f"Series {imgs.pdff.filename}",
                        "water_filename" : f"Series {imgs.water.filename}",
                    })
                file_path = os.path.join(directory_path, OUTPUT_DIR, f"{results['PatientName']}_{results['SeriesNumber_pdff']}_trace.json")
                with open(file_path, 'w') as file:
                    json.dump(trace_data, file, indent=4)
            return results
        except:
            logger.warning(f"Error computing per-slice for series {fw_serie.series_number_pdff} {fw_serie.series_description_pdff}")
            return {}

    def composite_statistics(self, stats_min_loc, stats_max_loc) -> dict:
        '''
        Calculate the composite stats for slices in range (min_loc, max_loc)
        Output (dict): means:[], stddevs:[], medians:[], mins:[], maxs:[], samples:[]
        '''
        # sort all circles and rois by x-coord
        # this is necessary so that when cast into an np array, all of the rois across
        # slices are properly grouped
        for img_pair in self.image_pairs:
            if img_pair.has_circles():
                img_pair.circles = sorted(img_pair.circles, key=lambda x: x[CX])
            if img_pair.has_rois():
                img_pair.rois = sorted(img_pair.rois, key=lambda x: x[CX])

        # quickly check that same number of ROIs in all images
        num_rois_in_images = []
        for img_pair in self.image_pairs:
            if img_pair.has_rois():
                num_rois_in_images.append(len(img_pair.rois))
        if len(set(num_rois_in_images)) > 1:
            logger.warning("WARNING: not all images have same number of ROIs! This may cause issues")

        # check that all of the rois are aligned across slices, by making sure their centers are withing the rois radius
        # so that we don't mix values from different ROIs
        for i in range(len(self.image_pairs)-1):
            if self.image_pairs[i].has_rois() and self.image_pairs[i+1].has_rois():
                for j in range(len(self.image_pairs[i].rois)-1):
                    if abs(float(self.image_pairs[i].rois[j][CX]) - float(self.image_pairs[i+1].rois[j][CX])) > self.image_pairs[i].rois[j][CR]:
                        logger.warning("WARNING: ROIs not aligned across slices!")
                    if abs(float(self.image_pairs[i].rois[j][CY]) - float(self.image_pairs[i+1].rois[j][CY])) > self.image_pairs[i].rois[j][CR]:
                        logger.warning("WARNING: ROIs not aligned across slices!")


        # to calc mean, create lists made up of all pixels value in rois across all slices in range
        num_rois_mode = mode(num_rois_in_images)
        masked_values = [[] for _ in range(num_rois_mode)] #create empty list of lists
        for img_pair in fw_series.image_pairs:
            if (float(img_pair.pdff.SliceLocation) > stats_max_loc) or (float(img_pair.pdff.SliceLocation) < stats_min_loc):
                continue
            if not img_pair.has_rois():
                continue
            for roi_index, roi in enumerate(img_pair.rois):
                # make a circle mask that can be applied to pdff
                vals = get_values_in_roi(img_pair.pdff, roi)
                masked_values[roi_index].extend(vals)

        # todo: rigid, doesn't handle situations where rois have different number of pixels
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
            stats["pdff_means"] = [-10]*5 # HACK - fixed for 5 ROIs
            stats["pdff_medians"] = [-10]*5
            stats["pdff_stddevs"] = [0]*5
            return stats
        # apply rois to PDFF
        pdff_means = []
        pdff_medians = []
        pdff_stddevs = []
        for r in self.rois:
            # make a circle mask that can be applied to pdff
            mask = np.zeros(img.pdff.pixel_array.shape, dtype=np.uint8)
            cv2.circle(mask, (r[CX], r[CY]), r[CR], color=1, thickness=-1) # solid circle (thickness = -1) filled with  1
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
                img_pair = FWImagePair(ff,wat_same_loc[0])
                img_pair.location = int(ff.SliceLocation)
                img_pair.location_full = ff.SliceLocation
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


# def plot_circles_ndarray(img, circles, name='image', waitkey=1):
#     cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
#     cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
#     np_circles = np.uint16(np.around(circles))
#     for c in np_circles[0,:]: # don't remembwer what packing needed this slice
#         cv2.circle(cimg,(c[CX],c[CY]),c[CR],(0,255,0),2)
#     cv2.imshow(name, cimg)
#     cv2.waitKey(waitkey)
#     cv2.destroyAllWindows()