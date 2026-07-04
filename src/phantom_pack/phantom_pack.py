import os
import argparse
import pydicom
import numpy as np
import cv2
import datetime
from .image_labels import identifying_labels
from .load_dicoms import load_dicoms
from .circle_grouping import find_circle_groups
from .circle_finder import circle_finder_water
from .fw import FWSeries, FWImagePair

from .pp_config import PP_CONST
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
        labeled_dicoms:list[pydicom.Dataset] | str | os.PathLike,
        directory_path: str | os.PathLike | None = None,
        output_dir: str | os.PathLike | None = None,
        vial_radius = PP_CONST.vial_radius_mm,
        radius_tolerance = PP_CONST.radius_tolerance, 
        vert_align_tol = PP_CONST.alignment_tolerance,
        roi_radius = PP_CONST.roi_radius_mm,
        span_mm = PP_CONST.analysis_span_mm,
    ) -> dict:
    '''
    process all labeled pdff data
    return a list of dictionaries containing results for each pdff/water pair
    '''
    if isinstance(labeled_dicoms, (str, os.PathLike)):
        return process_directory(
            labeled_dicoms,
            vial_radius = vial_radius,
            radius_tolerance = radius_tolerance,
            vert_align_tol = vert_align_tol,
            roi_radius = roi_radius,
            span_mm = span_mm,
            output_dir = output_dir,
        )

    if labeled_dicoms == None or len(labeled_dicoms) == 0:
        print("[phantom_pack] ERROR: Input data is empty")
        return {} 

    # prepare output directory
    if output_dir is None:
        output_parent = os.fspath(directory_path) if directory_path is not None else os.getcwd()
        output_dir = os.path.join(output_parent, PP_CONST.output_dir)
    else:
        output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(output_dir) or not os.access(output_dir, os.W_OK):
        print(f"[phantom_pack] ERROR: Output directory not writable: {output_dir}")
        return {}

    # find pdff/water pairs; img_packs = [[pair1],[pair2],...]
    fw_series = find_fw_pairs(labeled_dicoms) 
    print_paired_summary(fw_series, directory_path or "provided datasets")

    # save summary of loaded data: each pdff/water series and description
    log_seriesdata_to_file(output_dir, fw_series)
    # save summary of data loaded but unknown (no matching label)
    log_unknowns_to_file(output_dir, labeled_dicoms)

    # loop over series pairs to find phantom packs
    for fw in fw_series:
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

        # try:
        #     fw.stats_min_loc = fw.pack_midpoint-span_mm/2
        #     fw.stats_max_loc = fw.pack_midpoint+span_mm/2
        # except: 
        #     logger.warning("ERROR computing min and max slice location")
        #     return {}
        
        results = fw.compute_and_save_results(span_mm, output_dir)
    return results


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
    for fw_series in fw_series_paired:
        if len(fw_series.image_pairs) == 0:
            continue
        logger.info(f"PDFF  series {fw_series.series_number_pdff},  {fw_series.series_description_pdff}")
        logger.info(f"WATER series {fw_series.series_number_water}, {fw_series.series_description_water}")
        logger.info("")

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


def create_negative_image(img):
    return 255 - np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))


def find_packs_in_images(
        fw_series:FWSeries,
        vial_radius        = PP_CONST.vial_radius_mm,
        radius_tolerance   = PP_CONST.radius_tolerance,
        vial_separation    = PP_CONST.vial_separation_mm,
        vial_sep_tolerance = PP_CONST.separation_tolerance,
        vert_align_tol     = PP_CONST.alignment_tolerance,
        ):
    '''
        Finds circles in pdff/water image pairs ammends the fw_series.image_pairs to include those circles
    '''
    for ip in fw_series.image_pairs:
        px_size = ip.pixel_spacing
        min_radius, max_radius, min_vail_sep = vial_sizes_in_px(vial_radius, radius_tolerance, px_size)
        water_circles = circles_img_bottom(ip.water_img, min_radius, max_radius, min_vail_sep)
        ip.dbg_found_circles = water_circles
        # display_image_with_circles(ip.water_img, water_circles, name=str(ip.location_full), waitkey=0)
        num_circles_in_pack = 5
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
        ip.circles = pack_circles[0] # drop trivial first dimension/ hack: keep only first group

    count_circles = [pair for pair in fw_series.image_pairs if pair.has_circles()]
    logger.info(f"  {len(count_circles)} slices contain phantom pack")
    return


def circles_img_bottom(img, min_radius, max_radius, min_vail_sep):
    cropped_img = make_clipped_image(img)
    blur = max((max_radius - min_radius)//4, 3)
    circles = circle_finder_water(cropped_img, minDist=min_vail_sep, minRadius=min_radius, maxRadius=max_radius, blur_size=blur)
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
    # SliceLocation is used to separate images
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
                px_spacing = get_px_spacing(ff, wat_same_loc[0])
                img_pair = FWImagePair(ff.pixel_array, wat_same_loc[0].pixel_array, px_spacing, ff.SliceLocation)
                fw_series.image_pairs.append(img_pair)
                if fw_series.pdff_metadata == None:
                    fw_series.pdff_metadata = ff
                    # if "PixelData" in fw_series.pdff_metadata:
                    #     del fw_series.pdff_metadata.PixelData
                if fw_series.water_metadata == None:
                    fw_series.water_metadata = wat_same_loc[0]
                    # if "PixelData" in fw_series.water_metadata:
                    #     del fw_series.water_metadata.PixelData
        series_found.append(fw_series)
    return series_found


def get_px_spacing(ff:pydicom.Dataset, water:pydicom.Dataset) -> float:
    w_ps = water.PixelSpacing
    f_ps = ff.PixelSpacing
    if len(w_ps) < 2 or len(f_ps) < 2:
        logger.warning(f"WARNING: PixelSpacing missing for: {ff.SeriesDescription} {ff.SeriesNumber} {ff.SliceLocation}")
        return 0
    if w_ps[0] != w_ps[1] or f_ps[0] != f_ps[1] or w_ps[0] != f_ps[0]:
        logger.warning(f"WARNING: PixelSpacing mismatch for: {ff.SeriesDescription} {ff.SeriesNumber} {ff.SliceLocation}")
        return 0
    return w_ps[0]


def vial_sizes_in_px(vial_radius, radius_tolerance, px_size):
    min_radius = int(vial_radius/px_size) - int(radius_tolerance/px_size)
    max_radius = int(vial_radius/px_size) + int(np.ceil(radius_tolerance/px_size))
    min_vail_sep = vial_radius/px_size
    return min_radius,max_radius,min_vail_sep


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


def process_directory(directory_path: str | os.PathLike, **kwargs) -> dict:
    directory_path = os.fspath(directory_path)
    if not os.path.isdir(directory_path):
        print(f"Could not find directory {directory_path}")
        return {}
    logfile = os.path.join(directory_path, "phantom_pack.log")
    logging.basicConfig(filename=logfile, level=logging.INFO)
    logger.info(f"Processing {directory_path}")
    labeled_dicoms = load_and_label(directory_path)
    results = phantom_pack(labeled_dicoms, directory_path=directory_path, **kwargs)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find and analyze Calimetrix phantom pack vials in DICOM images.")
    parser.add_argument("input_directory", help="Directory containing one patient-exam of DICOM files.")
    args = parser.parse_args(argv)

    results = process_directory(args.input_directory)
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())

    # testimg_path = r'C:\testdata\PhantomPack\PQ024\SER00090\IMG00019.dcm'
    # directory_path = os.path.dirname(testimg_path)4
    # ds = pydicom.dcmread(testimg_path)
    # circles_in_pdff(ds, min_radius=1, max_radius=60, min_sep=1)


