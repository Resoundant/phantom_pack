import pydicom
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
        self.pack_midpoint:float|None = None

class FWImagePair:
    def __init__(self, pdff:pydicom.Dataset, water:pydicom.Dataset):
        self.pdff = pdff
        self.water = water
        self.circles = []
        self.rois = []
        self.location:int = -999
        self.location_full:float = -999.0
        self.pixel_spacing = water.PixelSpacing[0]
        if water.PixelSpacing[0] != pdff.PixelSpacing[0]:
            print(f"Pixel spacing mismatch!")

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
