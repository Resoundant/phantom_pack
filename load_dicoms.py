import os
import time
import pydicom
from image_labels import load_labels
import logging
logger = logging.getLogger(__name__)

def load_dicoms(directory_path:str, turbo_mode=False):
    time_load_start = time.perf_counter()
    if turbo_mode:
        logger.info("...Fast loading mode enabled")
        dicoms = load_dicoms_fromdir_quickly(directory_path)
    else:
        logger.info("...Standard loading mode enabled")
        dicoms = load_dicoms_fromdir(directory_path)
    time_load_end = time.perf_counter()
    logger.info(f"loaded {len(dicoms)} dicoms in {time_load_end - time_load_start:.2f} seconds")
    return dicoms

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
