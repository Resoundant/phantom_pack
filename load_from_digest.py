import os
from pathlib import Path
import pydicom
from typing import Union

PathLike = Union[str, Path]


    # {
    #     "image_label": "pdff",
    #     "search_in": "ImageType",
    #     "search_for": "FAT_FRACTION",
    #     "label_match": "water"
    # },
    # {
    #     "image_label": "water",
    #     "search_in": "ImageType",
    #     "search_for": "WATER",
    #     "label_match": "pdff"
    # },

def load_from_digest_and_label(digest_file:PathLike) -> list[pydicom.Dataset]:
    digest_lines = read_digest(digest_file)
    pdff = load_contrast_from_key('fw.pdff.', digest_lines)
    water = load_contrast_from_key('fw.water.', digest_lines)
    for ds in pdff:
        ds.image_label = "pdff"
        ds.label_match = "water"
    for ds in water:
        ds.image_label = "water"
        ds.label_match = "pdff"
    return pdff + water


def read_digest(digest_file:PathLike) -> list[str]:
    """Read a text file and return its lines without trailing newlines."""
    path = Path(digest_file)
    with path.open("r", encoding="utf-8") as fh:
        lines = [line.rstrip("\n") for line in fh]
    lines = [line for line in lines if line]
    return lines


def load_contrast_from_key(key_root:str, digest_lines:list) -> list[pydicom.Dataset]:
    files_to_load = [f for f in digest_lines if f.startswith(key_root)]
    dicoms  = []
    for f in files_to_load:
        try:
            ds = pydicom.dcmread(f)
            dicoms.append(ds)
        except:
            print(f"ERROR: could not load {f}")
    return dicoms