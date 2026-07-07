import os
from pathlib import Path
import pydicom
from typing import Union


def load_from_digest_and_label(digest_file:str|Path) -> list[pydicom.Dataset]:
    digest_lines = read_digest(digest_file)
    pdff = load_contrast_from_key('fw.ffrac.slice.', digest_lines) # very frigile, needs the trailing period
    water = load_contrast_from_key('fw.water.slice.', digest_lines)
    for ds in pdff:
        ds.image_label = "pdff"
        ds.label_match = "water"
    for ds in water:
        ds.image_label = "water"
        ds.label_match = "pdff"
    return pdff + water


def read_digest(digest_file:str|Path) -> list[str]:
    """Read a text file and return its lines without trailing newlines."""
    path = Path(digest_file)
    with path.open("r", encoding="utf-8") as fh:
        lines = [line.rstrip("\n") for line in fh]
    lines = [line for line in lines if line]
    return lines


def load_contrast_from_key(key_root:str, digest_lines:list) -> list[pydicom.Dataset]:
    digest_fp = [f for f in digest_lines if f.startswith(key_root)]
    files_to_load = [line.split("=", 1)[1].strip() for line in digest_fp]
    dicoms  = []
    for f in files_to_load:
        try:
            ds = pydicom.dcmread(f)
            dicoms.append(ds)
        except:
            print(f"ERROR: could not load {f}")
    return dicoms
