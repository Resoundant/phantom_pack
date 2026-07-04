# Phantom pack and analysis parameters
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PPConfig:
    output_dir: str = "phantompack_results"
    vial_radius_mm: float = 19 / 2        # radius of the phantom pack vials in mm
    vial_separation_mm: int = 31          # distance between vial centers
    roi_radius_mm: float = 13 / 2         # radius of the phantom pack ROI
    separation_tolerance: int = 9         # tolerance in vial sep
    radius_tolerance: int = 5             # only find circles VIAL_RADIUS +/- RADIUS_TOLERANCE
    alignment_tolerance: int = 7          # allows for skew in vial alignment through-slice
    analysis_span_mm: int = 20            # analyze a range of images centered at the midpoint
    analysis_center: float | None = None  # center span at a specific location, None to use midpoint
    pack_length_mm: float = 150


PP_CONST = PPConfig()


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
    # "0019109C",
    "InstitutionName",
    "StationName",
    "AcquisitionDate",
    "AcquisitionTime",
]
