# Phantom pack and analysis parameters
PP_CONST = {
    "OUTPUT_DIR" : "phantompack_results",
    "VIAL_RADIUS_MM" : 19/2,      # radius of the phantom pack vials
    "VIAL_SEP_MM" : 31,           # 20px*1.56mm/px
    "VIAL_SEP_TOLERANCE_MM" : 6,
    "ROI_RADIUS_MM" : 13/2,       # radius of the phantom pack ROI
    "RADIUS_TOLERANCE_MM" : 7,    # only find circles VIAL_RADIUS +/- RADIUS_TOLERANCE
    "VERT_ALIGN_TOLERANCE_MM" : 7,
    "ANALYSIS_SPAN_MM" : 20,      # analyze a range of images centered at the midpoint
    "ANALYSIS_CENTER_MM" : None,  # center span at a specific location, None to use midpoint
}


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