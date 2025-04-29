# these labels are used to determine what data to load
# series description should be lowercase
load_labels = {
    'image_type': ['WATER', 'FAT_FRACTION'],
    'series_description': ['feq-pdff', 'fatfrac', 'pdff', 'water']
}

identifying_labels = [
    {
        "image_label": "pdff_fam_bh_offline",
        "search_in": "SeriesDescription",
        "search_for": "FatFrac: BH new FAM Offline",
        "label_match": "water_fam_bh_offline"
    },
    {
        "image_label": "pdff_fam_bh_offline_nocorrection",
        "search_in": "SeriesDescription",
        "search_for": "FatFrac W/O correction: BH new FAM Offline",
        "label_match": "water_fam_bh_offline"
    },
    {
        "image_label": "water_fam_bh_offline",
        "search_in": "SeriesDescription",
        "search_for": "WATER: BH new FAM Offline",
        "label_match": "pdff_fam_bh_offline"
    },
    {
        "image_label": "pdff_fam_bh",
        "search_in": "SeriesDescription",
        "search_for": "FatFrac: BH new FAM",
        "label_match": "water_fam_bh"
    },
    {
        "image_label": "water_fam_bh",
        "search_in": "SeriesDescription",
        "search_for": "WATER: BH new FAM",
        "label_match": "pdff_fam_bh"
    },
    {
        "image_label": "pdff_fam_fb_offline",
        "search_in": "SeriesDescription",
        "search_for": "FatFrac: FB new FAM Offline",
        "label_match": "water_fam_fb_offline"
    },
    {
        "image_label": "pdff_fam_fb_offline_nocorrection",
        "search_in": "SeriesDescription",
        "search_for": "FatFrac W/O correction: FB new FAM Offline",
        "label_match": "water_fam_fb_offline"
    },
    {
        "image_label": "water_fam_fb_offline",
        "search_in": "SeriesDescription",
        "search_for": "WATER: FB new FAM Offline",
        "label_match": "pdff_fam_fb_offline"
    },
    {
        "image_label": "pdff_fam_fb",
        "search_in": "SeriesDescription",
        "search_for": "FatFrac: FB new FAM",
        "label_match": "water_fam_fb"
    },
    {
        "image_label": "water_fam_fb",
        "search_in": "SeriesDescription",
        "search_for": "WATER: FB new FAM",
        "label_match": "pdff_fam_fb"
    },
    {
        "image_label": "pdff_feq",
        "search_in": "SeriesDescription",
        "search_for": "FeQ-PDFF",
        "label_match": "water"
    },
    {
        "image_label": "pdff",
        "search_in": "ImageType",
        "search_for": "FAT_FRACTION",
        "label_match": "water"
    },
    {
        "image_label": "water",
        "search_in": "ImageType",
        "search_for": "WATER",
        "label_match": "pdff"
    },
]