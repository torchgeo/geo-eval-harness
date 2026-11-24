"""Coordinate benchmark names shared by lightweight validation and table loaders."""

DEEPMIND_EVAL_CONFIGS = (
    "africa_crop_mask",
    "aster_ged",
    "canada_crops_coarse",
    "canada_crops_fine",
    "descals",
    "ethiopia_crops",
    "glance",
    "lcmap_lc",
    "lcmap_lcc",
    "lcmap_lu",
    "lcmap_luc",
    "lucas_lc",
    "lucas_lu",
    "openet_ensemble",
    "us_trees",
)

USAVARS_LABELS = (
    "treecover",
    "elevation",
    "population",
    "nightlights",
    "income",
    "roads",
    "housing",
)

SUSTAINBENCH_TASKS = {
    "asset": "asset_index",
    "water": "water_index",
    "sanitation": "sanitation_index",
    "child_mortality": "under5_mort",
    "women_edu": "women_edu",
    "women_bmi": "women_bmi",
}

CDC_PLACES_MEASURES = {
    "phys_health": "PHLTH",
    "diabetes": "DIABETES",
    "copd": "COPD",
    "cancer": "CANCER",
    "chd": "CHD",
    "mental_health": "MHLTH",
    "checkup": "CHECKUP",
    "sleep_lt7": "SLEEP",
    "asthma": "CASTHMA",
    "obesity": "OBESITY",
    "smoking": "CSMOKING",
    "high_chol": "HIGHCHOL",
}

FAMILY_BENCHMARKS: dict[str, tuple[str, ...]] = {
    "pdfm": ("pdfm-conus27",),
    "air_temp": ("satclip-air-temp",),
    "california_housing": ("california-housing",),
    "satclip": (
        "satclip-country",
        "satclip-ecoregion",
        "satclip-biome",
        "satclip-population",
        "satclip-elevation",
    ),
    "sustainbench": tuple(f"sustainbench-{k}" for k in SUSTAINBENCH_TASKS),
    "better_together": (
        "bt-cropharvest",
        "bt-biomass",
        "bt-landcover",
        "bt-bioclim",
        "bt-population",
        "bt-distroad",
    ),
    "cdc_places": tuple(f"places-{k}" for k in CDC_PLACES_MEASURES),
    "usavars": tuple(f"mosaiks-{label}" for label in USAVARS_LABELS),
    "country": ("country",),
    "ecoregions": ("ecoregions",),
    "worldclim": ("worldclim-bio1", "worldclim-bio12"),
    "soilgrids": ("soilgrids-soc", "soilgrids-phh2o"),
    "deepmind": tuple(f"dm-{stem}" for stem in DEEPMIND_EVAL_CONFIGS),
}
