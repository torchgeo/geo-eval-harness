"""Task families for aggregating CoordBench results, keyed by benchmark name.

Encoders specialize, so results are grouped into three families and reported per
family:

* ``socio`` — socioeconomic / health / human-activity (income, population, DHS
  indices, CDC health prevalences, nightlights, roads).
* ``env`` — physical environment (climate, elevation, soil, biomass,
  evapotranspiration, tree cover).
* ``land`` — land cover / land use / crop type + biome/ecoregion/country
  classification.

Regression scores are floored at :data:`R2_FLOOR` before averaging.
"""

SOCIO = (
    {
        "pdfm-conus27",
        "california-housing",
        "mosaiks-population",
        "mosaiks-nightlights",
        "mosaiks-income",
        "mosaiks-housing",
        "mosaiks-roads",
        "satclip-population",
        "bt-population",
        "bt-distroad",
    }
    | {
        f"sustainbench-{k}"
        for k in ("asset", "water", "sanitation", "child_mortality", "women_edu", "women_bmi")
    }
    | {
        f"places-{k}"
        for k in (
            "asthma",
            "cancer",
            "chd",
            "checkup",
            "copd",
            "diabetes",
            "high_chol",
            "mental_health",
            "obesity",
            "phys_health",
            "sleep_lt7",
            "smoking",
        )
    }
)

ENV = {
    "worldclim-bio1",
    "worldclim-bio12",
    "soilgrids-soc",
    "soilgrids-phh2o",
    "dm-aster_ged",
    "dm-openet_ensemble",
    "mosaiks-elevation",
    "mosaiks-treecover",
    "satclip-elevation",
    "satclip-air-temp",
    "bt-biomass",
    "bt-bioclim",
}

LAND = {
    f"dm-{k}"
    for k in (
        "africa_crop_mask",
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
        "us_trees",
    )
} | {
    "satclip-biome",
    "satclip-country",
    "satclip-ecoregion",
    "country",
    "ecoregions",
    "bt-cropharvest",
    "bt-landcover",
}

FAMILIES = ("socio", "env", "land")

# Floor R² before averaging so extreme negative scores do not dominate.
R2_FLOOR = -1.0


def family_of(name: str) -> str:
    """Return the family (``socio``/``env``/``land``) for a benchmark name, or ``"other"``."""
    if name in SOCIO:
        return "socio"
    if name in ENV:
        return "env"
    if name in LAND:
        return "land"
    return "other"
