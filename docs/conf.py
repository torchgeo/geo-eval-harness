# Portions of this file are adapted from torchgeo.
# Copyright (c) TorchGeo Contributors. All rights reserved.

"""Sphinx configuration for the geo-eval-harness documentation site."""

import inspect
import os
import sys

# Let autodoc import the src/ package from a source checkout.
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

import geo_eval_harness

project = "geo-eval-harness"
copyright = "geo-eval-harness Contributors"  # noqa: A001 - Sphinx configuration key
author = geo_eval_harness.__author__
version = ".".join(geo_eval_harness.__version__.split(".")[:2])
release = geo_eval_harness.__version__


extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "plans", "examples"]

# Class docstrings with section titles need Sphinx 5.3+: https://github.com/sphinx-doc/sphinx/pull/10887
needs_sphinx = "5.3"

# Fail the docs build on broken links, except for the known missing targets below.
nitpicky = True
nitpick_ignore = [
    # Private types shown by the API documentation.
    ("py:class", "geo_eval_harness.datasets.geobench_v1._V1Dataset"),
    ("py:class", "geo_eval_harness.datasets.geobench_v2._V2Dataset"),
    ("py:class", "geo_eval_harness.datasets.geobench_v2._OffsetMaskV2Dataset"),
    ("py:meth", "_OffsetMaskV2Dataset.canonicalize_sample"),
    ("py:class", "geo_eval_harness.models.torchgeo_models._TorchGeoBackboneBench"),
    ("py:class", "geo_eval_harness.models._normalization.NormalizationStrategy"),
    ("py:class", "geo_eval_harness.models._input_units.InputUnit"),
    ("py:class", "geo_eval_harness.coordbench.models._RSHFEncoder"),
    ("py:class", "geo_eval_harness.segmentation_probe.CachedFeaturesDataset"),
    ("py:class", "geo_eval_harness.segmentation_probe.GPUTensorCache"),
    ("py:class", "CachedFeaturesDataset"),
    ("py:class", "GPUTensorCache"),
    # Names that autodoc cannot link to a documentation page.
    ("py:data", "_REGISTRY"),
    ("py:data", "OLMOEARTH_S2_BANDS"),
    ("py:data", "DEFAULT_V2_DATASETS"),
    ("py:class", "Normalizer"),
    ("py:class", "RCF"),
    ("py:class", "EvaluationResult"),
    ("py:class", "BandSpec"),
    ("py:class", "GeoBenchv2"),
    ("py:class", "Single-label"),
    ("py:class", "auto_resize /"),
    ("py:class", "DPTFeatureFusionLayer"),
    ("py:attr", "BenchModel.bands"),
    ("py:attr", "num_channels"),
    ("py:attr", "base.BenchDataset.supports_partitions"),
    ("py:meth", "_forward_patch_features"),
    ("py:meth", "SegmentationProbe.extract_segmentation_features"),
    ("py:mod", "geo_eval_harness.main"),
    # Third-party types without links in their published documentation.
    ("py:class", "faiss.swigfaiss_avx2.IndexFlatL2"),
    ("py:class", "geobench.task.Task"),
    ("py:class", "geobench_v2.GeoBenchDataModule"),
    ("py:class", "geobenchv2.task.Task"),
    ("py:class", "h5py._hl.dataset.Dataset"),
    ("py:class", "timm.models.resnet.ResNet"),
    ("py:class", "timm.models.vision_transformer.VisionTransformer"),
    ("py:class", "torchgeo.models.api.WeightsEnum"),
    ("py:class", "transformers.modeling_utils.PreTrainedModel"),
    # Self refers to the containing class, not a separate documented type.
    ("py:class", "Self"),
]

# Offline builds cannot resolve links to these external libraries.
nitpick_ignore_regex = [
    (r"py:.*", r"^(numpy|torch|sklearn|pandas|matplotlib|PIL|pillow|torchgeo|torchvision)(\..*)?$"),
    (r"py:.*", r"^(pathlib|abc|collections|typing|argparse)(\..*)?$"),
]


html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "collapse_navigation": False,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "navigation_depth": 4,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/torchgeo/geo-eval-harness",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "torchgeo",
    "github_repo": "geo-eval-harness",
    "github_version": "main",
    "doc_path": "docs",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]


autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "torchvision": ("https://docs.pytorch.org/vision/stable/", None),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "smartquotes",
]
myst_heading_anchors = 3
suppress_warnings = [
    "myst.header",
    "myst.xref_missing",
    "misc.highlighting_failure",
    "intersphinx.fetch_inventory",
]

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Resolve a GitHub URL for the given Python object."""
    if domain != "py":
        return None

    modname = info.get("module", "")
    fullname = info.get("fullname", "")
    if not modname:
        return None

    try:
        mod = sys.modules.get(modname)
        if mod is None:
            __import__(modname)
            mod = sys.modules[modname]

        obj = mod
        for part in fullname.split("."):
            obj = getattr(obj, part)

        obj = inspect.unwrap(obj)
        sourcefile = inspect.getsourcefile(obj)
        if sourcefile is None:
            return None
        source, lineno = inspect.getsource(obj), inspect.getsourcelines(obj)[1]
    except (
        ImportError,
        AttributeError,
        TypeError,
        OSError,
    ):  # allow-except: skip unavailable source links
        return None

    sourcefile = os.path.relpath(sourcefile, start=os.path.join(os.path.dirname(__file__), ".."))

    lineend = lineno + source.count("\n") - 1
    return (
        f"https://github.com/torchgeo/geo-eval-harness/blob/main/{sourcefile}#L{lineno}-L{lineend}"
    )
