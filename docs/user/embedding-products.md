# Embedding products

Precomputed embedding products, such as [AlphaEarth Foundations](https://source.coop/tge-labs/aef) (AEF, 64 channels), [TESSERA](https://github.com/ucam-eo/tessera) (128 channels) and [OlmoEarth v1.3](https://source.coop/olmoearth/olmoearth-v1-3-embeddings) (128 channels), are annual maps with one embedding per 10 m cell. `geo-eval-harness embeddings` reads them onto the pixel grid of any georeferenced sample, so each embedding cell covers the same ground as the matching block of sample pixels.

```bash
# Every sample of a registered dataset, paired with its label:
uv run geo-eval-harness embeddings download --product aef --dataset fotw --workers 8
# Any footprints: a table of bounds, or a folder of georeferenced GeoTIFFs:
uv run geo-eval-harness embeddings retrieve --product tessera --footprints fields.csv --output data/embedding-downloads/tessera/fields
uv run geo-eval-harness embeddings retrieve --product aef --rasters chips/ --year 2021 --output data/embedding-downloads/aef/chips
```

TESSERA and OlmoEarth need the `embeddings` extra (`pip install 'geo-eval-harness[embeddings]'`). Both commands write one `<sample>.npz` per sample, holding the float16 `embedding` `(channels, H, W)` with NaN for missing cells, the `crs` and rasterio-order `transform` of the embedding grid, the product `year` and `valid_fraction`, the share of cells with every channel present. `download` also stores the `label` or `mask`, and `metadata.json` summarizes coverage. Existing files are skipped, so an interrupted run resumes where it stopped. Samples are read in groups of neighbours so shared source blocks are fetched once, and `--workers N` reads groups in N parallel processes.

## How samples are aligned

1. **Footprint.** Each sample has a `Footprint`: its CRS, pixel grid and observation year.
2. **Embedding grid.** The footprint is divided into cells of about 10 m (`--cell-size` for `retrieve`), measured in the local UTM zone. The grid keeps the sample's CRS, extent and row order, so embedding cells line up with sample pixels.
3. **Year.** AEF, TESSERA and OlmoEarth cover 2017 to 2025. Each sample uses its own observation year, clamped to that range; for example, TreeSatAI's 2016 samples use 2017.
4. **Read.** The product is resampled onto the grid with nearest-neighbour lookup, mosaicking tiles where needed. TESSERA and OlmoEarth read only the UTM zone that contains the sample centre. Cells the product does not cover are NaN.

`download` gives every sample of a dataset the cell count of its first training sample, so samples batch together. `retrieve` sizes each footprint's grid separately.

## Footprints of any data

`retrieve --footprints` reads a CSV or Parquet table with one row per sample and the columns `id`, `crs`, `west`, `south`, `east`, `north` and `year`. Bounds are in the row's CRS, for example `EPSG:4326` degrees or UTM metres, and `id` names the output file, so it must be a unique file-name-safe string.

`retrieve --rasters` takes every `.tif` in a folder as one sample named by its file stem, with the raster's own CRS and pixel grid. `--year` sets the observation year of all rasters. With `--cell-size` equal to the raster's pixel size, embeddings come out pixel-aligned with the raster.

## Registered datasets

`download` takes each sample's footprint from `BenchDataset.footprint`, which is implemented for `benv2`, `treesatai`, `spacenet7`, `fotw`, `kuro_siwo` and `dynamic_earthnet`. To add a dataset, implement `footprint(dataset, index)` on its wrapper. For GeoBench V2 datasets, setting `footprint_asset` to the name of the upstream asset whose STAC grid and time describe the sample is usually enough. Override `footprint` only when that header is wrong, as SpaceNet7 does for its positive y spacing.

## OlmoEarth v1.3

`olmoearth-v1-3` is the full 128-dimensional embedding. `olmoearth-v1-3-64` reads only the first 64 dimensions, which form a complete embedding on their own (Matryoshka training), in half the time.

Only 2023 to 2025 are global; 2017 to 2022 cover selected countries while upstream backfills them. Samples from years that are not yet filled at their location are stored as missing and counted in `empty_samples`. Because runs resume by skipping existing files, delete an output directory and retrieve it again once the backfill is complete.

The embeddings are released under the [OlmoEarth Artifact License](https://source.coop/olmoearth/olmoearth-v1-3-embeddings/LICENSE.txt), which restricts military and extractive uses and requires citing Ai2 when redistributing derivatives.

## Adding a product

Each product has its own module in `geo_eval_harness/embeddings/products/` (`aef.py`, `tessera.py`, `olmo_earth.py`). Add a new module that subclasses `EmbeddingProduct` from `products/base.py`, set `name`, `channels` and `years`, and implement `read(grid)`, which returns float32 `(channels, H, W)` values on `grid` for `grid.year`, with NaN where data is missing. `geo_eval_harness.embeddings.grid.paste` resamples a source window onto the grid, and `EmbeddingProduct.read_zone` reads one year from a Zarr group per UTM zone, as TESSERA and OlmoEarth do. Then register the class in `PRODUCTS` in `products/__init__.py`.
