# Embedding products

Precomputed embedding products, such as [AlphaEarth Foundations](https://source.coop/tge-labs/aef) (AEF, 64 channels), [TESSERA](https://github.com/ucam-eo/tessera) (128 channels) and [OlmoEarth v1.3](https://source.coop/olmoearth/olmoearth-v1-3-embeddings) (128 channels), are evaluated like models. A product is downloaded once for a dataset, which pairs every sample's embeddings with the label that image models see, and exported as a fixed collection. `geo-eval-harness run` then evaluates the collection with the usual KNN, linear and segmentation probes, and each product is reported as a model.

```bash
# Fetch a published collection at its pinned revision, or build one locally:
uv run geo-eval-harness embeddings fetch --product aef --dataset benv2 benv2_pca16
uv run geo-eval-harness embeddings download --product aef --dataset fotw --workers 8
uv run geo-eval-harness embeddings export --product aef --dataset fotw --workers 8
# Optional: fit a 16-component PCA on training cells and store it as collection "fotw_pca16".
uv run geo-eval-harness embeddings pca --product aef --dataset fotw --components 16
# Evaluate; results go to results/models/aef.csv and aef-pca16.csv.
uv run geo-eval-harness run --model aef --dataset benv2
uv run geo-eval-harness run --model aef-pca16 --dataset fotw
```

TESSERA and OlmoEarth need the `embeddings` extra (`pip install 'geo-eval-harness[embeddings]'`). Downloads and exports read GeoBench V2 data from `data/geobenchv2`, so download the dataset first. A download is a resumable working copy in `data/embedding-downloads/`: an interrupted download resumes where it stopped, and `--limit N` stores only the first N samples of each split for a quick check. Samples are read in groups of neighbours so shared source blocks are fetched once, and `--workers N` reads groups in N parallel processes; remote products such as AEF download several times faster with 8 workers. `export` refuses incomplete downloads; its `--limit` takes one count for every split or one each for train, val and test, and is recorded in the metadata.

## Supported datasets

Embeddings need each sample's location and year. These come from the dataset's own metadata through `BenchDataset.footprint`, which is implemented for `benv2`, `treesatai`, `spacenet7`, `fotw`, `kuro_siwo` and `dynamic_earthnet`.

## How samples are paired

1. **Footprint.** The dataset gives the sample's CRS, pixel grid and observation year.
2. **Embedding grid.** The footprint is divided into cells of about 10 m, matching the products' native resolution. Every sample of a dataset uses the cell count of the first training sample, so samples batch together. The grid keeps the sample's CRS, extent and row order, so embedding cells line up with label pixels.
3. **Year.** AEF, TESSERA and OlmoEarth are annual products covering 2017 to 2025. Each sample uses its own observation year, clamped to that range; for example, TreeSatAI's 2016 samples use 2017.
4. **Read.** The product is resampled onto the grid with nearest-neighbour lookup, mosaicking tiles where needed. TESSERA and OlmoEarth read only the UTM zone that contains the sample centre. Cells the product does not cover are stored as NaN.
5. **Label.** The label or mask comes from the same dataset wrapper that image models use.

Classification probes average the valid cells of each sample. Segmentation heads receive the full embedding map, with missing cells set to zero, and upsample it to the mask like any coarse encoder feature map.

## OlmoEarth v1.3

OlmoEarth v1.3 has three presets:

- `olmoearth-v1-3`: the full 128-dimensional embedding.
- `olmoearth-v1-3-64`: the first 64 dimensions, which form a complete embedding on their own (Matryoshka training) and download in half the time. Download it as its own product with `--product olmoearth-v1-3-64`.
- `olmoearth-v1-3-pca16`: `embeddings pca --product olmoearth-v1-3`, a 16-component PCA of the 128-dimensional collection.

As of the initial release, only 2023 to 2025 are global; 2017 to 2022 cover selected countries and are being backfilled upstream. Samples from years that are not yet filled at their location are stored as missing and counted in `empty_samples`. Because downloads resume by skipping existing files, delete a download directory and download it again once the backfill is complete.

The embeddings are released under the [OlmoEarth Artifact License](https://source.coop/olmoearth/olmoearth-v1-3-embeddings/LICENSE.txt), which restricts military and extractive uses and requires citing Ai2 when redistributing derivatives.

## Stored collections

A collection is a flat directory of zstd-compressed Parquet shards, one row per sample in index order. `data/embeddings/<product>/` mirrors the product's Hugging Face dataset repository:

```
data/embeddings/<product>/
  <dataset>/
    train-00000.parquet     # index, embedding, label or mask, grid, years, valid_fraction
    val-00000.parquet
    test-00000.parquet
    metadata.json           # product version, codec, array dtypes and shapes, per-split coverage
    SHA256SUMS
  <dataset>_pca16/          # the same layout, plus pca.safetensors
```

Each row carries the sample's embedding grid (`crs`, `transform`, `height`, `width`), its `product_year` and `observation_year`, and `valid_fraction`, the share of cells with every channel present. Embeddings keep their product's native quantization: AEF and OlmoEarth as signed-square int8 codes (`-128` missing), TESSERA as int8 codes with a float32 scale per cell (scale 0 missing), and PCA collections as float16. Masks are stored as uint8. `EmbeddingDataset` decodes rows to float32 with NaN for missing cells, exactly the float16 values of the download. Evaluation includes a digest of `SHA256SUMS` in the resume hash, so results from a replaced collection are not resumed.

`metadata.json` reports per-split coverage. Samples without product coverage are kept, so every product is evaluated on the same samples as the image models, but they show up in `empty_samples`. A PCA is fitted on valid cells of the distinct training embeddings, so samples that share one annual embedding, such as DynamicEarthNet's monthly samples of one area, count once. Components are not whitened. The PCA collection records its source collection and digest, seed and explained variance ratios, and `pca.safetensors` holds the fitted mean, components and explained variance.

## Adding a product

Each product has its own module in `geo_eval_harness/embeddings/products/` (`aef.py`, `tessera.py`, `olmo_earth.py`). Add a new module that subclasses `EmbeddingProduct` from `products/base.py`. Set `name`, `channels`, `years` and `codec`, the name of the codec in `embeddings/codecs.py` that stores the product exactly, and implement `read(grid)`, which returns float32 `(channels, H, W)` values on `grid` for `grid.year`, with NaN where data is missing. `geo_eval_harness.embeddings.grid.paste` resamples a source window onto the grid. Then register the class in `PRODUCTS` in `products/__init__.py` and add a preset like `conf/model/aef.yaml`.

## Adding a dataset

Implement `footprint(dataset, index)` on the dataset wrapper. For GeoBench V2 datasets, setting `footprint_asset` to the name of the upstream asset whose STAC grid and time describe the sample is usually enough. Override `footprint` only when that header is wrong, as SpaceNet7 does for its positive y spacing.
