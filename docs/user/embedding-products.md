# Embedding products

Precomputed embedding products, such as [AlphaEarth Foundations](https://source.coop/tge-labs/aef) (AEF, 64 channels), [TESSERA](https://github.com/ucam-eo/tessera) (128 channels) and [OlmoEarth v1.3](https://source.coop/olmoearth/olmoearth-v1-3-embeddings) (128 channels), are evaluated like models. You download a product once for a dataset, which pairs every sample's embeddings with the label that image models see. `torchgeo-bench run` then evaluates the stored embeddings with the usual KNN, linear and segmentation probes, and each product is reported as a model.

```bash
# Read the product for every sample and store it with its label.
uv run torchgeo-bench embeddings download --product aef --dataset benv2 fotw
# Optional: fit a 16-component PCA on training cells and store it as product "aef-pca16".
uv run torchgeo-bench embeddings pca --product aef --dataset benv2 fotw --components 16
# Evaluate; results go to results/models/aef.csv and aef-pca16.csv.
uv run torchgeo-bench run --model aef --dataset benv2
uv run torchgeo-bench run --model aef-pca16 --dataset fotw
```

TESSERA and OlmoEarth need the `embeddings` extra (`pip install 'torchgeo-bench[embeddings]'`). Downloads read GeoBench V2 data from `data/geobenchv2`, so download the dataset first. An interrupted download resumes where it stopped, and `--limit N` stores only the first N samples of each split for a quick check.

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

As of the initial release, only 2023 to 2025 are global; 2017 to 2022 cover selected countries and are being backfilled upstream. Samples from years that are not yet filled at their location are stored as missing and counted in `empty_samples`. Because downloads resume by skipping existing files, delete a collection directory and download it again once the backfill is complete.

The embeddings are released under the [OlmoEarth Artifact License](https://source.coop/olmoearth/olmoearth-v1-3-embeddings/LICENSE.txt), which restricts military and extractive uses and requires citing Ai2 when redistributing derivatives.

## Stored collections

```
data/embeddings/<product>/<dataset>/
  metadata.json           # product, channels, samples per split, mean valid fraction, empty samples
  train/000000.npz        # embedding (C, H, W) float16, label or mask, year, valid_fraction
  val/...
  test/...
```

`metadata.json` reports per-split coverage. Samples without product coverage are kept, so every product is evaluated on the same samples as the image models, but they show up in `empty_samples`. A PCA collection also records its source product, seed and explained variance ratios.

## Adding a product

Each product has its own module in `torchgeo_bench/embeddings/products/` (`aef.py`, `tessera.py`, `olmo_earth.py`). Add a new module that subclasses `EmbeddingProduct` from `products/base.py`. Set `name`, `channels` and `years`, and implement `read(grid)`, which returns float32 `(channels, H, W)` values on `grid` for `grid.year`, with NaN where data is missing. `torchgeo_bench.embeddings.grid.paste` resamples a source window onto the grid. Then register the class in `PRODUCTS` in `products/__init__.py` and add a preset like `conf/model/aef.yaml`.

## Adding a dataset

Implement `footprint(dataset, index)` on the dataset wrapper. For GeoBench V2 datasets, setting `footprint_asset` to the name of the upstream asset whose STAC grid and time describe the sample is usually enough. Override `footprint` only when that header is wrong, as SpaceNet7 does for its positive y spacing.
