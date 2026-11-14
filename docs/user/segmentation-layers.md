# Segmentation backbone layer reference

This page records how the values for `segmentation.layers` (see
[Configuration](configuration.rst)) were determined for each model
family, and serves as a reference when adding new models. It complements
the [`SegmentationProbe`](../api/models.rst) and the segmentation-head
classes (`LinearHead`, `FPNHead`, `DPTHead`, …).

## Background

`SegmentationProbe` hooks into named PyTorch modules via `backbone.named_modules()`. The layer names in each model config must exactly match these module names. Layers should be listed **deepest first** (coarsest feature map first) for the FPN head.

## Inspecting layer names

To check a backbone's layer names and output shapes:
1. Instantiating each timm model with `pretrained=False, num_classes=0`
2. Running a dummy 224×224 input through the model with forward hooks on candidate layers
3. Recording the output spatial size of each layer

Discovery script:

```python
import timm, torch

model = timm.create_model("resnet50", pretrained=False, num_classes=0)
model.eval()
x = torch.zeros(1, 3, 224, 224)
shapes = {}
handles = []

for name in ["layer1", "layer2", "layer3", "layer4"]:
    module = dict(model.named_modules())[name]
    def hook(n):
        def fn(m, inp, out): shapes[n] = out.shape
        return fn
    handles.append(module.register_forward_hook(hook(name)))

with torch.no_grad():
    model(x)
for h in handles:
    h.remove()

print(shapes)
```

## Layer Names by Family

### Tessera V2 students

All four students expose `spatial_features`, the full-resolution NCHW pixel embedding map after the final MLP reducer, non-affine LayerNorm, and optional Matryoshka prefix selection. Its channel count is `embedding_dim` (128 by default), independent of student width. Use `layers: ["spatial_features"]`, `head: linear`, and no ViT neck. For UPerNet, use the average-pooled pyramid `pyramid.s32`, `pyramid.s16`, `pyramid.s8`, and `pyramid.s4`, in that order, with no neck. Temporal Transformer blocks are not spatial feature layers. Shape discovery uses an 8-by-8 input; actual images retain their configured grid size.

This is a single-snapshot protocol. PASTIS dates follow the actual selected S2 frame, and undated samples use the explicit fallback DOY. See [Models](models.rst) for input normalization, dataset exclusions, LoRA targets, and memory considerations.

For the following spatial backbones, sizes are given for a 224×224 input image.

### ResNet (resnet18, resnet34, resnet50, resnet101)

| Layer | Spatial size | Channels (resnet50) |
|---|---|---|
| `layer4` | 7×7 | 2048 |
| `layer3` | 14×14 | 1024 |
| `layer2` | 28×28 | 512 |
| `layer1` | 56×56 | 256 |

Config: `layers: ["layer4", "layer3", "layer2", "layer1"]`

### DenseNet (densenet121, densenet161)

| Layer | Spatial size | Channels (densenet121) |
|---|---|---|
| `features.denseblock4` | 7×7 | 1024 |
| `features.denseblock3` | 14×14 | 1024 |
| `features.denseblock2` | 28×28 | 512 |
| `features.denseblock1` | 56×56 | 256 |

Config: `layers: ["features.denseblock4", "features.denseblock3", "features.denseblock2", "features.denseblock1"]`

`features.transition1/2/3` downsample between denseblocks. The listed taps read each denseblock before that downsampling.

### VGG16

| Layer | Spatial size | Description |
|---|---|---|
| `features.30` | 7×7 | After pool5 |
| `features.23` | 14×14 | After pool4 |
| `features.16` | 28×28 | After pool3 |
| `features.9` | 56×56 | After pool2 |

Config: `layers: ["features.30", "features.23", "features.16", "features.9"]`

### VGG19

| Layer | Spatial size | Description |
|---|---|---|
| `features.36` | 7×7 | After pool5 |
| `features.27` | 14×14 | After pool4 |
| `features.18` | 28×28 | After pool3 |
| `features.9` | 56×56 | After pool2 |

Config: `layers: ["features.36", "features.27", "features.18", "features.9"]`

Note: VGG19 has more conv layers per stage than VGG16, so the MaxPool indices differ.

### EfficientNet (efficientnet_b0, b1, b2, b3)

| Layer | Spatial size | Channels (b0) |
|---|---|---|
| `blocks.6` | 7×7 | 320 |
| `blocks.5` | 7×7 | 192 |
| `blocks.3` | 14×14 | 80 |
| `blocks.1` | 56×56 | 24 |

Config: `layers: ["blocks.6", "blocks.5", "blocks.3", "blocks.1"]`

**Note:** `blocks.4`, `blocks.5`, and `blocks.6` all operate at 7×7 (EfficientNet uses multiple MBConv stages at the same stride). We include `blocks.5` to get 4 layers, though it shares spatial size with `blocks.6`. If 3 distinct pyramid levels are preferred, use `["blocks.6", "blocks.3", "blocks.1"]`.

### ConvNeXt (convnext_tiny, small, base, large; tiny/small/base/large_dinov3)

| Layer | Spatial size | Channels (tiny) |
|---|---|---|
| `stages.3` | 7×7 | 768 |
| `stages.2` | 14×14 | 384 |
| `stages.1` | 28×28 | 192 |
| `stages.0` | 56×56 | 96 |

Config: `layers: ["stages.3", "stages.2", "stages.1", "stages.0"]`

### MobileNetV3-Large (mobilenetv3_large_100)

| Layer | Spatial size | Channels |
|---|---|---|
| `blocks.6` | 7×7 | 960 |
| `blocks.5` | 7×7 | 160 |
| `blocks.3` | 14×14 | 80 |
| `blocks.1` | 56×56 | 24 |

Config: `layers: ["blocks.6", "blocks.5", "blocks.3", "blocks.1"]`

**Note:** `blocks.5` and `blocks.6` share 7×7 (same as EfficientNet note above).

### MobileNetV3-Small (mobilenetv3_small_100)

| Layer | Spatial size | Channels |
|---|---|---|
| `blocks.5` | 7×7 | 576 |
| `blocks.4` | 7×7 | 96 |
| `blocks.2` | 14×14 | 40 |
| `blocks.1` | 28×28 | 24 |

Config: `layers: ["blocks.5", "blocks.4", "blocks.2", "blocks.1"]`

Note: MobileNetV3-Small only has 6 blocks (0–5), unlike the Large variant which has 7 (0–6). The shallowest useful stage is 28×28 (no 56×56 stage).

### RegNet (regnetx_002, regnetx_008, regnety_002, regnety_008)

| Layer | Spatial size | Channels (regnetx_002) |
|---|---|---|
| `s4` | 7×7 | 368 |
| `s3` | 14×14 | 152 |
| `s2` | 28×28 | 56 |
| `s1` | 56×56 | 24 |

Config: `layers: ["s4", "s3", "s2", "s1"]`

## Adding a New Model

1. Instantiate the model with `timm.create_model(name, pretrained=False, num_classes=0)`
2. Print top-level modules: `[name for name, _ in model.named_children()]`
3. Run the discovery script above to confirm spatial sizes
4. Add to the model config in coarse-to-fine order (deepest first)
5. Note any stages that share spatial size (common in EfficientNet/MobileNet)
