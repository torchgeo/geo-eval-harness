"""Segmentation decoder heads for use with SegmentationProbe."""

import functools
import itertools
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers.models.upernet.modeling_upernet import UperNetPyramidPoolingModule


class LinearHead(nn.Module):
    """Per-layer BN + 1x1 conv heads with learned scale-weighted fusion.

    For a single layer the output is returned directly. For multiple layers,
    each head's logits are upsampled to the input resolution and combined via
    learned scalar weights.

    Args:
        channels_list: Channel count for each hooked feature layer.
        num_classes: Number of segmentation output classes.
    """

    def __init__(self, channels_list: list[int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.BatchNorm2d(c), nn.Conv2d(c, num_classes, kernel_size=1))
                for c in channels_list
            ]
        )
        if len(channels_list) > 1:
            self.scale_weights = nn.Parameter(torch.ones(len(channels_list)))

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Upsample and sum per-layer logits."""
        total_logits: torch.Tensor | int = 0
        for idx, (feat, head) in enumerate(zip(features, self.heads, strict=True)):
            logits = head(feat)
            if logits.shape[-2:] != (input_h, input_w):
                logits = F.interpolate(
                    logits, size=(input_h, input_w), mode="bilinear", align_corners=False
                )
            if len(self.heads) == 1:
                return logits
            total_logits = total_logits + logits * self.scale_weights[idx]
        return total_logits  # type: ignore[return-value]


class PatchLinearHead(nn.Module):
    """ViT-specific patch decoder using a per-token linear projection.

    The first hooked feature map is interpreted as a token grid. Each token is
    independently projected from ``D`` channels to ``C * P^2`` logits via a
    1x1 conv, then rearranged to pixel space with ``pixel_shuffle(P)``.

    Args:
        channels_list: Channel count for each hooked feature layer. Only the
            first entry is used.
        num_classes: Number of segmentation output classes.
    """

    def __init__(self, channels_list: list[int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = channels_list[0]
        self.norm = ChannelLayerNorm(self.in_channels)
        self.conv: nn.Conv2d | None = None
        self.patch_size: int | None = None

    def _ensure_conv(
        self,
        patch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Conv2d:
        if patch_size <= 0:
            raise ValueError(f"Patch size must be positive, got {patch_size}.")

        if self.conv is None:
            conv = nn.Conv2d(self.in_channels, self.num_classes * patch_size**2, kernel_size=1)
            self.conv = conv.to(device=device, dtype=dtype)
            self.patch_size = patch_size
            return self.conv

        if self.patch_size != patch_size:
            raise ValueError(
                f"PatchLinearHead was initialized for patch_size={self.patch_size}, "
                f"but got patch_size={patch_size}."
            )

        if self.conv.weight.device != device or self.conv.weight.dtype != dtype:
            self.conv = self.conv.to(device=device, dtype=dtype)
        return self.conv

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Project the token grid into pixel logits."""
        feat = features[0]
        grid_h, grid_w = feat.shape[-2:]
        patch_h = round(input_h / grid_h)
        patch_w = round(input_w / grid_w)
        if patch_h != patch_w:
            raise ValueError(
                "PatchLinearHead requires square patch geometry. "
                f"Got input=({input_h}, {input_w}) and grid=({grid_h}, {grid_w})."
            )

        conv = self._ensure_conv(patch_h, feat.device, feat.dtype)
        logits = conv(self.norm(feat))
        logits = F.pixel_shuffle(logits, patch_h)
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits


class SweepPatchLinearHead(nn.Module):
    """``n_heads`` independent :class:`PatchLinearHead` probes trained in one vectorised pass.

    Each head is LayerNorm + per-token linear projection to ``C * P^2`` logits + pixel shuffle,
    as in :class:`PatchLinearHead`. Every head has its own LayerNorm affine; the shared
    normalisation is computed once and the affine is folded into the projection, so no head
    materialises its own normalised copy of the features. This is the batched LayerNorm-linear
    probe of UniverSat's ``LP_eval.py`` (one head per learning-rate x weight-decay cell).

    Until :attr:`selected` is set, :meth:`forward` returns ``(n_heads, B, C, H, W)`` logits;
    afterwards it returns the selected head's ``(B, C, H, W)`` logits.
    """

    def __init__(self, channels_list: list[int], num_classes: int, n_heads: int) -> None:
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be positive, got {n_heads}.")
        self.num_classes = num_classes
        self.in_channels = channels_list[0]
        self.n_heads = n_heads
        self.ln_eps = 1e-5
        self.ln_weight = nn.ParameterList(
            [nn.Parameter(torch.ones(self.in_channels)) for _ in range(n_heads)]
        )
        self.ln_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.in_channels)) for _ in range(n_heads)]
        )
        self.weight: nn.ParameterList | None = None
        self.bias: nn.ParameterList | None = None
        self.patch_size: int | None = None
        self.selected: int | None = None

    def head_parameters(self, index: int) -> list[nn.Parameter]:
        """Return the trainable parameters of head ``index``."""
        if self.weight is None or self.bias is None:
            raise RuntimeError("SweepPatchLinearHead has not been materialised by a dry run.")
        return [
            self.ln_weight[index],
            self.ln_bias[index],
            self.weight[index],
            self.bias[index],
        ]

    def _ensure_projection(self, patch_size: int, device: torch.device) -> None:
        if patch_size <= 0:
            raise ValueError(f"Patch size must be positive, got {patch_size}.")
        if self.weight is None:
            out = self.num_classes * patch_size**2
            weights = []
            for _ in range(self.n_heads):
                w = torch.empty(out, self.in_channels, device=device)
                nn.init.trunc_normal_(w, std=0.02)
                weights.append(nn.Parameter(w))
            self.weight = nn.ParameterList(weights)
            self.bias = nn.ParameterList(
                [nn.Parameter(torch.zeros(out, device=device)) for _ in range(self.n_heads)]
            )
            self.patch_size = patch_size
        elif self.patch_size != patch_size:
            raise ValueError(
                f"SweepPatchLinearHead was initialized for patch_size={self.patch_size}, "
                f"but got patch_size={patch_size}."
            )

    def forward_all(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Return ``(n_heads, B, C, H, W)`` logits of every head."""
        feat = features[0]
        bsz, channels, grid_h, grid_w = feat.shape
        patch_h, patch_w = round(input_h / grid_h), round(input_w / grid_w)
        if patch_h != patch_w:
            raise ValueError(
                "SweepPatchLinearHead requires square patch geometry. "
                f"Got input=({input_h}, {input_w}) and grid=({grid_h}, {grid_w})."
            )
        self._ensure_projection(patch_h, feat.device)
        assert self.weight is not None
        assert self.bias is not None
        x_hat = F.layer_norm(feat.float().permute(0, 2, 3, 1), (channels,), eps=self.ln_eps)
        gamma = torch.stack(list(self.ln_weight))
        beta = torch.stack(list(self.ln_bias))
        weight = torch.stack(list(self.weight))
        bias = torch.stack(list(self.bias))
        # W(gamma * x_hat + beta) + b == (W * gamma) x_hat + (W beta + b), per head.
        weight_eff = weight * gamma[:, None, :]
        bias_eff = bias + torch.einsum("nod,nd->no", weight, beta)
        logits = torch.einsum("bhwd,nod->nbohw", x_hat, weight_eff)
        logits = logits + bias_eff[:, None, :, None, None].to(logits.dtype)
        logits = F.pixel_shuffle(logits.flatten(0, 1), patch_h)
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits.unflatten(0, (self.n_heads, bsz))

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Return every head's logits, or only the selected head's once one is selected."""
        logits = self.forward_all(features, input_h, input_w)
        return logits if self.selected is None else logits[self.selected]


class FeatureStandardizer(nn.Module):
    """Fixed per-channel z-scoring of each hooked feature map, fit on the training split.

    ``(x - mean) / sqrt(var + eps)`` per channel, as UniverSat's ``LP_eval.py``
    ``standardization=standard``. The statistics are buffers, so they move with the probe and are
    not trained.
    """

    def __init__(self, channels_list: list[int], eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.num_layers = len(channels_list)
        for index, channels in enumerate(channels_list):
            self.register_buffer(f"mean_{index}", torch.zeros(channels))
            self.register_buffer(f"std_{index}", torch.ones(channels))
        self.fitted = False

    def stats(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and standard deviation of layer ``index``."""
        return getattr(self, f"mean_{index}"), getattr(self, f"std_{index}")

    @torch.no_grad()
    def set_moments(
        self, sums: list[torch.Tensor], sq_sums: list[torch.Tensor], counts: list[int]
    ) -> None:
        """Set statistics from per-channel sums and squared sums over each layer's positions."""
        if not len(sums) == len(sq_sums) == len(counts) == self.num_layers:
            raise ValueError(f"Expected moments for {self.num_layers} layers.")
        if min(counts) <= 0:
            raise ValueError("Feature standardization needs at least one training position.")
        for index, (total, sq_total, count) in enumerate(zip(sums, sq_sums, counts, strict=True)):
            mean, std = self.stats(index)
            mu = total.double() / count
            var = (sq_total.double() / count - mu * mu).clamp(min=0.0)
            mean.copy_(mu.to(mean.dtype))
            std.copy_(torch.sqrt(var + self.eps).to(std.dtype))
        self.fitted = True

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Standardize each layer's ``(B, C, H, W)`` features channel-wise."""
        out = []
        for index, feat in enumerate(features):
            mean, std = self.stats(index)
            out.append((feat.float() - mean[:, None, None]) / std[:, None, None])
        return out


def feature_moments(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return per-channel float64 sums and squared sums of ``(B, C, H, W)`` features."""
    flat = features.double().transpose(0, 1).reshape(features.shape[1], -1)
    return flat.sum(dim=1), (flat * flat).sum(dim=1), flat.shape[1]


class StandardizedHead(nn.Module):
    """Apply a :class:`FeatureStandardizer` before ``head``; parameters are the head's."""

    def __init__(self, standardizer: FeatureStandardizer, head: nn.Module) -> None:
        super().__init__()
        self.standardizer = standardizer
        self.head = head

    @property
    def min_train_batch_size(self) -> int:
        """Forward the wrapped head's minimum training batch size."""
        return int(getattr(self.head, "min_train_batch_size", 1))

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Standardize the features, then run the head."""
        return self.head(self.standardizer(features), input_h, input_w)


class ConvBlockHead(nn.Module):
    """Per-layer 1x1 projection to hidden_dim, aligned concat, 1x1 classification head.

    All feature maps are projected to the same channel count, upsampled to the
    finest spatial resolution in the batch, concatenated, and classified with a
    single 1x1 conv.

    Args:
        channels_list: Channel count for each hooked feature layer.
        num_classes: Number of segmentation output classes.
        hidden_dim: Projection dimension (default 256).
    """

    def __init__(self, channels_list: list[int], num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(inplace=True),
                )
                for c in channels_list
            ]
        )
        self.head = nn.Conv2d(hidden_dim * len(channels_list), num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Project, upsample, concat, and classify features."""
        proj_feats = [proj(f) for f, proj in zip(features, self.projectors, strict=True)]

        target_h, target_w = 0, 0
        for f in proj_feats:
            if f.shape[-2] > target_h:
                target_h, target_w = f.shape[-2:]
        if target_h <= 1:
            target_h, target_w = 16, 16

        aligned = []
        for f in proj_feats:
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(
                    f, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            aligned.append(f)

        logits = self.head(torch.cat(aligned, dim=1))
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits


class FPNHead(nn.Module):
    """Feature Pyramid Network decoder head.

    Applies lateral 1x1 convs, a top-down merging pathway, 3x3 refinement
    convs, then upsamples all levels to the finest resolution, concatenates,
    and classifies with a 1x1 conv.

    Layers must be provided in **coarse-to-fine order** (deepest / lowest-
    resolution first). Example for ResNet:
    ``["layer4", "layer3", "layer2", "layer1"]``.

    Args:
        channels_list: Channel count for each hooked feature layer (coarse-to-fine).
        num_classes: Number of segmentation output classes.
        hidden_dim: Feature dimension used throughout the FPN (default 256).
    """

    def __init__(self, channels_list: list[int], num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        # CNN filters have consistent channel meanings, so BatchNorm can use their batch statistics.
        self.input_norms = nn.ModuleList([nn.BatchNorm2d(c) for c in channels_list])
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False) for c in channels_list]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in channels_list
            ]
        )
        self.fpn_head = nn.Conv2d(hidden_dim * len(channels_list), num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Top-down FPN forward pass.

        Args:
            features: Feature maps in coarse-to-fine order (index 0 = coarsest).
            input_h: Target output height (input image height).
            input_w: Target output width (input image width).
        """
        laterals = [
            lat(norm(f))
            for f, norm, lat in zip(features, self.input_norms, self.laterals, strict=True)
        ]

        # Top-down merging: from coarsest (0) to finest (-1)
        for i in range(len(laterals) - 1):
            target_size = laterals[i + 1].shape[-2:]
            laterals[i + 1] = laterals[i + 1] + F.interpolate(
                laterals[i], size=target_size, mode="bilinear", align_corners=False
            )

        fpn_outs = [conv(p) for p, conv in zip(laterals, self.fpn_convs, strict=True)]

        finest_size = fpn_outs[-1].shape[-2:]
        aligned = []
        for f in fpn_outs:
            if f.shape[-2:] != finest_size:
                f = F.interpolate(f, size=finest_size, mode="bilinear", align_corners=False)
            aligned.append(f)

        logits = self.fpn_head(torch.cat(aligned, dim=1))
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits


_INTERPOLATION_MATRICES: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _bilinear_matrix(size_in: int, size_out: int, device: torch.device) -> torch.Tensor:
    """Return the float32 ``(size_out, size_in)`` weights of 1-D bilinear resizing.

    The weights are PyTorch's own (``align_corners=False``), read off by resizing an identity
    in float64, and cached per ``(size_in, size_out, device)``.
    """
    key = (size_in, size_out, device)
    if key not in _INTERPOLATION_MATRICES:
        eye = torch.eye(size_in, dtype=torch.float64, device=device)[None, None]
        weights = F.interpolate(eye, size=(size_out, size_in), mode="bilinear", align_corners=False)
        _INTERPOLATION_MATRICES[key] = weights[0, 0].float()
    return _INTERPOLATION_MATRICES[key]


def separable_bilinear_upsample(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize ``(..., h, w)`` maps to *size* (``align_corners=False``) by matmuls.

    Computes ``A_h @ x @ A_w^T`` in float32 with autocast disabled and returns the dtype
    ``F.interpolate`` returns: float32 under CUDA autocast (which runs bilinear resizing in
    float32), otherwise *x*'s dtype. Upsampling tiny maps with ``F.interpolate`` has an
    atomic-add backward that is extremely slow on large outputs; the matmul backward is not.
    A length-1 axis is broadcast (a single-row matmul would dispatch to an unavailable kernel).

    The slow backward is PyTorch's atomic-add CUDA kernel (``UpSampleBilinear2d.cu``); upstream
    replaced it with an atomic-free kernel only for ROCm
    (https://github.com/pytorch/pytorch/pull/164572). Drop this workaround once CUDA does the same.
    """
    height, width = size
    cuda_autocast = x.device.type == "cuda" and torch.is_autocast_enabled("cuda")
    dtype = torch.float32 if cuda_autocast else x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        out = x.float()
        if out.shape[-2] == 1:
            out = out.expand(*out.shape[:-2], height, out.shape[-1])
        else:
            out = _bilinear_matrix(out.shape[-2], height, x.device) @ out
        if out.shape[-1] == 1:
            out = out.expand(*out.shape[:-1], width)
        else:
            out = out @ _bilinear_matrix(out.shape[-1], width, x.device).T
    return out.to(dtype)


@functools.cache
def _separable_pyramid_pooling_class() -> "type[UperNetPyramidPoolingModule]":
    from transformers.models.upernet.modeling_upernet import UperNetPyramidPoolingModule

    class SeparablePyramidPoolingModule(UperNetPyramidPoolingModule):
        """The ``transformers`` PSP module with matmul upsampling of its pooled maps."""

        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            size = (x.shape[-2], x.shape[-1])
            return [separable_bilinear_upsample(ppm(x), size) for ppm in self.blocks]

    return SeparablePyramidPoolingModule


def _separable_pyramid_pooling(module: nn.Module) -> None:
    """Make a ``transformers`` PSP module upsample its pooled maps with matmuls, keeping weights.

    Only the pyramid-pooling upsampling changes; the FPN and logit resizing stay ``F.interpolate``.
    """
    from transformers.models.upernet.modeling_upernet import UperNetPyramidPoolingModule

    if type(module) is not UperNetPyramidPoolingModule or module.align_corners:
        raise TypeError(f"Expected a transformers PSP module without align_corners, got {module}")
    module.__class__ = _separable_pyramid_pooling_class()


class UPerNetHead(nn.Module):
    """UPerNet decoder head using the reference ``transformers`` implementation.

    A pyramid pooling module on the coarsest map feeds a top-down FPN over the other maps,
    and all levels are fused before classification (Xiao et al., 2018). Layers must be in
    **coarse-to-fine order**; they may share one resolution (ViT blocks) or form a pyramid
    (CNN or Swin stages).

    Convolutions use the ``N(0, 0.02)`` initialisation of ``UperNetForSemanticSegmentation``.
    The pool-scale-1 branch applies BatchNorm to a ``1x1`` map, so training batches need at
    least two samples; :class:`~geo_eval_harness.segmentation_task.SegmentationSolver` skips
    smaller ones (``min_train_batch_size``) and rejects a training batch size below it.

    The pooled pyramid maps are upsampled by ``separable_bilinear_upsample``: the same
    bilinear weights as the reference ``F.interpolate``, applied as float32 matmuls whose backward
    avoids the reference's atomic accumulation into the tiny pooled maps (over 90% of a training
    step on large maps). Outputs have the reference dtype (float32 under CUDA autocast, as the
    reference computes them); results agree with the reference to float32 rounding.

    Args:
        channels_list: Channel count for each hooked feature layer (coarse-to-fine).
        num_classes: Number of segmentation output classes.
        hidden_dim: Channel width of the pyramid pooling and FPN layers (default 256).
    """

    min_train_batch_size = 2

    def __init__(self, channels_list: list[int], num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        from transformers import UperNetConfig
        from transformers.models.upernet.modeling_upernet import UperNetHead as ReferenceHead

        config = UperNetConfig(hidden_size=hidden_dim)
        config.num_labels = num_classes
        self.decoder = ReferenceHead(config, list(reversed(channels_list)))
        _separable_pyramid_pooling(self.decoder.psp_modules)
        for module in self.decoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Decode coarse-to-fine feature maps into logits at the input resolution."""
        logits = self.decoder(list(reversed(features)))
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits


class UNetHead(nn.Module):
    """U-Net decoder head using the ``segmentation_models_pytorch`` ``UnetDecoder``.

    The coarsest map is the decoder input and each finer map is one skip (no centre block, no
    attention, nearest upsampling, BatchNorm). Decoding stops at the finest hooked map (stride 4 for
    ConvNeXt stages or the ViT neck), the same resolution as :class:`UPerNetHead`; a 1x1 classifier
    then gives logits that are upsampled bilinearly to the input. There is no raw-image skip, so
    ViT blocks need ``neck: vit`` to form the pyramid.

    Every stage has ``hidden_dim`` channels, so at the default 256 the decoder is about as large as
    :class:`UPerNetHead`. Convolutions use UPerNet's ``N(0, 0.02)`` initialisation, and
    ``min_train_batch_size`` is 2 like :class:`UPerNetHead`, so both heads see identical batches.

    Args:
        channels_list: Channel count for each hooked feature layer (coarse-to-fine, at least 2).
        num_classes: Number of segmentation output classes.
        hidden_dim: Channel width of every decoder stage (default 256).
    """

    min_train_batch_size = 2

    def __init__(self, channels_list: list[int], num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

        if len(channels_list) < 2:
            raise ValueError(f"UNetHead needs at least 2 feature layers, got {len(channels_list)}.")
        n_blocks = len(channels_list) - 1
        # smp expects fine-to-coarse maps led by a full-resolution stage that it drops; repeating
        # the finest map keeps it as the last skip and ends decoding at its resolution.
        fine_to_coarse = list(reversed(channels_list))
        self.decoder = UnetDecoder(
            encoder_channels=[fine_to_coarse[0], *fine_to_coarse],
            decoder_channels=[hidden_dim] * n_blocks,
            n_blocks=n_blocks,
        )
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Decode coarse-to-fine feature maps into logits at the input resolution."""
        if any(
            fine.shape[-2] < coarse.shape[-2] or fine.shape[-1] < coarse.shape[-1]
            for coarse, fine in itertools.pairwise(features)
        ):
            raise ValueError(
                "UNetHead needs coarse-to-fine features; got sizes "
                f"{[tuple(f.shape[-2:]) for f in features]}."
            )
        fine_to_coarse = list(reversed(features))
        logits = self.classifier(self.decoder([fine_to_coarse[0], *fine_to_coarse]))
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits


# The fusion cascade is the reference implementation from
# ``transformers.models.dpt.modeling_dpt`` (Apache-2.0), itself a faithful port
# of Intel-ISL DPT. It is imported rather than reimplemented so the residual
# and upsampling arithmetic cannot silently drift from the paper.


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of a (B, C, H, W) feature map.

    ViT features can have outliers and large scale differences between layers.
    Per-position channel normalization avoids sharing outliers through batch statistics.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer norm over channels."""
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def _dpt_fusion_layer(hidden_dim: int) -> nn.Module:
    """Construct a reference ``DPTFeatureFusionLayer`` at ``hidden_dim`` channels.

    The fusion and residual layers read only three config values; the rest of
    ``DPTConfig`` stays at its defaults, which build neither ``DPTNeck`` nor a backbone.

    Stock ViT-DPT uses ``use_batch_norm_in_fusion_residual=False``.
    This disables BatchNorm and enables biased residual convolutions.

    The ``transformers`` import is deferred because only the optional ``dpt`` head needs it.
    """
    from transformers import DPTConfig
    from transformers.models.dpt.modeling_dpt import DPTFeatureFusionLayer

    config = DPTConfig(
        fusion_hidden_size=hidden_dim,
        use_batch_norm_in_fusion_residual=False,
        use_bias_in_fusion_residual=None,  # → ``not use_batch_norm`` → True
    )
    return DPTFeatureFusionLayer(config)


class DPTHead(nn.Module):
    """DPT decoder head using the reference ``transformers`` fusion cascade.

    Requires exactly **4** feature layers in **coarse-to-fine order** (same
    convention as FPN, e.g. ``["layer4", "layer3", "layer2", "layer1"]`` for
    ResNet). Features are normalised and projected to ``hidden_dim``, then run
    through a top-down cascade of :class:`DPTFeatureFusionLayer`.

    Imported fusion layers preserve Intel-ISL DPT's residual and upsampling arithmetic.
    They refine the skip input, project after fusion, and upsample 2x per layer.
    Four layers take a 14x14 grid to 224x224 without separate pre- or post-upsampling.

    ``SegmentationProbe`` supplies ``(B, D, H, W)`` grids, not tokens plus patch geometry.
    ``ChannelLayerNorm`` plus a 1x1 projection therefore replaces ``DPTReassembleStage``.

    Args:
        channels_list: Channel count for each hooked feature layer (coarse-to-fine).
            Must have exactly 4 entries.
        num_classes: Number of segmentation output classes.
        hidden_dim: Hidden channel dimension (default 256).
    """

    def __init__(
        self,
        channels_list: list[int],
        num_classes: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if len(channels_list) != 4:
            raise ValueError(
                f"DPTHead requires exactly 4 feature layers, got {len(channels_list)}. "
                "Specify exactly 4 layer names in coarse-to-fine order in the model config."
            )
        # Per-position LayerNorm avoids mixing sample statistics, including DOFA's outliers.
        self.input_norms = nn.ModuleList([ChannelLayerNorm(c) for c in channels_list])
        self.convs = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, kernel_size=1, padding=0) for c in channels_list]
        )
        self.ref = nn.ModuleList([_dpt_fusion_layer(hidden_dim) for _ in channels_list])
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """DPT forward pass.

        Args:
            features: Feature maps in coarse-to-fine order (index 0 = coarsest).
            input_h: Target output height.
            input_w: Target output width.
        """
        projected = [
            conv(norm(f))
            for norm, conv, f in zip(self.input_norms, self.convs, features, strict=True)
        ]

        # Match DPTFeatureFusionStage: running state first, then the finer map as skip.
        # Each layer doubles spatial resolution.
        out = self.ref[0](projected[0])
        for layer, feat in zip(self.ref[1:], projected[1:], strict=True):
            out = layer(out, feat)

        out = self.out_conv(out)
        if out.shape[-2:] != (input_h, input_w):
            out = F.interpolate(out, size=(input_h, input_w), mode="bilinear", align_corners=False)
        return out


class ViTNeck(nn.Module):
    """Turn four same-resolution ViT feature maps into a stride pyramid.

    Wraps ``transformers``' ``BeitFPNNeck``, the ``Feature2Pyramid`` neck of the BEiT/ViT UPerNet
    configs: coarse-to-fine inputs on one patch grid are rescaled by 1/2, 1, 2, and 4 (max-pool,
    identity, one 2x2 transposed conv, two 2x2 transposed convs with BatchNorm and GELU between).
    A 14x14 grid becomes 7, 14, 28, and 56, the strides 32, 16, 8, and 4 that pyramid backbones
    such as ConvNeXt and Swin give at 224 px.

    With ``input_norm="layernorm"`` each level first gets a learnable ``ChannelLayerNorm``.
    Raw block outputs of some ViTs carry outlier channels; without it the max-pool level passes
    them into UPerNet, whose BatchNorm running statistics then misrepresent eval batches.

    With ``native_finest=True`` (neck ``vit_fine``) the last, finest input is a map the backbone
    already produces at its own resolution (e.g. UniverSat's block-14 output on a finer output
    grid). It replaces the two transposed convs: after its input norm it is bilinearly resized to
    4x the grid of the other three inputs, unless already there.

    A single input (``channels_list`` of length 1) builds all four levels from that one map, the
    ViTDet simple feature pyramid (e.g. from UniverSat's block-14 output alone).

    Args:
        channels_list: Channel count for each of the four layers (coarse-to-fine, all equal), or
            one entry for a single-map pyramid.
        input_norm: ``"layernorm"`` (default) or ``"none"``.
        native_finest: Take the finest level from the last input instead of upsampling a trunk
            block.
    """

    def __init__(
        self,
        channels_list: list[int],
        input_norm: str = "layernorm",
        *,
        native_finest: bool = False,
    ) -> None:
        super().__init__()
        from transformers import BeitConfig
        from transformers.models.beit.modeling_beit import BeitFPNNeck

        self.single_input = len(channels_list) == 1
        if self.single_input and native_finest:
            raise ValueError("native_finest needs 4 inputs; a single-map pyramid has none finer.")
        if self.single_input:
            channels_list = list(channels_list) * 4
        if len(channels_list) != 4 or len(set(channels_list)) != 1:
            raise ValueError(
                f"ViTNeck needs 1 or 4 layers with one channel count, got {list(channels_list)}."
            )
        if input_norm not in ("layernorm", "none"):
            raise ValueError(f"input_norm must be 'layernorm' or 'none', got {input_norm!r}.")
        self.input_norms = nn.ModuleList(
            [
                ChannelLayerNorm(c) if input_norm == "layernorm" else nn.Identity()
                for c in channels_list
            ]
        )
        self.fpn = BeitFPNNeck(BeitConfig(hidden_size=channels_list[0]))
        self.native_finest = native_finest
        if native_finest:
            self.fpn.fpn1 = nn.Identity()

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Normalise each level (if enabled), then rescale coarse-to-fine maps by 1/2, 1, 2, 4."""
        if self.single_input:
            features = list(features) * 4
        normed = [norm(f) for norm, f in zip(self.input_norms, features, strict=True)]
        if self.native_finest:
            grid = normed[0].shape[-2:]
            finest_size = (4 * grid[0], 4 * grid[1])
            if normed[-1].shape[-2:] != finest_size:
                normed[-1] = F.interpolate(
                    normed[-1], size=finest_size, mode="bilinear", align_corners=False
                )
        return list(reversed(self.fpn(normed[::-1])))


class NeckedHead(nn.Module):
    """Apply a neck to the feature maps before a decoder head.

    Args:
        neck: Module mapping a list of feature maps to a list of feature maps.
        head: Decoder head called as ``head(features, input_h, input_w)``.
    """

    def __init__(self, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.neck = neck
        self.head = head
        self.min_train_batch_size = int(getattr(head, "min_train_batch_size", 1))

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Run the neck, then the head."""
        return self.head(self.neck(features), input_h, input_w)
