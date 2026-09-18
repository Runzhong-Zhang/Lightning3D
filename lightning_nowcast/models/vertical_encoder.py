import torch
from torch import nn


class VerticalEncoderBase(nn.Module):
    """Interface implemented by every altitude-reduction experiment.

    Subclasses accept ``(input_channels, input_depth)`` in their constructor,
    expose a positive ``output_channels`` value, and reduce separate radar and
    mask sequences to ``[B, T, output_channels, H, W]``.
    """

    @property
    def output_channels(self) -> int:
        raise NotImplementedError

    def encode_sequence(
        self,
        radar: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


VERTICAL_ENCODERS: dict[str, type[VerticalEncoderBase]] = {}


def register_vertical_encoder(name: str):
    """Register a vertical encoder class under a JSON configuration name."""
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("Vertical encoder names cannot be empty.")

    def register(cls: type[VerticalEncoderBase]) -> type[VerticalEncoderBase]:
        if normalized_name in VERTICAL_ENCODERS:
            raise ValueError(f"Vertical encoder {normalized_name!r} is already registered.")
        if not issubclass(cls, VerticalEncoderBase):
            raise TypeError("Registered vertical encoders must inherit VerticalEncoderBase.")
        VERTICAL_ENCODERS[normalized_name] = cls
        return cls

    return register


class VerticalAttentionAggregator(nn.Module):
    """Content-adaptive, per-variable aggregation over the reduced Z axis."""

    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.score = nn.Conv3d(
            channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0),
            groups=channels,
        )
        self.level_bias = nn.Parameter(torch.zeros(1, channels, depth, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[2] != self.depth:
            raise ValueError(f"Expected reduced depth {self.depth}, got {x.shape[2]}.")
        weights = torch.softmax(self.score(x) + self.level_bias, dim=2)
        return (x * weights).sum(dim=2), weights


class VerticalEncoder(VerticalEncoderBase):
    """Shared implementation for the current learned vertical encoders."""

    def __init__(
        self,
        in_channels: int,
        input_depth: int,
        num_layers: int = 3,
        out_channels: int | None = None,
        channel_mixing: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        self.in_channels = int(in_channels)
        self.input_depth = int(input_depth)
        self.out_channels = self.in_channels if out_channels is None else int(out_channels)
        self.channel_mixing = bool(channel_mixing)
        if self.out_channels < 1:
            raise ValueError("out_channels must be positive.")
        if not self.channel_mixing and self.out_channels % self.in_channels != 0:
            raise ValueError(
                "out_channels must be divisible by in_channels when channel_mixing is disabled."
            )

        groups = 1 if self.channel_mixing else self.in_channels
        layer_channels = [self.in_channels] + [self.out_channels] * num_layers
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv3d(
                    layer_channels[index], layer_channels[index + 1], kernel_size=(3, 1, 1),
                    stride=(2, 1, 1), padding=(1, 0, 0), groups=groups,
                ),
                nn.GELU(),
            )
            for index in range(num_layers)
        ])
        reduced_depth = input_depth
        for _ in range(num_layers):
            reduced_depth = (reduced_depth + 1) // 2
        self.aggregate = VerticalAttentionAggregator(self.out_channels, reduced_depth)

    @property
    def output_channels(self) -> int:
        return self.out_channels

    def encode_sequence(self, radar: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reduce a radar/mask sequence to 2-D feature maps."""
        if mask is not None:
            if mask.shape != radar.shape:
                raise ValueError("Radar and mask must have matching shapes.")
            radar = torch.cat((radar, mask), dim=2)
        if radar.ndim != 6 or radar.shape[2:4] != (self.in_channels, self.input_depth):
            raise ValueError("Expected radar [B,T,C,Z,H,W] with configured channels and depth.")
        b, t, c, z, h, w = radar.shape
        return self(radar.reshape(b * t, c, z, h, w)).reshape(b, t, self.output_channels, h, w)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected (B*T,C,Z,H,W), got {tuple(x.shape)}.")
        features, attention = self.aggregate(self.layers(x))
        return (features, attention) if return_attention else features

    def macs_per_volume(self, depth: int, height: int, width: int) -> int:
        """Count Conv3D MACs for one volume, including altitude attention."""
        macs = 0
        current_depth = int(depth)
        current_height = int(height)
        current_width = int(width)
        for block in self.layers:
            conv = block[0]
            output_depth = (
                current_depth + 2 * conv.padding[0]
                - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1
            ) // conv.stride[0] + 1
            output_height = (
                current_height + 2 * conv.padding[1]
                - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1
            ) // conv.stride[1] + 1
            output_width = (
                current_width + 2 * conv.padding[2]
                - conv.dilation[2] * (conv.kernel_size[2] - 1) - 1
            ) // conv.stride[2] + 1
            kernel_volume = conv.kernel_size[0] * conv.kernel_size[1] * conv.kernel_size[2]
            macs += (
                conv.out_channels
                * (conv.in_channels // conv.groups)
                * kernel_volume
                * output_depth
                * output_height
                * output_width
            )
            current_depth = output_depth
            current_height = output_height
            current_width = output_width

        attention = self.aggregate.score
        attention_kernel = (
            attention.kernel_size[0]
            * attention.kernel_size[1]
            * attention.kernel_size[2]
        )
        return macs + (
            attention.out_channels
            * (attention.in_channels // attention.groups)
            * attention_kernel
            * current_depth
            * current_height
            * current_width
        )


@register_vertical_encoder("baseline")
class BaselineVerticalEncoder(VerticalEncoder):
    def __init__(self, in_channels: int, input_depth: int):
        super().__init__(in_channels, input_depth)


@register_vertical_encoder("capacity")
class CapacityVerticalEncoder(VerticalEncoder):
    def __init__(self, in_channels: int, input_depth: int):
        super().__init__(in_channels, input_depth, out_channels=8)


@register_vertical_encoder("mixing")
class MixingVerticalEncoder(VerticalEncoder):
    def __init__(self, in_channels: int, input_depth: int):
        super().__init__(in_channels, input_depth, channel_mixing=True)


@register_vertical_encoder("max")
class MaxVerticalEncoder(VerticalEncoderBase):
    output_channels = 2

    def __init__(self, in_channels: int, input_depth: int):
        super().__init__()
        if in_channels != 2:
            raise ValueError("Max encoder requires two combined radar/mask channels.")
        self.input_depth = input_depth

    def encode_sequence(self, radar: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None or mask.shape != radar.shape:
            raise ValueError("Max encoder requires a matching radar validity mask.")
        if radar.ndim != 6 or radar.shape[2:4] != (1, self.input_depth):
            raise ValueError("Max encoder expects [B,T,1,Z,H,W] with configured depth.")
        valid = mask.bool()
        column_valid = valid.any(dim=3)
        column = radar.masked_fill(~valid, float('-inf')).amax(dim=3)
        column = torch.where(column_valid, column, -1.0)
        return torch.cat((column, column_valid.to(column.dtype)), dim=2)

    def forward(self, radar: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode_sequence(radar, mask)

    def macs_per_volume(self, depth: int, height: int, width: int) -> int:
        """No convolution MACs; excludes comparisons and masking."""
        return 0


def build_vertical_encoder(
    name: str,
    input_channels: int,
    input_depth: int,
) -> VerticalEncoderBase:
    """Construct a registered encoder selected by ``model.vertical_encoder``."""
    normalized_name = str(name).strip().lower()
    try:
        encoder_class = VERTICAL_ENCODERS[normalized_name]
    except KeyError as error:
        available = ", ".join(sorted(VERTICAL_ENCODERS))
        raise ValueError(
            f"Unknown vertical_encoder {name!r}. Available encoders: {available}."
        ) from error

    encoder = encoder_class(int(input_channels), int(input_depth))
    if int(encoder.output_channels) < 1:
        raise ValueError(
            f"Vertical encoder {normalized_name!r} must expose output_channels >= 1."
        )
    return encoder
