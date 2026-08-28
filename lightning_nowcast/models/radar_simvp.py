from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def relu_evidence(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def probability_and_uncertainty_from_dirichlet(raw_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = relu_evidence(raw_pred[:, :, :, 1:, :, :]) + 1.0
    strength = torch.sum(alpha, dim=3, keepdim=True)
    prob = alpha / strength
    positive_class_prob = prob[:, :, :, 1, :, :]
    uncertainty = (2.0 / strength).squeeze(3)
    return positive_class_prob, uncertainty


def load_simvp_model_class(openstl_root: Path) -> type[nn.Module]:
    module_path = openstl_root / "openstl" / "models" / "simvp_model_rev_crop_pred.py"
    spec = importlib.util.spec_from_file_location("openstl_simvp_model_rev_crop_pred", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SIMVP model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimVP_Model


_RADAR_SIMVP_FILE = Path(__file__).resolve()
_PROJ_ROOT = _RADAR_SIMVP_FILE.parents[3]  # src/lightning_nowcast/models/ → project root
_VENDORED_OPENSTL = _PROJ_ROOT / "third_party" / "openstl_src"


def _resolve_openstl_root(openstl_root: str | Path) -> Path:
    """Return the first valid openstl_root: config path → project-relative → vendored → raise."""
    p = Path(openstl_root)
    candidates = [
        p,
        _PROJ_ROOT / p if not p.is_absolute() else None,  # relative to project root
        _VENDORED_OPENSTL,
    ]
    for c in candidates:
        if c is not None and (c / "openstl" / "models" / "simvp_model_rev_crop_pred.py").exists():
            return c
    raise FileNotFoundError(
        f"Cannot find simvp_model_rev_crop_pred.py. "
        f"Tried: {[str(c) for c in candidates if c is not None]}"
    )


class OpenSTLRadarPredictor(nn.Module):
    def __init__(
        self,
        openstl_root: str | Path,
        checkpoint_path: str | Path,
        in_shape: tuple[int, int, int, int] = (13, 1, 240, 240),
        hid_S: int = 16,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 4,
        model_type: str = "incepu",
        drop: float = 0.0,
        drop_path: float = 0.1,
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
        out_seq_len: int | None = None,
        trainable_modules: tuple[str, ...] | list[str] | None = None,
    ):
        super().__init__()
        self.openstl_root = _resolve_openstl_root(openstl_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.out_seq_len = None if out_seq_len is None else int(out_seq_len)
        self.in_shape = tuple(int(value) for value in in_shape)
        self.latent_channels = int(hid_S)
        self.reduced_hw = (
            int(self.in_shape[2] / 2 ** (N_S / 2)),
            int(self.in_shape[3] / 2 ** (N_S / 2)),
        )
        self.trainable_modules = tuple(str(value) for value in (trainable_modules or ()))
        if str(self.openstl_root) not in sys.path:
            sys.path.insert(0, str(self.openstl_root))

        SimVP_Model = load_simvp_model_class(self.openstl_root)

        self.model = SimVP_Model(
            in_shape=in_shape,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            model_type=model_type,
            drop=drop,
            drop_path=drop_path,
            spatio_kernel_enc=spatio_kernel_enc,
            spatio_kernel_dec=spatio_kernel_dec,
            act_inplace=False,
            noise=None,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned_state = {key.replace("module.", ""): value for key, value in state_dict.items()}
        self.model.load_state_dict(cleaned_state, strict=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self._set_trainable_modules(self.trainable_modules)

    def _set_trainable_modules(self, trainable_modules: tuple[str, ...]) -> None:
        if not trainable_modules:
            self.model.eval()
            return

        for name, param in self.model.named_parameters():
            if any(name == module_name or name.startswith(f"{module_name}.") for module_name in trainable_modules):
                param.requires_grad = True
        self.model.train()

    @property
    def has_trainable_parameters(self) -> bool:
        return any(param.requires_grad for param in self.model.parameters())

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.model.parameters() if param.requires_grad]

    def _autograd_context(self):
        if self.has_trainable_parameters:
            return contextlib.nullcontext()
        return torch.no_grad()

    def predict_future_latent(
        self,
        radar_past: torch.Tensor,
        out_seq_len: int | None = None,
    ) -> torch.Tensor:
        if radar_past.ndim != 5:
            raise ValueError(f"Expected radar_past shape (B, T, H, W, C), got {tuple(radar_past.shape)}")
        target_seq_len = self.out_seq_len if out_seq_len is None else int(out_seq_len)
        batch = radar_past.permute(0, 1, 4, 2, 3).contiguous()
        batch_size, time_steps, channels, height, width = batch.shape
        x = batch.reshape(batch_size * time_steps, channels, height, width)
        embed, _ = self.model.enc(x)
        _, hidden_channels, hidden_h, hidden_w = embed.shape
        latent = embed.reshape(batch_size, time_steps, hidden_channels, hidden_h, hidden_w)
        latent = self.model.hid(latent)
        if target_seq_len is not None:
            latent = latent[:, :target_seq_len]
        return latent

    def predict_distribution(
        self,
        radar_past: torch.Tensor,
        out_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if radar_past.ndim != 5:
            raise ValueError(f"Expected radar_past shape (B, T, H, W, C), got {tuple(radar_past.shape)}")
        target_seq_len = self.out_seq_len if out_seq_len is None else int(out_seq_len)
        batch = radar_past.permute(0, 1, 4, 2, 3).contiguous()
        pred_u, _, _, _, _ = self.model(batch)
        if target_seq_len is not None:
            pred_u = pred_u[:, :target_seq_len]
        positive_prob, class_uncertainty = probability_and_uncertainty_from_dirichlet(pred_u)
        mean_intensity = pred_u[:, :, :, 0, :, :] * positive_prob
        return mean_intensity, class_uncertainty

    def build_future_condition(
        self,
        radar_past: torch.Tensor,
        out_seq_len: int | None = None,
    ) -> torch.Tensor:
        with self._autograd_context():
            mean_intensity, uncertainty = self.predict_distribution(radar_past, out_seq_len=out_seq_len)
        mean_intensity = mean_intensity.permute(0, 1, 3, 4, 2).contiguous()
        uncertainty = uncertainty.permute(0, 1, 3, 4, 2).contiguous()
        return torch.cat([mean_intensity, uncertainty], dim=-1)

    def build_future_feature(
        self,
        radar_past: torch.Tensor,
        out_seq_len: int | None = None,
    ) -> torch.Tensor:
        with self._autograd_context():
            latent = self.predict_future_latent(radar_past, out_seq_len=out_seq_len)
        return latent.permute(0, 1, 3, 4, 2).contiguous()

    def predict_latent_and_mean_intensity(
        self,
        radar_past: torch.Tensor,
        out_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return future latent (60×60, 16ch) and mean intensity (240×240, 1ch) in THWC format.

        Both are extracted from the same frozen SimVP model:
          - fut_lat comes from enc+hid (60×60, compact 16-ch representation)
          - mean_intensity comes from enc+hid+dec (240×240, decoded probability)

        Note: enc+hid runs twice internally (once for latent, once inside the full
        forward for mean_intensity).  Both are under no_grad so the cost is a
        memory-bandwidth duplicate, not a backward-pass issue.

        Returns:
            fut_lat:       (B, T_out, 60,  60,  16)
            mean_intensity:(B, T_out, 240, 240,  1)
        """
        with self._autograd_context():
            latent_bcthw = self.predict_future_latent(radar_past, out_seq_len=out_seq_len)
            mean_intensity, uncertainty = self.predict_distribution(radar_past, out_seq_len=out_seq_len)

        fut_lat = latent_bcthw.permute(0, 1, 3, 4, 2).contiguous()
        # (B, T_out, C, H, W) → (B, T_out, H, W, 1)
        mean_intensity_bthwc = mean_intensity.permute(0, 1, 3, 4, 2).contiguous()
        uncertainty_bthwc = uncertainty.permute(0, 1, 3, 4, 2).contiguous()

        return fut_lat, mean_intensity_bthwc, uncertainty_bthwc
