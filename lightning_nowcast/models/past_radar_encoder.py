"""Tunable past-radar encoder sharing SimVP encoder architecture.

Initialized from the same pretrained checkpoint as OpenSTLRadarPredictor but
all parameters are trainable (fine-tuned end-to-end with FlowCast+).
Outputs per-frame latent features at the same spatial resolution as the
future latents produced by the frozen radar SimVP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .radar_simvp import load_simvp_model_class


class PastRadarSimVPEncoder(nn.Module):
    """Encodes past radar frames to per-frame latent features.

    Uses the SimVP enc + hid sub-modules, initialized from a pretrained
    checkpoint, with all parameters kept trainable.

    Input:  radar_past (B, T_in, H, W, C=1) in THWC layout
    Output: latent     (B, T_in, H_lat, W_lat, C_lat) in THWC layout
                       e.g. (B, 13, 60, 60, 16) for hid_S=16, N_S=4
    """

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
    ):
        super().__init__()
        openstl_root = Path(openstl_root)
        if str(openstl_root) not in sys.path:
            sys.path.insert(0, str(openstl_root))

        SimVP_Model = load_simvp_model_class(openstl_root)

        # Build a full SimVP model just to load weights cleanly, then keep only enc+hid
        full_model = SimVP_Model(
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
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        full_model.load_state_dict(cleaned, strict=True)

        # Extract only enc + hid; dec is not needed
        self.enc = full_model.enc
        self.hid = full_model.hid

        self.latent_channels = hid_S
        self.reduced_hw = (
            int(in_shape[2] / 2 ** (N_S / 2)),
            int(in_shape[3] / 2 ** (N_S / 2)),
        )

    def forward(self, radar_past: torch.Tensor) -> torch.Tensor:
        """
        Args:
            radar_past: (B, T, H, W, C) in THWC layout

        Returns:
            latent: (B, T, H_lat, W_lat, C_lat) in THWC layout
        """
        B, T, H, W, C = radar_past.shape
        # → (B*T, C, H, W) for enc
        x = radar_past.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        embed, _ = self.enc(x)
        _, C_lat, H_lat, W_lat = embed.shape
        # reshape to (B, T, C_lat, H_lat, W_lat) then run hid
        latent = embed.reshape(B, T, C_lat, H_lat, W_lat)
        latent = self.hid(latent)  # (B, T, C_lat, H_lat, W_lat)
        return latent.permute(0, 1, 3, 4, 2).contiguous()  # → THWC
