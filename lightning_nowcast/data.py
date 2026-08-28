from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class LightningNowcastSample:
    radar_past: torch.Tensor
    lightning_past: torch.Tensor
    lightning_future: torch.Tensor
    radar_future: torch.Tensor
    sample_id: str


def counts_to_condition(counts: np.ndarray, clip_value: float) -> np.ndarray:
    clipped = np.minimum(counts.astype(np.float32), float(clip_value))
    return np.log1p(clipped) / np.log1p(float(clip_value))


class LightningNowcastNetCDFDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        lightning_clip_value: float = 50.0,
        radar_scale: float = 128.0,
        target_is_binary: bool = True,
        lightning_input_mode: str = "total",
    ):
        self.split_dir = Path(split_dir)
        self.lightning_clip_value = float(lightning_clip_value)
        self.radar_scale = float(radar_scale)
        self.target_is_binary = bool(target_is_binary)
        self.lightning_input_mode = str(lightning_input_mode).lower()
        if self.lightning_input_mode not in {"total", "ic_cg"}:
            raise ValueError(
                f"Unsupported lightning_input_mode {self.lightning_input_mode!r}. Expected 'total' or 'ic_cg'."
            )
        self.files = sorted(self.split_dir.glob("*.nc"))
        if not self.files:
            raise ValueError(f"No NetCDF files found in split directory: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        with h5py.File(path, "r") as dataset:
            radar_input = np.asarray(dataset["radar_input"], dtype=np.float32)
            radar_input = np.clip(radar_input, 0, self.radar_scale) / self.radar_scale
            radar_target = np.asarray(dataset["radar_target"], dtype=np.float32)
            radar_target = np.clip(radar_target, 0, self.radar_scale) / self.radar_scale
            ic_input = np.asarray(dataset["ic_input"], dtype=np.float32)
            cg_input = np.asarray(dataset["cg_input"], dtype=np.float32)
            ic_target = np.asarray(dataset["ic_target"], dtype=np.float32)
            cg_target = np.asarray(dataset["cg_target"], dtype=np.float32)
            total_lightning_input = (
                ic_input
                + cg_input
            )
            total_lightning_target = (
                ic_target
                + cg_target
            )

        if self.lightning_input_mode == "ic_cg":
            lightning_past = np.stack(
                [
                    counts_to_condition(ic_input, self.lightning_clip_value),
                    (cg_input > 0).astype(np.float32),
                ],
                axis=-1,
            )
        else:
            lightning_past = counts_to_condition(total_lightning_input, self.lightning_clip_value)[..., None]
        if self.target_is_binary:
            lightning_future = (total_lightning_target > 0).astype(np.float32)
        else:
            lightning_future = counts_to_condition(total_lightning_target, self.lightning_clip_value)

        stem = path.stem  # e.g. "20170101_0340_aid000001_idx0001"
        parts = stem.split("_")
        date_str, time_str = parts[0], parts[1]
        month = float(int(date_str[4:6]))
        hour = float(int(time_str[:2])) + float(int(time_str[2:])) / 60.0
        time_emb = torch.tensor(
            [
                math.sin(2.0 * math.pi * hour / 24.0),
                math.cos(2.0 * math.pi * hour / 24.0),
                math.sin(2.0 * math.pi * (month - 1.0) / 12.0),
                math.cos(2.0 * math.pi * (month - 1.0) / 12.0),
            ],
            dtype=torch.float32,
        )
        return {
            "radar_past": torch.from_numpy(radar_input).float().unsqueeze(-1),
            "lightning_past": torch.from_numpy(lightning_past).float(),
            "lightning_future": torch.from_numpy(lightning_future).float().unsqueeze(-1),
            "radar_future": torch.from_numpy(radar_target).float().unsqueeze(-1),
            "sample_id": path.stem,
            "time_emb": time_emb,
        }




class NEXRAD3DNetCDFDatasetSeparateCZ(Dataset):
    """Dataset adapter for 3-D NEXRAD NetCDF samples.

    Volume mode keeps dimensions separate:
        (time, field, altitude, H, W)
    
    Instead of flattening to (time, field * altitude, H, W),
    this loader keeps C and Z as separate dimensions:
        (time, C, Z, H, W)
    
    Radar and mask are returned separately .
    Invalid radar values are represented as -1 after normalization.
    A separate radar validity mask is also returned.
    """

    def __init__(
        self,
        split_dir: str | Path,
        lightning_clip_value: float = 50.0,
        radar_scale: float = 128.0,
        target_is_binary: bool = True,
        radar_field: str = "Reflectivity",
        vertical_reduction: str = "max",
        radar_representation: str = "volume",
        radar_fields: list[str] | None = None,
    ):
        if isinstance(split_dir, (str, Path)):
            self.split_dirs = [Path(split_dir)]
        elif isinstance(split_dir, (list, tuple)):
            self.split_dirs = [Path(p) for p in split_dir]
        else:
            raise TypeError(
                "split_dir must be a path string or a list of paths."
            )
        self.lightning_clip_value = float(lightning_clip_value)
        self.radar_scale = float(radar_scale)
        self.target_is_binary = bool(target_is_binary)
        self.radar_field = str(radar_field)
        self.vertical_reduction = str(vertical_reduction).lower()
        self.radar_representation = str(radar_representation).lower()

        self.radar_fields = [
            str(field)
            for field in (radar_fields or [self.radar_field])
        ]

        if self.vertical_reduction not in {"max", "mean"}:
            raise ValueError(
                "vertical_reduction must be 'max' or 'mean'."
            )

        if self.radar_representation not in {
            "column_max",
            "column_mean",
            "volume",
        }:
            raise ValueError(
                "radar_representation must be "
                "'column_max', 'column_mean', or 'volume'."
            )

        self.files = []

        for split_dir in self.split_dirs:
            if not split_dir.exists():
                raise ValueError(
                    f"Dataset directory does not exist: {split_dir}"
                )

            self.files.extend(
                split_dir.glob("*.nc")
            )

        self.files = sorted(self.files)
        if not self.files:
            raise ValueError(
                f"No NetCDF files found in directories: "
                f"{self.split_dirs}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def _load_radar(
        self,
        dataset: h5py.File,
        prefix: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load Reflectivity and its NaN/Inf validity mask.
        
        Returns SEPARATE radar and mask (not concatenated).

        Original Reflectivity:
            (T, Z, H, W)

        Validity mask:
            1 = finite Reflectivity
            0 = NaN or Inf Reflectivity

        For volume representation with Z=29:

            Reflectivity:
                (T, Z, H, W)

            Mask:
                (T, Z, H, W)

        Invalid Reflectivity values are represented as -1.
        """

        key = f"{prefix}_Reflectivity"

        if key not in dataset:
            raise KeyError(
                f"Dataset does not contain {key!r}."
            )

        volume = np.asarray(
            dataset[key],
            dtype=np.float32,
        )

        # Expected:
        # (T, Z, H, W)
        if volume.ndim != 4:
            raise ValueError(
                f"Expected {key!r} to have shape "
                f"(T, Z, H, W), got {volume.shape}"
            )

        # Create mask ONLY from NaN / Inf
        finite_mask = np.isfinite(volume)

        # 1 = valid Reflectivity
        # 0 = NaN / Inf
        radar_mask = finite_mask.astype(np.float32)
        #make clean volume
        # Replace invalid values with -1
        clean_volume = np.where(
            finite_mask,
            volume,
            -1.0,
        )

        '''Normalize Reflectivity
         Valid values:
            clip(value, 0, radar_scale) / radar_scale
        Invalid values remain -1.
        Therefore:
            valid Reflectivity -> [0, 1]
            invalid Reflectivity -> -1'''
        

        normalized_radar = np.where(
            finite_mask,
            np.clip(
                clean_volume,
                0.0,
                self.radar_scale,
            ) / self.radar_scale,
            -1.0,
        )

        if self.radar_representation == "volume":
            # Return as separate tensors: (T, Z, H, W) each
            # NOT concatenated along channel dimension
            return (
                normalized_radar.astype(np.float32),
                radar_mask.astype(np.float32),
            )

        # COLUMN REPRESENTATION (column_max or column_mean)
        if self.radar_representation not in {
            "column_max",
            "column_mean",
        }:
            raise ValueError(
                f"Unsupported radar representation: "
                f"{self.radar_representation}"
            )

        with np.errstate(
            invalid="ignore",
            divide="ignore",
        ):

            if self.radar_representation == "column_max":
                radar = np.nanmax(
                    np.where(
                        finite_mask,
                        volume,
                        np.nan,
                    ),
                    axis=1,
                )

            else:
                radar = np.nanmean(
                    np.where(
                        finite_mask,
                        volume,
                        np.nan,
                    ),
                    axis=1,
                )

        column_mask = np.any(
            finite_mask,
            axis=1,
        )

        radar = np.where(
            column_mask,
            np.clip(
                np.nan_to_num(
                    radar,
                    nan=0.0,
                    posinf=self.radar_scale,
                    neginf=0.0,
                ),
                0.0,
                self.radar_scale,
            ) / self.radar_scale,
            -1.0,
        )

        return (
            radar.astype(np.float32),
            column_mask.astype(np.float32),
        )

    @staticmethod
    def _time_embedding(
        stem: str,
    ) -> torch.Tensor:

        match = re.search(
            r"(\d{14})",
            stem,
        )

        if match is None:
            return torch.zeros(
                4,
                dtype=torch.float32,
            )

        timestamp = match.group(1)

        month = float(
            int(timestamp[4:6])
        )

        hour = (
            float(int(timestamp[8:10]))
            + float(int(timestamp[10:12])) / 60.0
        )

        return torch.tensor(
            [
                math.sin(
                    2.0 * math.pi * hour / 24.0
                ),
                math.cos(
                    2.0 * math.pi * hour / 24.0
                ),
                math.sin(
                    2.0 * math.pi
                    * (month - 1.0)
                    / 12.0
                ),
                math.cos(
                    2.0 * math.pi
                    * (month - 1.0)
                    / 12.0
                ),
            ],
            dtype=torch.float32,
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str]:

        path = self.files[index]

        with h5py.File(path, "r") as dataset:

            # Radar (returned separately)
            radar_input, radar_input_mask = (
                self._load_radar(
                    dataset,
                    "input",
                )
            )

            radar_target, radar_target_mask = (
                self._load_radar(
                    dataset,
                    "target",
                )
            )

            # Lightning
            lightning_input = np.asarray(
                dataset["input_lightning_map"],
                dtype=np.float32,
            )

            lightning_target = np.asarray(
                dataset["target_lightning_map"],
                dtype=np.float32,
            )

        # Lightning
        lightning_past = counts_to_condition(
            lightning_input,
            self.lightning_clip_value,
        )[..., None]

        lightning_future = (
            (lightning_target > 0).astype(
                np.float32
            )
            if self.target_is_binary
            else counts_to_condition(
                lightning_target,
                self.lightning_clip_value,
            )
        )

        # Convert radar to tensors
        # Shape: (T, Z, H, W)
        radar_past = torch.from_numpy(
            radar_input
        ).float()

        radar_future = torch.from_numpy(
            radar_target
        ).float()

        radar_past_mask = torch.from_numpy(
            radar_input_mask
        ).float()

        radar_future_mask = torch.from_numpy(
            radar_target_mask
        ).float()

        return {
            # Volume representation: (T, Z, H, W)
            # NOT flattened to (T, C*Z, H, W)
            "radar_past": radar_past,
            "radar_past_mask": radar_past_mask,

            "radar_future": radar_future,
            "radar_future_mask": radar_future_mask,

            "lightning_past": torch.from_numpy(
                lightning_past
            ).float(),

            "lightning_future": torch.from_numpy(
                lightning_future
            ).float().unsqueeze(-1),

            "sample_id": path.stem,

            "time_emb": self._time_embedding(
                path.stem
            ),
        }
        
        
        
class NEXRAD3DNetCDFDataset(Dataset):
    """Dataset adapter for 3-D NEXRAD NetCDF samples.

    Volume mode flattens:
        (time, field, altitude, H, W)
    into:
        (time, field * altitude, H, W)

    Invalid radar values are represented as -1 after normalization.
    A separate radar validity mask is also returned.
    """

    def __init__(
        self,
        split_dir: str | Path,
        lightning_clip_value: float = 50.0,
        radar_scale: float = 128.0,
        target_is_binary: bool = True,
        radar_field: str = "Reflectivity",
        vertical_reduction: str = "max",
        radar_representation: str = "column_max",
        radar_fields: list[str] | None = None,
        #radar_validity_mask_field: str | None = None,
        #radar_validity_minimum: float = 1.0,
    ):
        if isinstance(split_dir, (str, Path)):
            self.split_dirs = [Path(split_dir)]
        elif isinstance(split_dir, (list, tuple)):
            self.split_dirs = [Path(p) for p in split_dir]
        else:
            raise TypeError(
                "split_dir must be a path string or a list of paths."
            )
        self.lightning_clip_value = float(lightning_clip_value)
        self.radar_scale = float(radar_scale)
        self.target_is_binary = bool(target_is_binary)
        self.radar_field = str(radar_field)
        self.vertical_reduction = str(vertical_reduction).lower()
        self.radar_representation = str(radar_representation).lower()

        self.radar_fields = [
            str(field)
            for field in (radar_fields or [self.radar_field])
        ]

        if self.vertical_reduction not in {"max", "mean"}:
            raise ValueError(
                "vertical_reduction must be 'max' or 'mean'."
            )

        if self.radar_representation not in {
            "column_max",
            "column_mean",
            "volume",
        }:
            raise ValueError(
                "radar_representation must be "
                "'column_max', 'column_mean', or 'volume'."
            )

        self.files = []

        for split_dir in self.split_dirs:
            if not split_dir.exists():
                raise ValueError(
                    f"Dataset directory does not exist: {split_dir}"
                )

            self.files.extend(
                split_dir.glob("*.nc")
            )

        self.files = sorted(self.files)
        if not self.files:
            raise ValueError(
                f"No NetCDF files found in directories: "
                f"{self.split_dirs}"
            )
    def __len__(self) -> int:
        return len(self.files)

    # LOAD RADAR

    def _load_radar(
        self,
        dataset: h5py.File,
        prefix: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load Reflectivity and its NaN/Inf validity mask.

        Only Reflectivity is used.

        Original Reflectivity:
            (T, Z, H, W)

        Validity mask:
            1 = finite Reflectivity
            0 = NaN or Inf Reflectivity

        For volume representation with Z=29:

            Reflectivity:
                (T, 29, H, W)

            Mask:
                (T, 29, H, W)

            Combined:
                (T, 58, H, W)

        Invalid Reflectivity values are represented as -1.
        """

       #Reflectivity used
        

        key = f"{prefix}_Reflectivity"

        if key not in dataset:
            raise KeyError(
                f"Dataset does not contain {key!r}."
            )

        volume = np.asarray(
            dataset[key],
            dtype=np.float32,
        )

        # Expected:
        # (T, Z, H, W)
        if volume.ndim != 4:
            raise ValueError(
                f"Expected {key!r} to have shape "
                f"(T, Z, H, W), got {volume.shape}"
            )

     
        # Create mask ONLY from NaN / Inf
       

        finite_mask = np.isfinite(volume)

        # 1 = valid Reflectivity
        # 0 = NaN / Inf
        radar_mask = finite_mask.astype(np.float32)


        # Replace invalid values with -1
    

        clean_volume = np.where(
            finite_mask,
            volume,
            -1.0,
        )

   
        # Normalize Reflectivity
      
        #
        # Valid values:
        #
        #     clip(value, 0, radar_scale) / radar_scale
        #
        # Invalid values remain -1.
        #
        # Therefore:
        #
        #     valid Reflectivity -> [0, 1]
        #     invalid Reflectivity -> -1
        #

        normalized_radar = np.where(
            finite_mask,
            np.clip(
                clean_volume,
                0.0,
                self.radar_scale,
            ) / self.radar_scale,
            -1.0,
        )

   
        # VOLUME REPRESENTATION
      

        if self.radar_representation == "volume":

            # normalized_radar:
            #     (T, Z, H, W)
            #
            # radar_mask:
            #     (T, Z, H, W)
            #
            # concatenate along channel dimension:
            #
            #     (T, 2*Z, H, W)
            #
            # For Z=29:
            #
            #     (T, 58, H, W)

            combined = np.concatenate(
                [
                    normalized_radar,
                    radar_mask,
                ],
                axis=1,
            )

            return (
                combined.astype(np.float32),
                radar_mask.astype(np.float32),
            )

        # COLUMN REPRESENTATION

        if self.radar_representation not in {
            "column_max",
            "column_mean",
        }:
            raise ValueError(
                f"Unsupported radar representation: "
                f"{self.radar_representation}"
            )

        with np.errstate(
            invalid="ignore",
            divide="ignore",
        ):

            if self.radar_representation == "column_max":
                radar = np.nanmax(
                    np.where(
                        finite_mask,
                        volume,
                        np.nan,
                    ),
                    axis=1,
                )

            else:
                radar = np.nanmean(
                    np.where(
                        finite_mask,
                        volume,
                        np.nan,
                    ),
                    axis=1,
                )

        column_mask = np.any(
            finite_mask,
            axis=1,
        )

        radar = np.where(
            column_mask,
            np.clip(
                np.nan_to_num(
                    radar,
                    nan=0.0,
                    posinf=self.radar_scale,
                    neginf=0.0,
                ),
                0.0,
                self.radar_scale,
            ) / self.radar_scale,
            -1.0,
        )

        return (
            radar.astype(np.float32),
            column_mask.astype(np.float32),
        )
        
        # COLUMN REPRESENTATION
        if len(volumes) != 1:
            raise ValueError(
                "Column radar representations support "
                "exactly one radar field."
            )

        volume = volumes[0]
        volume_mask = masks[0]

        reduction = (
            "mean"
            if self.radar_representation == "column_mean"
            else "max"
        )

        
        # Reduce vertically while respecting invalid values.
        with np.errstate(
            invalid="ignore",
            divide="ignore",
        ):

            if reduction == "max":

                radar = np.nanmax(
                    volume,
                    axis=1,
                )

            else:

                radar = np.nanmean(
                    volume,
                    axis=1,
                )

        # A column is valid if at least one altitude level contains a valid observation.
        column_mask = np.any(
            volume_mask,
            axis=1,
        )

        radar = np.where(
            column_mask,
            np.clip(
                np.nan_to_num(
                    radar,
                    nan=0.0,
                    posinf=self.radar_scale,
                    neginf=0.0,
                ),
                0.0,
                self.radar_scale,
            ) / self.radar_scale,
            -1.0,
        )

        return (
            radar.astype(np.float32),
            column_mask.astype(np.float32),
        )

    

    @staticmethod
    def _time_embedding(
        stem: str,
    ) -> torch.Tensor:

        match = re.search(
            r"(\d{14})",
            stem,
        )

        if match is None:
            return torch.zeros(
                4,
                dtype=torch.float32,
            )

        timestamp = match.group(1)

        month = float(
            int(timestamp[4:6])
        )

        hour = (
            float(int(timestamp[8:10]))
            + float(int(timestamp[10:12])) / 60.0
        )

        return torch.tensor(
            [
                math.sin(
                    2.0 * math.pi * hour / 24.0
                ),
                math.cos(
                    2.0 * math.pi * hour / 24.0
                ),
                math.sin(
                    2.0 * math.pi
                    * (month - 1.0)
                    / 12.0
                ),
                math.cos(
                    2.0 * math.pi
                    * (month - 1.0)
                    / 12.0
                ),
            ],
            dtype=torch.float32,
        )

   
    # GET ITEM
    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str]:

        path = self.files[index]

        with h5py.File(path, "r") as dataset:

    
            # Radar
            radar_input, radar_input_mask = (
                self._load_radar(
                    dataset,
                    "input",
                )
            )

            radar_target, radar_target_mask = (
                self._load_radar(
                    dataset,
                    "target",
                )
            )

            
            
            # Lightning
            lightning_input = np.asarray(
                dataset["input_lightning_map"],
                dtype=np.float32,
            )

            lightning_target = np.asarray(
                dataset["target_lightning_map"],
                dtype=np.float32,
            )

        # Lightning
        lightning_past = counts_to_condition(
            lightning_input,
            self.lightning_clip_value,
        )[..., None]

        lightning_future = (
            (lightning_target > 0).astype(
                np.float32
            )
            if self.target_is_binary
            else counts_to_condition(
                lightning_target,
                self.lightning_clip_value,
            )
        )

        
        # Convert radar to tensors
        radar_past = torch.from_numpy(
            radar_input
        ).float()

        radar_future = torch.from_numpy(
            radar_target
        ).float()

        radar_past_mask = torch.from_numpy(
            radar_input_mask
        ).float()

        radar_future_mask = torch.from_numpy(
            radar_target_mask
        ).float()

        
        # Column representations need a singleton dimension
        if self.radar_representation != "volume":

            radar_past = radar_past.unsqueeze(-1)

            radar_future = radar_future.unsqueeze(-1)

            radar_past_mask = (
                radar_past_mask.unsqueeze(-1)
            )

            radar_future_mask = (
                radar_future_mask.unsqueeze(-1)
            )

        return {
            "radar_past": radar_past,

           
            # 1 = valid radar
            # 0 = invalid radar
            "radar_past_mask": radar_past_mask,

            "lightning_past": torch.from_numpy(
                lightning_past
            ).float(),

            "lightning_future": torch.from_numpy(
                lightning_future
            ).float().unsqueeze(-1),

            "radar_future": radar_future,

            # mask added
            "radar_future_mask": radar_future_mask,

            "sample_id": path.stem,

            "time_emb": self._time_embedding(
                path.stem
            ),
        }


def _to_numpy_sample_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sample_lightning_support(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = _to_numpy_sample_array(value)
    while array.ndim > 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4:
        array = np.any(array > 0.0, axis=-1)
    elif array.ndim == 3:
        array = array > 0.0
    else:
        raise ValueError(f"Expected lightning sample array with 3 or 4 dims, got shape {array.shape}")
    while array.ndim > 2:
        array = np.any(array, axis=0)
    return array.astype(bool, copy=False)


def infer_lightning_spatial_mask(
    dataset: Dataset,
    include_past: bool = True,
    include_future: bool = True,
) -> np.ndarray:
    if not include_past and not include_future:
        raise ValueError("At least one of include_past or include_future must be True.")

    if isinstance(dataset, Subset):
        indices = [int(index) for index in dataset.indices]
        base_dataset = dataset.dataset
    else:
        indices = list(range(len(dataset)))
        base_dataset = dataset

    if isinstance(base_dataset, LightningNowcastNetCDFDataset):
        support_mask = None
        for index in indices:
            path = base_dataset.files[index]
            with h5py.File(path, "r") as file_handle:
                sample_mask = None
                if include_past:
                    past_mask = np.asarray(file_handle["ic_input"]) > 0.0
                    past_mask |= np.asarray(file_handle["cg_input"]) > 0.0
                    sample_mask = np.any(past_mask, axis=0)
                if include_future:
                    future_mask = np.asarray(file_handle["ic_target"]) > 0.0
                    future_mask |= np.asarray(file_handle["cg_target"]) > 0.0
                    future_mask = np.any(future_mask, axis=0)
                    sample_mask = future_mask if sample_mask is None else np.logical_or(sample_mask, future_mask)
                if sample_mask is None:
                    continue
                support_mask = sample_mask if support_mask is None else np.logical_or(support_mask, sample_mask)
        if support_mask is None:
            raise ValueError("Could not infer a lightning spatial mask from the dataset.")
        return support_mask.astype(bool, copy=False)

    support_mask = None
    for index in indices:
        sample = base_dataset[index]
        sample_mask = None
        if include_past and "lightning_past" in sample:
            sample_mask = _sample_lightning_support(sample["lightning_past"])
        if include_future and "lightning_future" in sample:
            future_mask = _sample_lightning_support(sample["lightning_future"])
            sample_mask = future_mask if sample_mask is None else np.logical_or(sample_mask, future_mask)
        if sample_mask is None:
            continue
        support_mask = sample_mask if support_mask is None else np.logical_or(support_mask, sample_mask)

    if support_mask is None:
        raise ValueError("Could not infer a lightning spatial mask from the dataset.")
    return support_mask.astype(bool, copy=False)


def spatial_mask_to_bbox(mask: np.ndarray) -> dict[str, int] | None:
    array = np.asarray(mask, dtype=bool)
    ys, xs = np.where(array)
    if len(ys) == 0:
        return None
    return {
        "row_min": int(ys.min()),
        "row_max": int(ys.max()),
        "col_min": int(xs.min()),
        "col_max": int(xs.max()),
    }


def get_row_index(row: pd.Series, preferred_key: str | None = None) -> int:
    if preferred_key is not None and preferred_key in row:
        return int(row[preferred_key])
    for key in ("file_row", "file_index"):
        if key in row:
            return int(row[key])
    raise KeyError(
        "Could not find a row index column. Expected one of "
        f"{preferred_key!r}, 'file_row', or 'file_index'."
    )


def resolve_event_and_seq_index(cum_counts: np.ndarray, index: int) -> tuple[int, int]:
    event_idx = int(np.searchsorted(cum_counts, index, side="right"))
    if event_idx == 0:
        seq_idx = index
    else:
        seq_idx = index - int(cum_counts[event_idx - 1])
    return event_idx, seq_idx


class FlowCastPairedH5Dataset(Dataset):
    def __init__(
        self,
        meta_csv: str | Path,
        vil_file: str | Path,
        lightning_file: str | Path,
        vil_dataset: str = "vil",
        lightning_dataset: str = "lght",
        vil_row_key: str | None = None,
        lightning_row_key: str | None = None,
        raw_seq_len: int = 49,
        lag_time: int = 13,
        lead_time: int = 12,
        time_spacing: int = 1,
        stride: int = 12,
        normalize_vil: bool = True,
        lightning_clip_value: float = 20.0,
        channel_last: bool = True,
        debug_mode: bool = False,
        vil_output_hw: tuple[int, int] | None = None,
    ):
        self.meta_csv = str(meta_csv)
        self.vil_file = str(vil_file)
        self.lightning_file = str(lightning_file)
        self.vil_dataset = vil_dataset
        self.lightning_dataset = lightning_dataset
        self.vil_row_key = vil_row_key
        self.lightning_row_key = lightning_row_key
        self.raw_seq_len = int(raw_seq_len)
        self.lag_time = int(lag_time)
        self.lead_time = int(lead_time)
        self.time_spacing = int(time_spacing)
        self.seq_len = (self.lag_time + self.lead_time) * self.time_spacing
        self.stride = int(stride)
        self.normalize_vil = bool(normalize_vil)
        self.lightning_clip_value = float(lightning_clip_value)
        self.channel_last = bool(channel_last)
        self.debug_mode = bool(debug_mode)
        self.vil_output_hw = tuple(vil_output_hw) if vil_output_hw is not None else None

        self.metadata = pd.read_csv(self.meta_csv, parse_dates=["time_utc"])
        if self.metadata.empty:
            raise ValueError(f"No events found in metadata file: {self.meta_csv}")
        if self.debug_mode:
            self.metadata = self.metadata.iloc[:10].reset_index(drop=True)

        if self.raw_seq_len < self.seq_len:
            raise ValueError("raw_seq_len must be >= (lag_time + lead_time) * time_spacing")

        self.vil_h5 = None
        self.lightning_h5 = None
        self.n_seq_per_event = 1 + (self.raw_seq_len - self.seq_len) // self.stride
        if self.debug_mode:
            self.n_seq_per_event = 1
        self.event_seq_counts = np.full(len(self.metadata), self.n_seq_per_event, dtype=np.int32)
        self.cum_counts = np.cumsum(self.event_seq_counts)

    def __len__(self) -> int:
        return int(self.cum_counts[-1])

    def _open_hdf5(self) -> None:
        if self.vil_h5 is None:
            self.vil_h5 = h5py.File(self.vil_file, "r")
        if self.lightning_h5 is None:
            self.lightning_h5 = h5py.File(self.lightning_file, "r")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        self._open_hdf5()
        event_idx, seq_idx = resolve_event_and_seq_index(self.cum_counts, index)
        row = self.metadata.iloc[event_idx]
        vil_row = get_row_index(row, self.vil_row_key)
        lightning_row = get_row_index(row, self.lightning_row_key)

        vil_event = self.vil_h5[self.vil_dataset][vil_row].astype(np.float32)
        lightning_event = self.lightning_h5[self.lightning_dataset][lightning_row].astype(np.float32)
        if self.normalize_vil:
            vil_event /= 255.0

        start = seq_idx * self.stride
        end = start + self.seq_len
        vil_segment = vil_event[..., start:end]
        lightning_segment = lightning_event[..., start:end]
        x_indices = [i * self.time_spacing for i in range(self.lag_time)]
        y_end = self.seq_len - 1
        y_indices = [y_end - i * self.time_spacing for i in range(self.lead_time)]
        y_indices.reverse()

        radar_past = vil_segment[..., x_indices]
        lightning_past_count = lightning_segment[..., x_indices]
        lightning_future_count = lightning_segment[..., y_indices]
        lightning_past = counts_to_condition(lightning_past_count, self.lightning_clip_value)
        lightning_future = (lightning_future_count > 0).astype(np.float32)

        radar_past_tensor = torch.from_numpy(radar_past).float().permute(2, 0, 1).unsqueeze(0)
        if self.vil_output_hw is not None:
            h, w = radar_past_tensor.shape[2], radar_past_tensor.shape[3]
            if (h, w) != self.vil_output_hw:
                radar_past_tensor = torch.nn.functional.interpolate(
                    radar_past_tensor,
                    size=self.vil_output_hw,
                    mode="bilinear",
                    align_corners=False,
                )
        lightning_past_tensor = torch.from_numpy(lightning_past).float().permute(2, 0, 1).unsqueeze(0)
        lightning_future_tensor = torch.from_numpy(lightning_future).float().permute(2, 0, 1).unsqueeze(0)

        if self.channel_last:
            radar_past_tensor = radar_past_tensor.permute(1, 2, 3, 0)
            lightning_past_tensor = lightning_past_tensor.permute(1, 2, 3, 0)
            lightning_future_tensor = lightning_future_tensor.permute(1, 2, 3, 0)

        return {
            "radar_past": radar_past_tensor,
            "lightning_past": lightning_past_tensor,
            "lightning_future": lightning_future_tensor,
            "sample_id": f"{Path(self.meta_csv).stem}_{index}",
        }


class PresavedRadarConditionDataset(Dataset):
    """Wraps any dataset and attaches presaved SimVP fut_lat + radar_mean to each sample.

    Loads from two H5 files:
      predictions_h5 : contains 'predicted_future_radar' (N, T, H, W) in mm/h and 'sample_id'
      latents_h5     : contains 'future_radar_latent' (N, T, lat_h, lat_w, C) and 'sample_id'

    Adds to each batch item:
      presaved_radar_mean : (T, H, W, 1) float32 normalized to [0, 1]
      presaved_fut_lat    : (T, lat_h, lat_w, C) float32
    """

    def __init__(
        self,
        base_dataset: Dataset,
        predictions_h5: str | Path,
        latents_h5: str | Path,
        radar_scale: float = 128.0,
    ):
        self.base = base_dataset
        self.predictions_h5 = Path(predictions_h5)
        self.latents_h5 = Path(latents_h5)
        self.radar_scale = float(radar_scale)

        with h5py.File(self.predictions_h5, "r") as f:
            raw_ids = f["sample_id"][:]
        self._sid_to_pred_idx: dict[str, int] = {
            (sid.decode() if isinstance(sid, bytes) else sid): i
            for i, sid in enumerate(raw_ids)
        }
        with h5py.File(self.latents_h5, "r") as f:
            raw_ids = f["sample_id"][:]
        self._sid_to_lat_idx: dict[str, int] = {
            (sid.decode() if isinstance(sid, bytes) else sid): i
            for i, sid in enumerate(raw_ids)
        }

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.base[index])
        sid = item["sample_id"]
        pred_idx = self._sid_to_pred_idx[sid]
        lat_idx = self._sid_to_lat_idx[sid]

        with h5py.File(self.predictions_h5, "r") as f:
            radar_mean_np = f["predicted_future_radar"][pred_idx].astype(np.float32)  # (T, H, W)
            uncertainty_np = f["prediction_uncertainty"][pred_idx].astype(np.float32)  # (T, H, W)
        with h5py.File(self.latents_h5, "r") as f:
            fut_lat_np = f["future_radar_latent"][lat_idx].astype(np.float32)  # (T, lat_h, lat_w, C)

        radar_mean_np = np.clip(radar_mean_np, 0.0, self.radar_scale) / self.radar_scale
        # Stack mean + uncertainty as 2-channel radar condition: (T, H, W, 2)
        radar_cond_np = np.stack([radar_mean_np, uncertainty_np], axis=-1)
        item["presaved_radar_mean"] = torch.from_numpy(radar_cond_np)       # (T, H, W, 2)
        item["presaved_fut_lat"] = torch.from_numpy(fut_lat_np)             # (T, lat_h, lat_w, C)
        return item


def build_dataset(split_name: str, data_config: dict) -> Dataset:
    backend = str(data_config.get("backend", "netcdf")).lower()
    if backend == "nexrad_3d_netcdf":
        split_key = f"{split_name}_dir"
        if split_key not in data_config:
            raise KeyError(f"Missing {split_key!r} for nexrad_3d_netcdf backend.")
        return NEXRAD3DNetCDFDataset(
            split_dir=data_config[split_key],
            lightning_clip_value=float(data_config.get("lightning_clip_value", 50.0)),
            radar_scale=float(data_config.get("radar_scale", 128.0)),
            target_is_binary=bool(data_config.get("target_is_binary", True)),
            radar_field=str(data_config.get("radar_field", "Reflectivity")),
            vertical_reduction=str(data_config.get("vertical_reduction", "max")),
            radar_representation=str(data_config.get("radar_representation", "column_max")),
            radar_fields=data_config.get("radar_fields"),
            #radar_validity_mask_field=data_config.get("radar_validity_mask_field"),
            #radar_validity_minimum=float(data_config.get("radar_validity_minimum", 1.0)),
        )
    if backend == "netcdf":
        split_key = f"{split_name}_dir"
        if split_key not in data_config:
            raise KeyError(f"Missing {split_key!r} for netcdf backend.")
        base = LightningNowcastNetCDFDataset(
            split_dir=data_config[split_key],
            lightning_clip_value=float(data_config.get("lightning_clip_value", 50.0)),
            radar_scale=float(data_config.get("radar_scale", 128.0)),
            target_is_binary=bool(data_config.get("target_is_binary", True)),
            lightning_input_mode=str(data_config.get("lightning_input_mode", "total")),
        )
        radar_cond = data_config.get("radar_condition", {})
        if radar_cond.get("enabled") and radar_cond.get("source") == "presaved":
            presaved_cfg = radar_cond["presaved"]
            split_key_pred = f"{split_name}_predictions_path"
            split_key_lat = f"{split_name}_latents_path"
            return PresavedRadarConditionDataset(
                base_dataset=base,
                predictions_h5=presaved_cfg[split_key_pred],
                latents_h5=presaved_cfg[split_key_lat],
                radar_scale=float(data_config.get("radar_scale", 128.0)),
            )
        return base
    if backend == "flowcast_h5":
        split_cfg = data_config[split_name]
        return FlowCastPairedH5Dataset(
            meta_csv=split_cfg["meta_csv"],
            vil_file=split_cfg["vil_file"],
            lightning_file=split_cfg["lightning_file"],
            vil_dataset=split_cfg.get("vil_dataset", "vil"),
            lightning_dataset=split_cfg.get("lightning_dataset", "lght"),
            vil_row_key=split_cfg.get("vil_row_key"),
            lightning_row_key=split_cfg.get("lightning_row_key"),
            raw_seq_len=int(data_config.get("raw_seq_len", 49)),
            lag_time=int(data_config.get("lag_time", 13)),
            lead_time=int(data_config.get("lead_time", 12)),
            time_spacing=int(data_config.get("time_spacing", 1)),
            stride=int(data_config.get("stride", 12)),
            normalize_vil=bool(data_config.get("normalize_vil", True)),
            lightning_clip_value=float(data_config.get("lightning_clip_value", 20.0)),
            channel_last=bool(data_config.get("channel_last", True)),
            debug_mode=bool(data_config.get("debug_mode", False)),
            vil_output_hw=tuple(data_config["vil_output_hw"]) if data_config.get("vil_output_hw") is not None else None,
        )
    raise ValueError(f"Unsupported data backend {backend!r}.")
