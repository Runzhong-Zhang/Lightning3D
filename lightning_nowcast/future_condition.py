from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from lightning_nowcast.models.radar_simvp import OpenSTLRadarPredictor


class OnlineRadarFutureConditionSource:
    def __init__(self, predictor: OpenSTLRadarPredictor, output_type: str = "decoded"):
        self.predictor = predictor
        self.output_type = str(output_type).lower()
        if self.output_type not in {"decoded", "latent"}:
            raise ValueError(f"Unsupported output_type {self.output_type!r}. Expected 'decoded' or 'latent'.")

    def build_future_condition(self, batch: dict[str, Any], out_seq_len: int | None = None) -> torch.Tensor:
        if self.output_type == "latent":
            return self.predictor.build_future_feature(batch["radar_past"], out_seq_len=out_seq_len)
        return self.predictor.build_future_condition(batch["radar_past"], out_seq_len=out_seq_len)

    @property
    def has_trainable_parameters(self) -> bool:
        return self.predictor.has_trainable_parameters

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return self.predictor.trainable_parameters()

    def train(self) -> None:
        self.predictor.model.train()

    def eval(self) -> None:
        self.predictor.model.eval()

    def close(self) -> None:
        return None


class PresavedRadarFutureConditionSource:
    def __init__(self, h5_path: str | Path, radar_scale: float = 128.0):
        self.h5_path = Path(h5_path)
        self.radar_scale = float(radar_scale)
        self.h5_file = None
        self.predicted_ds = None
        self.uncertainty_ds = None
        with h5py.File(self.h5_path, "r") as h5_file:
            raw_sample_ids = h5_file["sample_id"][:]
        self.sample_index = {
            self._normalize_sample_id(sample_id): int(index)
            for index, sample_id in enumerate(raw_sample_ids)
        }

    def _ensure_open(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            self.predicted_ds = self.h5_file["predicted_future_radar"]
            self.uncertainty_ds = self.h5_file["prediction_uncertainty"]

    @staticmethod
    def _normalize_sample_id(sample_id: Any) -> str:
        if isinstance(sample_id, bytes):
            return sample_id.decode("utf-8")
        return str(sample_id)

    def build_future_condition_from_sample_ids(
        self,
        batch_sample_ids: list[str],
        out_seq_len: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        self._ensure_open()
        assert self.predicted_ds is not None
        assert self.uncertainty_ds is not None
        sample_indices = []
        for sample_id in batch_sample_ids:
            if sample_id not in self.sample_index:
                raise KeyError(f"Sample id {sample_id!r} not found in presaved future-radar file {self.h5_path}")
            sample_indices.append(self.sample_index[sample_id])

        index_array = np.asarray(sample_indices, dtype=np.int64)
        sort_order = np.argsort(index_array)
        sorted_indices = index_array[sort_order]
        restore_order = np.empty_like(sort_order)
        restore_order[sort_order] = np.arange(len(sort_order), dtype=np.int64)

        predicted_batch = np.asarray(self.predicted_ds[sorted_indices], dtype=np.float32)[restore_order] / self.radar_scale
        uncertainty_batch = np.asarray(self.uncertainty_ds[sorted_indices], dtype=np.float32)[restore_order]
        if out_seq_len is not None:
            predicted_batch = predicted_batch[:, : int(out_seq_len)]
            uncertainty_batch = uncertainty_batch[:, : int(out_seq_len)]

        future_condition = np.concatenate(
            [
                predicted_batch[..., None],
                uncertainty_batch[..., None],
            ],
            axis=-1,
        )
        tensor = torch.from_numpy(future_condition)
        if device is not None:
            tensor = tensor.to(device=device, non_blocking=True)
        return tensor

    def build_future_condition(self, batch: dict[str, Any], out_seq_len: int | None = None) -> torch.Tensor:
        batch_sample_ids = [self._normalize_sample_id(sample_id) for sample_id in batch["sample_id"]]
        target_device = batch["radar_past"].device if isinstance(batch.get("radar_past"), torch.Tensor) else None
        return self.build_future_condition_from_sample_ids(batch_sample_ids, out_seq_len=out_seq_len, device=target_device)

    def close(self) -> None:
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
            self.predicted_ds = None
            self.uncertainty_ds = None


class PresavedRadarFutureConditionDataset(Dataset):
    def __init__(self, dataset: Dataset, h5_path: str | Path, radar_scale: float = 128.0):
        self.dataset = dataset
        self.future_condition_source = PresavedRadarFutureConditionSource(h5_path, radar_scale=radar_scale)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        sample_id = self.future_condition_source._normalize_sample_id(sample["sample_id"])
        lead_time = int(sample["lightning_future"].shape[0])
        future_condition = self.future_condition_source.build_future_condition_from_sample_ids(
            [sample_id],
            out_seq_len=lead_time,
            device=None,
        )[0]
        sample["future_condition"] = future_condition.float()
        return sample

    def close(self) -> None:
        self.future_condition_source.close()

    def __del__(self):
        self.close()


def maybe_wrap_dataset_with_future_condition(dataset: Dataset, config: dict[str, Any], split: str) -> Dataset:
    radar_condition = config.get("radar_condition", {})
    if not radar_condition.get("enabled", False):
        return dataset
    source = str(radar_condition.get("source", "online")).lower()
    if source != "presaved":
        return dataset
    output_type = str(radar_condition.get("output_type", "decoded")).lower()
    if output_type == "latent":
        raise ValueError("Presaved latent future features are not implemented. Use radar_condition.source='online'.")
    presaved = radar_condition.get("presaved", {})
    h5_path = presaved.get(f"{split}_path") or presaved.get(split)
    if h5_path is None:
        raise ValueError(f"radar_condition.presaved must define a path for split {split!r}")
    radar_scale = float(config.get("data", {}).get("radar_scale", 128.0))
    return PresavedRadarFutureConditionDataset(dataset, h5_path, radar_scale=radar_scale)


def build_future_condition_source(config: dict[str, Any], split: str, device: torch.device):
    radar_condition = config.get("radar_condition", {})
    if not radar_condition.get("enabled", False):
        return None

    source = str(radar_condition.get("source", "online")).lower()
    output_type = str(
        radar_condition.get(
            "output_type",
            "latent" if str(config.get("model", {}).get("future_condition_mode", "simple")).lower() == "latent_predecoder" else "decoded",
        )
    ).lower()
    if source == "online":
        predictor = OpenSTLRadarPredictor(**radar_condition["predictor"]).to(device)
        return OnlineRadarFutureConditionSource(predictor, output_type=output_type)
    if source == "presaved":
        if output_type == "latent":
            raise ValueError("Presaved latent future features are not implemented. Use radar_condition.source='online'.")
        presaved = radar_condition.get("presaved", {})
        h5_path = presaved.get(f"{split}_path") or presaved.get(split)
        if h5_path is None:
            raise ValueError(f"radar_condition.presaved must define a path for split {split!r}")
        radar_scale = float(config.get("data", {}).get("radar_scale", 128.0))
        return PresavedRadarFutureConditionSource(h5_path, radar_scale=radar_scale)
    raise ValueError(f"Unsupported radar_condition source {source!r}. Expected 'online' or 'presaved'.")