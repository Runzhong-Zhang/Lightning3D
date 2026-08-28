"""Report NEXRAD validity-mask coverage without modifying data or checkpoints.

Example:
    python scripts/check_nexrad_mask.py --config configs/baseline_nexrad_subset_15_volume.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check NEXRAD validity-mask coverage for train/validation splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "baseline_nexrad_subset_15_volume.json",
        help="NEXRAD experiment JSON config.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Splits to inspect. Defaults to train and val, leaving test untouched.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for a quick representative check.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_split(
    split_dir: Path,
    mask_field: str,
    minimum: float,
    max_samples: int | None,
) -> dict:
    files = sorted(split_dir.glob("*.nc"))
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise ValueError(f"No NetCDF files found in {split_dir}")

    valid_count = 0
    total_count = 0
    per_sample_fraction: list[float] = []
    per_altitude_valid: np.ndarray | None = None
    per_altitude_total: np.ndarray | None = None

    for path in files:
        key = f"input_{mask_field}"
        with h5py.File(path, "r") as dataset:
            if key not in dataset:
                raise KeyError(f"{path.name} does not contain {key!r}")
            values = dataset[key]
            if values.ndim != 4:
                raise ValueError(f"Expected {key!r} shape (T, Z, H, W), got {values.shape} in {path.name}")
            altitude_levels = int(values.shape[1])
            if per_altitude_valid is None:
                per_altitude_valid = np.zeros(altitude_levels, dtype=np.int64)
                per_altitude_total = np.zeros(altitude_levels, dtype=np.int64)
            elif altitude_levels != len(per_altitude_valid):
                raise ValueError(f"Inconsistent number of altitude levels in {path.name}")

            sample_valid = 0
            sample_total = 0
            for altitude_idx in range(altitude_levels):
                layer = np.asarray(values[:, altitude_idx], dtype=np.float32)
                valid = np.isfinite(layer) & (layer >= minimum)
                count = int(valid.sum())
                total = int(valid.size)
                per_altitude_valid[altitude_idx] += count
                per_altitude_total[altitude_idx] += total
                sample_valid += count
                sample_total += total

        valid_count += sample_valid
        total_count += sample_total
        per_sample_fraction.append(sample_valid / max(sample_total, 1))

    assert per_altitude_valid is not None and per_altitude_total is not None
    return {
        "sample_count": len(files),
        "mask_key": f"input_{mask_field}",
        "validity_rule": f"finite value and {mask_field} >= {minimum}",
        "valid_voxels": valid_count,
        "total_voxels": total_count,
        "valid_fraction": valid_count / max(total_count, 1),
        "invalid_fraction": 1.0 - valid_count / max(total_count, 1),
        "per_sample_valid_fraction": {
            "min": float(np.min(per_sample_fraction)),
            "mean": float(np.mean(per_sample_fraction)),
            "max": float(np.max(per_sample_fraction)),
        },
        "per_altitude_valid_fraction": (per_altitude_valid / per_altitude_total).tolist(),
    }


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    data_config = config.get("data", {})
    if data_config.get("backend") != "nexrad_3d_netcdf":
        raise ValueError("This checker requires data.backend='nexrad_3d_netcdf'.")
    mask_field = data_config.get("radar_validity_mask_field")
    if not mask_field:
        raise ValueError("Set data.radar_validity_mask_field in the config before running this checker.")
    minimum = float(data_config.get("radar_validity_minimum", 1.0))

    report = {
        "config": str(args.config),
        "mask_field": str(mask_field),
        "minimum": minimum,
        "splits": {},
    }
    for split in args.splits:
        split_dir = Path(data_config[f"{split}_dir"])
        report["splits"][split] = check_split(split_dir, str(mask_field), minimum, args.max_samples)

    report_json = json.dumps(report, indent=2)
    print(report_json)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
