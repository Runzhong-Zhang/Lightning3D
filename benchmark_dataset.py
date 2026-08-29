import time
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lightning_nowcast.data import build_dataset


CONFIG = "configs/baseline_nexrad.json"


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())
    print()
    with open(CONFIG, "r") as f:
        config = json.load(f)

    print("Building dataset...")

    dataset = build_dataset("train", config["data"])

    print("Dataset length:", len(dataset))

    print("\nLoading sample 0...")
    start = time.perf_counter()

    sample = dataset[0]

    elapsed = time.perf_counter() - start

    print(f"Sample loading time: {elapsed:.3f} seconds")

    print("\nSample contents:")

    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(
                f"{key:25s} "
                f"shape={tuple(value.shape)} "
                f"dtype={value.dtype} "
                f"size={value.numel() * value.element_size() / 1024**2:.2f} MB"
            )
        else:
            print(
                f"{key:25s} "
                f"type={type(value).__name__} "
                f"value={value}"
            )


if __name__ == "__main__":
    main()
