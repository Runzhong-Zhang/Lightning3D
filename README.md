Training:

```bash
python scripts/train_lightning_simvp.py --config configs/baseline_nexrad.json
```

Evaluation:

```bash
python scripts/eval_lightning_simvp.py --checkpoint outputs/baseline_nexrad/latest.pt --split test
```

Modification:

```
benchmark_dataset.py
Line 14:
CONFIG = "/home/users/ntu/tejasvit/scratch/3D-radar-lightning-nowcasting/configs/baseline_nexrad_subset_15_volume.json"

configs/baseline_nexrad.json
Line 3, 4:
"experiment_name": "lightning_simvp_nexrad_refl_mask_volume_58ch_nradobs_mask",
"output_dir": "/home/users/ntu/tejasvit/scratch/3d-radar-nowcasting/3d-radar-nowcasting/outputs",
Line 7, 8, 9:
"train_dir":["/data/projects/13014407/datasets/3D_NEXRAD_GLM/2018/GRIDRAD_GLM_DATASET_2018", "/data/projects/13014407/datasets/3D_NEXRAD_GLM/2019/GRIDRAD_GLM_DATASET_2019","/data/projects/13014407/datasets/3D_NEXRAD_GLM/2020/GRIDRAD_GLM_DATASET_2020","/data/projects/13014407/datasets/3D_NEXRAD_GLM/2021/GRIDRAD_GLM_DATASET_2021"],
"val_dir": ["/data/projects/13014407/datasets/3D_NEXRAD_GLM/2022/GRIDRAD_GLM_DATASET_2022"],
"test_dir": ["/data/projects/13014407/datasets/3D_NEXRAD_GLM/2023/GRIDRAD_GLM_DATASET_2023"],
Line 38: "batch_size": 15,
Line 40: "epochs": 2,
```