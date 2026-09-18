For the NEXRAD backend, `data.radar_representation="volume"` automatically
selects the SeparateCZ dataset, preserving radar and mask altitude dimensions.
Column representations (`column_max` and `column_mean`) use the 2D dataset path.

Choose `model.vertical_encoder` from `baseline`, `capacity`, `mixing`, or `max`.
Volume models require `radar_input_layout="channel_first_3d"`,
`radar_input_channels=2`, and the dataset's `radar_vertical_levels` (29 here).

Training:

```bash
python scripts/train_lightning_simvp.py --config configs/baseline_nexrad.json
python scripts/train_lightning_simvp.py --config configs/baseline_vertical_capacity.json
python scripts/train_lightning_simvp.py --config configs/baseline_vertical_mixing.json
python scripts/train_lightning_simvp.py --config configs/baseline_vertical_max.json
```

Evaluation:

```bash
python scripts/eval_lightning_simvp.py --checkpoint outputs/baseline_nexrad/best.pt --split test
python scripts/eval_lightning_simvp.py --checkpoint outputs/baseline_vertical_capacity/best.pt --split test
python scripts/eval_lightning_simvp.py --checkpoint outputs/baseline_vertical_mixing/best.pt --split test
python scripts/eval_lightning_simvp.py --checkpoint outputs/baseline_vertical_max/best.pt --split test
```

Important note:

Please modify the path/environment info in pbs files for different users.
