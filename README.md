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