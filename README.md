# esm-runs (private)

Run queue, results and model weights for **Evolving the Stain Map** (quality-diversity stain/scanner augmentation, Camelyon17-WILDS).

- `code/esm_core.py`: the single copy of the training code that every worker uses
- `runs.json`: the fixed run list (B = dev track, ResNet-18; C = official-style, DenseNet-121 from scratch)
- `claims/<run>.json`: who is running what (a claim older than 40 min (B) or 4 h (C) without a result is taken over)
- `results/<run>/`: metrics.json, log.jsonl, preds_test.npz, pool_final.npy
- Release **weights**: one `<run>.pt` per run (EMA weights, fp16)

Workers need the secret `GH_TOKEN` (fine-grained token: this repo only, Contents read/write). Never commit tokens.
