# Setup and evaluation

The eight JSON configurations specify all training settings and seeds for Addition and Dyck-1. Data generation and task support for Unique Set and Copy are included without configurations. Both configured tasks use uniform input lengths 1–19, batch size 64, AdamW with learning rate 0.0001 and weight decay 0.01, gradient clipping at 1.0, and delayed cosine decay starting after zero-based update index 32,000. Addition trains for 50,001 updates; Dyck-1 trains for 100,001.

The shared looped model has 3 transformer layers, width 256, 8 attention heads, a linear input embedding, and a 6-token vocabulary. Dropout is zero. Configuration seeds control initialization, data generation, and training randomness separately; all evaluation seeds are 42.

| Method | Training depth | Policy evaluation |
|---|---|---|
| K=20 | 20 | 20 |
| K=n | Input length n | n |
| K=n±5 | Inclusive Python `randint(max(2,n−5), max(2,n+5))`, then capped at 22 | n |
| RL-Halt | Learned distribution over depths 1–30; 4 samples per example | Most probable depth within 1–30 |

RL-Halt uses negative cross-entropy reward, entropy weight 0.01, moving-average baseline decay 0.9, and gate-loss weight 1. The gate input remains attached to the computation graph.

Evaluation uses 6,400 examples at each length 5, 10, …, 60. Oracle accuracy counts an example as correct if its entire supervised output is correct at any depth from 1 through `max(20,n+2)`. This includes depth 20 for short inputs; OOD lengths retain the `n+2` horizon. Oracle evaluation uses the reference answer to choose a depth. Policy accuracy uses the stopping rule above and is recorded separately. The reported OOD mean equally weights lengths 20, 25, …, 60.

## Commands

Run all eight configurations sequentially on one GPU:

```sh
for config in configs/*.json; do
  name=$(basename "$config" .json)
  looped-paper train --config "$config" --output "runs/$name" --device cuda || break
done
```

Resume a checkpoint with the same configuration:

```sh
looped-paper train --config configs/addition-fixed20.json --output runs/addition-fixed20 --resume runs/addition-fixed20/last.pt --device cuda
```

Evaluate a trained checkpoint:

```sh
looped-paper evaluate --checkpoint runs/addition-fixed20/last.pt --output runs/addition-fixed20/evaluation-repeat.json --device cuda
```

Each run saves its resolved configuration, training metrics, `last.pt` checkpoint, and per-length evaluation counts. Checkpoints are written every 1,000 updates and include optimizer, scheduler, and random states for resumption. Training protects nonempty output folders unless `--resume` is supplied; evaluation requires a new output filename.
