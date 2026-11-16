# Looped Transformers with Learned Stochastic Stopping

Code for our NeurIPS 2026 paper, **Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping**.

The code supports Binary Addition, Dyck-1, Unique Set, and Copy with four methods: K=20, K=n, K=n±5, and RL-Halt. 

## Abstract

Looped Transformers — which repeatedly apply a shared transformer block — are an architecturally natural fit for variable-length algorithmic tasks. Although they can exhibit strong length generalization beyond the length of training sequences, this behavior is brittle, yielding high out-of-distribution (OOD) variance, even across well-performing in-distribution solutions. We trace this variance to the spurious correlation in simple algorithmic tasks between sequence length and number of loops. Introducing stochasticity into the number of loops during training sharply reduces OOD variance and stabilizes predictions across inference-time loop counts. To improve upon heuristic randomization schemes, we further analyze RL-Halting as a learned stochastic schedule and find that it generally improves the accuracy–stability trade-off. Across binary addition, Dyck-1, Unique Set, and Copy, learned stochastic stopping often improves this trade-off but can also stabilize a suboptimal computation. Our work suggests that “when to stop” should be treated as a training-time design choice, not merely an inference-time computation-allocation rule.

## Lay Summary

A looped transformer is an AI model that can repeat the same step before giving an answer. This makes it a natural fit for tasks like addition, where the same rule works for numbers of any length. Does giving the model one step per digit help it learn a rule it can use on longer numbers? In our experiments, models trained this way could be equally accurate on short numbers yet differ sharply on longer ones. We then asked how many steps to give the model during training. Across four tasks, varying that number at random made performance on longer examples more consistent across training runs. Letting the model learn how likely it should be to stop at each step often combined that consistency with better accuracy on examples longer than any it had seen during training.

## Install

Use Python 3.9–3.11.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run

Choose a configuration from the table below. Data are generated during training; no dataset or pretrained model download is needed.

```sh
looped-paper train --config configs/addition-fixed20.json --output runs/addition-fixed20 --device cuda
```

For a small installation check, use `--smoke --device cpu` and a separate output folder. See [setup and evaluation](docs/SETUP.md) for the exact protocol and resume commands.

## Recorded results

Mean OOD oracle exact-sequence accuracy over input lengths 20, 25, …, 60. Each recorded result is one run with the listed configuration; the paper column contains averages across runs. Exact scores can vary across environments and configs.

| Task | Method | Recorded accuracy | Paper mean |
|---|---|---:|---:|
| Addition | [K=20](configs/addition-fixed20.json) | 32.28% | 34.20% |
| Addition | [K=n](configs/addition-length_matched.json) | 14.56% | 22.10% |
| Addition | [K=n±5](configs/addition-random_window.json) | 33.98% | 30.90% |
| Addition | [RL-Halt](configs/addition-rl_halt.json) | 42.14% | 45.00% |
| Dyck-1 | [K=20](configs/dyck1-fixed20.json) | 82.15% | 84.60% |
| Dyck-1 | [K=n](configs/dyck1-length_matched.json) | 49.66% | 66.60% |
| Dyck-1 | [K=n±5](configs/dyck1-random_window.json) | 41.74% | 88.20% |
| Dyck-1 | [RL-Halt](configs/dyck1-rl_halt.json) | 99.87% | 97.50% |

## Reference

If you use this code, please cite our paper:

Hsun-Yu Kuo, El Mahdi Chayti, Patrik Reizinger, Wieland Brendel, and Martin Jaggi. **Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping.** NeurIPS 2026. [Official conference page](https://neurips.cc/virtual/2026/poster/152714) · [OpenReview](https://openreview.net/forum?id=Qv0Lwp7s5A).

```bibtex
@inproceedings{kuo2026stabilizing,
  title = {Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping},
  author = {Hsun-Yu Kuo and El Mahdi Chayti and Patrik Reizinger and Wieland Brendel and Martin Jaggi},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  url = {https://neurips.cc/virtual/2026/poster/152714}
}
```

Model and dependency attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
