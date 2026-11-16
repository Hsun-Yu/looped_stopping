"""Four algorithmic tasks with four stopping methods and local outputs."""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import (
    generate_prompt_matrix_addition, generate_prompt_matrix_copy,
    generate_prompt_matrix_dict, generate_prompt_matrix_dyck,
)
from .halting import compute_step_losses, sample_exit_steps, MovingAverageBaseline
from .model import GeneralTransformerModel

GENERATORS = {
    'addition': generate_prompt_matrix_addition,
    'copy': generate_prompt_matrix_copy,
    'unique_set': generate_prompt_matrix_dict,
    'dyck1': generate_prompt_matrix_dyck,
}
METHODS = ('fixed20', 'length_matched', 'random_window', 'rl_halt')
ORACLE_HORIZONS = ('min20_n_plus_2',)


def oracle_depth_limit(protocol, length):
    """Include K=20 for short inputs; use n+2 for OOD inputs."""
    if protocol != 'min20_n_plus_2':
        raise ValueError('This release uses min20_n_plus_2 oracle evaluation.')
    return max(20, length + 2)


def validate_config(cfg):
    if cfg['task'] not in GENERATORS or cfg['method'] not in METHODS:
        raise ValueError('Only Addition, Dyck-1, Unique Set, Copy, and the four release methods are supported.')
    data = cfg['data']
    if data['sampling'] != 'uniform' or not 1 <= data['min_length'] <= data['max_length'] <= 19:
        raise ValueError('Training requires uniform input lengths within 1–19.')
    for key in ('steps', 'batch_size', 'log_every', 'checkpoint_every'):
        if cfg['training'][key] < 1:
            raise ValueError(f'training.{key} must be positive')
    if cfg['training']['decay_start_step'] < 0:
        raise ValueError('Learning-rate decay start must be nonnegative.')
    if cfg['method'] == 'random_window':
        if cfg['training'].get('depth_sampling') != 'archive_uniform_window':
            raise ValueError('Use the random-window protocol in the supplied configurations.')
        if cfg['training']['depth_max'] < 2:
            raise ValueError('Random-window depth cap must be at least 2.')
    if cfg['model']['n_embd'] % cfg['model']['n_head']:
        raise ValueError('Model width must be divisible by attention heads.')
    if cfg['method'] == 'rl_halt':
        h = cfg['halting']
        if h['max_depth'] < 1 or h['samples_per_example'] < 1:
            raise ValueError('Halting depth and sample count must be positive.')
        if h['reward'] != 'negative_loss' or h['detach_gate_input']:
            raise ValueError('RL-Halt requires negative-loss rewards and an attached gate input.')
    ev = cfg['evaluation']
    if ev['oracle_horizon'] not in ORACLE_HORIZONS:
        raise ValueError(f'oracle_horizon must be one of {ORACLE_HORIZONS}.')
    if not ev['lengths'] or min(ev['lengths']) < 1 or min(ev['examples_per_length'], ev['batch_size']) < 1:
        raise ValueError('Evaluation lengths and batch sizes must be positive.')


def build_model(cfg, device):
    seeds = cfg['seeds']
    random.seed(seeds['runtime'])
    np.random.seed(seeds['runtime'])
    torch.manual_seed(seeds['runtime'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds['runtime'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # manual_seed also changes CUDA RNGs. Restore them after model initialization
    # so a distinct model_init seed cannot silently replace the runtime seed.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seeds['model_init'])
        model = GeneralTransformerModel(**cfg['model'])
    return model.to(device)


def batch(cfg, size, rng, device, length=None):
    lo = cfg['data']['min_length'] if length is None else length
    hi = cfg['data']['max_length'] if length is None else length
    values = GENERATORS[cfg['task']](
        size, min_num_digits=lo, max_num_digits=hi + 1, max_len=hi + 2, rng=rng,
    )
    xs, lengths, targets, mask = values
    if cfg['model']['linear_embedding']:
        xs = F.one_hot(xs.long(), cfg['model']['n_dims']).float()
    return xs.to(device), lengths.flatten().to(device), targets.to(device), mask.to(device)


def sample_depths(method, lengths, generator=None, protocol='archive_uniform_window', max_depth=22):
    if method == 'fixed20':
        return torch.full_like(lengths, 20)
    if method == 'length_matched':
        return lengths.clone()
    if method == 'random_window':
        if protocol != 'archive_uniform_window':
            raise ValueError('Use the random-window protocol in the supplied configurations.')
        rng = random if generator is None else generator
        draws = [rng.randint(max(2, int(n) - 5), max(2, int(n) + 5))
                 for n in lengths.tolist()]
        return torch.tensor(draws, device=lengths.device, dtype=lengths.dtype).clamp(2, max_depth)
    raise ValueError('RL-Halt draws from its learned distribution instead.')


def selected_logits(states, depths):
    stacked = torch.stack(states)
    return stacked[depths.long() - 1, torch.arange(depths.numel(), device=depths.device)]


def training_loss(model, cfg, values, baseline):
    xs, lengths, targets, mask = values
    if cfg['method'] != 'rl_halt':
        depths = sample_depths(cfg['method'], lengths,
                              protocol=cfg['training'].get('depth_sampling', 'archive_uniform_window'),
                              max_depth=cfg['training'].get('depth_max', 22))
        states = model.looped_forward(xs, int(depths.max()))
        logits = selected_logits(states, depths)
        loss = F.cross_entropy(logits[mask.bool()], targets[mask.bool()])
        return loss, {'task_loss': float(loss.detach()), 'mean_depth': float(depths.float().mean())}
    h = cfg['halting']
    states, probabilities = model.looped_forward_with_exit_prob(xs, h['max_depth'])
    q = torch.stack(probabilities)
    q = q / q.sum(0, keepdim=True).clamp_min(1e-8)
    per_step_losses = compute_step_losses(states, targets, mask)
    tau, logp = sample_exit_steps(q, h['samples_per_example'])
    chosen = per_step_losses.T.unsqueeze(0).expand(tau.shape[0], -1, -1).gather(2, tau.unsqueeze(-1)).squeeze(-1)
    reward = -chosen.detach()  # No compute penalty.
    advantage = (reward - baseline.update(reward)).detach()
    policy_loss = -(advantage * logp).mean()
    entropy = -(q * q.clamp_min(1e-8).log()).sum(0).mean()
    task_loss = chosen.mean()
    loss = task_loss + h['gate_weight'] * policy_loss - h['entropy_weight'] * entropy
    return loss, {
        'task_loss': float(task_loss.detach()), 'policy_loss': float(policy_loss.detach()),
        'entropy': float(entropy.detach()), 'mean_depth': float(tau.float().mean() + 1),
    }


def exact_match(logits, targets, mask):
    return ((logits.argmax(-1) == targets) | ~mask.bool()).all(-1)


@torch.no_grad()
def evaluate(model, cfg, device):
    model.eval()
    ev = cfg['evaluation']
    rows = []
    for length in ev['lengths']:
        # Fixed per-length seed makes task examples independent of evaluation order.
        rng = np.random.RandomState(cfg['seeds']['eval'] + 1009 * length)
        oracle_horizon = oracle_depth_limit(ev['oracle_horizon'], length)
        policy_horizon = 20 if cfg['method'] == 'fixed20' else (ev['policy_max_depth'] if cfg['method'] == 'rl_halt' else length)
        oracle_correct = policy_correct = total = 0
        policy_depth_sum = 0.0
        while total < ev['examples_per_length']:
            size = min(ev['batch_size'], ev['examples_per_length'] - total)
            xs, lengths, targets, mask = batch(cfg, size, rng, device, length)
            # Oracle and policy see the same examples. Their horizons stay distinct.
            states = model.looped_forward(xs, max(oracle_horizon, policy_horizon))
            correct = torch.stack([exact_match(s, targets, mask) for s in states[:oracle_horizon]])
            oracle_correct += int(correct.any(0).sum())
            if cfg['method'] == 'rl_halt':
                policy_states, probabilities = model.looped_forward_with_exit_prob(xs, policy_horizon)
                depths = torch.stack(probabilities).argmax(0) + 1
                logits = selected_logits(policy_states, depths)
            else:
                depths = torch.full_like(lengths, policy_horizon)
                logits = states[policy_horizon - 1]
            policy_correct += int(exact_match(logits, targets, mask).sum())
            policy_depth_sum += float(depths.sum())
            total += size
        rows.append({'length': length, 'examples': total, 'oracle_correct': oracle_correct,
                     'policy_correct': policy_correct, 'oracle_accuracy': oracle_correct / total,
                     'policy_accuracy': policy_correct / total, 'mean_policy_depth': policy_depth_sum / total,
                     'oracle_max_depth': oracle_horizon})
    ood = [r for r in rows if r['length'] in range(20, 61, 5)]
    return {'oracle_horizon': ev['oracle_horizon'], 'rows': rows,
            'ood_lengths': [r['length'] for r in ood],
            'oracle_ood_mean': sum(r['oracle_accuracy'] for r in ood) / len(ood) if ood else None,
            'policy_ood_mean': sum(r['policy_accuracy'] for r in ood) / len(ood) if ood else None}


def rng_state(rng):
    name, keys, pos, has_gauss, cached = rng.get_state()
    return [name, keys.tolist(), pos, has_gauss, cached]


def restore_numpy(rng, state):
    rng.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))


def write_checkpoint(path, model, optimizer, scheduler, baseline, cfg, next_step, data_rng):
    payload = {'format_version': 1, 'config': cfg, 'next_step': next_step,
               'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'baseline': baseline.value if baseline else None,
               'data_rng': rng_state(data_rng), 'numpy_rng': rng_state(np.random),
               'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train(cfg, output, device, resume=None):
    validate_config(cfg)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError(f'{output} is not empty; choose a new output directory or use --resume.')
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, device)
    t = cfg['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, t['steps'] - t['decay_start_step']))
    baseline = MovingAverageBaseline(cfg['halting']['baseline_decay']) if cfg['method'] == 'rl_halt' else None
    data_rng = np.random.RandomState(cfg['seeds']['data'])
    start = 0
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=True)
        if saved['config'] != cfg:
            raise ValueError('Resume configuration differs from the checkpoint configuration.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        if baseline:
            baseline.value = saved['baseline']
        restore_numpy(data_rng, saved['data_rng'])
        restore_numpy(np.random, saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        start = saved['next_step']
    (output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    model.train()
    with (output / 'metrics.jsonl').open('a') as log:
        for step in range(start, t['steps']):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = training_loss(model, cfg, batch(cfg, t['batch_size'], data_rng, device), baseline)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Preserve the recorded delayed-cosine update timing.
            if t['decay_start_step'] == 0 or step > t['decay_start_step']:
                scheduler.step()
            if step % t['log_every'] == 0 or step == t['steps'] - 1:
                record = {'step': step + 1, 'loss': float(loss.detach()), 'grad_norm': float(norm),
                          'learning_rate': optimizer.param_groups[0]['lr'],
                          'training_max_length': cfg['data']['max_length'], **metrics}
                log.write(json.dumps(record) + '\n')
                log.flush()
                print(json.dumps(record), flush=True)
            if (step + 1) % t['checkpoint_every'] == 0 or step == t['steps'] - 1:
                write_checkpoint(output / 'last.pt', model, optimizer, scheduler, baseline, cfg, step + 1, data_rng)
    result = evaluate(model, cfg, device)
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('train')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--resume', type=Path)
    run.add_argument('--smoke', action='store_true', help='Two tiny CPU-compatible steps; not paper results.')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--checkpoint', type=Path, required=True)
    ev.add_argument('--output', type=Path, required=True)
    for child in (run, ev):
        child.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        child.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if args.command == 'train':
        cfg = json.loads(args.config.read_text())
        if args.smoke:
            cfg = copy.deepcopy(cfg)
            cfg['smoke_test'] = True
            cfg['model'].update(n_embd=16, n_layer=1, n_head=2, n_positions=128)
            cfg['data']['max_length'] = 4
            cfg['training'].update(steps=2, batch_size=2, decay_start_step=0, log_every=1, checkpoint_every=1)
            cfg['evaluation'].update(lengths=[2, 4], examples_per_length=4, batch_size=2)
        result = train(cfg, args.output, device, args.resume)
    else:
        if args.output.exists():
            raise ValueError(f'{args.output} already exists; choose a new evaluation output path.')
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        cfg = saved['config']
        validate_config(cfg)
        model = build_model(cfg, device)
        model.load_state_dict(saved['model'])
        result = evaluate(model, cfg, device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'oracle_ood_mean': result['oracle_ood_mean'], 'policy_ood_mean': result['policy_ood_mean']}))


if __name__ == '__main__':
    main()
