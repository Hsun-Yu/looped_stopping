import torch
import torch.nn.functional as F
from torch.distributions import Categorical

def sample_exit_steps(exit_q, num_samples=1):
    """Sample exit steps per example.

    Args:
        exit_q: Tensor [T, B]
        num_samples: int >= 1
    Returns:
        tau: LongTensor [S, B] sampled step indices (S=num_samples)
        logp: Tensor [S, B] log-probabilities
    """
    if exit_q.dim() != 2:
        raise ValueError('exit_q must be [T, B]')
    if num_samples < 1:
        raise ValueError('num_samples must be >= 1')
    probs = exit_q.transpose(0, 1).clamp_min(1e-08)
    probs = probs / probs.sum(dim=1, keepdim=True)
    dist = Categorical(probs=probs)
    if num_samples == 1:
        tau = dist.sample().unsqueeze(0)
        logp = dist.log_prob(tau.squeeze(0)).unsqueeze(0)
    else:
        tau = dist.sample(sample_shape=torch.Size([num_samples]))
        logp = dist.log_prob(tau)
    return (tau, logp)

def compute_step_losses(logits_list, targets, mask):
    """Return per-step, per-example CE losses.

    logits_list: list length T of [B, L, V]
    targets: [B, L]
    mask: [B, L]
    Returns: tensor [T, B]
    """
    if not logits_list:
        raise ValueError('logits_list must not be empty')
    device = logits_list[0].device
    targets = targets.to(device)
    mask_flat = mask.to(device).view(targets.size(0), -1).float()
    batch_size = targets.size(0)
    safe_counts = mask_flat.sum(dim=1).clamp_min(1.0)
    target_flat = targets.view(-1)
    per_step_losses = []
    for logits in logits_list:
        vocab = logits.size(-1)
        logits_flat = logits.view(-1, vocab)
        ce = F.cross_entropy(logits_flat, target_flat, reduction='none')
        ce = ce.view(batch_size, -1)
        masked = (ce * mask_flat).sum(dim=1)
        per_example = masked / safe_counts
        per_step_losses.append(per_example)
    return torch.stack(per_step_losses, dim=0)

class MovingAverageBaseline:

    def __init__(self, decay=0.9, device=None):
        self.decay = float(decay)
        self.value = None
        self.device = device

    def update(self, rewards):
        mean_reward = rewards.mean().detach()
        if self.device is not None:
            mean_reward = mean_reward.to(self.device)
        if self.value is None:
            self.value = mean_reward
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * mean_reward
        return self.value

    def get(self):
        if self.value is None:
            return torch.tensor(0.0, device=self.device)
        return self.value
