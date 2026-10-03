"""Small, inspectable objectives shared by post-training and math tests."""
import torch
from torch.nn import functional as F


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(margin), margin


def clipped_policy_loss(logp, old_logp, advantage, clip):
    ratio = (logp - old_logp).exp()
    loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    return loss, ratio


def clipped_value_loss(values, old_values, returns, clip):
    clipped = old_values + (values - old_values).clamp(-clip, clip)
    return .5 * torch.maximum((values - returns).square(), (clipped - returns).square())


def gae(rewards, values, gamma=.99, lam=.95):
    """One unpadded episode; EOS and response cap both terminate this text task.

    No bootstrap beyond the returned answer. Call separately per response,
    rather than propagating advantages across padding or episode boundaries.
    """
    advantages = torch.zeros_like(rewards)
    carry = rewards.new_zeros(())
    for t in range(len(rewards) - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < len(values) else values.new_zeros(())
        delta = rewards[t] + gamma * next_value - values[t]
        carry = delta + gamma * lam * carry
        advantages[t] = carry
    return advantages, advantages + values


def group_advantages(rewards, epsilon=1e-6):
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + epsilon)


def reference_kl(logp, ref_logp):
    """Nonnegative k3 sample estimator used by the GRPO objective."""
    delta = ref_logp - logp
    return delta.exp() - delta - 1
