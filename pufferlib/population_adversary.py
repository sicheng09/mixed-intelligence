"""Stateless population policy for behavior-conditioned self-play."""

import math

import numpy as np
import torch


LATENT_DIM = 12
CONDITION_DIM = 6


def _logit(value):
    return math.log(value / (1.0 - value))


def initial_population_mean(rho_min=0.02, rho_max=0.5):
    """Return the latent mean whose Beta distributions are uniform."""
    target_rho = 1.0 / 3.0
    if not rho_min < target_rho < rho_max:
        raise ValueError("rho_min < 1/3 < rho_max is required for uniform initialization")
    relative = (target_rho - rho_min) / (rho_max - rho_min)
    return np.asarray([0.0] * CONDITION_DIM + [_logit(relative)] * CONDITION_DIM,
                      dtype=np.float32)


def latent_to_beta(latent, epsilon=0.02, rho_min=0.02, rho_max=0.5):
    """Map unconstrained 12-D latent actions to six valid Beta distributions."""
    latent = np.asarray(latent, dtype=np.float64)
    if latent.shape[-1] != LATENT_DIM:
        raise ValueError(f"latent last dimension must be {LATENT_DIM}, got {latent.shape}")
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
    mu = epsilon + (1.0 - 2.0 * epsilon) * sigmoid(latent[..., :CONDITION_DIM])
    rho = rho_min + (rho_max - rho_min) * sigmoid(latent[..., CONDITION_DIM:])
    kappa = 1.0 / rho - 1.0
    alpha = mu * kappa
    beta = (1.0 - mu) * kappa
    return alpha.astype(np.float32), beta.astype(np.float32), mu.astype(np.float32), rho.astype(np.float32)


def cosine_exploration_std(sample_count, decay_samples, start=0.5, end=0.05):
    """Cosine exploration schedule indexed by completed map-window samples."""
    if decay_samples <= 0:
        return float(end)
    progress = min(max(float(sample_count) / float(decay_samples), 0.0), 1.0)
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress)))


class GaussianPopulationPolicy(torch.nn.Module):
    """A context-free Gaussian policy over 12-D latent Beta parameters."""

    def __init__(self, rho_min=0.02, rho_max=0.5):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.from_numpy(
            initial_population_mean(rho_min, rho_max)
        ))

    def distribution(self, std):
        std = torch.as_tensor(std, dtype=self.mean.dtype, device=self.mean.device)
        return torch.distributions.Normal(self.mean, std)


def population_ppo_loss(policy, latent, old_mean, old_std, advantage, clip_coef=0.2):
    """Clipped PPO loss for externally scheduled, fixed-covariance Gaussians."""
    new_logprob = torch.distributions.Normal(policy.mean, old_std[:, None]).log_prob(latent).sum(-1)
    with torch.no_grad():
        old_logprob = torch.distributions.Normal(old_mean, old_std[:, None]).log_prob(latent).sum(-1)
    ratio = (new_logprob - old_logprob).exp()
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantage
    return -torch.minimum(unclipped, clipped).mean(), ratio


def population_anchors():
    """Canonical NumPy anchors shared by worker sampling and the learner."""
    anchors = np.tile(initial_population_mean(), (5, 1))
    anchors[1, 2] = -4
    anchors[2, 5] = -4
    anchors[3:, [2, 4, 5]] = -4
    anchors[4, 6:] = -4
    return anchors


def mixture_payload(std=0.35):
    # sequence, five probabilities, descriptor std, policy version
    return np.asarray([0, .2, .2, .2, .2, .2, std, 0], dtype=np.float64)


def publish_mixture(target, probabilities, std, version):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (probabilities.shape != (5,) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1)
            or not np.isfinite(std) or std <= 0 or version < 0):
        raise ValueError("invalid population mixture payload")
    # Single writer; readers retry if publication overlaps a window reset.
    sequence = int(target[0])
    target[0] = sequence + 1
    target[1:6] = probabilities / probabilities.sum()
    target[6:8] = (std, version)
    target[0] = sequence + 2


def read_mixture(source):
    while True:
        sequence = source[0]
        if int(sequence) % 2:
            continue
        snapshot = source.copy()
        if sequence == source[0]:
            return snapshot[1:6], float(snapshot[6]), int(snapshot[7])


class AnchoredPopulationPolicy(torch.nn.Module):
    """Learn categorical choices; keep Gaussian descriptor anchors fixed.

    The recorded joint action is (component, latent). With fixed anchors and
    sigma during collection/update, its PPO ratio is the categorical ratio.
    """

    names = ("uniform", "low_goal", "narrow_view", "combined", "combined_concentrated")

    def __init__(self):
        super().__init__()
        anchors = population_anchors()
        self.register_buffer("anchors", torch.from_numpy(anchors))
        self.logits = torch.nn.Parameter(torch.zeros(5))

    def distribution(self):
        return torch.distributions.Categorical(logits=self.logits)

    @torch.no_grad()
    def sample(self, count, std, generator=None):
        if not math.isfinite(std) or std <= 0:
            raise ValueError("descriptor std must be finite and positive")
        component = torch.multinomial(self.logits.softmax(-1), count,
                                      replacement=True, generator=generator)
        noise = torch.randn((count, LATENT_DIM), generator=generator,
                            device=self.logits.device)
        latent = self.anchors[component] + std * noise
        return component, latent, self.distribution().log_prob(component)


def anchored_population_ppo_loss(policy, component, old_logprob, advantage, clip_coef=0.2):
    ratio = (policy.distribution().log_prob(component) - old_logprob).exp()
    loss = -torch.minimum(ratio * advantage,
                          ratio.clamp(1 - clip_coef, 1 + clip_coef) * advantage).mean()
    return loss, ratio
