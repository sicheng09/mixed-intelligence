"""Opt-in frozen-driver experiments; not imported by the production trainer."""

import copy
import hashlib
import math

import numpy as np
import torch
from torch import nn

from pufferlib.population_adversary import AnchoredPopulationPolicy


CONTEXT_NAMES = (
    "log_agent_count_over_log129", "simple_speed_mean_over_20mps",
    "simple_speed_std_over_20mps", "simple_neighbors_30m_over_slots",
    "simple_initial_value_mean", "simple_initial_value_std",
)
VARIANTS = {
    "global": dict(context_actor=False, scene_baseline=False, probability_floor=0.0),
    "floor": dict(context_actor=False, scene_baseline=False, probability_floor=0.05),
    "baseline": dict(context_actor=False, scene_baseline=True, probability_floor=0.05),
    "context": dict(context_actor=True, scene_baseline=True, probability_floor=0.05),
}


def scene_context(observations, simple_indices, values, ego_features, partner_slots,
                  partner_features=8):
    """Read the current Drive observation schema, before sampling a q action.

    Partners are capped visible neighbors, not an uncapped physical density.
    No condition coordinates, map IDs or outcome statistics are used.
    """
    obs = np.asarray(observations)[simple_indices]
    values = np.asarray(values)
    if len(obs) == 0 or len(values) != len(obs) or partner_features != 8 or partner_slots < 1:
        raise ValueError("unsupported or empty scene context")
    partners = obs[:, ego_features:ego_features + partner_slots * 8].reshape(-1, partner_slots, 8)
    valid = (partners[..., 2] > 0) & (partners[..., 3] > 0)
    near = valid & ((partners[..., :2] ** 2).sum(-1) <= 0.6 ** 2)
    speed = np.abs(obs[:, 2]) * 5.0  # C stores signed speed / 100 m/s.
    context = np.asarray([
        math.log1p(len(observations)) / math.log(129), speed.mean(), speed.std(),
        near.sum(-1).mean() / partner_slots, values.mean(), values.std(),
    ], dtype=np.float32)
    if not np.isfinite(context).all():
        raise ValueError("non-finite scene context")
    return context


def fingerprint(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _mlp(output_size, hidden_size):
    net = nn.Sequential(nn.Linear(len(CONTEXT_NAMES), hidden_size), nn.Tanh(),
                        nn.Linear(hidden_size, output_size))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class ScenePopulationPolicy(AnchoredPopulationPolicy):
    """Fixed-anchor joint policy with an optional context actor and exact floor."""

    def __init__(self, context_actor=False, probability_floor=0.0, hidden_size=32):
        super().__init__()
        if not math.isfinite(probability_floor) or not 0 <= probability_floor < 1 / len(self.names):
            raise ValueError("probability_floor must be in [0, 1 / components)")
        self.register_buffer("probability_floor", torch.tensor(float(probability_floor)))
        self.context_net = _mlp(len(self.names), hidden_size) if context_actor else None

    def distribution(self, context):
        if context.ndim != 2 or context.shape[1] != len(CONTEXT_NAMES):
            raise ValueError("context must have shape (batch, context_features)")
        logits = self.logits.expand(len(context), -1)
        if self.context_net is not None:
            logits = logits + self.context_net(context)
        floor = self.probability_floor
        probs = (1 - len(self.names) * floor) * logits.softmax(-1) + floor
        return torch.distributions.Categorical(probs=probs)

    @torch.no_grad()
    def sample(self, context, std, generator=None):
        if not math.isfinite(std) or std <= 0:
            raise ValueError("descriptor std must be finite and positive")
        distribution = self.distribution(context)
        component = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
        noise = torch.randn((len(context), 12), generator=generator, device=context.device)
        latent = self.anchors[component] + std * noise
        return component, latent, distribution.log_prob(component), distribution.probs


class SceneBaseline(nn.Module):
    """Predict raw challenge; publish its fitted parameters only at phase boundaries."""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.net = _mlp(1, hidden_size)

    def forward(self, context):
        return self.net(context).squeeze(-1)


def scene_ppo_loss(policy, context, component, old_logprob, advantage, clip_coef=0.2):
    ratio = (policy.distribution(context).log_prob(component) - old_logprob).exp()
    return -torch.minimum(ratio * advantage,
                          ratio.clamp(1 - clip_coef, 1 + clip_coef) * advantage).mean(), ratio


class FrozenPhaseLearner:
    """One fixed-version collection, one full-batch actor update, then baseline fit."""

    def __init__(self, variant="global", seed=917, lr=0.05, actor_lr=None,
                 epochs=8, target_kl=0.02, baseline_lr=1e-3, baseline_epochs=40,
                 probability_floor=None):
        options = dict(VARIANTS[variant])
        if probability_floor is not None:
            options["probability_floor"] = probability_floor
        if epochs < 1 or baseline_epochs < 1 or target_kl <= 0:
            raise ValueError("epochs and target KL must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.policy = ScenePopulationPolicy(options["context_actor"], options["probability_floor"])
            torch.random.default_generator.manual_seed(seed + 1)
            self.baseline = SceneBaseline() if options["scene_baseline"] else None
        groups = [{"params": [self.policy.logits], "lr": lr}]
        if self.policy.context_net is not None:
            groups.append({"params": self.policy.context_net.parameters(),
                           "lr": lr if actor_lr is None else actor_lr})
        self.optimizer = torch.optim.Adam(groups)
        self.baseline_optimizer = (torch.optim.Adam(self.baseline.parameters(), lr=baseline_lr)
                                   if self.baseline is not None else None)
        self.epochs, self.target_kl, self.baseline_epochs = epochs, target_kl, baseline_epochs
        self.version = 0
        self.last_seen = [-1] * len(self.policy.names)

    def snapshot(self):
        return dict(version=self.version, policy_fingerprint=fingerprint(self.policy),
                    baseline_fingerprint=fingerprint(self.baseline) if self.baseline is not None else None)

    @torch.no_grad()
    def predict_baseline(self, context):
        return self.baseline(context) if self.baseline is not None else torch.zeros(len(context))

    def update(self, records, std):
        if len(records) < 2:
            raise ValueError("a phase requires at least two samples")
        snapshot = self.snapshot()
        for record in records:
            if any(record[k] != v for k, v in snapshot.items()) or record["std"] != std:
                raise ValueError("mixed or stale policy/baseline version or descriptor std")
        context = torch.tensor(np.asarray([r["context"] for r in records]), dtype=torch.float32)
        component = torch.tensor([r["component"] for r in records])
        old_lp = torch.tensor([r["old_logprob"] for r in records])
        old_probs = torch.tensor([r["old_probs"] for r in records])
        reward = torch.tensor([r["challenge"] for r in records])
        predictions = torch.tensor([r["scene_baseline"] for r in records])
        if not all(torch.isfinite(x).all() for x in (context, old_lp, old_probs, reward, predictions)):
            raise ValueError("non-finite phase data")
        with torch.no_grad():
            current = self.policy.distribution(context)
            if (not torch.allclose(current.probs, old_probs, atol=1e-6, rtol=1e-6)
                    or not torch.allclose(current.log_prob(component), old_lp, atol=1e-6, rtol=1e-6)
                    or not torch.allclose(self.predict_baseline(context), predictions, atol=1e-6, rtol=1e-6)):
                raise ValueError("recorded behavior probability or pre-phase baseline differs")
        residual = reward - predictions
        # Same normalization for all ablations. Raw V-G is always retained.
        advantage = (residual - residual.mean()) / residual.std().clamp_min(1e-8)
        accepted_epochs, backtracks = 0, 0
        for epoch in range(self.epochs):
            loss, _ = scene_ppo_loss(self.policy, context, component, old_lp, advantage)
            self.optimizer.zero_grad()
            loss.backward()
            previous_policy = copy.deepcopy(self.policy.state_dict())
            previous_optimizer = copy.deepcopy(self.optimizer.state_dict())
            original_lrs = [group["lr"] for group in self.optimizer.param_groups]
            accepted = False
            # A context MLP can overshoot in a single step. Retry the same gradient
            # with smaller steps, restoring Adam moments too; all variants use this.
            for attempt in range(9):
                self.policy.load_state_dict(previous_policy)
                self.optimizer.load_state_dict(previous_optimizer)
                for group, lr in zip(self.optimizer.param_groups, original_lrs):
                    group["lr"] = lr * 0.5 ** attempt
                self.optimizer.step()
                with torch.no_grad():
                    new_probs = self.policy.distribution(context).probs
                    kl_by_context = (old_probs * (old_probs.log() - new_probs.log())).sum(-1)
                if torch.isfinite(kl_by_context).all() and float(kl_by_context.mean()) <= self.target_kl:
                    accepted = True
                    accepted_epochs += 1
                    backtracks += attempt
                    break
            if not accepted:
                self.policy.load_state_dict(previous_policy)
                self.optimizer.load_state_dict(previous_optimizer)
                backtracks += 9
                break
            for group, lr in zip(self.optimizer.param_groups, original_lrs):
                group["lr"] = lr
        with torch.no_grad():
            new_probs = self.policy.distribution(context).probs
            kl_by_context = (old_probs * (old_probs.log() - new_probs.log())).sum(-1)
        # Only now may the scene baseline see this phase's rewards.
        fit_mse = None
        if self.baseline is not None:
            for _ in range(self.baseline_epochs):
                fit_loss = (self.baseline(context) - reward).square().mean()
                self.baseline_optimizer.zero_grad()
                fit_loss.backward()
                self.baseline_optimizer.step()
            with torch.no_grad():
                fit_mse = float((self.baseline(context) - reward).square().mean())
        counts = torch.bincount(component, minlength=len(self.policy.names)).tolist()
        for k, count in enumerate(counts):
            if count:
                self.last_seen[k] = self.version
        components = []
        for k, name in enumerate(self.policy.names):
            selected = reward[component == k]
            components.append(dict(name=name, count=counts[k],
                mean_challenge=float(selected.mean()) if len(selected) else None,
                challenge_std=float(selected.std(unbiased=False)) if len(selected) else None,
                phases_without_sample=self.version - self.last_seen[k]))
        by_map = []
        for map_id in sorted({r["map"] for r in records}):
            group = [r for r in records if r["map"] == map_id]
            by_map.append(dict(map=map_id, count=len(group), **{
                key: float(np.mean([r[key] for r in group]))
                for key in ("baseline", "bootstrapped_return", "challenge", "scene_baseline")}))
        info = dict(version=self.version, samples=len(records), epochs=accepted_epochs, backtracks=backtracks,
            update_kl=float(kl_by_context.mean()), max_context_kl=float(kl_by_context.max()),
            entropy=float(-(new_probs * new_probs.log()).sum(-1).mean()),
            mean_probabilities=new_probs.mean(0).tolist(), min_probability=float(new_probs.min()),
            component_diagnostics=components, by_map=by_map,
            raw_challenge_variance=float(reward.var(unbiased=False)),
            residual_variance=float(residual.var(unbiased=False)),
            baseline_pre_mse=float(residual.square().mean()), baseline_post_fit_mse=fit_mse)
        self.version += 1
        return info

    def state_dict(self):
        return dict(policy=self.policy.state_dict(), optimizer=self.optimizer.state_dict(),
                    baseline=self.baseline.state_dict() if self.baseline is not None else None,
                    baseline_optimizer=self.baseline_optimizer.state_dict() if self.baseline is not None else None,
                    version=self.version, last_seen=self.last_seen)

    def load_state_dict(self, state):
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        if (self.baseline is None) != (state["baseline"] is None):
            raise ValueError("checkpoint baseline configuration differs")
        if self.baseline is not None:
            self.baseline.load_state_dict(state["baseline"])
            self.baseline_optimizer.load_state_dict(state["baseline_optimizer"])
        self.version, self.last_seen = state["version"], state["last_seen"]
