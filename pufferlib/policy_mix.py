"""Policy identity helpers for mixed PPO training."""

import torch


POLICY_STAT_KEYS = (
    "n", "score", "offroad_rate", "collision_rate", "episode_length",
    "episode_return", "dnf_rate", "completion_rate", "lane_alignment_rate",
    "lane_aligned_steps", "lane_distance_count", "speed_limit_rate",
    "lane_distance_avg", "velocity_reward_total", "comfort_violations",
    "offroad_per_agent", "collisions_per_agent", "goals_sampled_this_episode",
    "goals_reached_this_episode", "speed_at_goal",
)

LOSS_DISPLAY_NAMES = {
    "policy_loss": "policy",
    "value_loss": "value",
    "old_approx_kl": "old_kl",
    "approx_kl": "kl",
    "importance": "ratio",
}


def parse_policy_mix(spec):
    """Parse ``name:fraction`` entries and normalize their fractions."""
    if spec is None or spec == "":
        return ["learner"], [1.0]

    names = []
    fractions = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            names.append(item)
            fractions.append(1.0)
            continue
        name, fraction = item.split(":", 1)
        names.append(name.strip())
        fractions.append(float(fraction.strip()))

    if not names:
        return ["learner"], [1.0]
    total = sum(fractions)
    if total <= 0:
        raise ValueError("Policy mix fractions must sum to a positive value")
    return names, [fraction / total for fraction in fractions]


def assign_policy_ids(total_agents, fractions, device=None):
    """Deficit-interleave stable policy IDs over flattened agent slots."""
    if total_agents <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    total = sum(fractions)
    if total <= 0:
        raise ValueError("Policy fractions must sum to a positive value")
    fractions = [float(fraction) / total for fraction in fractions]
    counts = [0] * len(fractions)
    policy_ids = []
    for total_assigned in range(1, total_agents + 1):
        deficits = [
            fraction * total_assigned - counts[policy_id]
            for policy_id, fraction in enumerate(fractions)
        ]
        policy_id = max(range(len(fractions)), key=lambda i: (deficits[i], -i))
        counts[policy_id] += 1
        policy_ids.append(policy_id)
    return torch.tensor(policy_ids, dtype=torch.long, device=device)


def group_policy_losses(losses):
    grouped = {}
    for key, value in losses.items():
        if not key.startswith("policy_") or "/" not in key:
            continue
        policy, metric = key.split("/", 1)
        try:
            policy_idx = int(policy.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        grouped.setdefault(policy_idx, {})[metric] = value
    return grouped


def flatten_policy_stats(policy_stats):
    return {
        f"mix_ppo/policy_{policy_idx}/{key}": value
        for policy_idx, stats in policy_stats.items()
        for key, value in stats.items()
    }
