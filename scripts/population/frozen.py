"""Auditable frozen-driver rollouts with pre-action scene context."""

import configparser
import hashlib
import inspect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.torch import Drive as SimplePolicy, DriveBehaviorConditioned, Recurrent
from pufferlib.population_adversary import initial_population_mean
from pufferlib.population_experiment import fingerprint, scene_context


def observation_digest(observations):
    return hashlib.sha256(np.ascontiguousarray(observations[:, :-6]).tobytes()).hexdigest()


class FrozenEvaluator:
    def __init__(self, config):
        self.config = config
        cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        if not cfg.read(config["env_config"]):
            raise FileNotFoundError(config["env_config"])
        self.env_options = {}
        for key, value in cfg["env"].items():
            if key not in inspect.signature(Drive).parameters:
                continue
            value = value.strip('"\'')
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            self.env_options[key] = value
        self.env_options.update(data_root=str(Path(config["data_root"]).resolve()),
            split=config["split"], num_maps=1, num_agents=128,
            policy_log_ids=[0, 1], policy_log_count=2, behavior_view_angle_max=360,
            population_adversary=True, behavior_conditioning=True)
        if (self.env_options.get("action_type") != "discrete"
                or self.env_options.get("mix_traffic", False)
                or self.env_options.get("idm_others", False)
                or self.env_options.get("reward_conditioning", False)
                or self.env_options.get("include_global_state", False)):
            raise ValueError("this experiment requires the audited discrete, two-policy Drive observation layout")
        if config["horizon"] >= self.env_options.get("resample_frequency", 910):
            raise ValueError("diagnostic horizon must precede map resampling")
        self.policy_options = {k: int(v) for k, v in cfg["policy"].items()}
        self.rnn_options = {k: int(v) for k, v in cfg["rnn"].items()}
        self.hidden_size = self.rnn_options["hidden_size"]
        self.envs = OrderedDict()
        self.policies = None
        self.initial_policy_hashes = None

    def close(self):
        for env in self.envs.values():
            env.close()
        self.envs.clear()

    def assert_frozen(self):
        if self.policies is not None:
            if [fingerprint(p) for p in self.policies] != self.initial_policy_hashes:
                raise RuntimeError("driver parameters changed during frozen experiment")
            if any(p.requires_grad for policy in self.policies for p in policy.parameters()):
                raise RuntimeError("driver gradients were enabled")

    def _env(self, map_id):
        if map_id not in self.envs:
            if len(self.envs) >= self.config.get("env_cache_size", 16):
                _, old_env = self.envs.popitem(last=False)
                old_env.close()
            self.envs[map_id] = Drive(**self.env_options, map_id=map_id, seed=42)
        self.envs.move_to_end(map_id)
        env = self.envs[map_id]
        if env.num_envs != 1 or list(env.map_ids) != [map_id]:
            raise RuntimeError("requested fixed map differs from binding map")
        if env.num_agents < 2:
            raise ValueError(f"map {map_id} cannot supply both driving policies")
        if self.policies is None:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(42)
                self.policies = [Recurrent(env, cls(env, **self.policy_options), **self.rnn_options)
                                 for cls in (SimplePolicy, DriveBehaviorConditioned)]
            for policy, checkpoint in zip(self.policies, self.config["checkpoints"]):
                policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                policy.eval().requires_grad_(False)
            self.initial_policy_hashes = [fingerprint(p) for p in self.policies]
        return env

    def _state(self, count):
        return dict(lstm_h=torch.zeros(count, self.hidden_size),
                    lstm_c=torch.zeros(count, self.hidden_size))

    @torch.no_grad()
    def prepare(self, map_id, seed):
        """Neutral reset and value/context snapshot; q has not yet sampled an action."""
        env = self._env(map_id)
        env.set_population_policy(initial_population_mean(), 0.0)
        obs, _ = env.reset(seed=seed)
        simple = np.arange(0, env.num_agents, 2)
        _, value = self.policies[0].forward_eval(torch.as_tensor(obs[simple]), self._state(len(simple)))
        value = value.flatten().numpy()
        context = scene_context(obs, simple, value, env.ego_features,
                                env.max_partner_objects, env.partner_features)
        return dict(context=context.tolist(), initial_digest=observation_digest(obs),
                    pre_action_baseline=float(value.mean()))

    @torch.no_grad()
    def run(self, map_id, seed, latent, prepared):
        env = self._env(map_id)
        torch.random.default_generator.manual_seed(seed)
        env.set_population_policy(latent, 0.0)
        obs, _ = env.reset(seed=seed)
        if observation_digest(obs) != prepared["initial_digest"]:
            raise RuntimeError("descriptor changed the pre-action physical observation")
        ids = [np.arange(i, env.num_agents, 2) for i in (0, 1)]
        states = [self._state(len(idx)) for idx in ids]
        alive = np.ones(env.num_agents, dtype=bool)
        ret, bootstrap = np.zeros(env.num_agents), np.zeros(env.num_agents)
        discount = np.ones(env.num_agents)
        terminal_count = np.zeros(env.num_agents, dtype=bool)
        # Simple observation feature 5 is a collision indicator, not a collision subtype.
        collided = np.zeros(env.num_agents, dtype=bool)
        baseline = None

        def bootstrap_values(mask):
            for i, policy in enumerate(self.policies):
                idx = ids[i]
                if not mask[idx].any():
                    continue
                state = {k: v.clone() for k, v in states[i].items()}
                _, value = policy.forward_eval(torch.as_tensor(obs[idx]), state)
                selected = mask[idx]
                bootstrap[idx[selected]] = discount[idx[selected]] * value.flatten().numpy()[selected]

        for step in range(self.config["horizon"]):
            actions = np.zeros((env.num_agents, 1), np.int32)
            values = np.zeros(env.num_agents)
            for i, policy in enumerate(self.policies):
                idx = ids[i]
                logits, value = policy.forward_eval(torch.as_tensor(obs[idx]), states[i])
                values[idx] = value.flatten().numpy()
                actions[idx, 0] = torch.distributions.Categorical(logits=logits[0]).sample().numpy()
            if baseline is None:
                baseline = values.copy()
                simple = ids[0]
                actual_context = scene_context(obs, simple, baseline[simple], env.ego_features,
                                               env.max_partner_objects, env.partner_features)
                if (abs(float(baseline[simple].mean()) - prepared["pre_action_baseline"]) > 1e-7
                        or not np.allclose(actual_context, prepared["context"], atol=1e-7, rtol=0)):
                    raise RuntimeError("descriptor changed baseline or scene context")
            obs, reward, terminal, truncated, _ = env.step(actions)
            ret[alive] += discount[alive] * np.clip(reward[alive], -1, 1)
            discount[alive] *= self.config["gamma"]
            collided |= alive & (obs[:, 5] > 0)
            finished = alive & ((terminal > 0) | (truncated > 0))
            terminal_count[finished] = terminal[finished] > 0
            bootstrap_values(finished & (terminal == 0))
            alive[finished] = False
            if not alive[ids[0]].any():
                break
        bootstrap_values(alive)
        simple = ids[0]
        return dict(map=map_id, seed=seed, n_simple=len(simple), steps=step + 1,
            baseline=float(baseline[simple].mean()), ret=float(ret[simple].mean()),
            bootstrap=float(bootstrap[simple].mean()),
            bootstrapped_return=float((ret[simple] + bootstrap[simple]).mean()),
            challenge=float((baseline[simple] - ret[simple] - bootstrap[simple]).mean()),
            horizon_survivors=int(alive[simple].sum()), terminal_count=int(terminal_count[simple].sum()),
            collision_indicator_rate=float(collided[simple].mean()), **prepared)
