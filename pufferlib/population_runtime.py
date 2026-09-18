"""Optional main-MixPPO population phases; driver rollouts remain in PuffeRL."""
import copy
from contextlib import contextmanager
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from pufferlib.population_experiment import FrozenPhaseLearner, fingerprint
from scripts.population.frozen import FrozenEvaluator


@contextmanager
def isolated_cpu_rng():
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        np.random.set_state(numpy_state)


class LiveSnapshotEvaluator(FrozenEvaluator):
    """Reuse the audited first-episode evaluator with current driver snapshots."""

    def __init__(self, env_options, gamma):
        self.env_options = dict(env_options)
        self.env_options.update(population_sampling="legacy", population_adversary=True,
            policy_log_ids=[0, 1], policy_log_count=2, num_agents=128,
            resample_frequency=max(910, env_options.get("resample_frequency", 910)))
        if self.env_options.get("split", "training") != "training":
            raise ValueError("main-training q phases must use the training split")
        self.config = dict(gamma=gamma, horizon=int(env_options.get("episode_length", 91))-1,
                           env_cache_size=4)
        if self.config["horizon"] < 1:
            raise ValueError("population phase requires an episode horizon")
        self.envs = OrderedDict()
        self.policies = None
        self.initial_policy_hashes = None

    def snapshot_drivers(self, policies):
        self.policies = [copy.deepcopy(p).cpu().eval().requires_grad_(False) for p in policies]
        self.hidden_size = self.policies[0].hidden_size
        self.initial_policy_hashes = [fingerprint(p) for p in self.policies]


class MainPopulationRuntime:
    """First integration: fixed mixture or global adaptive probabilities.

    Each q phase is blocking with respect to driver optimization. It snapshots
    the current two drivers, collects fresh first episodes, updates only q, then
    publishes probabilities for future map windows. No q data enters driver PPO.
    """

    def __init__(self, trainer, env_options):
        self.trainer = trainer
        c = trainer.config
        self.mode = c["population_mode"]
        if self.mode not in ("fixed", "adaptive"):
            raise ValueError("invalid main population mode")
        if (not trainer.mix_ppo or trainer.mix_ppo_sampling != "stratified"
                or len(trainer.uncompiled_policies) != 2 or torch.distributed.is_initialized()):
            raise ValueError("main population phases require single-rank stratified two-policy MixPPO")
        if env_options.get("mix_traffic", False) or env_options.get("idm_others", False):
            raise ValueError("main population phases currently require two-policy-only traffic")
        self.std = float(c.get("population_descriptor_std", .35))
        self.interval = int(c.get("population_phase_interval", 16))
        self.batch = int(c.get("population_phase_samples", 320))
        if self.std <= 0 or self.interval < 1 or self.batch < 2:
            raise ValueError("invalid population phase configuration")
        self.learner = FrozenPhaseLearner(variant="floor", seed=int(c["seed"]),
            lr=float(c.get("population_phase_lr", .05)), epochs=int(c.get("population_phase_epochs", 8)),
            target_kl=float(c.get("population_phase_kl", .02)),
            probability_floor=float(c.get("population_probability_floor", .05)))
        self.learner_spec = dict(epochs=self.learner.epochs, target_kl=self.learner.target_kl,
            floor=float(self.learner.policy.probability_floor),
            lr=float(c.get("population_phase_lr", .05)))
        self.generator = torch.Generator().manual_seed(int(c["seed"]) + 711)
        self.scene_rng = np.random.default_rng(int(c["seed"]) + 712)
        self.evaluator = LiveSnapshotEvaluator(env_options, float(c["gamma"])) if self.mode == "adaptive" else None
        self.map_count = int(env_options["num_maps"])
        self.driver_updates = 0
        self.samples = self.env_steps = self.agent_steps = 0
        self.last_metrics = {}
        self.publish()

    def probabilities(self):
        with torch.no_grad():
            return self.learner.policy.distribution(torch.zeros(1, 6)).probs[0].numpy()

    def publish(self):
        self.trainer.vecenv.set_population_mixture(self.probabilities(), self.std, self.learner.version)

    def diagnostics(self):
        return {**self.last_metrics, "population/main_version": self.learner.version,
            "population/q_samples": self.samples, "population/q_env_steps": self.env_steps,
            "population/q_agent_steps": self.agent_steps,
            **{f"population/component_{i}_prob": float(p) for i,p in enumerate(self.probabilities())}}

    def _phase_metrics(self, info, records, behavior_probabilities, phase_seconds):
        """Flatten phase evidence for the trainer dashboard and checkpoint audit.

        The q phase JSON keeps every record, but a training dashboard must make
        the distinction between an actual categorical-policy update and noisy
        challenge values visible without post-processing a large JSONL file.
        These are diagnostics only: they neither alter q's objective nor the
        driving rollout budget.
        """
        current_probabilities = self.probabilities()
        behavior_probabilities = np.asarray(behavior_probabilities, dtype=np.float64)
        component_info = info["component_diagnostics"]
        counts = np.asarray([entry["count"] for entry in component_info], dtype=np.int64)
        challenges = np.asarray([r["challenge"] for r in records], dtype=np.float64)
        returns = np.asarray([r["bootstrapped_return"] for r in records], dtype=np.float64)
        baselines = np.asarray([r["baseline"] for r in records], dtype=np.float64)
        bootstrap = np.asarray([r["bootstrap"] for r in records], dtype=np.float64)
        metrics = {
            "population/phase_seconds": float(phase_seconds),
            "population/challenge_mean": float(challenges.mean()),
            "population/return_mean": float(returns.mean()),
            "population/baseline_mean": float(baselines.mean()),
            "population/bootstrap_mean": float(bootstrap.mean()),
            "population/update_kl": float(info["update_kl"]),
            "population/max_context_kl": float(info["max_context_kl"]),
            "population/behavior_entropy": float(-(behavior_probabilities * np.log(behavior_probabilities)).sum()),
            "population/policy_entropy": float(-(current_probabilities * np.log(current_probabilities)).sum()),
            "population/distribution_tv": float(.5 * np.abs(current_probabilities - behavior_probabilities).sum()),
            "population/max_probability_change": float(np.abs(current_probabilities - behavior_probabilities).max()),
            "population/phase_min_component_count": int(counts.min()),
            "population/phase_accepted_epochs": int(info["epochs"]),
            "population/phase_backtracks": int(info["backtracks"]),
            "population/raw_challenge_variance": float(info["raw_challenge_variance"]),
            "population/residual_variance": float(info["residual_variance"]),
        }
        for index, entry in enumerate(component_info):
            metrics[f"population/phase_component_{index}_count"] = int(entry["count"])
            metrics[f"population/phase_component_{index}_fraction"] = float(entry["count"] / len(records))
            # A zero-count component has no empirical challenge estimate. Do
            # not publish NaN into standard metric backends.
            if entry["mean_challenge"] is not None:
                metrics[f"population/phase_component_{index}_challenge"] = float(entry["mean_challenge"])
        return metrics

    def after_driver_update(self):
        self.driver_updates += 1
        if self.mode == "fixed" or self.driver_updates % self.interval:
            return self.diagnostics()
        started = time.monotonic()
        evaluator = self.evaluator
        source_hashes = [fingerprint(p) for p in self.trainer.uncompiled_policies]
        evaluator.snapshot_drivers(self.trainer.uncompiled_policies)
        snapshot = self.learner.snapshot()
        behavior_probabilities = self.probabilities().copy()
        records = []
        # Isolate CPU action RNG used by the frozen evaluator from main training.
        with isolated_cpu_rng():
            for attempt in range(self.batch * 20):
                map_id = int(self.scene_rng.integers(self.map_count))
                seed = int(self.scene_rng.integers(1, 2**31-1))
                try:
                    prepared = evaluator.prepare(map_id, seed)
                except ValueError as exc:
                    if "cannot supply both driving policies" not in str(exc):
                        raise
                    continue
                context = torch.tensor([prepared["context"]], dtype=torch.float32)
                with torch.no_grad():
                    k, latent, lp, probabilities = self.learner.policy.sample(context, self.std, self.generator)
                record = evaluator.run(map_id, seed, latent[0].numpy(), prepared)
                record.update(snapshot, component=int(k[0]), latent=latent[0].tolist(),
                    old_logprob=float(lp[0]), old_probs=probabilities[0].tolist(),
                    std=self.std, scene_baseline=0.)
                record["n_agents"] = evaluator._env(map_id).num_agents
                records.append(record)
                if len(records) == self.batch:
                    break
            if len(records) != self.batch:
                raise RuntimeError("not enough eligible training scenes for population phase")
            evaluator.assert_frozen()
            info = self.learner.update(records, self.std)
        if source_hashes != [fingerprint(p) for p in self.trainer.uncompiled_policies]:
            raise RuntimeError("population phase changed a driving policy")
        self.samples += len(records)
        self.env_steps += sum(r["steps"] for r in records)
        self.agent_steps += sum(r["steps"] * r["n_agents"] for r in records)
        self.last_metrics = self._phase_metrics(
            info, records, behavior_probabilities, time.monotonic() - started)
        self.publish()
        output = Path(self.trainer.config["data_dir"]) / (
            f"{self.trainer.config['env']}_{self.trainer.logger.run_id}")
        output.mkdir(parents=True, exist_ok=True)
        with (output / "population_phases.jsonl").open("a") as f:
            f.write(json.dumps(dict(driver_updates=self.driver_updates, driver_hashes=source_hashes,
                behavior_probabilities=behavior_probabilities.tolist(), info=info,
                records=records, totals=self.diagnostics())) + "\n")
        print(f"[population phase] driver_updates={self.driver_updates} version={self.learner.version} "
              f"probabilities={self.probabilities().round(4).tolist()} KL={info['update_kl']:.5f} "
              f"TV={self.last_metrics['population/distribution_tv']:.5f} "
              f"counts={[entry['count'] for entry in info['component_diagnostics']]}", flush=True)
        return self.diagnostics()

    def state_dict(self):
        return dict(mode=self.mode, std=self.std, interval=self.interval, batch=self.batch, learner_spec=self.learner_spec,
            learner=self.learner.state_dict(), generator=self.generator.get_state(),
            scene_rng=self.scene_rng.bit_generator.state, driver_updates=self.driver_updates,
            samples=self.samples, env_steps=self.env_steps, agent_steps=self.agent_steps,
            last_metrics=self.last_metrics)

    def load_state_dict(self, state):
        for key in ("mode", "std", "interval", "batch", "learner_spec"):
            if state[key] != getattr(self, key):
                raise ValueError(f"population checkpoint {key} mismatch")
        self.learner.load_state_dict(state["learner"])
        self.generator.set_state(state["generator"])
        self.scene_rng.bit_generator.state = state["scene_rng"]
        for key in ("driver_updates", "samples", "env_steps", "agent_steps"):
            setattr(self, key, state[key])
        self.last_metrics = state.get("last_metrics", {})
        self.publish()

    def close(self):
        if self.evaluator is not None:
            self.evaluator.close()
