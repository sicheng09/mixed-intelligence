"""Invariants of the opt-in population experiment, independent of driver training."""

import copy
import unittest

import numpy as np
import torch

from pufferlib.population_adversary import AnchoredPopulationPolicy
from pufferlib.population_experiment import (
    CONTEXT_NAMES, FrozenPhaseLearner, ScenePopulationPolicy,
    fingerprint, scene_context, scene_ppo_loss,
)
from scripts.population.run import map_ids, phase_scenes, validate_config


def records_for(learner, generator, count=80, std=.35):
    context = torch.zeros(count, len(CONTEXT_NAMES))
    context[:, 0] = torch.arange(count) % 2
    k, latent, lp, probs = learner.policy.sample(context, std, generator)
    prediction = learner.predict_baseline(context)
    snapshot = learner.snapshot()
    records = []
    for i in range(count):
        # Both a scene-dependent offset and a component effect.
        reward = float(context[i, 0] + (k[i] == 3) * .1)
        records.append(dict(snapshot, map=int(context[i, 0]), context=context[i].tolist(),
            component=int(k[i]), latent=latent[i].tolist(), old_logprob=float(lp[i]),
            old_probs=probs[i].tolist(), std=std, scene_baseline=float(prediction[i]),
            challenge=reward, baseline=0., bootstrapped_return=-reward))
    return records


class PopulationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_floor_global_matches_legacy_distribution_and_state(self):
        old = AnchoredPopulationPolicy()
        new = ScenePopulationPolicy()
        self.assertEqual(set(old.state_dict()), {"anchors", "logits"})
        with torch.no_grad():
            old.logits.copy_(torch.arange(5.) / 3)
            new.logits.copy_(old.logits)
        context = torch.randn(4, len(CONTEXT_NAMES))
        torch.testing.assert_close(new.distribution(context).probs, old.distribution().probs.expand(4, -1))

    def test_floor_applies_to_sampling_and_joint_ratio(self):
        q = ScenePopulationPolicy(probability_floor=.05)
        with torch.no_grad():
            q.logits[0] = 12
        context = torch.zeros(2000, len(CONTEXT_NAMES))
        k, latent, lp, probs = q.sample(context, .35, torch.Generator().manual_seed(8))
        self.assertGreaterEqual(float(probs.min()), .05 - 1e-7)
        self.assertGreaterEqual(len(k.unique()), 5)
        empirical = torch.bincount(k, minlength=5) / len(k)
        self.assertLess(float((empirical - probs[0]).abs().max()), .04)
        normal_lp = torch.distributions.Normal(q.anchors[k], .35).log_prob(latent).sum(-1)
        old_joint = lp + normal_lp
        with torch.no_grad():
            q.logits[0] = 1
        new_joint = q.distribution(context).log_prob(k) + normal_lp
        _, ratio = scene_ppo_loss(q, context, k, lp, torch.ones(len(k)))
        torch.testing.assert_close(ratio, (new_joint - old_joint).exp(), atol=3e-6, rtol=2e-6)

    def test_context_is_condition_independent_and_ignores_padding(self):
        obs = np.zeros((4, 7 + 2 * 8 + 6), dtype=np.float32)
        obs[:, 2] = [.1, .2, .3, .4]
        obs[:, 7:11] = [.2, .1, .2, .2]
        first = scene_context(obs, [0, 2], [1., 3.], 7, 2)
        obs[:, -6:] = 999
        second = scene_context(obs, [0, 2], [1., 3.], 7, 2)
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(first[3]), .5)  # One real partner; padding is not a neighbor.
        self.assertAlmostEqual(float(first[4]), 2.)

    def test_context_actor_can_learn_different_choices(self):
        torch.manual_seed(3)
        q = ScenePopulationPolicy(context_actor=True, probability_floor=.05)
        x = torch.zeros(2, len(CONTEXT_NAMES))
        x[:, 0] = torch.tensor([-1., 1.])
        torch.testing.assert_close(q.distribution(x).probs, torch.full((2, 5), .2))
        optimizer = torch.optim.Adam(q.parameters(), lr=.03)
        anchors = q.anchors.clone()
        for _ in range(80):
            probabilities = q.distribution(x).probs
            loss = -(probabilities[0, 0] + probabilities[1, 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertEqual(q.distribution(x).probs.argmax(-1).tolist(), [0, 1])
        torch.testing.assert_close(q.anchors, anchors)

    def test_baseline_is_used_from_previous_phase_only(self):
        learner = FrozenPhaseLearner("baseline", baseline_epochs=80)
        generator = torch.Generator().manual_seed(4)
        batch = records_for(learner, generator)
        self.assertTrue(all(r["scene_baseline"] == 0 for r in batch))
        old_baseline = batch[0]["baseline_fingerprint"]
        summary = learner.update(batch, .35)
        self.assertTrue(all(r["scene_baseline"] == 0 for r in batch))
        self.assertNotEqual(fingerprint(learner.baseline), old_baseline)
        self.assertLess(summary["baseline_post_fit_mse"], summary["baseline_pre_mse"])
        next_batch = records_for(learner, generator)
        self.assertTrue(any(abs(r["scene_baseline"]) > .1 for r in next_batch))
        self.assertEqual(next_batch[0]["version"], 1)

    def test_stale_or_tampered_records_rejected_before_mutation(self):
        learner = FrozenPhaseLearner("baseline")
        batch = records_for(learner, torch.Generator().manual_seed(7))
        for key, value in [("version", 9), ("std", .2), ("policy_fingerprint", "stale"),
                           ("baseline_fingerprint", "stale"), ("old_logprob", 0.),
                           ("scene_baseline", 10.)]:
            changed = copy.deepcopy(batch)
            changed[0][key] = value
            before = learner.snapshot()
            with self.assertRaises(ValueError, msg=key):
                learner.update(changed, .35)
            self.assertEqual(learner.snapshot(), before)

    def test_large_context_step_respects_mean_behavior_kl_budget(self):
        learner = FrozenPhaseLearner("context", lr=1., target_kl=.02)
        batch = records_for(learner, torch.Generator().manual_seed(21))
        result = learner.update(batch, .35)
        self.assertLessEqual(result["update_kl"], .02000001)
        self.assertGreater(result["backtracks"], 0)
        self.assertTrue(all(group["lr"] == 1. for group in learner.optimizer.param_groups))

    def test_stage_resume_reproduces_next_sampling_and_update(self):
        original = FrozenPhaseLearner("context", seed=33)
        generator = torch.Generator().manual_seed(33)
        original.update(records_for(original, generator), .35)
        state, rng_state = copy.deepcopy(original.state_dict()), generator.get_state().clone()
        continued_batch = records_for(original, generator)
        continued_result = original.update(continued_batch, .35)
        restored = FrozenPhaseLearner("context", seed=33)
        restored.load_state_dict(state)
        restored_generator = torch.Generator()
        restored_generator.set_state(rng_state)
        resumed_batch = records_for(restored, restored_generator)
        self.assertEqual(continued_batch, resumed_batch)
        self.assertEqual(continued_result, restored.update(resumed_batch, .35))
        self.assertEqual(original.snapshot(), restored.snapshot())

    def test_map_design_disjoint_and_balanced(self):
        import json
        from pathlib import Path
        config = json.loads(Path("scripts/population/design.json").read_text())
        validate_config(config)
        scenes = phase_scenes(config, 917, 0)
        counts = {m: sum(a == m for a, _ in scenes) for m in map_ids(config["train_maps"])}
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(scenes, phase_scenes(config, 917, 0))
        config["confirmation_maps"] = "0:2"
        with self.assertRaises(ValueError):
            validate_config(config)
        with self.assertRaises(ValueError):
            map_ids("0:3,2")


if __name__ == "__main__":
    unittest.main()
