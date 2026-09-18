"""Known-value trajectories distinguish terminal, truncation and horizon rewards."""

import unittest

import numpy as np
import torch

from pufferlib.population_adversary import initial_population_mean
from scripts.population.frozen import FrozenEvaluator


class ConstantCritic:
    def forward_eval(self, observations, state):
        return [torch.zeros(len(observations), 2)], torch.ones(len(observations), 1)


class OneStepScene:
    num_agents = 2
    ego_features = 7
    max_partner_objects = 2
    partner_features = 8

    def __init__(self, terminal=False, truncated=False, change_initial=False):
        self.terminal, self.truncated = terminal, truncated
        self.change_initial = change_initial

    def set_population_policy(self, latent, std):
        self.latent = latent

    def reset(self, seed):
        self.obs = np.zeros((2, 7 + 16 + 6), dtype=np.float32)
        if self.change_initial:
            self.obs[:, 0] = self.latent[0]
        return self.obs, []

    def step(self, actions):
        return (self.obs, np.zeros(2), np.full(2, self.terminal),
                np.full(2, self.truncated), [])


def evaluator_for(env):
    evaluator = FrozenEvaluator.__new__(FrozenEvaluator)
    evaluator.config = dict(horizon=1, gamma=1.)
    evaluator.hidden_size = 4
    evaluator.policies = [ConstantCritic(), ConstantCritic()]
    evaluator._env = lambda map_id: env
    return evaluator


class FrozenRolloutTests(unittest.TestCase):
    def test_perfect_continuation_value_cancels_truncation_and_horizon_challenge(self):
        for truncated in (True, False):
            with self.subTest(truncated=truncated):
                evaluator = evaluator_for(OneStepScene(truncated=truncated))
                prepared = evaluator.prepare(0, 1)
                result = evaluator.run(0, 1, initial_population_mean(), prepared)
                self.assertEqual(result["challenge"], 0.)
                self.assertEqual(result["bootstrap"], 1.)
                self.assertEqual(result["horizon_survivors"], int(not truncated))

    def test_true_terminal_never_bootstraps_even_when_also_truncated(self):
        evaluator = evaluator_for(OneStepScene(terminal=True, truncated=True))
        prepared = evaluator.prepare(0, 1)
        result = evaluator.run(0, 1, initial_population_mean(), prepared)
        self.assertEqual(result["challenge"], 1.)
        self.assertEqual(result["bootstrap"], 0.)

    def test_changed_initial_scene_is_rejected(self):
        evaluator = evaluator_for(OneStepScene(change_initial=True))
        prepared = evaluator.prepare(0, 1)
        latent = initial_population_mean()
        latent[0] = 1.
        with self.assertRaisesRegex(RuntimeError, "physical observation"):
            evaluator.run(0, 1, latent, prepared)


if __name__ == "__main__":
    unittest.main()
