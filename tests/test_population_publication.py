"""Real worker tests for deterministic population publication boundaries."""
import multiprocessing
import random

import gymnasium
import numpy as np
import pytest

import pufferlib
from pufferlib.population_adversary import mixture_payload, read_mixture
from pufferlib.vector import Multiprocessing, make


class SnapshotEnv(pufferlib.PufferEnv):
    def __init__(self, entered, release, stall_seed=-1, buf=None, seed=1):
        self.num_agents = 1
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(1)
        super().__init__(buf)
        self.entered, self.release, self.stall_seed = entered, release, stall_seed
        self.seed = seed
        self.constructor_draws = [random.random(), np.random.random()]
        self.mean = buf['population_policy_mean'] if buf else np.zeros(12)
        self.std = buf['population_policy_std'] if buf else np.ones(1)
        self.mixture = buf['population_mixture'] if buf else mixture_payload()

    def reset(self, seed=0):
        self.seed, self.steps = seed, 0
        return self._observe(), []

    def _observe(self):
        probabilities, std, version = read_mixture(self.mixture)
        self.observations[:] = [self.mean[0], self.std[0], probabilities[0],
                                std, version, *self.constructor_draws]
        return self.observations

    def step(self, actions):
        if self.seed == self.stall_seed and self.steps == 0:
            self.entered.set()
            if not self.release.wait(10):
                raise RuntimeError('test worker release timed out')
        self.steps += 1
        return self._observe(), self.rewards, self.terminals, self.truncations, []

    def close(self):
        pass


def make_vector(reproducible=True, seed=42, stall_seed=-1, num_envs=2,
                num_workers=2, batch_size=1, zero_copy=True):
    entered, release = multiprocessing.Event(), multiprocessing.Event()
    vector = make(SnapshotEnv, env_kwargs=dict(entered=entered, release=release,
        stall_seed=stall_seed), backend=Multiprocessing, num_envs=num_envs,
        num_workers=num_workers, batch_size=batch_size, zero_copy=zero_copy,
        seed=seed, reproducible=reproducible)
    return vector, entered, release


@pytest.mark.parametrize('reproducible', [True, False])
def test_in_flight_worker_population_obeys_selected_publication_mode(reproducible):
    vector, entered, release = make_vector(reproducible, stall_seed=43)
    actions = np.zeros(1, dtype=np.int32)
    try:
        vector.set_population_policy(np.ones(12), .5)
        vector.set_population_mixture([1,0,0,0,0], .3, 1)
        vector.async_reset(42)
        vector.recv(); vector.send(actions)
        vector.recv(); vector.send(actions)
        assert entered.wait(5), 'slow worker never entered its step'
        vector.recv()  # Fast worker returned; slow worker remains in flight.
        mean = np.full(12, 2., dtype=np.float32)
        probabilities = np.array([0.,1.,0.,0.,0.])
        vector.set_population_policy(mean, .2)
        vector.set_population_mixture(probabilities, .1, 2)
        mean[:] = 99  # Publication must own the caller's data.
        probabilities[:] = 0
        vector.send(actions)
        release.set()
        observation = vector.recv()[0].copy()
        expected = [1,.5,1,.3,1] if reproducible else [2,.2,0,.1,2]
        np.testing.assert_allclose(observation[0,:5], expected)
        vector.send(actions)
        vector.recv(); vector.send(actions)
        observation = vector.recv()[0].copy()
        np.testing.assert_allclose(observation[0,:5], [2,.2,0,.1,2])
    finally:
        release.set()
        vector.close()


def constructor_draws(seed):
    vector, _, release = make_vector(seed=seed)
    try:
        vector.async_reset(seed)
        a = vector.recv()[0][0,5:].copy()
        vector.send(np.zeros(1, dtype=np.int32))
        b = vector.recv()[0][0,5:].copy()
        return np.stack([a,b])
    finally:
        release.set()
        vector.close()


def test_worker_constructor_rngs_repeat_and_different_seeds_diverge():
    first = constructor_draws(42)
    np.testing.assert_array_equal(first, constructor_draws(42))
    assert not np.array_equal(first, constructor_draws(43))
    expected = np.array([[random.Random(seed).random(),
                         np.random.RandomState(seed).random_sample()]
                        for seed in [42,43]], dtype=np.float32)
    np.testing.assert_array_equal(first, expected)


@pytest.mark.parametrize('num_envs,num_workers,batch_size,zero_copy',
    [(2,2,2,True), (4,2,2,True), (4,2,4,True), (6,3,4,False)])
def test_command_and_reset_publications_cover_batch_layouts(
        num_envs,num_workers,batch_size,zero_copy):
    vector, _, release = make_vector(num_envs=num_envs,num_workers=num_workers,
                                     batch_size=batch_size,zero_copy=zero_copy)
    try:
        vector.set_population_policy(np.full(12,2), .2)
        vector.set_population_mixture([0,1,0,0,0], .1, 2)
        vector.async_reset(42)
        observation = vector.recv()[0].copy()
        np.testing.assert_allclose(observation[:,:5], np.tile([2,.2,0,.1,2],(batch_size,1)))
        vector.set_population_policy(np.full(12,3), .3)
        vector.set_population_mixture([1,0,0,0,0], .2, 3)
        np.testing.assert_array_equal(vector.buf['population_policy_mean'], 2)
        expected = np.full((num_workers,12),2.,dtype=np.float32)
        expected[vector.w_slice] = 3
        vector.send(np.zeros(batch_size,dtype=np.int32))
        np.testing.assert_array_equal(vector.buf['population_policy_mean'],expected)
        vector.async_reset(42)
        observation = vector.recv()[0].copy()
        np.testing.assert_allclose(observation[:,:5], np.tile([3,.3,1,.2,3],(batch_size,1)))
    finally:
        release.set()
        vector.close()
