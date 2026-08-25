from types import SimpleNamespace

import gymnasium
import numpy as np
import torch

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.torch import DriveBehaviorConditioned
from pufferlib.policy_mix import assign_policy_ids, parse_policy_mix
from pufferlib.policy_sampling import (
    build_policy_segment_pools,
    parse_stratified_policy_specs,
    resolve_policy_update_steps,
)


def _mock_env():
    return SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7 + 3 * 8 + 4 * 7 + 6,), dtype=np.float32
        ),
        single_action_space=gymnasium.spaces.MultiDiscrete([91]),
        max_partner_objects=3,
        partner_features=8,
        max_road_objects=4,
        road_features=7,
        dynamics_model="classic",
        behavior_condition_features=6,
        behavior_view_angle_max=180.0,
    )


def test_agent_slot_policy_assignment_is_stable_and_balanced():
    names, fractions = parse_policy_mix("simple:0.5,conditioned:0.5")
    policy_ids = assign_policy_ids(10, fractions)
    assert names == ["simple", "conditioned"]
    assert policy_ids.tolist() == [0, 1] * 5
    assert torch.equal(policy_ids, policy_ids.clone())


def test_stratified_pools_keep_policy_segments_disjoint():
    ids = torch.tensor([0, 1, 0, 1])
    pools = build_policy_segment_pools(ids, 2)
    specs = parse_stratified_policy_specs(
        {
            "mix_ppo_policy_minibatch_sizes": "64,64",
            "mix_ppo_policy_update_steps": "2,2",
        },
        policy_count=2,
        horizon=32,
        trainable=[True, True],
    )
    assert pools[0].tolist() == [0, 2]
    assert pools[1].tolist() == [1, 3]
    assert resolve_policy_update_steps(
        specs, pools, horizon=32, update_epochs=1, trainable=[True, True]
    ) == [2, 2]


def test_behavior_policy_masks_partner_by_radius_and_view_angle():
    policy = DriveBehaviorConditioned(_mock_env(), input_size=8, hidden_size=16)
    partners = torch.zeros(1, 3, 8)
    partners[0, 0, :3] = torch.tensor([0.5, 0.0, 1.0])
    partners[0, 1, :3] = torch.tensor([0.0, 0.5, 1.0])
    partners[0, 2, :3] = torch.tensor([0.9, 0.0, 1.0])
    # radius=35m and view=180 degrees (z_view_angle=1 at the configured max).
    condition = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    masked = policy.apply_partner_mask(partners, condition)
    assert torch.equal(masked[0, 0], partners[0, 0])
    assert torch.equal(masked[0, 1], partners[0, 1])
    assert torch.equal(masked[0, 2], torch.zeros(8))


def test_behavior_condition_sampling_keeps_simple_baseline():
    env = Drive.__new__(Drive)
    env.behavior_conditioning = True
    env.behavior_conditioned_policy_id = 1
    env.behavior_condition_features = 6
    env.behavior_view_angle_max = 180.0
    env.num_agents = 4
    env.policy_log_ids = [0, 1, 0, 1]
    env.behavior_conditions = np.zeros((4, 6), dtype=np.float32)
    env._behavior_rng = np.random.default_rng(7)
    env._sample_behavior_conditions()
    expected = np.array([0.5, 0.5, 1.0, 0.25, 50.0, 180.0], dtype=np.float32)
    np.testing.assert_allclose(
        env.behavior_conditions[[0, 2]], np.repeat(expected[None, :], 2, axis=0)
    )
    assert np.all((env.behavior_conditions[[1, 3], 0] >= 0.0))
    assert np.all((env.behavior_conditions[[1, 3], 0] <= 3.0))
    assert np.all((env.behavior_conditions[[1, 3], 5] >= 0.0))
    assert np.all((env.behavior_conditions[[1, 3], 5] <= 180.0))


def test_behavior_view_angle_max_is_configurable():
    env = Drive.__new__(Drive)
    env.behavior_conditioning = True
    env.behavior_conditioned_policy_id = 1
    env.behavior_condition_features = 6
    env.behavior_view_angle_max = 90.0
    env.num_agents = 2
    env.policy_log_ids = [1, 1]
    env.behavior_conditions = np.zeros((2, 6), dtype=np.float32)
    env._behavior_rng = np.random.default_rng(11)
    env._sample_behavior_conditions()
    assert np.all(env.behavior_conditions[:, 5] >= 0.0)
    assert np.all(env.behavior_conditions[:, 5] <= 90.0)
