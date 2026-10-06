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
from pufferlib.population_adversary import (
    GaussianPopulationPolicy,
    cosine_exploration_std,
    initial_population_mean,
    latent_to_beta,
    population_ppo_loss,
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


def test_disabled_population_adversary_preserves_uniform_rng_sequence():
    env = Drive.__new__(Drive)
    env.behavior_conditioning = True
    env.population_adversary = False
    env.behavior_conditioned_policy_id = 1
    env.behavior_condition_features = 6
    env.behavior_view_angle_max = 180.0
    env.num_agents = 4
    env.policy_log_ids = [0, 1, 0, 1]
    env.behavior_conditions = np.zeros((4, 6), dtype=np.float32)
    env._behavior_rng = np.random.default_rng(7)
    reference_rng = np.random.default_rng(7)
    expected = np.column_stack([
        reference_rng.uniform(0.0, 3.0, 2),
        reference_rng.uniform(0.0, 3.0, 2),
        reference_rng.uniform(0.0, 2.0, 2),
        reference_rng.uniform(0.0, 0.5, 2),
        reference_rng.uniform(20.0, 50.0, 2),
        reference_rng.uniform(0.0, 180.0, 2),
    ])
    env._sample_behavior_conditions()
    np.testing.assert_allclose(env.behavior_conditions[[1, 3]], expected)


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


def test_population_initialization_recovers_uniform_beta():
    latent = initial_population_mean(0.02, 0.5)
    alpha, beta, mu, rho = latent_to_beta(latent, 0.02, 0.02, 0.5)
    np.testing.assert_allclose(alpha, np.ones(6), atol=1e-6)
    np.testing.assert_allclose(beta, np.ones(6), atol=1e-6)
    np.testing.assert_allclose(mu, np.full(6, 0.5), atol=1e-6)
    np.testing.assert_allclose(rho, np.full(6, 1.0 / 3.0), atol=1e-6)


def test_population_latent_always_produces_valid_beta_parameters():
    latent = np.linspace(-100.0, 100.0, 36, dtype=np.float32).reshape(3, 12)
    alpha, beta, mu, rho = latent_to_beta(latent)
    assert np.all(alpha > 0.0)
    assert np.all(beta > 0.0)
    assert np.all((mu >= 0.02) & (mu <= 0.98))
    assert np.all((rho >= 0.02) & (rho <= 0.5))


def test_population_cosine_std_has_fixed_endpoints():
    assert cosine_exploration_std(0, 100, 0.5, 0.05) == 0.5
    assert np.isclose(cosine_exploration_std(100, 100, 0.5, 0.05), 0.05)
    assert np.isclose(cosine_exploration_std(200, 100, 0.5, 0.05), 0.05)


def test_population_ppo_moves_mean_toward_positive_advantage_action():
    policy = GaussianPopulationPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    old_mean = policy.mean.detach().repeat(2, 1)
    latent = old_mean.clone()
    latent[0] += 0.1
    latent[1] -= 0.1
    advantage = torch.tensor([1.0, -1.0])
    old_std = torch.full((2,), 0.5)
    before = policy.mean.detach().clone()
    loss, _ = population_ppo_loss(
        policy, latent, old_mean, old_std, advantage, clip_coef=0.2
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.all(policy.mean.detach() > before)


def test_population_window_reward_is_discounted_sum_not_time_average():
    env = Drive.__new__(Drive)
    env.population_adversary = True
    env.population_gamma = 0.5
    env.population_episode_decay = 0.8
    env.episode_length = 10
    env.num_agents = 3
    env.num_envs = 1
    env.agent_offsets = np.asarray([0, 3])
    env.policy_log_ids = [0, 0, 1]
    env.behavior_conditioned_policy_id = 1
    env._population_simple_mask = np.asarray([True, True, False])
    env._population_return = np.zeros(3, dtype=np.float32)
    env._population_discount = np.ones(3, dtype=np.float32)
    env._population_baseline = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    env._population_episode_start = np.asarray([10, 10, 10], dtype=np.int32)
    env._population_active = np.asarray([True, True, False])
    env._population_degradation = np.full(3, np.nan, dtype=np.float32)
    env._population_window_score = np.zeros(1, dtype=np.float32)
    env._population_episode_count = np.zeros(1, dtype=np.int32)
    env._population_bootstrap_coeff = np.zeros(3, dtype=np.float32)
    env.terminals = np.ones(3, dtype=np.float32)
    env.truncations = np.zeros(3, dtype=np.float32)
    env.rewards = np.asarray([0.5, 0.0, 100.0], dtype=np.float32)
    done = np.ones(3, dtype=bool)
    env._population_record_step(done, done)
    # mean(V-G)=mean(0.5, 1.0)=0.75, weighted by 0.8^(10/10).
    assert np.isclose(env._population_window_score[0], 0.6)
    assert env._population_episode_count[0] == 1


def test_population_first_episode_mode_ignores_later_cohorts_without_decay():
    env = Drive.__new__(Drive)
    env.population_adversary = True
    env.population_reward_mode = "first_episode"
    env.population_gamma = 0.5
    env.population_episode_decay = 0.8
    env.episode_length = 10
    env.num_agents = 3
    env.num_envs = 1
    env.agent_offsets = np.asarray([0, 3])
    env.policy_log_ids = [0, 0, 1]
    env.behavior_conditioned_policy_id = 1
    env._population_simple_mask = np.asarray([True, True, False])
    env._population_return = np.zeros(3, dtype=np.float32)
    env._population_discount = np.ones(3, dtype=np.float32)
    env._population_baseline = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    env._population_episode_start = np.asarray([10, 10, 10], dtype=np.int32)
    env._population_active = np.asarray([True, True, False])
    env._population_degradation = np.full(3, np.nan, dtype=np.float32)
    env._population_window_score = np.zeros(1, dtype=np.float32)
    env._population_episode_count = np.zeros(1, dtype=np.int32)
    env._population_bootstrap_coeff = np.zeros(3, dtype=np.float32)
    env.terminals = np.ones(3, dtype=np.float32)
    env.truncations = np.zeros(3, dtype=np.float32)
    done = np.ones(3, dtype=bool)

    env.rewards = np.asarray([0.5, 0.0, 100.0], dtype=np.float32)
    env._population_record_step(done, done)
    # First cohort: mean(V-G)=.75, with no legacy time decay.
    assert np.isclose(env._population_window_score[0], .75)
    assert env._population_episode_count[0] == 1

    env._population_active[:] = [True, True, False]
    env._population_return[:] = 0.
    env._population_discount[:] = 1.
    env._population_baseline[:] = [3., 3., 0.]
    env._population_episode_start[:] = 20
    env.rewards = np.asarray([0., 0., 0.], dtype=np.float32)
    env._population_record_step(done, done)
    assert np.isclose(env._population_window_score[0], .75)
    assert env._population_episode_count[0] == 1


def test_population_reward_mode_reaches_environment(monkeypatch):
    from pufferlib.pufferl import load_env
    import pufferlib.vector

    captured = {}
    sentinel = object()

    def capture(make_env, *, env_kwargs, **vec_kwargs):
        captured.update(env_kwargs)
        return sentinel

    monkeypatch.setattr(pufferlib.vector, "make", capture)
    args = {
        "package": "ocean", "env": {}, "vec": {},
        "train": {"population_adversary": True,
                  "population_reward_mode": "first_episode"},
    }
    assert load_env("puffer_drive", args) is sentinel
    assert captured["population_reward_mode"] == "first_episode"


def _population_test_trainer(chunk_size=64, target_kl=0.03):
    from pufferlib.pufferl import PuffeRL
    trainer = PuffeRL.__new__(PuffeRL)
    trainer.population_adversary = True
    trainer.population_monitor_dir = None
    trainer.population_update_count = 0
    trainer.config = dict(device='cpu', population_batch_size=256,
                          population_minibatch_size=chunk_size,
                          population_update_epochs=4, population_target_kl=target_kl)
    trainer.population_policy = GaussianPopulationPolicy()
    trainer.population_optimizer = torch.optim.Adam(trainer.population_policy.parameters(), lr=3e-4)
    trainer.vecenv = SimpleNamespace(set_population_policy=lambda mean, std: None)
    rng = np.random.default_rng(123)
    mean = trainer.population_policy.mean.detach().numpy().copy()
    latent = rng.normal(mean, 0.5, (513, 12))
    trainer.population_samples = [dict(latent=x.tolist(), old_mean=mean.tolist(),
                                      old_std=0.5, advantage=float(x[0]), episode_count=10)
                                  for x in latent]
    trainer.population_sample_count = len(latent)
    return trainer


def test_population_collection_consumed_and_chunk_invariant():
    a, b = _population_test_trainer(64), _population_test_trainer(513)
    metrics = a._train_population_adversary()
    b._train_population_adversary()
    assert not a.population_samples
    assert metrics['population/train_samples'] == 513
    assert metrics['population/optimizer_steps'] == 4
    assert metrics['population/behavior_kl'] == 0
    assert metrics['population/signal_corr_0'] > 0.99
    assert a.population_policy.mean[0].item() > 0
    assert torch.allclose(a.population_policy.mean, b.population_policy.mean, atol=1e-6)
    assert all(np.isfinite(v) for v in metrics.values())


def test_population_kl_guard_stops_remaining_epochs():
    trainer = _population_test_trainer(target_kl=1e-12)
    metrics = trainer._train_population_adversary()
    assert metrics['population/optimizer_steps'] == 1
    assert metrics['population/update_kl'] > 1e-12


def test_population_small_collection_waits_without_mutation():
    trainer = _population_test_trainer()
    trainer.population_samples = trainer.population_samples[:12]
    before = trainer.population_policy.mean.detach().clone()
    trainer._train_population_adversary()
    assert len(trainer.population_samples) == 12
    assert torch.equal(before, trainer.population_policy.mean)


def test_population_bootstrap_corrects_partial_return_once():
    env = Drive.__new__(Drive)
    env.population_adversary = True
    env.num_agents, env.num_envs = 3, 1
    env.agent_offsets = np.asarray([0, 3])
    env._population_agent_env = np.zeros(3, dtype=np.int32)
    env.population_gamma = 0.5
    env.population_episode_decay = 0.8
    env.episode_length = 10
    env.tick = 1
    env._population_simple_mask = np.asarray([True, True, False])
    env._population_active = np.asarray([True, True, False])
    env._population_baseline = np.asarray([1.0, 1.0, 0.0])
    env._population_return = np.zeros(3)
    env._population_discount = np.ones(3)
    env._population_episode_start = np.zeros(3, dtype=np.int32)
    env._population_degradation = np.full(3, np.nan)
    env._population_window_score = np.zeros(1)
    env._population_episode_count = np.zeros(1, dtype=np.int32)
    env._population_bootstrap_coeff = np.zeros(3)
    env._population_bootstrap_total = np.zeros(1)
    env.rewards = np.asarray([0.0, 1.0, 0.0])
    # Agent 0 is truncated with future value 2; agent 1 truly terminates.
    env.terminals = np.asarray([0, 1, 1])
    env.truncations = np.ones(3)
    env._population_record_step(np.ones(3, bool), np.ones(3, bool))
    assert np.isclose(env._population_window_score[0], 0.5)
    env._policy_values = np.asarray([2.0, 999.0, 999.0])
    env._population_apply_bootstrap()
    assert np.isclose(env._population_window_score[0], 0.0)
    assert np.isclose(env._population_bootstrap_total[0], 0.5)
    env._population_apply_bootstrap()
    assert np.isclose(env._population_window_score[0], 0.0)
