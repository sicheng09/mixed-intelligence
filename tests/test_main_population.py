"""Main-path sampler and phase-state regression tests (CPU, no training runs)."""
from types import SimpleNamespace
import numpy as np
import torch
from pufferlib.population_adversary import mixture_payload, publish_mixture, read_mixture, population_anchors
from pufferlib.population_runtime import MainPopulationRuntime


def test_payload_validation_and_version():
    payload = mixture_payload()
    before = payload.copy()
    try:
        publish_mixture(payload, [1, 1, 1, 1, 1], .35, 0)
        raise AssertionError('invalid payload accepted')
    except ValueError:
        pass
    assert np.array_equal(payload, before)
    publish_mixture(payload, [0, 0, 0, 1, 0], .1, 7)
    probabilities, std, version = read_mixture(payload)
    assert probabilities.tolist() == [0, 0, 0, 1, 0] and std == .1 and version == 7


def test_runtime_fixed_and_resume():
    published = []
    trainer = SimpleNamespace(config=dict(population_mode='fixed', seed=42),
        mix_ppo=True, mix_ppo_sampling='stratified', uncompiled_policies=[None,None],
        vecenv=SimpleNamespace(set_population_mixture=lambda *x:published.append(x)))
    a = MainPopulationRuntime(trainer, dict(num_maps=10))
    before = a.learner.policy.logits.detach().clone()
    for _ in range(40):a.after_driver_update()
    assert torch.equal(before, a.learner.policy.logits)
    assert a.diagnostics()["population/main_version"] == 0
    b = MainPopulationRuntime(trainer, dict(num_maps=10))
    b.load_state_dict(a.state_dict())
    assert b.driver_updates == 40
    assert b.last_metrics == a.last_metrics
    assert torch.equal(a.generator.get_state(),b.generator.get_state())
    assert np.allclose(b.probabilities(), .2)


def test_phase_metrics_expose_policy_motion_and_component_coverage():
    trainer = SimpleNamespace(config=dict(population_mode='fixed', seed=42),
        mix_ppo=True, mix_ppo_sampling='stratified', uncompiled_policies=[None, None],
        vecenv=SimpleNamespace(set_population_mixture=lambda *x: None))
    runtime = MainPopulationRuntime(trainer, dict(num_maps=10))
    records = [dict(challenge=float(i), bootstrapped_return=float(i + 1),
                    baseline=float(i + 2), bootstrap=.25)
               for i in range(5)]
    components = [dict(name=name, count=1, mean_challenge=float(i),
                       challenge_std=0., phases_without_sample=0)
                  for i, name in enumerate(runtime.learner.policy.names)]
    info = dict(update_kl=.01, max_context_kl=.015, epochs=3, backtracks=1,
                raw_challenge_variance=2., residual_variance=2.,
                component_diagnostics=components)
    metrics = runtime._phase_metrics(info, records, np.full(5, .2), 1.5)
    assert np.isclose(metrics['population/distribution_tv'], 0.)
    assert metrics['population/phase_min_component_count'] == 1
    assert metrics['population/phase_component_3_challenge'] == 3.
    assert np.isclose(metrics['population/bootstrap_mean'], .25)


def test_real_drive_window_publication():
    from pufferlib.ocean.drive.drive import Drive
    from pathlib import Path
    env = Drive(data_root=str(Path('data/binaries_full').resolve()), split='training',
        num_maps=10, num_agents=32, behavior_conditioning=True,
        population_sampling='mixture', policy_log_ids=[0,1],
        episode_length=91, resample_frequency=4, goal_behavior=3,
        collision_behavior=2, offroad_behavior=2)
    try:
        env.reset(42)
        latent = env._population_latent.copy()
        env.set_population_mixture([0,0,0,1,0], .1,7)
        for _ in range(4):
            env.step(np.zeros_like(env.actions))
            assert np.array_equal(latent,env._population_latent)
        env.step(np.zeros_like(env.actions))
        assert env._population_version == 7
        assert (env._population_component == 3).all()
        assert np.allclose(env._population_old_mean,population_anchors()[3])
        assert not env.population_adversary
    finally:env.close()


def test_rng_isolation():
    from pufferlib.population_runtime import isolated_cpu_rng
    state = torch.get_rng_state().clone()
    np_state = np.random.get_state()
    with isolated_cpu_rng():
        torch.random.default_generator.manual_seed(11)
        np.random.seed(12)
        torch.randn(32)
    assert torch.equal(state,torch.get_rng_state())
    assert np.array_equal(np_state[1],np.random.get_state()[1])


def test_real_vector_publication():
    import pufferlib.vector
    from pufferlib.ocean.drive.drive import Drive
    from pathlib import Path
    kwargs=dict(data_root=str(Path('data/binaries_full').resolve()),split='training',
        num_maps=10,num_agents=16,behavior_conditioning=True,population_sampling='mixture',
        policy_log_ids=[0,1],episode_length=91,resample_frequency=4,
        goal_behavior=3,collision_behavior=2,offroad_behavior=2)
    for backend, envs, workers in [('Serial',2,2),('Multiprocessing',2,2),('Multiprocessing',4,2)]:
        vec = pufferlib.vector.make(Drive,env_kwargs=kwargs,backend=backend,
            num_envs=envs,num_workers=workers,batch_size=envs)
        try:
            vec.set_population_mixture([0,0,0,1,0],.1,17)
            vec.async_reset(42)
            seen=[]
            for _ in range(7):
                obs,r,d,t,infos,ids,mask=vec.recv()
                for info in infos:
                    # Serial flattens nested dictionaries before returning infos.
                    data=info.get('population_mixture',info)
                    version=data.get('version',data.get('population_mixture/version'))
                    fraction=data.get('component_3_fraction',data.get('population_mixture/component_3_fraction'))
                    if version is not None:seen.append((version,fraction))
                vec.send(np.zeros((len(obs),1),dtype=np.int32))
            assert seen, (backend,envs,'no population window metadata')
            assert all(v==17 and p==1 for v,p in seen),seen
        finally:vec.close()
