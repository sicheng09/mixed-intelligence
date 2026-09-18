"""Evaluate trained SimplePPO against one fixed ConditionedPPO checkpoint.

This is a development protocol, not q training. All models share conditions,
maps, action seeds and a fixed opponent. Reports physical first-episode events
in addition to (critic-dependent) bootstrapped returns.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.population.frozen import FrozenEvaluator
from pufferlib.population_adversary import initial_population_mean, population_anchors


class EventEvaluator(FrozenEvaluator):
    def run(self, map_id, seed, latent, prepared):
        env = self._env(map_id)
        simple = np.arange(0, env.num_agents, 2)
        alive = np.ones(len(simple), dtype=bool)
        events = np.zeros((len(simple), 3), dtype=bool)
        original = env.step
        def step(actions):
            result = original(actions)
            raw = env.reward_components_raw[simple, :3]
            events[alive] |= raw[alive] > 0
            alive[:] &= ~((result[2][simple] > 0) | (result[3][simple] > 0))
            return result
        env.step = step
        try:
            result = super().run(map_id, seed, latent, prepared)
        finally:
            env.step = original
        result.update(collision_rate=float(events[:, 0].mean()),
            offroad_rate=float(events[:, 1].mean()), completion_rate=float(events[:, 2].mean()))
        return result


def profiles():
    anchors = population_anchors()
    result = {"uniform": initial_population_mean(), "combined": anchors[3],
              "combined_concentrated": anchors[4]}
    unseen = initial_population_mean()
    unseen[2], unseen[5] = 2., -2.
    result['high_goal_narrow_view'] = unseen
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='root containing uniform/fixed/adaptive run directories')
    parser.add_argument('--output', required=True)
    parser.add_argument('--maps', type=int, default=20)
    parser.add_argument('--start-map', type=int, default=20)
    parser.add_argument('--seeds', default='1901,2903,3907')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    rows = []
    selected = {}
    for mode in ('uniform', 'fixed', 'adaptive'):
        states = list((Path(args.root)/mode).glob('*/trainer_state.pt'))
        if len(states) != 1:
            raise ValueError(f'exactly one checkpoint directory required for {mode}')
        state = torch.load(states[0], map_location='cpu', weights_only=False)
        selected[mode] = str(states[0].parent/state['mix_policy_model_names'][0])
        config = dict(env_config='pufferlib/config/ocean/drive_population_main.ini',
            data_root='data/binaries_full', split='validation', horizon=90, gamma=.98,
            checkpoints=[selected[mode], 'experiments/puffer_drive_178958541908/model_policy_1_puffer_drive_002000.pt'])
        evaluator = EventEvaluator(config)
        try:
            for map_id in range(args.start_map, args.start_map+args.maps):
                for seed in map(int,args.seeds.split(',')):
                    prepared = evaluator.prepare(map_id,seed)
                    for profile, latent in profiles().items():
                        row = evaluator.run(map_id,seed,latent,prepared)
                        row.update(mode=mode, profile=profile)
                        rows.append(row)
                print(mode,map_id,flush=True)
            evaluator.assert_frozen()
        finally:
            evaluator.close()
    # Physical initial states must be identical across student checkpoints too.
    digests = {}
    for row in rows:
        key=(row['map'],row['seed'])
        if key in digests and digests[key] != row['initial_digest']:
            raise RuntimeError('cross-model physical scenes differ')
        digests[key]=row['initial_digest']
    summary = {}
    for mode in selected:
        summary[mode] = {}
        for profile in profiles():
            group = [r for r in rows if r['mode']==mode and r['profile']==profile]
            summary[mode][profile] = {k:float(np.mean([r[k] for r in group])) for k in
                ('completion_rate','collision_rate','offroad_rate','ret','bootstrapped_return')}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(models=selected, args=vars(args), rows=rows, summary=summary),indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
