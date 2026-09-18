"""Reports must preserve scene-specific weights and reject unpaired results."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.population.run import map_ids
from scripts.population.summarize import summarize


def document():
    config = json.loads(Path("scripts/population/smoke.json").read_text())
    rows = []
    for i, map_id in enumerate(map_ids(config["development_maps"])):
        for seed in config["evaluation_seeds"]:
            probs = [.05] * 5
            probs[i] = .8
            components = [dict(component=k, bootstrapped_return=float(k != i),
                               collision_indicator_rate=0.) for k in range(5)]
            rows.append(dict(map=map_id, seed=seed, context=[float(i)] * 6,
                             probabilities=probs, components=components,
                             uniform=dict(bootstrapped_return=1.)))
    return dict(config=config, training_complete=True, variant="context", seed=917,
                evaluations=dict(development=rows))


class PopulationSummaryTests(unittest.TestCase):
    def write_results(self, root, documents):
        paths = []
        for i, data in enumerate(documents):
            path = Path(root) / f"{i}.json"
            path.write_text(json.dumps(data))
            paths.append(path)
        return paths

    def test_weights_are_applied_per_scene(self):
        with tempfile.TemporaryDirectory() as root:
            result = summarize(self.write_results(root, [document()]), bootstrap_samples=100)
        self.assertAlmostEqual(result["runs"][0]["vs_uniform"]["mean_return"], .2)
        self.assertFalse(result["complete_matrix"])

    def test_unpaired_component_rollout_is_rejected(self):
        first, second = document(), document()
        second["variant"] = "baseline"
        second["evaluations"]["development"][0]["components"][0]["bootstrapped_return"] = .1
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "component rollouts differ"):
                summarize(self.write_results(root, [first, second]), bootstrap_samples=100)

    def test_incomplete_evaluation_is_rejected(self):
        data = document()
        data["evaluations"]["development"].pop()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "evaluation is incomplete"):
                summarize(self.write_results(root, [data]), bootstrap_samples=100)

    def test_training_seed_is_kept_separate(self):
        first, second = document(), copy.deepcopy(document())
        first["config"]["seeds"] = [917, 918]
        second["config"]["seeds"] = [917, 918]
        second["seed"] = 918
        with tempfile.TemporaryDirectory() as root:
            result = summarize(self.write_results(root, [first, second]), bootstrap_samples=100)
        self.assertEqual([r["seed"] for r in result["runs"]], [917, 918])

    def test_float32_probability_roundoff_does_not_create_challenge(self):
        data = document()
        for row in data["evaluations"]["development"]:
            row["probabilities"] = [.20000000298023224] * 5
            row["uniform"]["bootstrapped_return"] = .4321
            for component in row["components"]:
                component["bootstrapped_return"] = .4321
        with tempfile.TemporaryDirectory() as root:
            result = summarize(self.write_results(root, [data]), bootstrap_samples=100)
        self.assertEqual(result["runs"][0]["vs_uniform"]["delta"], 0.)
        self.assertEqual(result["runs"][0]["vs_initial_mixture"]["map_bootstrap_95"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
