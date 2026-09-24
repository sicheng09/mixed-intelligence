import numpy as np
import unittest
from unittest.mock import patch

from pufferlib.ocean.drive import drive


class TestDriveReproducibility(unittest.TestCase):
    def test_reproducible_drive_seed_fits_signed_c_int(self):
        upper_bound = np.iinfo(np.int32).max

        def largest_sampled_seed(low, high):
            self.assertEqual(low, 0)
            self.assertEqual(high, upper_bound)
            return high - 1

        with patch.object(np.random, "randint", side_effect=largest_sampled_seed):
            seed = drive._sample_reproducible_seed()

        self.assertEqual(seed, upper_bound - 1)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, upper_bound)
