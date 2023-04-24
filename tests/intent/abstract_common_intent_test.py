import unittest
import os
import shutil
import pandas as pd
from ds_discovery import SyntheticBuilder
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class AbstractCommonIntentTest(unittest.TestCase):

    def setUp(self):
        os.environ['HADRON_PM_PATH'] = os.path.join('work', 'config')
        os.environ['HADRON_DEFAULT_PATH'] = os.path.join('work', 'data')
        try:
            os.makedirs(os.environ['HADRON_PM_PATH'])
            os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        except:
            pass
        PropertyManager._remove_all()

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_smoke(self):
        tools: SyntheticIntentModel = SyntheticBuilder.from_memory().tools

    def test_freq_dist_size(self):
        tools: SyntheticIntentModel = SyntheticBuilder.from_memory().tools
        size = 125117 # prime
        freq = [1,5,12]
        result = tools._freq_dist_size(relative_freq=freq, size=size)
        self.assertEqual(3, len(result))
        self.assertEqual(size, sum(result))
        self.assertTrue(result[0] < result[1] < result[2])
        result = tools._freq_dist_size(relative_freq=freq, size=size, seed=31)
        other = tools._freq_dist_size(relative_freq=freq, size=size, seed=31)
        self.assertEqual(size, sum(result))
        self.assertEqual(other, result)

    def test_freq_dist_size_elements_small(self):
        tools: SyntheticIntentModel = SyntheticBuilder.from_memory().tools
        size = 100
        freq = [0.1,1,2,3,2,1,0.1]
        # both
        result = tools._freq_dist_size(relative_freq=freq, size=size, dist_length=5, dist_on='both', seed=31)
        self.assertEqual([16, 18, 39, 18, 9], result)
        self.assertEqual(size, sum(result))
        # left
        result = tools._freq_dist_size(relative_freq=freq, size=size, dist_length=5, dist_on='left', seed=31)
        self.assertEqual([30, 32, 24, 14, 0], result)
        self.assertEqual(size, sum(result))
        # right
        result = tools._freq_dist_size(relative_freq=freq, size=size, dist_length=5, dist_on='right', seed=31)
        self.assertEqual([3, 8, 28, 39, 22], result)
        self.assertEqual(size, sum(result))
        # default
        result = tools._freq_dist_size(relative_freq=freq, size=size, dist_length=5, seed=31)
        self.assertEqual([16, 18, 39, 18, 9], result)
        self.assertEqual(size, sum(result))

    def test_freq_dist_size_elements_long(self):
        tools: SyntheticIntentModel = SyntheticBuilder.from_memory().tools
        size = 100
        freq = [0.1,1,2,3,2,1,0.1]
        # both
        result = tools._freq_dist_size(relative_freq=freq, size=size, dist_length=10, dist_on='both', seed=31)
        self.assertEqual([2, 0, 1, 10, 23, 29, 17, 17, 0, 1], result)
        self.assertEqual(size, sum(result))
        # left
        result = tools._freq_dist_size(relative_freq=freq, size=size, dist_length=10, dist_on='left', seed=31)
        self.assertEqual([2, 0, 1, 1, 12, 14, 30, 30, 9, 1], result)
        self.assertEqual(size, sum(result))
        # right
        result = tools._freq_dist_size(relative_freq=freq, size=size, dist_length=10, dist_on='right', seed=31)
        self.assertEqual([2, 6, 23, 29, 17, 17, 0, 1, 4, 1], result)
        self.assertEqual(size, sum(result))
        # default
        result = tools._freq_dist_size(relative_freq=freq, size=size, dist_length=10, seed=31)
        self.assertEqual([2, 0, 1, 10, 23, 29, 17, 17, 0, 1], result)
        self.assertEqual(size, sum(result))


    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
