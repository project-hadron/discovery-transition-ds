import unittest
import os
import shutil
import pandas as pd
from ds_discovery import SyntheticBuilder
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class SyntheticWeightingTest(unittest.TestCase):

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

    @property
    def tools(self) -> SyntheticIntentModel:
        return SyntheticBuilder.scratch_pad()

    def test_relative(self):
        size = 125117 # prime
        freq = [1.,3.,.4]
        result = self.tools._freq_dist_size(relative_freq=freq, size=size)
        self.assertEqual(3, len(result))
        self.assertEqual(size, sum(result))
        result = self.tools._freq_dist_size(relative_freq=freq, size=size, seed=31)
        other = self.tools._freq_dist_size(relative_freq=freq, size=size, seed=31)
        self.assertEqual(size, sum(result))
        self.assertEqual(other, result)

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
