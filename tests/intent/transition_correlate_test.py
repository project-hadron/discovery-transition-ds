import unittest
import os
from pathlib import Path
import shutil
import pandas as pd
from pprint import pprint

from build.lib.ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from ds_discovery import Transition, Wrangle, Controller, SyntheticBuilder
from aistac.properties.property_manager import PropertyManager
from ds_discovery.intent.transition_intent import TransitionIntentModel


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # clean out any old environments
        for key in os.environ.keys():
            if key.startswith('HADRON'):
                del os.environ[key]
        # Local Domain Contract
        os.environ['HADRON_PM_PATH'] = os.path.join('working', 'contracts')
        os.environ['HADRON_PM_TYPE'] = 'json'
        # Local Connectivity
        os.environ['HADRON_DEFAULT_PATH'] = Path('working/data').as_posix()
        # Specialist Component
        try:
            os.makedirs(os.environ['HADRON_PM_PATH'])
        except:
            pass
        try:
            os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        except:
            pass
        PropertyManager._remove_all()
        builder = SyntheticBuilder.from_env('builder', has_contract=False)
        builder.set_persist()
        sample_size = 10
        df = pd.DataFrame()
        df['cat'] = builder.tools.get_category(selection=['a', 'b', 'c', 'd'], size=sample_size, column_name='cat')
        df['norm'] = builder.tools.get_dist_normal(mean=4, std=1, size=sample_size, column_name='norm')
        df['pois'] = builder.tools.get_dist_poisson(interval=7, size=sample_size, column_name='pois')
        df['norm_std'] = builder.tools.correlate_numbers(df, header='norm', standardize=True, column_name='norm_std')
        df['jitter1'] = builder.tools.correlate_numbers(df, header='pois', jitter=0.1, column_name='jitter1')
        df['jitter2'] = builder.tools.correlate_numbers(df, header='pois', jitter=0.8, column_name='jitter2')
        df['jitter3'] = builder.tools.correlate_numbers(df, header='pois', jitter=1.5, column_name='jitter3')
        df['jitter4'] = builder.tools.correlate_numbers(df, header='pois', jitter=2, column_name='jitter4')
        df['jitter5'] = builder.tools.correlate_numbers(df, header='pois', jitter=3, column_name='jitter5')
        builder.run_component_pipeline()

    def tearDown(self):
        try:
            shutil.rmtree('working')
        except:
            pass

    def test_filter_correlate(self):
        builder = SyntheticBuilder.from_env('builder')
        tr = Transition.from_env("tr1", has_contract=False)
        cleaners: TransitionIntentModel = tr.cleaners
        tr.set_source_uri(builder.get_persist_contract().raw_uri)
        tr.set_persist()
        df = tr.load_source_canonical()
        self.assertEqual((1000, 9), df.shape)
        df = cleaners.auto_drop_correlated(df)
        self.assertEqual((1000, 7), df.shape)
        df = cleaners.auto_drop_correlated(df, threshold=0.8)
        self.assertEqual((1000, 3), df.shape)

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
