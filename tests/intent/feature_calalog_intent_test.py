import unittest
import os
from pathlib import Path
import shutil
import pandas as pd
from pprint import pprint
from ds_discovery import SyntheticBuilder, FeatureCatalog
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from ds_discovery.intent.feature_catalog_intent import FeatureCatalogIntentModel
from aistac.properties.property_manager import PropertyManager


class TestFeatureCatalog(unittest.TestCase):

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

    def tearDown(self):
        try:
            shutil.rmtree('working')
        except:
            pass

    def test_smoke(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        sample_size = 10
        seed =27
        df = pd.DataFrame()
        df['cat'] = builder.tools.get_category(selection=['a', 'b', 'c', 'd'], size=sample_size, column_name='cat', seed=seed)
        df['norm'] = builder.tools.get_dist_normal(mean=4, std=1, size=sample_size, column_name='norm', seed=seed)
        df['pois'] = builder.tools.get_dist_poisson(interval=7, size=sample_size, column_name='pois', seed=seed)
        df['norm_std'] = builder.tools.correlate_numbers(df, header='norm', standardize=True, column_name='norm_std', seed=seed)
        df['jitter1'] = builder.tools.correlate_numbers(df, header='pois', jitter=0.1, column_name='jitter1', seed=seed)
        df['jitter2'] = builder.tools.correlate_numbers(df, header='pois', jitter=0.8, column_name='jitter2', seed=seed)
        df['jitter3'] = builder.tools.correlate_numbers(df, header='pois', jitter=1.5, column_name='jitter3', seed=seed)
        df['jitter4'] = builder.tools.correlate_numbers(df, header='pois', jitter=2, column_name='jitter4', seed=seed)
        df['jitter5'] = builder.tools.correlate_numbers(df, header='pois', jitter=7, column_name='jitter5', seed=seed)
        fc = FeatureCatalog.from_memory()
        catalog: FeatureCatalogIntentModel = fc.tools
        result = catalog.select_correlate(df, target ='pois', threshold=0.8, train_size=0.3, seed=seed)
        self.assertCountEqual(['pois', 'cat', 'norm_std', 'jitter3'], result.columns.to_list())


    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
