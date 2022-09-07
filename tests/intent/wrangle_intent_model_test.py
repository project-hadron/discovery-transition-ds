import unittest
import os
from pathlib import Path
import shutil
import pandas as pd
from pprint import pprint

from ds_discovery.intent.wrangle_intent import WrangleIntentModel

from ds_discovery import SyntheticBuilder, Wrangle
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class WrangleIntentModelTest(unittest.TestCase):

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

    def test_model_drop_outliers(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(tools.get_dist_normal(2,1, size=1000, seed=99), columns=['number'])
        self.assertEqual((1000, 1), df.shape)
        result = tools.model_drop_outliers(canonical=df, header="number", measure=1.5, method='interquartile')
        self.assertEqual((992,1), result.shape)
        df = pd.DataFrame(tools.get_dist_normal(2,1, size=1000, seed=99), columns=['number'])
        result = tools.model_drop_outliers(canonical=df, header="number", measure=3, method='empirical')
        self.assertEqual((995,1), result.shape)
        df = pd.DataFrame(tools.get_dist_normal(2,1, size=1000, seed=99), columns=['number'])
        result = tools.model_drop_outliers(canonical=df, header="number", measure=0.002, method='probability')
        self.assertEqual((996,1), result.shape)

    def test_model_encode_one_hot(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 3, 2, 1], "B": list("ABCDCBA"), 'C': list("BCDECFB")})
        result = tools.model_encode_one_hot(df, headers=['B', 'C'])
        self.assertCountEqual(['A', 'B_A', 'B_B', 'B_C', 'B_D', 'C_B', 'C_C', 'C_D', 'C_E', 'C_F'], result.columns.to_list())

    def test_model_encode_ordinal(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 3, 2, 1], "B": list("ABCDCBA"), 'C': list("BCDECFB")})
        result = tools.model_encode_ordinal(df, headers=['B', 'C'])
        self.assertCountEqual([1, 2, 3, 4, 3, 2, 1], result['B'].to_list())
        self.assertCountEqual([1, 2, 3, 4, 2, 5, 1], result['C'].to_list())

    def test_model_encode_count(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame()
        df['B'] = tools.get_category(['a','b','c','d'], size=1000, relative_freq=[40,8,6,4], quantity=0.998, seed=99)
        df['C'] = tools.get_category(['a','b','c','d'], size=1000, relative_freq=[40,8,6,4], quantity=0.998, seed=99)
        result = tools.model_encode_count(df, headers=['B'])
        self.assertCountEqual([685, 137, 102, 74, 2], result['B'].value_counts().to_list())
        result = tools.model_encode_count(df, headers=['C'], top=3)
        self.assertCountEqual([685, 137, 102, 76], result['C'].value_counts().to_list())
        # print(result['C'].value_counts().to_list())

    def test_model_encode_woe(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame()
        df['A'] = tools.get_category([1,0], size=10000, relative_freq=[75,20], seed=99)
        df['B'] = tools.get_category(['a','b','c','d'], size=10000, seed=99)
        result = tools.model_encode_woe(df, headers=['B'], target='A')
        # self.assertCountEqual([1, 2, 3, 4, 3, 2, 1], result['B'].to_list())
        # self.assertCountEqual([1, 2, 3, 4, 2, 5, 1], result['C'].to_list())
        print(result.head())



    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
