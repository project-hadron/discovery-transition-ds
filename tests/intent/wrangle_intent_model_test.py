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

    def test_model_to_category_titanic(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        builder.set_source_uri('../_test_data/titanic_features.pickle')
        df = builder.load_source_canonical()
        self.assertEqual('int64', df.family.dtype)
        self.assertEqual('string', df.deck.dtype)
        result = tools.model_to_category(df, headers=['family', 'is_alone', 'deck'])
        self.assertEqual('category', result.family.dtype)
        self.assertEqual('category', result.deck.dtype)

    def test_model_to_numeric(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [.1, .2, .3, .4, .3, .2, .1], "B": list("ABCDCBA"), 'C': list("B2DE56B"),
                                'D': [True, False, False, False, True, False, True], 'E': [0,0,1,0,1,1,0]})
        df['E'] = df['E'].astype('bool')
        self.assertEqual('float64 object object', f'{df.A.dtype} {df.B.dtype} {df.C.dtype}')
        result = tools.model_to_numeric(df, headers='A')
        self.assertEqual('float64 object object', f'{result.A.dtype} {result.B.dtype} {result.C.dtype}')
        result = tools.model_to_numeric(df, headers=['A', 'C'])
        self.assertEqual('float64 object float64', f'{result.A.dtype} {result.B.dtype} {result.C.dtype}')
        result = tools.model_to_numeric(df, headers=['D', 'E'])
        self.assertEqual('int64 int64', f'{result.D.dtype} {result.E.dtype}')


    def test_model_to_category(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 3, 2, 1], "B": list("ABCDCBA"), 'C': list("BCDECFB")})
        self.assertEqual('int64 object object', f'{df.A.dtype} {df.B.dtype} {df.C.dtype}')
        result = tools.model_to_category(df, headers='A')
        self.assertEqual('category object object', f'{result.A.dtype} {result.B.dtype} {result.C.dtype}')
        result = tools.model_to_category(df, headers=['A','B'])
        self.assertEqual('category category object', f'{result.A.dtype} {result.B.dtype} {result.C.dtype}')

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

    def test_model_encode_one_hot(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 3, 2, 1], "B": list("ABCDCBA"), 'C': list("BCDECFB")})
        result = tools.model_encode_one_hot(df, headers=['B', 'C'])
        self.assertCountEqual(['A', 'B_A', 'B_B', 'B_C', 'B_D', 'C_B', 'C_C', 'C_D', 'C_E', 'C_F'], result.columns.to_list())

    def test_model_encode_integer(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 3, 2, 1], "B": list("ABCDEFA"), 'C': list("BCDECFB")})
        result = tools.model_encode_integer(df, headers=['B', 'C'], seed=31)
        self.assertCountEqual([0, 1, 2, 3, 4, 5, 0], result['B'].to_list())
        self.assertCountEqual([0, 1, 2, 3, 1, 4, 0], result['C'].to_list())
        result = tools.model_encode_integer(df, headers=['B', 'C'], ranking=list('ABCDX'), seed=31)
        self.assertCountEqual([0, 1, 2, 3, 5, 5, 0], result['B'].to_list())
        self.assertCountEqual([1, 2, 3, 5, 2, 5, 1], result['C'].to_list())

    def test_model_encode_count(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame()
        df['B'] = tools.get_category(['a','b','c','d'], size=1000, relative_freq=[40,8,6,4], quantity=0.998, seed=99)
        result = tools.model_encode_count(df, headers=['B'])
        self.assertCountEqual([685, 137, 102, 74, 2], result['B'].value_counts().to_list())

    def test_model_difference_num(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 5, 6, 7], "B": [1, 2, 3, 4, 3, 3, 1], 'C': [0, 2, 0, 4, 3, 2, 1]})
        target = pd.DataFrame(data={"A": [1, 2, 3, 4, 5, 6, 7], "B": [1, 2, 5, 4, 3, 3, 1], 'C': [1, 2, 1, 4, 3, 2, 1]})
        builder.add_connector_persist('target', uri_file='working/data/target.csv')
        builder.save_canonical('target', target)
        result = tools.model_difference(df, 'target')
        self.assertEqual([0,7,2,9], result.index.tolist())

    def test_model_difference_str(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 5, 6, 7], "B": list("ABCFCBA"), 'C': list("BCDECFB")})
        target = pd.DataFrame(data={"A": [1, 2, 3, 4, 5, 6, 7], "B": list("ABCDCBA"), 'C': list("BCDECFP")})
        builder.add_connector_persist('target', uri_file='working/data/target.csv')
        builder.save_canonical('target', target)
        result = tools.model_difference(df, 'target')
        self.assertEqual([10,3,6,13], result.index.tolist())

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
