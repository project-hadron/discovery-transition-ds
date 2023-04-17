import unittest
import os
from pathlib import Path
import shutil
import pandas as pd
from pprint import pprint

from ds_discovery import SyntheticBuilder, Wrangle
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager

# Pandas setup
pd.set_option('max_colwidth', 320)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('expand_frame_repr', True)



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

    def test_model_difference_summary(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"X":  list("ABCDEFG"), "Y":  list("ABCDEFG"), "B": [1, 2, 3, 4, 3, 3, 1], 'C': [0, 2, 0, 4, 3, 2, 1]})
        target = pd.DataFrame(data={"X": list("ABCDEFG"), "Y":  list("ABCDEFG"), "B": [1, 2, 5, 4, 3, 3, 1], 'C': [1, 2, 3, 4, 3, 2, 1]})
        builder.add_connector_persist('target', uri_file='working/data/target.csv')
        builder.save_canonical('target', target)
        # summary connector
        builder.add_connector_persist('summary', uri_file='summary.csv')
        _ = tools.model_difference(df, 'target', on_key='X', summary_connector='summary')
        result = builder.load_canonical('summary')
        self.assertEqual(result.shape, (3,2))
        self.assertEqual(result.Attribute.to_list(), ['Y','B','C'])
        self.assertEqual(result.Summary.to_list(), [0,1,2])
        _ = tools.model_difference(df, 'target', on_key='X', summary_connector='summary', drop_no_diff=True)
        result = builder.load_canonical('summary')
        self.assertEqual(result.shape, (2,2))
        self.assertEqual(result.Attribute.to_list(), ['B','C'])
        self.assertEqual(result.Summary.to_list(), [1,2])
        _ = tools.model_difference(df, 'target', on_key='X', summary_connector='summary', ordered=True)
        result = builder.load_canonical('summary')
        self.assertEqual(result.shape, (3,2))
        self.assertEqual(result.Attribute.tolist(), ['B','C','Y'])
        self.assertEqual(result.Summary.to_list(), [1,2,0])

    def test_model_difference_detail(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"X":  list("CBADEFG"), "Y":  list("ABCDEFG"), "B": [1, 2, 3, 4, 3, 3, 1], 'C': [0, 2, 0, 4, 3, 2, 1]})
        target = pd.DataFrame(data={"X": list("CBADEFG"), "Y":  list("ABCDEFG"), "B": [1, 2, 5, 4, 3, 3, 1], 'C': [1, 2, 3, 4, 3, 2, 1]})
        builder.add_connector_persist('target', uri_file='working/data/target.csv')
        builder.save_canonical('target', target)
        # detail connector
        builder.add_connector_persist('detail', uri_file='detail.csv')
        _ = tools.model_difference(df, 'target', on_key='X', detail_connector='detail')
        result = builder.load_canonical('detail')
        self.assertEqual(result.shape, (2,4))
        self.assertEqual(result.columns.to_list(), ['X','Y','B','C'])
        _ = tools.model_difference(df, 'target', on_key='X', detail_connector='detail', drop_no_diff=True)
        result = builder.load_canonical('detail')
        self.assertEqual(result.shape, (2,3))
        self.assertEqual(result.columns.to_list(), ['X','B','C'])
        _ = tools.model_difference(df, 'target', on_key='X', detail_connector='detail', ordered=True)
        result = builder.load_canonical('detail')
        self.assertEqual(result.shape, (2,4))
        self.assertEqual(result.columns.to_list(), ['X','Y','B','C'])
        self.assertEqual(['A', 'C'], result['X'].to_list())

    def test_model_difference_num(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data=    {"A": list("ABCDEFG"), "B": [1, 2, 5, 5, 3, 3, 1], 'C': [0,  2, 3, 3, 3, 2, 1]})
        target = pd.DataFrame(data={"A": list("ABCDEFG"), "B": [1, 2, 5, 4, 3, 3, 1], 'C': [11, 2, 3, 4, 3, 2, 1]})
        builder.add_connector_persist('target', uri_file='working/data/target.csv')
        builder.save_canonical('target', target)
        # normal
        result = tools.model_difference(df, 'target', on_key='A')
        self.assertEqual((2,3), result.shape)
        self.assertEqual(['A', 'D'], result['A'].tolist())
        self.assertEqual([0,1], result['B'].tolist())
        self.assertEqual([1,1], result['C'].tolist())
        # with levenshtein distance
        result = tools.model_difference(df, 'target', on_key='A', distance=True)
        self.assertEqual((2,3), result.shape)
        self.assertEqual(['A', 'D'], result['A'].tolist())
        self.assertEqual([0,1], result['B'].tolist())
        self.assertEqual([2,1], result['C'].tolist())

    def test_model_difference_str(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data=    {"A": list("ABCDEFG"), "B": ['B', 'C', 'A', 'A', 'F', 'E', 'G'], 'C': ['L', 'L',  'M', 'N', 'J', 'K', 'M']})
        target = pd.DataFrame(data={"A": list("ABCDEFG"), "B": ['B', 'C', 'D', 'A', 'F', 'E', 'G'], 'C': ['L', 'FX', 'M', 'N', 'P', 'K', 'M']})
        builder.add_connector_persist('target', uri_file='working/data/target.csv')
        builder.save_canonical('target', target)
        # normal
        result = tools.model_difference(df, 'target', on_key='A')
        self.assertEqual((3,3), result.shape)
        self.assertEqual(['B', 'C', 'E'], result['A'].tolist())
        self.assertEqual([0, 1, 0], result['B'].tolist())
        self.assertEqual([1,0,1], result['C'].tolist())
        # with levenshtein distance
        result = tools.model_difference(df, 'target', on_key='A', distance=True)
        self.assertEqual((3,3), result.shape)
        self.assertEqual(['B', 'C', 'E'], result['A'].tolist())
        self.assertEqual([0, 1, 0], result['B'].tolist())
        self.assertEqual([2,0,1], result['C'].tolist())

    def test_model_difference_order(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": list("ABCDEFG"), "B": list("ABCFCBA"), 'C': list("BCDECFB"), 'D': [0, 2, 0, 4, 3, 2, 1]})
        target = pd.DataFrame(data={"A": list("ABCDEFG"), "B": list("BBCDCAA"), 'C': list("BCDECFB"), 'D': [0, 2, 0, 4, 1, 2, 1]})
        target.sample(frac = 1)
        builder.add_connector_persist('target', uri_file='working/data/target.csv')
        builder.save_canonical('target', target)
        result = tools.model_difference(df, 'target', on_key='A', drop_no_diff=True)
        self.assertEqual((4,3), result.shape)
        self.assertEqual(['A', 'B', 'D'], result.columns.to_list())

    def test_model_difference_drop(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": list("ABCDEFG"), "B": list("ABCFCBA"), 'C': list("BCDECFB"), 'D': [0, 2, 0, 4, 3, 2, 1]})
        target = pd.DataFrame(data={"A": list("ABCDEFG"), "B": list("BBCDCAA"), 'C': list("BCDECFB"), 'D': [0, 2, 0, 4, 1, 2, 1]})
        builder.add_connector_persist('target', uri_file='working/data/target.csv')
        builder.save_canonical('target', target)
        result = tools.model_difference(df, 'target', on_key='A', drop_no_diff=True)
        self.assertEqual((4,3), result.shape)
        self.assertEqual(['A', 'B', 'D'], result.columns.to_list())
        self.assertEqual(df['C'].to_list(), target['C'].to_list())

    def test_model_difference_index(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": list("ABCDFEG"), "B": list("ABCFCBA"), 'C': list("BCDECFB"), 'D': [0, 2, 0, 4, 3, 2, 1]})
        target = pd.DataFrame(data={"A": list("ABCDFEG"), "B": list("BBCDCAA"), 'C': list("ACDECFB"), 'D': [0, 2, 0, 4, 1, 2, 1]})
        builder.add_connector_persist('target', uri_file='working/data/target.csv')
        builder.save_canonical('target', target)
        result = tools.model_difference(df, 'target', on_key='A', index_on_key=True)
        self.assertEqual((4,3), result.shape)
        self.assertEqual(['B', 'C', 'D'], result.columns.to_list())
        self.assertEqual(['A', 'D', 'E', 'F'], result.index.tolist())

    def test_model_difference_extra(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        # add an extra row to source
        df = pd.DataFrame(data={"A": list("ABCDEFGH"), "B": list("ABCFCBAF"), 'C': list("BCDECFBB")})
        target = pd.DataFrame(data={"A": list("ABCDEFG"), "B": list("ABCDCBA"), 'C': list("BCDECFP")})
        builder.add_connector_persist('target', uri_file='working/data/target.csv')
        builder.save_canonical('target', target)
        result = tools.model_difference(df, 'target', on_key='A')
        self.assertEqual((2,3), result.shape)
        self.assertEqual(['D', 'G'], result['A'].tolist())

    def test_model_profiling(self):
        sb = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = sb.tools
        sb.add_connector_persist('quality', 'hadron_quality.csv')
        sb.add_connector_persist('dictionary', 'hadron_dictionary.csv')
        sb.add_connector_persist('schema', 'hadron_schema.csv')
        df = tools.model_synthetic_data_types(2000)
        result = sb.tools.model_profiling(df, profiling='schema', headers=['cat', 'num', 'date'])
        pprint(result)


    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
