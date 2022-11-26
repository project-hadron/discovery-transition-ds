import unittest
import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from pprint import pprint
from ds_discovery import SyntheticBuilder
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class SyntheticTest(unittest.TestCase):

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
        builder = SyntheticBuilder.from_env('tester', has_contract=False)
        builder.set_persist()
        tools: SyntheticIntentModel = builder.tools

    def tearDown(self):
        try:
            shutil.rmtree('working')
        except:
            pass

    def test_get_noise(self):
        builder = SyntheticBuilder.from_env('tester', default_save_intent=False)
        tools: SyntheticIntentModel = builder.tools
        result = tools.get_noise(10)
        self.assertEqual(list(np.ones(10)), result)
        result = tools.get_noise(10, ones=False)
        self.assertEqual(list(np.zeros(10)), result)

    def test_correlate_date(self):
        builder = SyntheticBuilder.from_env('tester', default_save_intent=False)
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame()
        df['date1'] = tools.get_datetime(start='2020-01-01', until='2020-02-01', seed=31, size=10)
        df['date2'] = tools.get_datetime(start='2020-03-01', until='2020-04-01', seed=31, size=10)
        result = tools.correlate_date_diff(df, first_date='date1', second_date='date2')
        self.assertEqual([60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0], result)

    def test_model_categories(self):
        builder = SyntheticBuilder.from_env('tester', default_save_intent=False)
        tools: SyntheticIntentModel = builder.tools
        sample_size = 10
        df = pd.DataFrame()
        df['cat'] = tools.get_category(selection=['A', 'B', 'C'], size=sample_size)
        df['num'] = tools.get_number(to_value=2, size=sample_size)
        self.assertEqual('object', df.cat.dtype)
        self.assertEqual('int64', df.num.dtype)
        result = tools.model_to_category(df, headers=['cat', 'num'])
        self.assertEqual('category', result.cat.dtype)
        self.assertEqual('category', result.num.dtype)


    def test_model_onehot(self):
        builder = SyntheticBuilder.from_env('tester', default_save_intent=False)
        tools: SyntheticIntentModel = builder.tools
        sample_size = 10
        df = pd.DataFrame()
        df['cat'] = tools.get_category(selection=['A', 'B', 'C'], size=sample_size)
        df['num'] = tools.get_number(from_value=1, to_value=9, size=sample_size)
        df['value'] = tools.get_number(from_value=1, to_value=3, size=sample_size)
        result = tools.model_multihot(df, header='cat')
        self.assertEqual(['num', 'value', 'cat_A', 'cat_B', 'cat_C'], result.columns.to_list())

    def test_flatten_onehot(self):
        builder = SyntheticBuilder.from_env('tester', default_save_intent=False)
        tools: SyntheticIntentModel = builder.tools
        sample_size = 10
        df = pd.DataFrame()
        df['profile'] = tools.get_number(from_value=1, to_value=9, at_most=3, size=sample_size)
        df['cat'] = tools.get_category(selection=['A', 'B', 'C'], size=sample_size)
        df['num'] = tools.get_number(from_value=1, to_value=9, size=sample_size)
        df['value'] = tools.get_number(from_value=1, to_value=3, size=sample_size)
        result = tools.model_multihot(df, header='cat')
        result = tools.model_group(result, group_by='profile', headers=['cat', 'value'], regex=True)
        self.assertCountEqual(['profile', 'value', 'cat_A', 'cat_B', 'cat_C'], result.columns.to_list())


    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
