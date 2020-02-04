import os
import shutil
import unittest

import numpy as np
import pandas as pd

from ds_behavioral import DataBuilderTools as tools
from ds_behavioral.sample.sample_data import ProfileSample
from ds_foundation.handlers.abstract_handlers import ConnectorContract

from ds_discovery import Transition
from ds_discovery.intent.transition_intent import TransitionIntentModel as Cleaner, TransitionIntentModel
from ds_discovery.managers.transition_property_manager import TransitionPropertyManager


class CleanerTest(unittest.TestCase):
    """Test: """

    def setUp(self):
        property_manager = TransitionPropertyManager('test')
        property_manager.set_property_connector(ConnectorContract(uri='', handler='DummyPersistHandler',
                                                                  module_name='ds_foundation.handlers.dummy_handlers'))
        property_manager.remove_intent()
        self.clean = TransitionIntentModel(property_manager=property_manager)
        try:
            os.makedirs('data')
        except:
            pass

    def tearDown(self):
        try:
            shutil.rmtree('data')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        TransitionIntentModel(property_manager=TransitionPropertyManager('test'), default_save_intent=False)

    def test_auto_remove(self):
        df = pd.DataFrame()
        df['single_num'] = tools.get_number(1, 1, quantity=0.7, size=100)
        df['two_num'] = tools.get_number(2, quantity=0.7, size=100)
        df['null_num'] = tools.get_number(1, 100, quantity=0, size=100)
        df['normal'] = tools.get_number(1, 100, size=100)
        df.loc[1:4, 'normal'] = 'None'
        df['none_num'] = tools.get_number(1, 1, quantity=0.7, size=100)
        df.loc[1:4, 'none_num'] = 'None'
        df.loc[7:9, 'none_num'] = ''
        self.clean.auto_remove_columns(df, nulls_list=True, inplace=True)
        self.assertEqual(['two_num', 'normal'], df.columns.tolist())

    def test_auto_remove_predom(self):
        df = pd.DataFrame()
        df['single_num'] = tools.get_number(1, 1, size=100)
        df['two_num'] = tools.get_number(2, size=100)
        df['weight_num'] = tools.get_number(2, weight_pattern=[98, 1], size=100)
        df['null_num'] = tools.get_number(1, 100, quantity=0, size=100)
        df['normal_num'] = tools.get_number(1, 100, size=100)
        df['single_cat'] = tools.get_category(['A'], size=100)
        df['two_cat'] = tools.get_category(['A', 'B'], quantity=0.9, size=100)
        df['weight_cat'] = tools.get_category(['A', 'B'], weight_pattern=[95, 1], size=100)
        df['normal_cat'] = tools.get_category(list('ABCDE'), size=100)
        self.clean.auto_remove_columns(df, predominant_max=0.8, inplace=True)
        self.assertEqual(['two_num', 'normal_num', 'two_cat', 'normal_cat'], df.columns.tolist())

    def test_clean_headers(self):
        df = tools.get_profiles()
        control = ['surname', 'forename', 'gender']
        result = df.columns
        self.assertTrue(control, result)
        rename = {'forename': 'first_name'}
        control = {'clean_header': {'case': 'title', 'rename': {'surname: last_name'}}}
        result = self.clean.auto_clean_header(df, rename_map=rename, case='title', inplace=True)
        self.assertTrue(control, result)
        control = ['Surname', 'First_Name', 'Gender']
        self.assertTrue(control, df.columns)

    def test_remove_columns(self):
        clean = self.clean
        df = tools.get_profiles(size=10)
        result = clean.to_remove(df, headers=['surname'])
        self.assertNotIn('surname', result.columns.values)

        df = tools.get_profiles(size=10)
        clean.to_remove(df, headers=['surname', 'gender'], inplace=True)
        self.assertNotIn('surname', df.columns.values)
        self.assertNotIn('gender', df.columns.values)

    def test_select_columns(self):
        clean = self.clean
        df = tools.get_profiles(size=10)
        control = ['surname']
        result = clean.to_select(df, headers=['surname'])
        self.assertEqual(['surname'], result.columns.values)

        df = tools.get_profiles(size=10)
        clean.to_select(df, headers=['surname', 'gender'], inplace=True)
        self.assertIn('surname', df.columns.values)
        self.assertIn('gender', df.columns.values)
        self.assertEqual((10,2), df.shape)

    def test_contract_pipeline(self):
        clean = self.clean
        df = pd.DataFrame()
        df['int'] = tools.get_number(100, size=10)
        df['float'] = tools.get_number(0, 1.0, size=10)
        df['object'] = tools.get_category(list('abcdef'), size=10)
        df['date'] = tools.get_datetime('01/01/2010', '01/01/2018', date_format='%Y-%m-%d', size=10)
        df['category'] = tools.get_category(list('vwxyz'), size=10)
        # no intent set
        result = clean.run_intent_pipeline(df)
        self.assertTrue(result.equals(df))
        # set intent
        self.assertEqual((10,5), df.shape)
        control = self.clean.to_remove(df, headers=['date'], inplace=False, save_intent=True)
        self.assertEqual((10,4), control.shape)
        control = self.clean.to_category_type(control, headers=['category'], inplace=False, save_intent=True)
        self.assertEqual('category', control['category'].dtype)
        # run pipeline
        result = clean.run_intent_pipeline(df)
        self.assertTrue(result.equals(control))

    def test_to_float_with_mode(self):
        clean = self.clean
        col = 'X'
        df = pd.DataFrame()
        df[col] = tools.get_number(5.0, precision=5, size=5, seed=101)
        df.loc[[2, 4], col] = np.nan
        control = df.copy()
        mode = df[df[col].notna()]
        result = clean.to_float_type(df, headers='X', errors='coerce', fillna='mode')
        self.assertEqual(control.iloc[0,0], result.iloc[0,0])
        self.assertEqual(control.iloc[3,0], result.iloc[3,0])

    def test_to_float_type(self):
        clean = self.clean
        col = 'X'
        df = pd.DataFrame()
        df[col] = tools.get_number(5.0, precision=5, size=5, seed=101)
        df.loc[[2, 4], col] = np.nan
        df.loc[1, col] = 'text'
        control = df.copy()
        result = clean.to_float_type(df, headers='X', errors='coerce', fillna=-1)
        self.assertEqual(-1, result.iloc[1,0])
        self.assertEqual(-1, result.iloc[2,0])
        self.assertEqual(control.iloc[0,0], result.iloc[0,0])
        self.assertEqual(control.iloc[3,0], result.iloc[3,0])

    def test_make_list(self):
        for value in ['', 0, 0.0, pd.Timestamp(2018,1,1), [], (), pd.Series(), list(), tuple(),
                      'name', ['list1', 'list2'], ('tuple1', 'tuple2'), pd.Series(['series1', 'series2']),
                      {'key1': 'value1', 'key2': 'value2'}, {}, dict()]:
            result = Transition.list_formatter(value)
            self.assertTrue(isinstance(result, list), value)
        self.assertEqual(None, Transition.list_formatter(None))

    def test_to_date(self):
        cleaner = self.clean
        df = pd.DataFrame()
        df['date'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', date_format='%d-%m-%Y', seed=101)
        df['datetime'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', date_format='%d-%m-%Y %H:%M:%S', seed=102)
        df['number'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', date_format='%Y%m%d.0', seed=101)
        result = cleaner.to_date_type(df, headers=['date', 'datetime'])
        self.assertEqual(df['date'].iloc[0], result['date'].iloc[0].strftime(format='%d-%m-%Y'))
        self.assertEqual(df['datetime'].iloc[0], result['datetime'].iloc[0].strftime(format='%d-%m-%Y %H:%M:%S'))

        df['numtime'] = df['datetime']
        result = cleaner.to_date_type(df, headers=['numtime'], as_num=True)
        self.assertEqual(float, result['numtime'].iloc[0].dtype)

    def test_currency(self):
        cleaner = self.clean
        df = pd.DataFrame()
        df['currency'] = ['$3,320.12', '£1,001.34', '€34', 23.4, '5 220.12']
        df['control'] = [3320.12, 1001.34, 34, 23.4, 5220.12]
        df = cleaner.to_float_type(df, headers='currency')
        self.assertEqual(list(df.control), list(df.currency))

    def test_get_cols(self):
        cleaner = self.clean
        df = pd.DataFrame()
        df['int'] = tools.get_number(100, size=10)
        df['float'] = tools.get_number(0, 1.0, size=10)
        df['object'] = tools.get_category(list('abcdef'), size=10)
        df['date'] = tools.get_datetime('01/01/2010', '01/01/2018', size=10)
        df['category'] = tools.get_category(list('vwxyz'), size=10)
        df = cleaner.to_category_type(df, headers='category')
        df = cleaner.to_date_type(df, headers='date')
        control = ['float', 'object', 'date', 'category', 'int']
        result = cleaner.filter_headers(df)
        self.assertTrue(set(result).intersection(control))
        control = ['object']
        result = cleaner.filter_headers(df, dtype=[object])
        self.assertEqual(control, result)
        result = cleaner.filter_headers(df, dtype=['object'])
        self.assertEqual(control, result)
        control = ['float', 'int']
        result = cleaner.filter_headers(df, dtype=['number'])
        self.assertTrue(set(result).intersection(control))
        control = ['date']
        result = cleaner.filter_headers(df, dtype=['datetime'])
        self.assertEqual(control, result)

        # self.assertTrue(set(cleaner.filter_headers(self.df, dtype=['object'])).intersection(df['object']))
        # self.assertTrue(set(cleaner.filter_headers(self.df, dtype=['float'])).intersection(self.float_col))
        # self.assertTrue(set(cleaner.filter_headers(self.df, dtype=['float'])).intersection(self.float_col))
        # self.assertTrue(set(cleaner.filter_headers(self.df, dtype=['float', 'object'])).intersection(self.df.columns))
        #
        # self.assertTrue(set(cleaner.filter_headers(self.df, dtype=['object'], exclude=False)).intersection(self.object_col))
        # self.assertTrue(set(cleaner.filter_headers(self.df, dtype=['object'], exclude=True)).difference(self.object_col))
        #
        # test_col = ['Dummy_Policy_Number', 'ACTIVITY_ID', 'PRODUCT_CATEGORY', 'SYSTEM_NAME']
        # test_rev = self.df.columns.difference(test_col)
        #
        # self.assertTrue(set(cleaner.filter_headers(self.df, headers=test_col, drop=False)).intersection(test_col))
        # self.assertTrue(set(cleaner.filter_headers(self.df, headers=test_col, drop=True)).intersection(test_rev))
        #
        # self.assertTrue(set(cleaner.filter_headers(self.df, headers=test_col, dtype=['object'], exclude=False)).
        #                 intersection(['PRODUCT_CATEGORY', 'SYSTEM_NAME']))
        # self.assertTrue(set(cleaner.filter_headers(self.df, dtype=['object'], exclude=True)).
        #                 difference(['Dummy_Policy_Number', 'ACTIVITY_ID']))

    def test_filter(self):
        cleaner = self.clean
        sample_size = 1000
        df = pd.DataFrame()
        df['normal_num'] = tools.get_number(1, 10, size=sample_size, seed=31)
        df['single num'] = tools.get_number(1, 1, quantity=0.8, size=sample_size, seed=31)
        df['weight_num'] = tools.get_number(1, 2, weight_pattern=[90, 1], size=sample_size, seed=31)
        df['null'] = tools.get_number(1, 100, quantity=0, size=sample_size, seed=31)
        df['single cat'] = tools.get_category(['A'], quantity=0.6, size=sample_size, seed=31)
        df['weight_cat'] = tools.get_category(['A', 'B', 'C'], weight_pattern=[80, 1, 1], size=sample_size, seed=31)
        df['normal_cat'] = tools.get_category(['A', 'B', 'C'], size=sample_size, seed=31)
        result = cleaner.filter_headers(df, headers=['normal_num', 'single num'])
        control = ['normal_num', 'single num']
        self.assertCountEqual(control, result)
        result = cleaner.filter_headers(df, dtype=['number'])
        control = ['weight_num', 'normal_num', 'single num']
        self.assertCountEqual(control, result)




if __name__ == '__main__':
    unittest.main()