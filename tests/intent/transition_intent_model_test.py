import os
import shutil
import unittest

import numpy as np
import pandas as pd
from aistac.properties.property_manager import PropertyManager
from ds_behavioral import SyntheticBuilder

from ds_discovery import Transition
from ds_discovery.intent.transition_intent import TransitionIntentModel
from ds_discovery.managers.transition_property_manager import TransitionPropertyManager
from ds_discovery.components.commons import Commons


class IntentModelTest(unittest.TestCase):
    """Test: """

    def setUp(self):
        os.environ['HADRON_PM_PATH'] = os.path.join('work', 'config')
        os.environ['HADRON_DEFAULT_PATH'] = os.path.join('work', 'data')
        try:
            os.makedirs(os.environ['HADRON_PM_PATH'])
            os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        except:
            pass
        PropertyManager._remove_all()
        self.tools = SyntheticBuilder.scratch_pad()
        self.clean = Transition.scratch_pad()

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        TransitionIntentModel(property_manager=TransitionPropertyManager('test', username='TestUser'), default_save_intent=False)

    def test_to_date_element(self):
        tools = self.tools
        df = pd.DataFrame()
        df['dates'] = tools.get_datetime("01/01/2018", "01/02/2018", day_first=False, size=5)
        result = self.clean.to_date_element(df, matrix=['yr', 'dec'])
        self.assertCountEqual(['dates', 'dates_yr', 'dates_dec'], result.columns.to_list())
        result = self.clean.to_date_element(df, matrix=[])
        self.assertCountEqual(['dates'], result.columns.to_list())

    def test_to_sample(self):
        tools = self.tools
        df = pd.DataFrame(tools.model_sample_map({'@empty'}, sample_map='us_zipcode_primary').iloc[:200])
        result = self.clean.to_sample(df, sample_size=100)
        self.assertEqual((100, 13), result.shape)
        self.assertCountEqual(df.iloc[0], result.iloc[0])
        result = self.clean.to_sample(df, sample_size=100, shuffle=True)
        self.assertEqual((100, 13), result.shape)
        self.assertNotEqual(df.iloc[0].to_list(), result.iloc[0].to_list())

        result = self.clean.to_sample(df, sample_size=0.2)
        self.assertEqual((200, 13), result.shape)
        self.assertCountEqual(df.iloc[0], result.iloc[0])
        result = self.clean.to_sample(df, sample_size=0.2, shuffle=True)
        self.assertEqual((200, 13), result.shape)
        self.assertNotEqual(df.iloc[0].to_list(), result.iloc[0].to_list())

    def test_to_date_from_mdates(self):
        tools = self.tools
        df = pd.DataFrame()
        df['dates'] = tools.get_datetime("01/01/2018", "01/02/2018", day_first=False, as_num=True, size=5)
        result = self.clean.to_date_from_mpldates(df, headers='dates', date_format="%Y-%m-%d")
        self.assertEqual(['2018-01-01']*5, result['dates'].to_list())

    def test_auto_transition(self):
        tools = self.tools
        sample_size = 100
        df = pd.DataFrame()
        df['nums'] = tools.get_number(1, 100, size=sample_size)
        df['floats'] = tools.get_number(1, 100, quantity=0.9, size=sample_size)
        df['num_str'] = tools.get_category(list(range(100)), quantity=0.9, size=sample_size)
        df['bools'] = tools.get_category([True, False], quantity=0.9, size=sample_size)
        df['bool_str'] = tools.get_category(['1', '0'], quantity=0.9, size=sample_size)
        df['bool_num'] = tools.get_category([1, 0], quantity=0.9, size=sample_size)
        df['cats'] = tools.get_category(list('ABC'), quantity=0.9, size=sample_size)
        df = self.clean.auto_transition(df)
        self.assertTrue(df['nums'].dtype.name.startswith('int'))
        self.assertTrue(df['floats'].dtype.name.startswith('float'))
        self.assertTrue(df['num_str'].dtype.name.startswith('float'))
        self.assertTrue(df['bools'].dtype.name.startswith('bool'))
        self.assertTrue(df['bool_str'].dtype.name.startswith('category'))
        self.assertTrue(df['bool_num'].dtype.name.startswith('bool'))
        self.assertTrue(df['cats'].dtype.name.startswith('category'))

    def test_auto_remove(self):
        tools = self.tools
        df = pd.DataFrame()
        df['single_num'] = tools.get_number(1, 2, quantity=0.7, size=100)
        df['two_num'] = tools.get_number(2, quantity=0.7, size=100)
        df['null_num'] = tools.get_number(1, 100, quantity=0, size=100)
        df['normal'] = tools.get_number(1, 100, size=100)
        df.loc[1:4, 'normal'] = 'None'
        df['none_num'] = tools.get_number(1, 2, quantity=0.7, size=100)
        df.loc[1:4, 'none_num'] = 'None'
        df.loc[7:9, 'none_num'] = ''
        self.clean.auto_remove_columns(df, nulls_list=True, inplace=True)
        self.assertEqual(['two_num', 'normal'], df.columns.tolist())

    def test_auto_remove_predom(self):
        tools = self.tools
        df = pd.DataFrame()
        df['single_num'] = tools.get_number(1, 2, size=100)
        df['two_num'] = tools.get_number(2, size=100)
        df['weight_num'] = tools.get_number(2, relative_freq=[98, 1], size=100)
        df['null_num'] = tools.get_number(1, 100, quantity=0, size=100)
        df['normal_num'] = tools.get_number(1, 100, size=100)
        df['single_cat'] = tools.get_category(['A'], size=100)
        df['two_cat'] = tools.get_category(['A', 'B'], quantity=0.9, size=100)
        df['weight_cat'] = tools.get_category(['A', 'B'], relative_freq=[95, 1], size=100)
        df['normal_cat'] = tools.get_category(list('ABCDE'), size=100)
        self.clean.auto_remove_columns(df, predominant_max=0.8, inplace=True)
        self.assertEqual(['two_num', 'normal_num', 'two_cat', 'normal_cat'], df.columns.tolist())

    def test_clean_headers(self):
        tools = self.tools
        df = pd.DataFrame(tools.model_sample_map('@empty', sample_map='us_zipcode_primary'))
        control = ['City', 'State', 'Zipcode']
        result = df.columns
        self.assertTrue(control, result)
        rename = {'City': 'town'}
        control = {'clean_header': {'case': 'title', 'rename': {'City': 'town'}}}
        result = self.clean.auto_clean_header(df, rename_map=rename, case='title', inplace=True)
        self.assertTrue(control, result)
        control = ['town', 'State', 'Zipcode']
        self.assertTrue(control, df.columns)

    def test_remove_columns(self):
        tools = self.tools
        clean = self.clean
        df = pd.DataFrame(tools.model_sample_map('@empty', sample_map='us_zipcode_primary'))
        result = clean.to_remove(df, headers=['City'])
        self.assertNotIn('City', result.columns.values)

        df = pd.DataFrame(tools.model_sample_map('@empty', sample_map='us_zipcode_primary'))
        clean.to_remove(df, headers=['City', 'State'], inplace=True)
        self.assertNotIn('City', df.columns.values)
        self.assertNotIn('State', df.columns.values)

    def test_select_columns(self):
        tools = self.tools
        clean = self.clean
        df = pd.DataFrame(tools.model_sample_map('@empty', sample_map='us_zipcode_primary'))
        control = ['City']
        result = clean.to_select(df, headers=['City'])
        self.assertEqual(['City'], result.columns.values)

        df = pd.DataFrame(tools.model_sample_map('@empty', sample_map='us_zipcode_primary').iloc[:10])
        clean.to_select(df, headers=['City', 'State'], inplace=True)
        self.assertIn('City', df.columns.values)
        self.assertIn('State', df.columns.values)
        self.assertEqual((10,2), df.shape)

    def test_contract_pipeline(self):
        tools = self.tools
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
        tools = self.tools
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
        tools = self.tools
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

    def test_to_str_type(self):
        clean = self.clean
        df = pd.DataFrame()
        df['X'] = [1,2,3,4]
        print(df['X'])
        df['X'] = clean.to_str_type(df, headers='X')
        print(df['X'])

    def test_make_list(self):
        for value in ['', 0, 0.0, pd.Timestamp(2018,1,1), [], (), pd.Series(dtype=str), list(), tuple(),
                      'name', ['list1', 'list2'], ('tuple1', 'tuple2'), pd.Series(['series1', 'series2']),
                      {'key1': 'value1', 'key2': 'value2'}, {}, dict()]:
            result = Commons.list_formatter(value)
            self.assertTrue(isinstance(result, list), value)
        self.assertEqual([], Commons.list_formatter(None))

    def test_to_date(self):
        tools = self.tools
        cleaner = self.clean
        df = pd.DataFrame()
        df['date'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', date_format='%d-%m-%Y', seed=101)
        df['datetime'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', date_format='%d-%m-%Y %H:%M:%S', seed=102)
        df['number'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', date_format='%Y%m%d.0', seed=101)
        result = cleaner.to_date_type(df, headers=['date', 'datetime'])
        self.assertEqual(df['date'].iloc[0], result['date'].iloc[0].strftime(format='%m-%d-%Y'))
        self.assertEqual(df['datetime'].iloc[0], result['datetime'].iloc[0].strftime(format='%m-%d-%Y %H:%M:%S'))

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
        tools = self.tools
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
        result = Commons.filter_headers(df)
        self.assertTrue(set(result).intersection(control))
        control = ['object']
        result = Commons.filter_headers(df, dtype=[object])
        self.assertEqual(control, result)
        result = Commons.filter_headers(df, dtype=['object'])
        self.assertEqual(control, result)
        control = ['float', 'int']
        result = Commons.filter_headers(df, dtype=['number'])
        self.assertTrue(set(result).intersection(control))
        control = ['date']
        result = Commons.filter_headers(df, dtype=['datetime'])
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



if __name__ == '__main__':
    unittest.main()
