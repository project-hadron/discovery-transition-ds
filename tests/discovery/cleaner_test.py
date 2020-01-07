import os
import shutil
import unittest

import numpy as np
import pandas as pd

from ds_behavioral import DataBuilderTools as tools
from ds_behavioral.sample.sample_data import ProfileSample

from ds_discovery import Transition
from ds_discovery.intent.pandas_cleaners import PandasCleaners as Cleaner



class CleanerTest(unittest.TestCase):
    """Test: """

    def setUp(self):
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
        Cleaner()

    def test_auto_remove(self):
        clean = Cleaner()
        df = pd.DataFrame()
        df['single_num'] = tools.get_number(1, 1, quantity=0.7, size=100)
        df['two_num'] = tools.get_number(1, 2, quantity=0.7, size=100)
        df['null_num'] = tools.get_number(1, 100, quantity=0, size=100)
        df['normal'] = tools.get_number(1, 100, size=100)
        df.loc[1:4, 'normal'] = 'None'
        df['none_num'] = tools.get_number(1, 1, quantity=0.7, size=100)
        df.loc[1:4, 'none_num'] = 'None'
        df.loc[7:9, 'none_num'] = ''
        clean.auto_remove_columns(df, nulls_list=True, inplace=True)
        self.assertEqual(['two_num', 'normal'], df.columns.tolist())

    def test_auto_remove_predom(self):
        clean = Cleaner()
        df = pd.DataFrame()
        df['single_num'] = tools.get_number(1, 1, size=100)
        df['two_num'] = tools.get_number(1, 2, size=100)
        df['weight_num'] = tools.get_number(1, 2, weight_pattern=[98, 1], size=100)
        df['null_num'] = tools.get_number(1, 100, quantity=0, size=100)
        df['normal_num'] = tools.get_number(1, 100, size=100)
        df['single_cat'] = tools.get_category(['A'], size=100)
        df['two_cat'] = tools.get_category(['A', 'B'], quantity=0.9, size=100)
        df['weight_cat'] = tools.get_category(['A', 'B'], weight_pattern=[95, 1], size=100)
        df['normal_cat'] = tools.get_category(list('ABCDE'), size=100)
        clean.auto_remove_columns(df, predominant_max=0.8, inplace=True)
        self.assertEqual(['two_num', 'normal_num', 'two_cat', 'normal_cat'], df.columns.tolist())

    def test_clean_headers(self):
        clean = Cleaner()
        df = tools.get_profiles()
        control = ['surname', 'forename', 'gender']
        result = df.columns
        self.assertTrue(control, result)
        rename = {'forename': 'first_name'}
        control = {'clean_header': {'case': 'title', 'rename': {'surname: last_name'}}}
        result = clean.auto_clean_header(df, rename_map=rename, case='title', inplace=True)
        self.assertTrue(control, result)
        control = ['Surname', 'First_Name', 'Gender']
        self.assertTrue(control, df.columns)

    def test_remove_columns(self):
        clean = Cleaner()
        df = tools.sample_dataset(size=10)
        result = clean.to_remove(df, headers=['surname'])
        self.assertNotIn('surname', result.columns.values)

        df = tools.sample_dataset(size=10)
        clean.to_remove(df, headers=['surname', 'gender'], inplace=True)
        self.assertNotIn('surname', df.columns.values)
        self.assertNotIn('gender', df.columns.values)

    def test_select_columns(self):
        clean = Cleaner()
        df = tools.sample_dataset(size=10)
        control = ['surname']
        result = clean.to_select(df, headers=['surname'])
        self.assertEqual(['surname'], result.columns.values)

        df = tools.sample_dataset(size=10)
        result = clean.to_select(df, headers=['surname', 'gender'], inplace=True)
        self.assertIn('surname', df.columns.values)
        self.assertIn('gender', df.columns.values)
        control = {'to_select': {'drop': False, 'headers': ['surname', 'gender']}}
        self.assertEqual(control, result)

    def test_contract_pipeline(self):
        clean = Cleaner()
        cleaner_cfg = {'auto_category': {'null_max': 0.7, 'unique_max': 120},
                       'clean_header': True
                       }
        df = pd.DataFrame()
        df['int'] = tools.get_number(100, size=10)
        df['float'] = tools.get_number(0, 1.0, size=10)
        df['object'] = tools.get_category(list('abcdef'), size=10)
        df['date'] = tools.get_datetime('01/01/2010', '01/01/2018', size=10)
        df['category'] = tools.get_category(list('vwxyz'), size=10)

        result = clean.run_contract_pipeline(df, cleaner_contract=cleaner_cfg)
        print(result)

    def test_to_float_with_mode(self):
        clean = Cleaner()
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
        clean = Cleaner()
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
        cleaner = Cleaner()
        df = pd.DataFrame()
        df['date'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', date_format='%d-%m-%Y', seed=101)
        df['datetime'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', date_format='%d-%m-%Y %H:%M:%S', seed=102)
        df['number'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', date_format='%Y%m%d.0', seed=101)
        df = cleaner.to_date_type(df, headers=['date', 'datetime'])
        self.assertEqual('10-08-2010', df['date'].iloc[0].strftime(format='%d-%m-%Y'))
        self.assertEqual('17-10-2007', df['datetime'].iloc[0].strftime(format='%d-%m-%Y'))
        df = cleaner.to_date_type(df, headers=['number'], date_format='%Y%m%d')
        self.assertEqual('08-10-2010', df['number'].iloc[0].strftime(format='%d-%m-%Y'))

        df['numtime'] = df['datetime']
        df = cleaner.to_date_type(df, headers=['numtime'], as_num=True)
        self.assertEqual(732966.241887, round(df['numtime'].iloc[0],6))

    def test_currency(self):
        cleaner = Cleaner()
        df = pd.DataFrame()
        df['currency'] = ['$3,320.12', '£1,001.34', '€34', 23.4, '5 220.12']
        df['control'] = [3320.12, 1001.34, 34, 23.4, 5220.12]
        df = cleaner.to_float_type(df, headers='currency')
        self.assertEqual(list(df.control), list(df.currency))

    def test_get_cols(self):
        cleaner = Cleaner()
        df = pd.DataFrame()
        df['int'] = tools.get_number(100, size=10)
        df['float'] = tools.get_number(0, 1.0, size=10)
        df['object'] = tools.get_category(list('abcdef'), size=10)
        df['date'] = tools.get_datetime('01/01/2010', '01/01/2018', size=10)
        df['category'] = tools.get_category(list('vwxyz'), size=10)
        df = cleaner.to_category_type(df, headers='category')
        control = ['float', 'object', 'date', 'category', 'int']
        result = cleaner.filter_headers(df)
        self.assertTrue(set(result).intersection(control))
        control = ['object']
        result = cleaner.filter_headers(df, dtype=[object])
        self.assertEqual(result, control)
        result = cleaner.filter_headers(df, dtype=['object'])
        self.assertEqual(result, control)
        control = ['date']
        result = cleaner.filter_headers(df, dtype=['datetime'])
        self.assertEqual(result, control)
        control = ['float', 'int']
        result = cleaner.filter_headers(df, dtype=['number'])
        self.assertTrue(set(result).intersection(control))

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

    def test_data_builder(self):
        sample_size = 1000
        df = tools.get_profiles(size=sample_size, mf_weighting=[5, 3])
        df['id'] = tools.unique_identifiers(from_value=10000, to_value=99999, prefix='CU_', size=sample_size)
        value_distribution = [0.01, 0.8, 1, 3, 9, 8, 3, 2, 1] + list(np.flip(np.exp(np.arange(-5, 0.0, 0.2)).round(2)))
        df['balance'] = tools.get_number(0.0, 1000, precision=2, weight_pattern=value_distribution, size=sample_size)
        age_pattern = [3, 5, 6, 10, 6, 5, 7, 15, 5, 2, 1, 0.5, 0.2, 0.1]
        df['age'] = tools.get_number(20.0, 90.0, weight_pattern=age_pattern, size=sample_size)
        df['start'] = tools.get_datetime(start='01/01/2018', until='31/12/2018', date_format='%m-%d-%y',
                                         size=sample_size)
        prof_pattern = [10, 8, 5, 4, 3, 2] + [1] * 9
        profession = ProfileSample.professions(size=15)
        df['profession'] = tools.get_category(selection=profession, weight_pattern=prof_pattern, quantity=0.7,
                                              size=sample_size)
        contract = {'clean_header': True,
                    'to_category': {'headers': ['profession', 'gender'], 'drop': False, 'exclude': False},
                    'to_date': {'headers': ['start'], 'drop': False, 'exclude': False},
                    'to_int': {'headers': ['age'], 'drop': False, 'exclude': False, 'fillna': 0}}
        df_clean = Cleaner.run_contract_pipeline(df, contract)
        print(df_clean)

    def test_filter(self):
        cleaner = Cleaner()
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
