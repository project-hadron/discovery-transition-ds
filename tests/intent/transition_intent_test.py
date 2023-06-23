import os
import shutil
import unittest

import numpy as np
import pandas as pd
from sklearn import datasets
from aistac.properties.property_manager import PropertyManager

from ds_discovery import SyntheticBuilder

from ds_discovery import Transition
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from ds_discovery.intent.transition_intent import TransitionIntentModel
from ds_discovery.managers.transition_property_manager import TransitionPropertyManager
from ds_discovery.components.commons import Commons


class TransitionIntentModelTest(unittest.TestCase):
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
        TransitionIntentModel(property_manager=TransitionPropertyManager('test', creator='TestUser'), default_save_intent=False)

    def test_to_date_element(self):
        tools = self.tools
        df = pd.DataFrame()
        df['dates'] = tools.get_datetime("01/01/2018", "01/02/2018", day_first=False, size=5)
        result = self.clean.to_date_element(df.copy(), matrix=['yr', 'dec'])
        self.assertCountEqual(['dates', 'dates_yr', 'dates_dec'], result.columns.to_list())
        result = self.clean.to_date_element(df.copy(), matrix=[])
        self.assertCountEqual(['dates'], result.columns.to_list())

    def test_to_sample(self):
        tools = self.tools
        df = tools.model_sample_map(200, sample_map='us_zipcode')
        result = self.clean.to_sample(df.copy(), sample_size=100)
        self.assertEqual((100, 6), result.shape)
        self.assertCountEqual(df.iloc[0], result.iloc[0])
        result = self.clean.to_sample(df.copy(), sample_size=100, shuffle=True)
        self.assertEqual((100, 6), result.shape)
        self.assertNotEqual(df.iloc[0].to_list(), result.iloc[0].to_list())

        result = self.clean.to_sample(df.copy(), sample_size=0.2)
        self.assertEqual((40, 6), result.shape)
        self.assertCountEqual(df.iloc[0], result.iloc[0])
        result = self.clean.to_sample(df.copy(), sample_size=0.2, shuffle=True)
        self.assertEqual((40, 6), result.shape)
        self.assertNotEqual(df.iloc[0].to_list(), result.iloc[0].to_list())

    def test_auto_transition(self):
        sb = SyntheticBuilder.from_memory()
        tr = Transition.from_memory()
        df = sb.tools.model_synthetic_data_types(100)
        df = tr.tools.auto_transition(df.copy(), inc_category=True)
        self.assertTrue(df['bool'].dtype.name.startswith('bool'))
        self.assertTrue(df['cat'].dtype.name.startswith('cat'))
        self.assertTrue(df['date'].dtype.name.startswith('datetime'))
        self.assertTrue(df['int'].dtype.name.startswith('int'))
        self.assertTrue(df['num'].dtype.name.startswith('float'))
        self.assertTrue(df['binary'].dtype.name.startswith('object'))
        self.assertTrue(df['str'].dtype.name.startswith('string'))

    def test_auto_reinstate_nulls(self):
        tools = self.tools
        sample_size = 10
        df = pd.DataFrame()
        df['nums'] = tools.get_number(4, size=sample_size, seed=31)
        df['bool_num'] = tools.get_category([1, 0, ' '], size=sample_size, seed=31)
        df['cats'] = tools.get_category(list('ABC '), size=sample_size, seed=31)
        result = self.clean.auto_reinstate_nulls(df.copy())
        self.assertEqual(['C', ' ', 'B', 'A', 'A', 'C', 'C', 'A', 'A', ' '], df['cats'].to_list())
        self.assertEqual(['C', 'B', 'A', 'A', 'C', 'C', 'A', 'A'], result['cats'].dropna().to_list())
        self.assertEqual([' ', ' ', 1, 1, 1, ' ', 0, 1, 1, ' '], df['bool_num'].to_list())
        self.assertEqual([1, 1, 1,  0, 1, 1], result['bool_num'].dropna().to_list())
        result = self.clean.auto_reinstate_nulls(df.copy(),  headers='cats')
        self.assertEqual(['C', 'B', 'A', 'A', 'C', 'C', 'A', 'A'], result['cats'].dropna().to_list())
        self.assertEqual([' ', ' ', 1, 1, 1, ' ', 0, 1, 1, ' '], result['bool_num'].dropna().to_list())
        result = self.clean.auto_reinstate_nulls(df.copy(), nulls_list=[0], headers='nums')
        self.assertEqual([1, 0, 2, 2, 1, 2, 1, 2, 0, 3], df['nums'].to_list())
        self.assertEqual([1, 2, 2, 1, 2, 1, 2, 3], result['nums'].dropna().to_list())

    def test_auto_projection(self):
        tr = Transition.from_memory()
        sb = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = sb.tools
        df = tools.model_synthetic_data_types(1000, True).iloc[:,:11]
        result = tr.tools.auto_projection(df.copy(), n_components=2)
        control = ['cat', 'date', 'str', 'binary', 'pca_A', 'pca_B']
        self.assertEqual(control, result.columns.to_list())
        self.assertEqual((1000, 6), result.shape)

    def test_auto_clean_header(self):
        tr = Transition.from_memory()
        df = pd.DataFrame(data={"A": list("ABCDEFG"), "B": list("ABCFCBA"), 'C': list("BCDECFB")})
        # save map
        mapper = {'A': 'X', 'B': 'Y', 'F': 'Z'}
        # test
        result = tr.tools.auto_clean_header(df.copy(), rename_map=mapper)
        self.assertEqual(['X', 'Y', 'C'], result.columns.to_list())
        result = tr.tools.auto_clean_header(df.copy(), rename_map=mapper, case='lower')
        self.assertEqual(['x', 'y', 'c'], result.columns.to_list())

    def test_auto_clean_header_list(self):
        tr = Transition.from_memory()
        df = pd.DataFrame(data={"A": list("ABCDEFG"), "B": list("ABCFCBA"), 'C': list("BCDECFB")})
        # save map
        mapper = list('XYZA')
        result = tr.tools.auto_clean_header(df.copy(), rename_map=mapper)
        self.assertEqual(list('ABC'), result.columns.to_list())
        mapper = list('XYZ')
        result = tr.tools.auto_clean_header(df.copy(), rename_map=mapper)
        self.assertEqual(list('XYZ'), result.columns.to_list())


    def test_auto_clean_header_connector(self):
        tr = Transition.from_memory()
        df = pd.DataFrame(data={"A": list("ABCDEFG"), "B": list("ABCFCBA"), 'C': list("BCDECFB")})
        connector_name = 'other'
        tr.add_connector_persist(connector_name, 'hadron_other.csv')
        # dict
        header_map = pd.DataFrame({'master': list('ABF'), 'other': list('XYZ')})
        handler = tr.pm.get_connector_handler(connector_name)
        handler.persist_canonical(header_map)
        result = tr.tools.auto_clean_header(df.copy(), rename_map='other')
        self.assertEqual(['X', 'Y', 'C'], result.columns.to_list())
        # list
        header_map = pd.DataFrame({'other': list('XYZ')})
        handler = tr.pm.get_connector_handler(connector_name)
        handler.persist_canonical(header_map)
        result = tr.tools.auto_clean_header(df.copy(), rename_map='other')
        self.assertEqual(['X', 'Y', 'Z'], result.columns.to_list())
        # no file
        result = tr.tools.auto_clean_header(df.copy(), rename_map='empty')
        self.assertEqual(['A', 'B', 'C'], result.columns.to_list())


    def test_auto_drop_duplicates(self):
        tools = self.tools
        df = pd.DataFrame()
        df['single_num'] = tools.get_number(1, 2, quantity=0.7, size=100)
        df['two_num'] = tools.get_number(2, quantity=0.7, size=100)
        df['normal_cat'] = tools.get_category(list('ABCDE'), size=100)
        df['normal'] = tools.get_number(1, 100, size=100)
        df['dup1'] = df['normal']
        df['dup2'] = df['normal']
        df['dup3'] = df['single_num']
        self.assertEqual((100, 7), df.shape)
        result = self.clean.auto_drop_duplicates(df.copy())
        self.assertEqual((100, 4), result.shape)

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
        df = self.clean.auto_drop_columns(df.copy(), nulls_list=True)
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
        result = self.clean.auto_drop_columns(df.copy(), predominant_max=0.8)
        self.assertEqual(['two_num', 'normal_num', 'two_cat', 'normal_cat'], result.columns.tolist())
        result = self.clean.auto_drop_columns(df.copy(), drop_predominant=False)
        self.assertEqual(8, len(result.columns))
        self.assertNotIn('null_num', result.columns.tolist())

    def test_auto_remove_unknown(self):
        sb = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = sb.tools
        df = tools.model_synthetic_data_types(1000, extended=True)
        prev = df.columns.to_list()
        result = self.clean.auto_drop_columns(df.copy(), null_min=0.999, drop_unknown=True, drop_predominant=False)
        new = result.columns.to_list()
        self.assertCountEqual(['binary', 'nulls'], Commons.list_diff(new, prev))

    def test_clean_headers(self):
        tools = self.tools
        df = pd.DataFrame(tools.model_sample_map(0, sample_map='us_zipcode'))
        control = ['city', 'state', 'Zipcode']
        result = df.columns
        self.assertTrue(control, result)
        rename = {'city': 'town'}
        control = {'clean_header': {'case': 'title', 'rename': {'city': 'town'}}}
        result = self.clean.auto_clean_header(df.copy(), rename_map=rename, case='title')
        self.assertTrue(control, result)
        control = ['town', 'state', 'Zipcode']
        self.assertTrue(control, df.columns)

    def test_remove_columns(self):
        tools = self.tools
        clean = self.clean
        df = pd.DataFrame(tools.model_sample_map(0, sample_map='us_zipcode'))
        result = clean.to_remove(df.copy(), headers=['city'])
        self.assertNotIn('city', result.columns.values)

        df = pd.DataFrame(tools.model_sample_map(0, sample_map='us_zipcode'))
        df = clean.to_remove(df.copy(), headers=['city', 'state'])
        self.assertNotIn('city', df.columns.values)
        self.assertNotIn('state', df.columns.values)

    def test_select_columns(self):
        tools = self.tools
        clean = self.clean
        df = pd.DataFrame(tools.model_sample_map(0, sample_map='us_zipcode'))
        control = ['city']
        result = clean.to_select(df.copy(), headers=['city'])
        self.assertEqual(['city'], result.columns.values)

        df = pd.DataFrame(tools.model_sample_map(0, sample_map='us_zipcode').iloc[:10])
        df = clean.to_select(df.copy(), headers=['city', 'state'])
        self.assertIn('city', df.columns.values)
        self.assertIn('state', df.columns.values)
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
        result = clean.run_intent_pipeline(df.copy())
        self.assertTrue(result.equals(df))
        # set intent
        self.assertEqual((10,5), df.shape)
        control = self.clean.to_remove(df.copy(), headers=['date'], save_intent=True)
        self.assertEqual((10,4), control.shape)
        control = self.clean.to_category_type(control, headers=['category'], save_intent=True)
        self.assertEqual('category', control['category'].dtype)
        # run pipeline
        result = clean.run_intent_pipeline(df.copy())
        self.assertTrue(result.equals(control))

    def test_to_bool_type(self):
        tr = Transition.from_memory()
        df = pd.DataFrame()
        df['bool'] = [False, True, False, True, None]
        df['two_num'] = [1,1,0,0,1]
        result = tr.tools.to_bool_type(df.copy())
        self.assertEqual([False, True, False, True, False], result['bool'].to_list())
        self.assertEqual([True, True, False, False, True], result['two_num'].to_list())


    def test_to_float_with_mode(self):
        tools = self.tools
        clean = self.clean
        col = 'X'
        df = pd.DataFrame()
        df[col] = tools.get_number(5.0, precision=5, size=5, seed=101)
        df.loc[[2, 4], col] = np.nan
        control = df.copy()
        mode = df[df[col].notna()]
        result = clean.to_float_type(df.copy(), headers='X', errors='coerce', fillna='mode')
        self.assertEqual(control.iloc[0,0], result.iloc[0,0])
        self.assertEqual(control.iloc[3,0], result.iloc[3,0])

    def test_to_float_type(self):
        tools = self.tools
        clean = self.clean
        col = 'X'
        df = pd.DataFrame()
        df['X'] = tools.get_number(5.0, precision=10, size=7, seed=101)
        df['Y'] = tools.get_number(5, precision=10, size=7, seed=101)
        df['Z'] = list('1234567')

        df.loc[[2, 4], 'X'] = np.nan
        control = df.copy()
        result = clean.to_float_type(df.copy(), headers=['X','Y','Z'], errors='coerce', fillna=-1, precision=3)
        print(result)
        # self.assertEqual(-1, result.iloc[1,0])
        # self.assertEqual(-1, result.iloc[2,0])
        # self.assertEqual(control.iloc[0,0], result.iloc[0,0])
        # self.assertEqual(control.iloc[3,0], result.iloc[3,0])

    def test_to_list_type(self):
        clean = self.clean
        df = pd.DataFrame()
        df['X'] = ["['A', 'B']", "[1, 2, 3]", "['Fred', 'Jim']", "", " ", 'bob', 23, np.nan]
        df['X'] = clean.to_list_type(df.copy(), headers='X')
        self.assertEqual(['A', 'B'], df['X'][0])
        self.assertEqual([1, 2, 3], df['X'][1])
        self.assertEqual([], df['X'][3])
        self.assertEqual([], df['X'][4])
        self.assertEqual(['bob'], df['X'][5])
        self.assertEqual([23], df['X'][6])
        self.assertEqual([], df['X'][7])
        # test it is idempotent
        df['X'] = clean.to_list_type(df.copy(), headers='X')
        self.assertEqual(['A', 'B'], df['X'][0])
        self.assertEqual([1, 2, 3], df['X'][1])

    def test_to_str_type(self):
        clean = self.clean
        df = pd.DataFrame()
        df['X'] = [1,2,3,4]
        df = clean.to_str_type(df.copy(), headers='X')
        self.assertCountEqual(['1','2','3','4'], df['X'])
        df['Y'] = ['ABC', 'ABCD', '', 'A']
        df = clean.to_str_type(df.copy(), headers='Y', fixed_len_pad='0')
        self.assertCountEqual(['0ABC', 'ABCD', np.nan, '000A'], df['Y'])

    def test_make_list(self):
        for value in ['', 0, 0.0, pd.Timestamp(2018,1,1), [], (), pd.Series(dtype=str), list(), tuple(),
                      'name', ['list1', 'list2'], ('tuple1', 'tuple2'), pd.Series(['series1', 'series2']),
                      {'key1': 'value1', 'key2': 'value2'}, {}, dict()]:
            result = Commons.list_formatter(value)
            self.assertTrue(isinstance(result, list), value)
        self.assertEqual([], Commons.list_formatter(None))

    def test_model_custom(self):
        tools = self.tools
        df = pd.DataFrame()
        df['gender'] = ['M', 'F', 'U']
        code_str ='''
            \n@['new_gender'] = [True if x in $value else False for x in @[$header]]
            \n@['value'] = [4, 5, 6]
        '''
        result = tools.model_custom(canonical=df, code_str=code_str, header='"gender"', value=['M', 'F'])
        print(result)

    def test_auto_to_date(self):
        df = SyntheticBuilder.from_memory().tools.model_synthetic_data_types(100, extended=False)
        self.assertEqual([np.dtype('O'), np.dtype('float64'), np.dtype('int64'), np.dtype('int64'), np.dtype('O'), np.dtype('O'), np.dtype('O')], df.dtypes.values.tolist())
        tr = Transition.from_memory()
        df = tr.tools.auto_to_date(df.copy())
        self.assertEqual([np.dtype('O'), np.dtype('float64'), np.dtype('int64'), np.dtype('int64'), np.dtype('datetime64[ns]'), np.dtype('O'), np.dtype('O')], df.dtypes.values.tolist())
        df = SyntheticBuilder.from_memory().tools.model_synthetic_data_types(100, extended=False, seed=31)
        df = tr.tools.auto_to_date(df.copy(), iso_format=True)
        self.assertEqual(['2023-02-27T02:21:01'], df['date'].iloc[:1].values.tolist())
        df = SyntheticBuilder.from_memory().tools.model_synthetic_data_types(100, extended=True)
        df = tr.tools.auto_to_date(df.copy())
        self.assertCountEqual(['date', 'date_tz', 'dup_date', 'date_null'], Commons.filter_headers(df, dtype='datetime64[ns]'))


    def test_to_date(self):
        tools = self.tools
        cleaner = self.clean
        df = pd.DataFrame()
        df['date'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', day_first=True, size=1, date_format='%d-%m-%Y', seed=101)
        df['datetime'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', day_first=True, size=1, date_format='%d-%m-%Y %H:%M:%S', seed=102)
        df['number'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', day_first=True, size=1, date_format='%Y%m%d', seed=101)
        result = cleaner.to_date_type(df.copy(), day_first=True, headers=['date', 'datetime'])
        self.assertEqual(df['date'].iloc[0], result['date'].iloc[0].strftime(format='%d-%m-%Y'))
        self.assertEqual(df['datetime'].iloc[0], result['datetime'].iloc[0].strftime(format='%d-%m-%Y %H:%M:%S'))

    def test_to_date_tz(self):
        tools = self.tools
        cleaner = self.clean
        df = pd.DataFrame()
        df['date'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', size=5, seed=101)
        df['datetime'] = tools.get_datetime(start='10/01/2000', until='01/01/2018', size=5, seed=102)
        result = cleaner.to_date_type(df.copy(), headers=['date', 'datetime'])
        self.assertIsNone(result['date'].dt.tz)
        result = cleaner.to_date_type(df.copy(), headers=['date'], timezone='UTC')
        self.assertIsNotNone(result['date'].dt.tz)
        result = cleaner.to_date_type(df.copy(), headers=['numtime'])
        self.assertIsNone(result['date'].dt.tz)



    def test_get_cols(self):
        tools = self.tools
        cleaner = self.clean
        df = pd.DataFrame()
        df['int'] = tools.get_number(100, size=10)
        df['float'] = tools.get_number(0, 1.0, size=10)
        df['object'] = tools.get_category(list('abcdef'), size=10)
        df['date'] = tools.get_datetime('01/01/2010', '01/01/2018', size=10)
        df['category'] = tools.get_category(list('vwxyz'), size=10)
        df = cleaner.to_category_type(df.copy(), headers='category')
        df = cleaner.to_date_type(df.copy(), headers='date')
        control = ['float', 'object', 'date', 'category', 'int']
        result = Commons.filter_headers(df.copy())
        self.assertTrue(set(result).intersection(control))
        control = ['object']
        result = Commons.filter_headers(df.copy(), dtype=[object])
        self.assertEqual(control, result)
        result = Commons.filter_headers(df.copy(), dtype=['object'])
        self.assertEqual(control, result)
        control = ['float', 'int']
        result = Commons.filter_headers(df.copy(), dtype=['number'])
        self.assertTrue(set(result).intersection(control))
        control = ['date']
        result = Commons.filter_headers(df.copy(), dtype=['datetime'])
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
