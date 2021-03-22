import os
import shutil
import unittest

import numpy as np
import pandas as pd
from aistac.properties.property_manager import PropertyManager
from ds_discovery import SyntheticBuilder

from ds_discovery.intent.feature_catalog_intent import FeatureCatalogIntentModelModel
from ds_discovery.managers import FeatureCatalogPropertyManager
from ds_discovery.components.feature_catalog import FeatureCatalog


class FeatureCatalogIntentTest(unittest.TestCase):
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
        self.fc: FeatureCatalogIntentModelModel = FeatureCatalog.scratch_pad()

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        FeatureCatalogIntentModelModel(property_manager=FeatureCatalogPropertyManager('test', username='UserTest'), default_save_intent=False)

    def test_run_pipeline(self):
        local_fc = FeatureCatalog.from_env('tester', default_save=False, has_contract=False)
        df = pd.DataFrame()
        df['cu_id'] = self.tools.get_number(100000, 1000000, at_most=1, size=1000)
        df['age'] = self.tools.get_number(20, 90, relative_freq=[5, 2, 4, 3, 2, 0.5, 0.1], size=1000)
        df['salary'] = self.tools.get_number(0, 100.0, relative_freq=[10, 5, 3, 10], size=1000)
        _ = local_fc.intent_model.interval_categorical(df, key='cu_id', column='salary', granularity=[(0, 20), (80, 100)], precision=2, feature_name='First')
        local_fc.intent_model.group_features(df, headers=['age', 'salary'], aggregator='sum', group_by=['cu_id'], feature_name='First', unindex= True, intent_order=1)
        _ = local_fc.intent_model.interval_categorical(df, key='cu_id', column='age', granularity=[0.9, 0.1], categories=['younger', 'average', 'older'], precision=0, feature_name='Second')
        _ = local_fc.intent_model.run_intent_pipeline(df, feature_name='First')
        _ = local_fc.intent_model.run_intent_pipeline(df, feature_name='Second')
        _ = local_fc.intent_model.run_intent_pipeline(df, feature_name='Second', train_size=0.75)
        _ = local_fc.intent_model.run_intent_pipeline(df, feature_name='Second', train_size=750)
        _ = local_fc.intent_model.run_intent_pipeline(df, feature_name='Second', train_size=0.75, shuffle=True)
        _ = local_fc.intent_model.run_intent_pipeline(df, feature_name='Second', train_size=750, shuffle=True)

    def test_run_pipeline_from_connector(self):
        catalog: FeatureCatalog = FeatureCatalog.from_env('merge', default_save=False, has_contract=False)
        catalog.set_catalog_feature('test')
        df = pd.DataFrame()
        df['key'] = list(range(10))
        df['values'] = [1,3,np.nan,5,4,1,3,np.nan,1,6]
        df['cats'] = list('AAC') + [np.nan] + list('CCBBAA')
        catalog.save_catalog_feature(feature_name='test', canonical=df)
        # DataFrame
        _ = catalog.intent_model.apply_missing(df, key='key', headers='values', feature_name='test')
        _ = catalog.pm.get_intent(level='test', intent='apply_missing')
        result = catalog.intent_model.run_intent_pipeline(df, feature_name='test')
        self.assertEqual((10,2),result.shape)
        # Connector
        _ = catalog.intent_model.apply_missing('test', key='key', headers='cats', feature_name='test')
        result = catalog.intent_model.run_intent_pipeline(df, feature_name='test')
        self.assertEqual((10,2),result.shape)

    def test_run_pipeline_intent_order(self):
        catalog: FeatureCatalog = FeatureCatalog.from_env('merge', default_save=False, has_contract=False)
        catalog.set_catalog_feature('test')
        df = pd.DataFrame()
        df['key'] = list(range(6))
        df['age'] = [23, 18, 47, 32, 29, 61]
        df['gender'] = ['M', 'F', np.nan, 'M', 'F', 'M']
        df['values'] = [1,3,5,1,2,3]
        df['cats'] = list('AACBCA')
        df['location'] = ['NY', 'PA', 'TX', 'IL', 'TX', 'CA']
        catalog.save_catalog_feature(feature_name='test', canonical=df)
        conditions_list = [self.fc.select2dict(column='age', condition="20", operator='>')]
        _ = catalog.intent_model.select_where(df, key='key', selection=conditions_list, intent_order=0, feature_name='test')
        _ = catalog.intent_model.apply_missing(df, key='key', headers='gender', unindex=True, intent_order=1, feature_name='test')
        result = catalog.intent_model.run_intent_pipeline(df, feature_name='test')

    def test_apply_merge(self):
        catalog: FeatureCatalog = FeatureCatalog.from_env('merge', has_contract=False)
        catalog.set_catalog_feature('test')
        merge_df = pd.DataFrame()
        merge_df['cu_id'] = [1,2,3,4,5,6]
        merge_df['age'] = [23, 18, 47, 32, 29, 61]
        merge_df['gender'] = ['M', 'F', 'M', 'M', 'F', 'M']
        catalog.save_catalog_feature(feature_name='test', canonical=merge_df)
        df = pd.DataFrame()
        df['cu_id'] = [1,2,4,6]
        df['location'] = ['NY', 'PA', 'TX', 'IL']
        result = catalog.intent_model.apply_merge(canonical=df, merge_connector='test', key='cu_id', on='cu_id', save_intent=False)
        self.assertEqual([1, 2, 4, 6], result.index.to_list())
        self.assertCountEqual(['location', 'age', 'gender'], result.columns.to_list())

    def test_date_diff(self):
        df = pd.DataFrame()
        df['key'] = range(1000)
        df['primary'] = self.tools.get_datetime(start='2000/01/01', until='2000/01/2', year_first=True, size=1000)
        df['secondary'] = self.tools.get_datetime(start='2000/02/01', until='2000/02/2', year_first=True, size=1000)
        result = self.fc.apply_date_diff(df, key='key', first_date='primary', second_date='secondary', units='D', save_intent=False)
        self.assertEqual([30], result.min().values)
        self.assertEqual([32], result.max().values)
        result = self.fc.apply_date_diff(df, key='key', first_date='primary', second_date='secondary', units='W', save_intent=False)
        self.assertEqual([4], result.min().values)
        self.assertEqual([5], result.max().values)
        # check nulls work
        df['primary'] = self.tools.get_datetime(start='2000/01/01', until='2000/01/2', year_first=True, quantity=0.1, size=1000)
        df['secondary'] = self.tools.get_datetime(start='2000/02/01', until='2000/02/2', year_first=True, quantity=0.5, size=1000)
        result = self.fc.apply_date_diff(df, key='key', first_date='primary', second_date='secondary', units='D', save_intent=False)
        self.assertEqual(1000, result.size)

    def test_interval_categorical(self):
        df = pd.DataFrame()
        df['cu_id'] = self.tools.get_number(100000, 1000000, at_most=1, size=1000)
        df['age'] = self.tools.get_number(20, 90, relative_freq=[5, 2, 4, 3, 2, 0.5, 0.1], size=1000)
        df['salary'] = self.tools.get_number(0, 100.0, relative_freq=[10, 5, 3, 10], size=1000)
        result = self.fc.interval_categorical(df, key='cu_id', column='salary', granularity=[(0, 20), (80, 100)],
                                                  precision=2)
        self.assertEqual('category', result['salary_cat'].dtype.name)
        self.assertCountEqual(['<NA>', '80->100', '0->20'], result['salary_cat'].value_counts().index.to_list())
        result = self.fc.interval_categorical(df, key='cu_id', column='age', granularity=[0.9, 0.1],
                                                  categories=['younger', 'average', 'older'], precision=0)
        self.assertEqual('category', result['age_cat'].dtype.name)
        self.assertCountEqual(['younger', 'average', 'older'], result['age_cat'].value_counts().index.to_list())
        result = self.fc.interval_categorical(df, key='cu_id', column='age', rename='age_gap', lower=0, upper=100)
        self.assertCountEqual(['33->66', '0->33', '66->100'], result['age_gap'].value_counts().index.to_list())

    def test_select_where(self):
        df = pd.DataFrame()
        df['cid'] = [1,2,3,4,5,6]
        df['starts'] = list('ABABCD')
        df['ends'] = list("XZZXYZ")
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='==')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([2,4], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='=='),
                           self.fc.select2dict(column='ends', condition=".str.contains('Z')")]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([2], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='=='),
                           self.fc.select2dict(column='ends', condition=".str.contains('Z')", logic='AND')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([2], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='=='),
                           self.fc.select2dict(column='ends', condition=".str.contains('Z')", logic='NOT')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([4], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='=='),
                           self.fc.select2dict(column='ends', condition=".str.contains('Z')", logic='OR')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([2,3,4,6], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='=='),
                           self.fc.select2dict(column='ends', condition=".str.contains('Z')", logic='XOR')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([3,4,6], result.index.tolist())
        dates = []
        for offset in range (-2, 4):
            dates.append(pd.Timestamp.now()-pd.Timedelta(days=offset))
        df['dates'] = dates
        conditions_list = [self.fc.select2dict(column='dates', condition="date.now", operator='>')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([1,2,3], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='dates', condition="date.now", operator='<=', offset=-2)]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([6], result.index.tolist())

    def test_group_features(self):
        df = pd.DataFrame()
        df['genre'] = ['Comedy', 'Drama', 'Drama', 'Drama', 'Undefined']
        df['end_type'] = ['Ad End', 'Ad Start', 'Ad End', 'Ad Start', 'Ad End']
        df['spend'] = [1, 3, 2, 4, 0]
        df['viewed'] = [1, 2, 1, 3, 1]
        result = self.fc.group_features(df, headers=['viewed', 'spend'], aggregator='nunique', group_by=['genre'],
                                            drop_group_by=True)
        print(result)

    def test_remove_outliers(self):
        df = pd.DataFrame()
        df['key'] = [1, 2, 3, 4, 5, 6]
        df['values'] = [10, 3, 1, 5, 6, 10]
        result = self.fc.remove_outliers(df, key='key', column='values', lower_quantile=0.1, upper_quantile=0.9)
        self.assertCountEqual([2, 4, 5], result.index.to_list())
        self.assertCountEqual([3, 5, 6], result['values'].to_list())

    def test_apply_condition(self):
        df = pd.DataFrame()
        df['key'] = [1, 2, 3, 4, 5, 6, 3, 5, 1]
        df['values'] = [10, 3, 1, 5, 6, 10, 2, 4, 5]
        conditions = [('< 5', '-1'), ('> 5', '1')]
        result = self.fc.apply_condition(df, key='key', header='values', conditions=conditions, default=0, save_intent=False)
        self.assertEqual(['1', '-1', '-1', '0', '1', '1', '-1', '-1', '0'], result['values'].to_list())

    def test_apply_map(self):
        df = pd.DataFrame()
        df['key'] = list(range(10))
        df['values'] = [1,3,2,5,4,1,3,2,1,6]
        result = self.fc.apply_map(df, key='key', header='values', value_map={1: "A", 2: "B", 3: "C"})
        self.assertEqual(['A', 'C', 'B', 'A', 'C', 'B', 'A'], result['values'].to_list())
        result = self.fc.apply_map(df, key='key', header='values', value_map={1: "A", 2: "B", 3: "C"}, default_to='Z')
        self.assertEqual(['A', 'C', 'B', 'Z', 'Z', 'A', 'C', 'B', 'A', 'Z'], result['values'].to_list())

    def test_apply_replace(self):
        df = pd.DataFrame()
        df['key'] = list(range(10))
        df['values'] = [1,3,2,5,4,1,3,2,1,6]
        result = self.fc.apply_replace(df, key='key', header='values', to_replace={1: 10, 2: 11})
        self.assertEqual([10, 3, 11, 5, 4, 10, 3, 11, 10, 6], result['values'].to_list())
        # nulls
        result = self.fc.apply_replace(df, key='key', header='values', to_replace={1: '$null'})
        self.assertEqual(7, result['values'].dropna().size)
        df['values'] = [np.nan, 3, 2, np.nan, 4, np.nan, 3, 2, 1, 6]
        result = self.fc.apply_replace(df, key='key', header='values', to_replace={'$null': 0})
        self.assertEqual([0, 3, 2, 0, 4, 0, 3, 2, 1, 6], result['values'].to_list())

    def test_apply_missing(self):
        df = pd.DataFrame()
        df['key'] = list(range(10))
        df['values'] = [1,3,np.nan,5,4,1,3,np.nan,1,6]
        df['cats'] = list('ABCDEFGHIJ')
        self.assertEqual(8, df['values'].dropna().size)
        result = self.fc.apply_missing(df, key='key', headers='values')
        self.assertEqual(10, result['values'].dropna().size)

    def test_apply_category_typing(self):
        df = pd.DataFrame()
        df['key'] = list(range(10))
        df['values'] = [1,3,4,5,4,1,3,2,1,6]
        df['cats'] = list('ABCAEDEABA')
        result = self.fc.apply_category_typing(df, key='key', header='cats')
        self.assertTrue(result['cats'].dtype.name, 'category')
        self.assertEqual(df['cats'].to_list(), result['cats'].to_list())
        result = self.fc.apply_category_typing(df, key='key', header='cats', as_num=True)
        self.assertTrue(result['cats'].dtype.name, int)
        self.assertEqual([0, 1, 2, 0, 4, 3, 4, 0, 1, 0], result['cats'].to_list())

    def test_apply_numeric_typing(self):
        df = pd.DataFrame()
        df['key'] = list(range(6))
        df['ints'] = [1, 8, np.nan, 1, 2, 4]
        df['floats'] = [1.1, 3.8, 1.1, 2.4, np.nan, 2.5]
        df['cats'] = list('ABCABA')
        result = self.fc.apply_numeric_typing(df, key='key', header='ints', precision=0)
        self.assertEqual([1, 8, 0, 1, 2, 4], result['ints'].to_list())
        result = self.fc.apply_numeric_typing(df, key='key', header='floats', fillna='mode')
        self.assertEqual([1.1, 3.8, 1.1, 2.4, 1.1, 2.5], result['floats'].to_list())

    def test_get_canonical(self):
        df = pd.DataFrame()
        value = self.fc._get_canonical(df)
        self.assertIsInstance(value, pd.DataFrame)
        self.assertEqual([], value.columns.to_list())
        df['cu_id'] = [1,2,3,4,5,6]
        df['age'] = [23, 18, 47, 32, 29, 61]
        df['gender'] = ['M', 'F', 'M', 'M', 'F', 'M']
        catalog: FeatureCatalog = FeatureCatalog.from_env('tester', has_contract=False)
        catalog.set_catalog_feature('data')
        catalog.save_catalog_feature(feature_name='data', canonical=df)
        value = catalog.intent_model._get_canonical('data')
        self.assertIsInstance(value, pd.DataFrame)
        self.assertEqual(['cu_id', 'age', 'gender'], value.columns.to_list())

    def test_set_intend_signature(self):
        catalog: FeatureCatalog = FeatureCatalog.from_env('merge', default_save=False, has_contract=False)
        catalog.set_catalog_feature('test')
        df = pd.DataFrame()
        df['key'] = list(range(10))
        df['values'] = [1,3,np.nan,5,4,1,3,np.nan,1,6]
        df['cats'] = list('ABCDEFGHIJ')
        catalog.save_catalog_feature(feature_name='test', canonical=df)
        # DataFrame
        _ = catalog.intent_model.apply_missing(df, key='key', headers='values', feature_name='test')
        result = catalog.pm.get_intent(level='test', intent='apply_missing')
        self.assertFalse('canonical' in result.get('apply_missing').keys())
        # Connector
        _ = catalog.intent_model.apply_missing('test', key='key', headers='values', feature_name='test')
        result = catalog.pm.get_intent(level='test', intent='apply_missing')
        self.assertTrue('canonical' in result.get('apply_missing').keys())
        self.assertEqual('test', result.get('apply_missing').get('canonical'))


if __name__ == '__main__':
    unittest.main()
