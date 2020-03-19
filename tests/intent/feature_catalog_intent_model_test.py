import decimal
import os
import shutil
import unittest
from pprint import pprint

import numpy as np
import pandas as pd

from ds_behavioral import SyntheticBuilder
from ds_behavioral.sample.sample_data import ProfileSample
from ds_engines.engines.event_books.pandas_event_book import PandasEventBook

from ds_discovery.intent.feature_catalog_intent import FeatureCatalogIntentModel
from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager
from aistac.handlers.abstract_handlers import ConnectorContract


class FeatureCatalogIntentTest(unittest.TestCase):
    """Test: """

    def setUp(self):
        property_manager = FeatureCatalogPropertyManager('test')
        property_manager.set_property_connector(
            ConnectorContract(uri='contracts/property.pickle', handler='PandasPersistHandler',
                              module_name='ds_discovery.handlers.pandas_handlers'))
        property_manager.reset_intents()
        self.pm = property_manager
        self.intent = FeatureCatalogIntentModel(property_manager=property_manager)
        self.tools = SyntheticBuilder.from_env('tester', default_save=False).intent_model
        self.cc = ConnectorContract(uri='contracts/feature.pickle', handler='PandasPersistHandler', module_name='ds_discovery.handlers.pandas_handlers')
        try:
            os.makedirs('contracts')
        except:
            pass

    def tearDown(self):
        try:
            shutil.rmtree('contracts')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        FeatureCatalogIntentModel(property_manager=FeatureCatalogPropertyManager('test'), default_save_intent=False)

    def test_run_pipeline(self):
        df = pd.DataFrame()
        df['cu_id'] = self.tools.get_number(100000, 1000000, at_most=1, size=1000)
        df['age'] = self.tools.get_number(20, 90, weight_pattern=[5, 2, 4, 3, 2, 0.5, 0.1], size=1000)
        df['salary'] = self.tools.get_number(0, 100.0, weight_pattern=[10, 5, 3, 10], size=1000)
        self.pm.set_connector_contract(connector_name='First', connector_contract=self.cc)
        _ = self.intent.interval_categorical(df, key='cu_id', column='salary', granularity=[(0, 20), (80, 100)],
                                             precision=2, feature_name='First')
        self.intent.group_features(df, headers=['age', 'salary'], aggregator='sum', group_by=['cu_id'],
                                   feature_name='First', unindex= True, intent_order=1)
        self.pm.set_connector_contract(connector_name='Second', connector_contract=self.cc)
        _ = self.intent.interval_categorical(df, key='cu_id', column='age', granularity=[0.9, 0.1],
                                             categories=['younger', 'average', 'older'], precision=0,
                                             feature_name='Second')
        self.intent.run_intent_pipeline(df)


    def test_interval_categorical(self):
        df = pd.DataFrame()
        df['cu_id'] = self.tools.get_number(100000, 1000000, at_most=1, size=1000)
        df['age'] = self.tools.get_number(20, 90, weight_pattern=[5, 2, 4, 3, 2, 0.5, 0.1], size=1000)
        df['salary'] = self.tools.get_number(0, 100.0, weight_pattern=[10, 5, 3, 10], size=1000)
        result = self.intent.interval_categorical(df, key='cu_id', column='salary', granularity=[(0, 20), (80, 100)],
                                                  precision=2)
        self.assertEqual('category', result['salary_cat'].dtype.name)
        self.assertCountEqual(['<NA>', '80->100', '0->20'], result['salary_cat'].value_counts().index.to_list())
        result = self.intent.interval_categorical(df, key='cu_id', column='age', granularity=[0.9, 0.1],
                                                  categories=['younger', 'average', 'older'], precision=0)
        self.assertEqual('category', result['age_cat'].dtype.name)
        self.assertCountEqual(['younger', 'average', 'older'], result['age_cat'].value_counts().index.to_list())
        result = self.intent.interval_categorical(df, key='cu_id', column='age', label='age_gap', lower=0, upper=100)
        self.assertCountEqual(['33->66', '0->33', '66->100'], result['age_gap'].value_counts().index.to_list())

    def test_apply_condition(self):
        df = pd.DataFrame()
        df['cu_id'] = self.tools.get_number(100000, 1000000, at_most=1, size=20)
        df['genre'] = self.tools.get_category(
            selection=['Comedy', 'Drama', 'News and Information', 'Reality and Game Show', 'Undefined'], size=20)
        df['EndType'] = self.tools.get_category(
            selection=['Ad End', 'Ad Start', 'Undefined', 'Video End', 'Video Start'],
            weight_pattern=[1, 3, 1, 6, 2], size=20)
        result1 = self.intent.apply_where(df, key='cu_id', column='EndType', condition="== 'Video End'")
        self.assertEqual(1, result1['EndType'].nunique())
        eb = PandasEventBook('test_book')
        result2 = self.intent.run_intent_pipeline(df, event_book=eb)
        self.assertEqual(1, result2['EndType'].nunique())
        self.assertEqual(result1.shape, result2.shape)
        self.assertCountEqual(result1['genre'], result2['genre'])

    def test_group_features(self):
        df = pd.DataFrame()
        df['genre'] = ['Comedy', 'Drama', 'Drama', 'Drama', 'Undefined']
        df['end_type'] = ['Ad End', 'Ad Start', 'Ad End', 'Ad Start', 'Ad End']
        df['spend'] = [1, 3, 2, 4, 0]
        df['viewed'] = [1, 2, 1, 3, 1]
        result = self.intent.group_features(df, headers=['viewed', 'spend'], aggregator='nunique', group_by=['genre'],
                                            drop_group_by=True)
        print(result)

    def test_date_diff(self):
        df = pd.DataFrame()
        df['primary'] = self.tools.get_datetime(start='2000/01/01', until='2000/01/2', year_first=True, size=1000)
        df['secondary'] = self.tools.get_datetime(start='2000/02/01', until='2000/02/2', year_first=True, size=1000)
        result = self.intent.flatten_date_diff(df, first_date='primary', second_date='secondary', units='D',
                                               save_intent=False)
        self.assertEqual(30, result.min())
        self.assertEqual(32, result.max())
        result = self.intent.flatten_date_diff(df, first_date='primary', second_date='secondary', units='W',
                                               save_intent=False)
        self.assertEqual(4, result.min())
        self.assertEqual(5, result.max())
        # check nulls work
        df['primary'] = self.tools.get_datetime(start='2000/01/01', until='2000/01/2', year_first=True, quantity=0.1,
                                                size=1000)
        df['secondary'] = self.tools.get_datetime(start='2000/02/01', until='2000/02/2', year_first=True, quantity=0.5,
                                                  size=1000)
        result = self.intent.flatten_date_diff(df, first_date='primary', second_date='secondary', units='D',
                                               save_intent=False)

    def test_remove_outliers(self):
        df = pd.DataFrame()
        df['key'] = [1, 2, 3, 4, 5, 6]
        df['values'] = [10, 3, 1, 5, 6, 10]
        result = self.intent.remove_outliers(df, key='key', column='values', lower_quantile=0.1, upper_quantile=0.9)
        self.assertCountEqual([2, 4, 5], result.index.to_list())
        self.assertCountEqual([3, 5, 6], result['values'].to_list())

    def test_date_matrix(self):
        df = pd.DataFrame()
        df['cu_id'] = self.tools.get_number(1000, 10000, at_most=1, size=20)
        df['primary'] = self.tools.get_datetime(start='2000/01/01', until='2000/02/2', year_first=True, size=20)
        result = self.intent.date_categorical(df, key='cu_id', column='primary')
        self.assertEqual(10, result.shape[1])
        result = self.intent.date_categorical(df, key='cu_id', column='primary', matrix=['day', 'dow', 'woy', 'doy'])
        self.assertCountEqual(['primary_woy', 'primary_doy', 'primary_dow', 'primary_day'], result.columns)
        result = self.intent.date_categorical(df, key='cu_id', column='primary', matrix=['day', 'dow'], label='cat')
        self.assertCountEqual(['cat_dow', 'cat_day'], result.columns)

    def test_apply_selection(self):
        df = pd.DataFrame()
        df['key'] = [1, 2, 3, 4, 5, 6, 3, 5, 1]
        df['values'] = [10, 3, 1, 5, 6, 10, 2, 4, 5]
        conditions = [('< 5', '-1'), ('> 5', '1')]
        result = self.intent.apply_condition(df, key='key', column='values', conditions=conditions, default=0)
        print(result['values'].to_list())


if __name__ == '__main__':
    unittest.main()
