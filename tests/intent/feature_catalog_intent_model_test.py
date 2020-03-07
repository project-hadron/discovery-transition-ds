import decimal
import os
import shutil
import unittest

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
        property_manager.set_property_connector(ConnectorContract(uri='data/dummyfile.pickle', handler='DummyPersistHandler',
                                                                  module_name='aistac.handlers.dummy_handlers'))
        property_manager.reset_intents()
        self.intent = FeatureCatalogIntentModel(property_manager=property_manager)
        self.tools = SyntheticBuilder.from_env('testing', default_save=False).intent_model
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
        FeatureCatalogIntentModel(property_manager=FeatureCatalogPropertyManager('test'), default_save_intent=False)

    def test_interval_categorical(self):
        df = pd.DataFrame()
        df['age'] = self.tools.get_number(20, 90, weight_pattern=[1,2,4,3,2,0.5,0.1], size=1000)
        df['salary'] = self.tools.get_number(0, 100.0, weight_pattern=[30,5,0.5,0.1,0.05], size=1000)
        result = self.intent.interval_categorical(df, headers='salary', granularity=3)
        print(result)

    def test_apply_condition(self):
        df = pd.DataFrame()
        df['genre'] = self.tools.get_category( selection=['Comedy', 'Drama', 'News and Information', 'Reality and Game Show', 'Undefined'], size=20)
        df['EndType'] = self.tools.get_category(selection=['Ad End', 'Ad Start', 'Undefined', 'Video End', 'Video Start'],
                                                weight_pattern=[1, 3, 1, 6, 2], size=20)
        result1 = self.intent.apply_condition(df, headers='EndType', condition="== 'Video End'")
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
        result1 = self.intent.group_features(df, headers=['viewed', 'spend'], aggregator='sum', group_by=['genre'])
        eb = PandasEventBook('test_book')
        result2 = self.intent.run_intent_pipeline(df, event_book=eb)
        self.assertCountEqual(['Comedy', 'Drama', 'Undefined'], list(result1.index))
        self.assertCountEqual(['Comedy', 'Drama', 'Undefined'], list(result2.index))
        self.assertEqual(result1.shape, result2.shape)

    def test_date_diff(self):
        df = pd.DataFrame()
        df['primary'] = self.tools.get_datetime(start='2000/01/01', until='2000/01/2', year_first=True, size=1000)
        df['secondary'] = self.tools.get_datetime(start='2000/02/01', until='2000/02/2', year_first=True, size=1000)
        result = self.intent.flatten_date_diff(df, first_date='primary', second_date='secondary', units='D', save_intent=False)
        self.assertEqual(30, result.min())
        self.assertEqual(32, result.max())
        result = self.intent.flatten_date_diff(df, first_date='primary', second_date='secondary', units='W', save_intent=False)
        self.assertEqual(4, result.min())
        self.assertEqual(5, result.max())
        # check nulls work
        df['primary'] = self.tools.get_datetime(start='2000/01/01', until='2000/01/2', year_first=True, quantity=0.1, size=1000)
        df['secondary'] = self.tools.get_datetime(start='2000/02/01', until='2000/02/2', year_first=True, quantity=0.5, size=1000)
        result = self.intent.flatten_date_diff(df, first_date='primary', second_date='secondary', units='D', save_intent=False)





if __name__ == '__main__':
    unittest.main()
