import os
import shutil
import unittest

import numpy as np
import pandas as pd

from ds_behavioral import SyntheticBuilder
from ds_behavioral.sample.sample_data import ProfileSample

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

    def test_apply_condition(self):
        df = pd.DataFrame()
        df['genre'] = self.tools.get_category( selection=['Comedy', 'Drama', 'News and Information', 'Reality and Game Show', 'Undefined'], size=20)
        df['EndType'] = self.tools.get_category(selection=['Ad End', 'Ad Start', 'Undefined', 'Video End', 'Video Start'],
                                           weight_pattern=[1, 3, 1, 6, 2], size=20)
        result = self.intent.apply_condition(df, headers='EndType', condition='== value', value='Video End')
        self.assertEqual(1, result['EndType'].nunique())

    def test_group_features(self):
        df = pd.DataFrame()
        df['genre'] = ['Comedy', 'Drama', 'Drama', 'Drama', 'Undefined']
        df['end_type'] = ['Ad End', 'Ad Start', 'Ad End', 'Ad Start', 'Ad End']
        df['spend'] = [1, 3, 2, 4, 0]
        df['viewed'] = [1, 2, 1, 3, 1]
        result = self.intent.group_features(df, headers=['viewed', 'spend'], aggregator='sum', group_by=['genre'], drop_group_by=True)
        print(result)




if __name__ == '__main__':
    unittest.main()
