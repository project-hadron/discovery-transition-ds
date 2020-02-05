import os
import shutil
import unittest

import numpy as np
import pandas as pd

from ds_behavioral import DataBuilderTools as tools
from ds_behavioral.sample.sample_data import ProfileSample

from ds_discovery.intent.feature_catalog_intent import FeatureCatalogIntentModel
from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager
from ds_foundation.handlers.abstract_handlers import ConnectorContract



class FeatureCatalogIntentTest(unittest.TestCase):
    """Test: """

    def setUp(self):
        property_manager = FeatureCatalogPropertyManager('test')
        property_manager.set_property_connector(ConnectorContract(uri='', handler='DummyPersistHandler',
                                                                  module_name='ds_foundation.handlers.dummy_handlers'))
        property_manager.remove_intent()
        self.clean = FeatureCatalogIntentModel(property_manager=property_manager)
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

    def test_auto_remove(self):
        df = pd.DataFrame()
        df['gendre'] = tools.get_category( selection=['Comedy', 'Drama', 'News and Information', 'Reality and Game Show', 'Undefined'], size=20)
        df['EndType'] = tools.get_category(selection=['Ad End', 'Ad Start', 'Undefined', 'Video End', 'Video Start'],
                                           weight_pattern=[1, 3, 1, 6, 2], size=20)
        result = self.clean.apply_condition(df, headers='EndType', condition='== value', value='Video End')
        self.assertEqual(1, result['EndType'].nunique())




if __name__ == '__main__':
    unittest.main()
