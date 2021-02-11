import os
import shutil
import unittest

import pandas as pd
from aistac.properties.property_manager import PropertyManager
from ds_discovery import SyntheticBuilder

from ds_discovery.components.feature_catalog import FeatureCatalog


class MyTestCase(unittest.TestCase):

    def setUp(self):
        os.environ['HADRON_PM_PATH'] = os.path.join('work', 'config')
        os.environ['HADRON_DEFAULT_PATH'] = os.path.join('work', 'data')
        try:
            os.makedirs(os.environ['HADRON_PM_PATH'])
            os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        except:
            pass
        PropertyManager._remove_all()
        self.tools = SyntheticBuilder.from_env('tester', default_save=False, default_save_intent=False, has_contract=False).intent_model
        self.fc: FeatureCatalog = FeatureCatalog.from_env('tester', default_save=False, has_contract=False)

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        pass

    def test_run_feature_pipeline(self):
        df = pd.DataFrame()
        df['cu_id'] = self.tools.get_number(100000, 1000000, at_most=1, size=1000)
        df['age'] = self.tools.get_number(20, 90, relative_freq=[5, 2, 4, 3, 2, 0.5, 0.1], size=1000)
        df['salary'] = self.tools.get_number(0, 100.0, relative_freq=[10, 5, 3, 10], size=1000)
        _ = self.fc.intent_model.interval_categorical(df, key='cu_id', column='salary', granularity=[(0, 20), (80, 100)],
                                                      precision=2, feature_name='salary_cat')
        _ = self.fc.intent_model.interval_categorical(df, key='cu_id', column='age', granularity=10.0,
                                                      precision=0, feature_name='age_cat')
        self.fc.run_feature_pipeline(df)
        result = list(self.fc.pm.get_intent().keys())
        print(result)

    def test_report_feature(self):
        df = pd.DataFrame()
        df['cu_id'] = self.tools.get_number(100000, 1000000, at_most=1, size=1000)
        df['age'] = self.tools.get_number(20, 90, relative_freq=[5, 2, 4, 3, 2, 0.5, 0.1], size=1000)
        df['salary'] = self.tools.get_number(0, 100.0, relative_freq=[10, 5, 3, 10], size=1000)
        _ = self.fc.intent_model.interval_categorical(df, key='cu_id', column='salary', granularity=[(0, 20), (80, 100)],
                                                      precision=2, feature_name='salary_cat')
        self.fc.add_feature_description(feature_name='salary_cat', description='')
        _ = self.fc.intent_model.interval_categorical(df, key='cu_id', column='age', granularity=10.0,
                                                      precision=0, feature_name='age_cat')
        result = self.fc.report_feature_catalog(stylise=False)
        print(result)


if __name__ == '__main__':
    unittest.main()
