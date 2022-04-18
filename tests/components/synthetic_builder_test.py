import unittest
import os
import shutil
import pandas as pd
from ds_discovery import SyntheticBuilder
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager


class SyntheticBuilderTest(unittest.TestCase):

    def setUp(self):
        os.environ['HADRON_PM_PATH'] = os.path.join('work', 'config')
        os.environ['HADRON_DEFAULT_PATH'] = os.path.join('work', 'data')
        try:
            os.makedirs(os.environ['HADRON_PM_PATH'])
            os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        except:
            pass
        PropertyManager._remove_all()

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        self.assertEqual(SyntheticBuilder, type(SyntheticBuilder.from_env('tester', has_contract=False)))

    def test_run_synthetic_pipeline_seed(self):
        builder = SyntheticBuilder.from_env('tester', has_contract=False)
        builder.set_persist()
        tools: SyntheticIntentModel = builder.tools
        _ = tools.get_category(selection=['M', 'F'], relative_freq=[4, 3], column_name='gender')
        _ = tools.get_number(from_value=18, to_value=80, column_name='age')
        builder.run_component_pipeline(size=1000, seed=23)
        df = builder.load_persist_canonical()
        dist = df['gender'].value_counts().values
        mean = df['age'].mean()
        builder.run_component_pipeline(size=1000, seed=23)
        df = builder.load_persist_canonical()
        self.assertCountEqual(dist, df['gender'].value_counts().values)
        self.assertEqual(mean, df['age'].mean())

    def test_set_report_persist(self):
        builder = SyntheticBuilder.from_env('tester', default_save=False, has_contract=False)
        builder.setup_bootstrap(domain='domain', project_name='project_name', path=None)
        report = builder.report_connectors(stylise=False)
        _, file = os.path.split(report.uri.iloc[-1])
        self.assertTrue(file.startswith('project_name'))

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
