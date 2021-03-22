import unittest
import os
import shutil

from ds_discovery import SyntheticBuilder
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager

from ds_discovery.intent.controller_intent import ControllerIntentModel
from ds_discovery.managers.controller_property_manager import ControllerPropertyManager


class ControllerIntentTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # clean out any old environments
        for key in os.environ.keys():
            if key.startswith('HADRON'):
                del os.environ[key]

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

    @property
    def instance(self):
        pm = ControllerPropertyManager('test', username='TestUser')
        return ControllerIntentModel(property_manager=pm)

    @property
    def tools(self) -> SyntheticIntentModel:
        return SyntheticBuilder.scratch_pad()

    def test_smoke(self):
        """Basic smoke test"""
        ControllerIntentModel(property_manager=ControllerPropertyManager('test', username='TestUser'))

    def test_register_task(self):
        repo_uri = "https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/bundles/samples/hk_income_sample/contracts/"
        dc = self.instance
        synth_df = dc.synthetic_builder(task_name='hk_income', size=1000, uri_pm_repo=repo_uri, run_task=True)
        result = dc.transition(canonical=synth_df, task_name='hk_income', uri_pm_repo=repo_uri, run_task=True, intent_order=2)
        control = ['ref_id', 'industry', 'age-group', 'ethnicity', 'area', 'gender', 'district', 'salary']
        self.assertEqual(control, result.columns.to_list())
        self.assertEqual((1000, 8), result.shape)
        # Check the intent in the properties
        intent = dc._pm.get_intent()
        self.assertEqual(['0', '2'], list(intent.get('primary_swarm').keys()))
        self.assertEqual(['activate_synthetic'], list(intent.get('primary_swarm').get('0').keys()))
        self.assertEqual(['activate_transition'], list(intent.get('primary_swarm').get('2').keys()))
        # run the pipeline
        result = dc.run_intent_pipeline()
        control = ['ref_id', 'industry', 'age-group', 'ethnicity', 'area', 'gender', 'district', 'salary']
        self.assertEqual(control, result.columns.to_list())
        self.assertEqual((1000, 8), result.shape)

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
