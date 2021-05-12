import unittest
import os
import shutil
import pandas as pd
from pprint import pprint

from ds_discovery import SyntheticBuilder, Transition
from ds_discovery.components.commons import Commons
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager

from ds_discovery import Controller


class ControllerTest(unittest.TestCase):

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
    def tools(self) -> SyntheticIntentModel:
        return SyntheticBuilder.scratch_pad()

    def test_smoke(self):
        """Basic smoke test"""
        Controller.from_env(has_contract=False)

    def test_run_controller(self):
        uri_pm_repo = "https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/contracts/factory/healthcare/"
        tr = Transition.from_env(task_name='members', uri_pm_repo=uri_pm_repo)
        controller = Controller.from_env(uri_pm_repo=uri_pm_repo)
        roadmap = [
            controller.runbook2dict(task='members_sim', source=1000),
            controller.runbook2dict(task='members_gen', source='members_sim', persist=True),
         ]
        controller.run_controller(run_book=roadmap)
        self.assertEqual(1000, tr.load_persist_canonical().shape[0])
        # test mod_task
        controller.run_controller(run_book=roadmap, mod_tasks={'members_sim': {'source': 10}})
        self.assertEqual(10, tr.load_persist_canonical().shape[0])

    def test_run_intent_pipeline(self):
        uri_pm_repo = "https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/contracts/factory/healthcare/"
        # os.environ['HADRON_PM_REPO'] = uri_pm_repo
        controller = Controller.from_env(uri_pm_repo=uri_pm_repo)
        result = controller.intent_model.run_intent_pipeline(canonical=100, intent_level='members_sim', controller_repo=uri_pm_repo)
        self.assertEqual(100, result.shape[0])

    def test_report_tasks(self):
        uri_pm_repo = "https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/contracts/factory/healthcare/"
        controller = Controller.from_env(uri_pm_repo=uri_pm_repo)
        result = controller.report_tasks(stylise=False)
        self.assertEqual(['level', 'order', 'component', 'task', 'parameters', 'creator'], list(result.columns))

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
