import unittest
import os
import shutil
from pprint import pprint

import pandas as pd

from ds_discovery import SyntheticBuilder, Controller, Transition, Wrangle
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
            raise IOError('Unable to create directories')
        PropertyManager._remove_all()
        builder = SyntheticBuilder.from_env('task1', has_contract=False)
        builder.set_persist()
        builder.pm_persist()
        tr = Transition.from_env('task2', has_contract=False)
        tr.set_source_uri(builder.get_persist_contract().raw_uri)
        tr.set_persist()
        tr.pm_persist()
        wr = Wrangle.from_env('task3', has_contract=False)
        wr.set_source_uri(tr.get_persist_contract().raw_uri)
        wr.set_persist()
        wr.pm_persist()

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_register_task(self):
        controller = Controller.from_env(has_contract=False)
        controller.intent_model.synthetic_builder(canonical=100, task_name='task1', intent_level='task1_build')
        # Check the intent in the properties
        intent = controller.report_intent(stylise=False)
        self.assertEqual(['synthetic_builder'], intent['intent'].to_list())
        controller.intent_model.transition(canonical=pd.DataFrame(), task_name='task2', intent_level='task2_clean')
        intent = controller.report_intent(stylise=False)
        self.assertEqual(['synthetic_builder', 'transition'], intent['intent'].to_list())


    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
