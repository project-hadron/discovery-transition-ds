import unittest
import os
import shutil
from pathlib import Path

import pandas as pd
from pprint import pprint

from build.lib.ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from ds_discovery import Transition, Wrangle, Controller, SyntheticBuilder
from aistac.properties.property_manager import PropertyManager


class MyTestCase(unittest.TestCase):

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
        # Local Domain Contract
        os.environ['HADRON_PM_PATH'] = os.path.join('working', 'contracts')
        os.environ['HADRON_PM_TYPE'] = 'json'
        # Local Connectivity
        os.environ['HADRON_DEFAULT_PATH'] = Path('working/data').as_posix()
        # Specialist Component
        try:
            os.makedirs(os.environ['HADRON_PM_PATH'])
        except:
            pass
        try:
            os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        except:
            pass
        PropertyManager._remove_all()
        tr = Transition.from_env('task1', has_contract=False)
        tr.set_source_uri("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        tr.set_persist()
        wr = Wrangle.from_env('task2', has_contract=False)
        wr.set_source_uri(tr.get_persist_contract().raw_uri)
        wr.set_persist()
        controller = Controller.from_env(has_contract=False)
        controller.intent_model.transition(canonical=pd.DataFrame(), task_name='task1', intent_level='transition')
        controller.intent_model.wrangle(canonical=pd.DataFrame(), task_name='task2', intent_level='wrangle')

    def tearDown(self):
        try:
            shutil.rmtree('working')
        except:
            pass

    def test_model_iterator(self):
        builder = SyntheticBuilder.from_env(
            "test", default_save=False, default_save_intent=False, has_contract=False
        )
        tools: SyntheticIntentModel = builder.tools
        builder.add_connector_uri(
            "titanic",
            uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        )
        # do nothing
        result = tools.model_iterator(canonical="titanic")
        self.assertEqual(builder.load_canonical("titanic").shape, result.shape)
        # add marker
        result = tools.model_iterator(canonical="titanic", marker_col="marker")
        self.assertEqual(
            builder.load_canonical("titanic").shape[1] + 1, result.shape[1]
        )
        # with selection
        selection = [tools.select2dict(column="survived", condition="@==1")]
        control = tools.frame_selection(canonical="titanic", selection=selection)
        result = tools.model_iterator(
            canonical="titanic", marker_col="marker", selection=selection
        )
        self.assertEqual(control.shape[0], result.shape[0])
        # with iteration
        result = tools.model_iterator(
            canonical="titanic", marker_col="marker", iter_stop=3
        )
        self.assertCountEqual(
            [0, 1, 2], result["marker"].value_counts().index.to_list()
        )
        # with actions
        actions = {2: (tools.action2dict(method="get_category", selection=[4, 5]))}
        result = tools.model_iterator(
            canonical="titanic",
            marker_col="marker",
            iter_stop=3,
            iteration_actions=actions,
        )
        self.assertCountEqual(
            [0, 1, 4, 5], result["marker"].value_counts().index.to_list()
        )

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
