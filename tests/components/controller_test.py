import unittest
import os
import shutil
import pandas as pd
import numpy as np
from pprint import pprint

from ds_discovery import SyntheticBuilder, Transition, Wrangle
from ds_discovery.components.commons import Commons
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager

from ds_discovery import Controller


class ControllerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pd.set_option('display.width', 300)
        np.set_printoptions(linewidth=300)
        pd.set_option('display.max_columns', 10)

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
        tr = Transition.from_env('task1', has_contract=False)
        tr.set_source_uri("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        tr.set_persist()
        wr = Wrangle.from_env('task2' , has_contract=False)
        wr.set_source_uri(tr.get_persist_contract().raw_uri)
        wr.set_persist()
        controller = Controller.from_env(has_contract=False)
        controller.intent_model.transition(canonical=pd.DataFrame(), task_name='task1', intent_level='task1_tr')
        controller.intent_model.wrangle(canonical=pd.DataFrame(), task_name='task2', intent_level='task2_wr')

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_smoke(self):
        """Basic smoke test"""
        Controller.from_env(has_contract=False)

    def test_run_controller(self):
        controller = Controller.from_env()
        # test errors
        with self.assertRaises(ValueError) as context:
            controller.run_controller(run_book='noname')
        self.assertTrue("The run book or intent level" in str(context.exception))
        controller.run_controller()
        self.assertEqual(['hadron_wrangle_task2_primary_persist.pickle'], os.listdir('work/data/'))

    def test_repeat_iterations(self):
        wr = Wrangle.from_env('task2')
        wr.set_persist(wr.pm.file_pattern(name='tester', prefix='result1_', file_type='parquet'))
        controller = Controller.from_env()
        controller.run_controller(repeat=1)
        self.assertEqual(['result1_tester.parquet'], os.listdir('work/data/'))
        shutil.rmtree('work/data')
        os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        wr.set_persist(wr.pm.file_pattern(name='tester', prefix='result1_', file_type='parquet', stamped='ns'))
        controller.run_controller(repeat=3)
        self.assertEqual(3, len(os.listdir('work/data/')))
        shutil.rmtree('work/data')
        os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        controller.run_controller(repeat=2, sleep=1)
        self.assertEqual(2, len(os.listdir('work/data/')))
        shutil.rmtree('work/data')
        os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        controller.run_controller(repeat=2, sleep=1, run_time=4)
        self.assertEqual(4, len(os.listdir('work/data/')))

    def test_synthetic_with_no_source(self):
        shutil.rmtree('work/config')
        os.makedirs(os.environ['HADRON_PM_PATH'])
        PropertyManager._remove_all()
        builder = SyntheticBuilder.from_env('task3', has_contract=False)
        tools: SyntheticIntentModel = builder.tools
        builder.set_persist()
        df = pd.DataFrame(index=range(10))
        tools.model_noise(df, num_columns=5, column_name='noise')
        controller = Controller.from_env(has_contract=False)
        controller.intent_model.synthetic_builder(df, task_name='task3')
        controller.run_controller()
        self.assertIn(builder.CONNECTOR_PERSIST, builder.report_connectors(stylise=False)['connector_name'].to_list())
        self.assertNotIn(builder.CONNECTOR_SOURCE, builder.report_connectors(stylise=False)['connector_name'].to_list())

    def test_controller_log(self):
        controller = Controller.from_env()
        controller.run_controller(run_cycle_report='report.csv')
        df = controller.load_canonical(connector_name='run_cycle_report')
        control = ['start run-cycle 0', 'start run count 0', 'running task1_tr', 'canonical shape is (891, 15)',
                   'running task2_wr', 'canonical shape is (891, 15)', 'tasks complete', 'end of report']
        self.assertEqual(control, df['text'].to_list())
        controller.run_controller(run_time=3, run_cycle_report='report.csv')
        df = controller.load_canonical(connector_name='run_cycle_report')
        self.assertEqual((15, 2), df.shape)

    def test_controller_check_changed(self):
        tr = Transition.from_env('task1')
        tr.set_source(uri_file='sample.csv')
        tr.set_persist(uri_file='sample.csv')
        df = pd.DataFrame({'A': [1,2,3,4]})
        tr.save_persist_canonical(df)
        controller = Controller.from_env()
        controller.run_controller(repeat=2, source_check_uri=tr.get_persist_contract().raw_uri, run_cycle_report='report.csv')
        df = controller.load_canonical(connector_name='run_cycle_report')
        control = ['start run-cycle 0', 'start run count 0', 'running task1_tr', 'canonical shape is (4, 1)',
                   'running task2_wr', 'canonical shape is (4, 1)', 'tasks complete',
                   'start run count 1', 'Source has not changed', 'end of report']
        self.assertEqual(control, df['text'].to_list())

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
