import unittest
import os
import shutil
from pprint import pprint
import pandas as pd
from aistac.properties.property_manager import PropertyManager
from aistac.properties.abstract_properties import AbstractPropertyManager
from ds_discovery import *
from ds_discovery.components.commons import Commons

pd.set_option('max_colwidth', 200)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 99)
pd.set_option('expand_frame_repr', True)
# Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))


class AbstractCommonComponentTest(unittest.TestCase):

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

    def test_multi_environments_from_env(self):
        os.environ['HADRON_TRANSITION_PATH'] = "hadron/transition/path"
        os.environ['HADRON_TRANSITION_PERSIST_PATH'] = "hadron/transition/persist/path"
        tr = Transition.from_memory()
        tr.set_source('source.csv')
        tr.set_persist('persist.p')
        self.assertEqual('hadron/transition/path/source.csv', tr.pm.get_connector_contract('primary_source').uri)
        self.assertEqual('hadron/transition/persist/path/persist.p', tr.pm.get_connector_contract('primary_persist').uri)
        wr = Wrangle.from_memory()
        wr.set_source('source.csv')
        wr.set_persist('persist.p')
        self.assertEqual('work/data/source.csv', wr.pm.get_connector_contract('primary_source').uri)
        self.assertEqual('work/data/persist.p', wr.pm.get_connector_contract('primary_persist').uri)

    def test_save_report_canonical(self):
        tr = Transition.from_memory()
        df = pd.DataFrame({'A': [1,2,3,4]})
        tr.save_report_canonical(reports=tr.REPORT_SUMMARY, report_canonical=df)
        result = tr.load_canonical(connector_name=tr.REPORT_SUMMARY)
        self.assertEqual(df.shape, result.shape)

    def test_save_report_canonical_params(self):
        tr = Transition.from_memory()
        df = pd.DataFrame({'A': [1,2,3,4]})
        reports = [Commons.param2dict(reports=tr.REPORT_SUMMARY, )]
        tr.save_report_canonical(reports=tr.REPORT_SUMMARY, report_canonical=df)



    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
