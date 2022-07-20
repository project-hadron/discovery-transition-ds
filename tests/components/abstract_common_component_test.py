import unittest
import os
import shutil
from pprint import pprint
import pandas as pd
from aistac.properties.property_manager import PropertyManager
from aistac.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.intent.wrangle_intent import WrangleIntentModel
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

    def test_load_canonical_return_empty(self):
        tr = Transition.from_env('task', has_contract=False)
        tr.set_source(uri_file='sample.csv')
        tr.set_persist(uri_file='sample.csv')
        df = pd.DataFrame({'A': [1, 2, 3, 4]})
        tr.save_persist_canonical(df)
        sample = tr.load_source_canonical(has_changed=True, return_empty=True)
        self.assertEqual(False, tr.pm.get_connector_handler(tr.CONNECTOR_SOURCE).has_changed())
        self.assertEqual(sample.shape, df.shape)
        sample = tr.load_source_canonical(has_changed=True, return_empty=True)
        self.assertEqual((0, 0), sample.shape)


    def test_load_canonical_has_changed(self):
        tr = Transition.from_env('task', has_contract=False)
        tr.set_source(uri_file='sample.csv')
        tr.set_persist(uri_file='sample.csv')
        df = pd.DataFrame({'A': [1,2,3,4]})
        tr.save_persist_canonical(df)
        self.assertEqual(True, tr.pm.get_connector_handler(tr.CONNECTOR_SOURCE).has_changed())
        sample = tr.load_source_canonical(reset_changed=False)
        self.assertEqual(False, tr.pm.get_connector_handler(tr.CONNECTOR_SOURCE).has_changed())
        self.assertEqual(sample.shape, df.shape)
        with self.assertRaises(ConnectionAbortedError) as context:
            sample = tr.load_source_canonical(has_changed=True)
        self.assertTrue("The connector name primary_source has been aborted" in str(context.exception))
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7 ,8]})
        tr.save_persist_canonical(df)
        self.assertEqual(True, tr.pm.get_connector_handler(tr.CONNECTOR_SOURCE).has_changed())
        sample = tr.load_source_canonical(has_changed=True)
        self.assertEqual(False, tr.pm.get_connector_handler(tr.CONNECTOR_SOURCE).has_changed())
        self.assertEqual(sample.shape, df.shape)

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

    def test_default_handler(self):
        tr = Transition.from_env('tester', has_contract=False, default_save=False)
        tr.set_source_uri("s3://project-hadron-cs-repo/factory/tester")
        self.assertEqual("ds_discovery.handlers.s3_handlers", tr.pm.get_connector_contract(tr.CONNECTOR_SOURCE).module_name)
        self.assertEqual("S3PersistHandler", tr.pm.get_connector_contract(tr.CONNECTOR_SOURCE).handler)
        tr.add_connector_uri("tester", "s3://project-hadron-cs-repo/factory/tester")
        self.assertEqual("ds_discovery.handlers.s3_handlers", tr.pm.get_connector_contract("tester").module_name)
        self.assertEqual("S3PersistHandler", tr.pm.get_connector_contract("tester").handler)

    def test_report_task(self):
        tr = Transition.from_env("tr1", has_contract=False)
        data = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl.csv'
        tr.set_source_uri(uri=data)
        tr.set_persist()
        tr.set_version('0.0.1')
        tr.set_status('testing')
        tr.set_description("A description of the component")
        result = tr.report_task(stylise=False)
        self.assertEqual(['contract', 'task', 'description', 'status', 'version'], result.loc[:,'name'].tolist())
        self.assertEqual(['transition', 'tr1', 'A description of the component', 'testing', '0.0.1'], result.loc[:,'value'].tolist())

    def test_schema_report(self):
        wr = Wrangle.from_env("titanic", has_contract=False)
        tools: WrangleIntentModel = wr.tools
        data = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl.csv'
        wr.set_source_uri(uri=data)
        wr.set_persist()
        df = wr.load_source_canonical()
        df = tools.frame_starter(df, headers=['survived', 'sex', 'deck'], column_name='starter')
        wr.run_component_pipeline()
        wr.save_canonical_schema()
        result = wr.report_canonical_schema(stylise=False)
        print(result)



    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
