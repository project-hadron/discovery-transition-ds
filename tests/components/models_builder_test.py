import unittest
import os
from pathlib import Path
import shutil
from ds_discovery import SyntheticBuilder
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager
from ds_discovery import ModelsBuilder


class SyntheticTest(unittest.TestCase):

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
        builder = SyntheticBuilder.from_env('tester', has_contract=False)
        builder.set_persist()
        tools: SyntheticIntentModel = builder.tools

    def tearDown(self):
        try:
            shutil.rmtree('working')
        except:
            pass


    def test_add_trained_model(self):
        ml = ModelsBuilder.from_env('tester', has_contract=False)
        ml.add_trained_model(b'12345')
        self.assertTrue(ml.pm.has_connector('ml_trained_connector'))
        ml.add_trained_model(b'12345', model_name='logreg')
        self.assertTrue(ml.pm.has_connector('logreg'))


    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
