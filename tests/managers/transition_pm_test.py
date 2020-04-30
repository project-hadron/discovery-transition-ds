import unittest
import os
import shutil

from ds_discovery.managers.transition_property_manager import TransitionPropertyManager


class TransitionPMTest(unittest.TestCase):

    def setUp(self):
        os.environ['AISTAC_PM_PATH'] = os.path.join(os.environ['PWD'], 'work')
        pass

    def tearDown(self):
        try:
            shutil.rmtree(os.path.join(os.environ['PWD'], 'work'))
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        TransitionPropertyManager('test')

    def test_catalog(self):
        pm = TransitionPropertyManager('test')
        catalog = pm.knowledge_catalog
        self.assertCountEqual(['transition', 'observations', 'actions', 'schema', 'intent', 'attributes'], catalog)

    def test_manager_name(self):
        pm = TransitionPropertyManager('test')
        result = pm.manager_name()
        self.assertEqual('transition', result)


if __name__ == '__main__':
    unittest.main()
