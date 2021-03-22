import os
import shutil
import unittest

from ds_discovery.managers import FeatureCatalogPropertyManager


class FeaturePMTest(unittest.TestCase):

    def setUp(self):
        os.environ['HADRON_PM_PATH'] = os.path.join(os.environ['PWD'], 'work')
        pass

    def tearDown(self):
        try:
            shutil.rmtree(os.path.join(os.environ['PWD'], 'work'))
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        FeatureCatalogPropertyManager('test', username='TestUser')

    def test_catalog(self):
        pm = FeatureCatalogPropertyManager('test', username='TestUser')
        catalog = pm.knowledge_catalog
        self.assertCountEqual(['features', 'observations', 'actions', 'schema', 'intent'], catalog)

    def test_manager_name(self):
        pm = FeatureCatalogPropertyManager('test', username='TestUser')
        result = pm.manager_name()
        self.assertEqual('feature_catalog', result)


if __name__ == '__main__':
    unittest.main()
