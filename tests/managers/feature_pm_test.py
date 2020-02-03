import unittest
import os
import shutil

from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager


class FeaturePMTest(unittest.TestCase):

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
        FeatureCatalogPropertyManager('test')

    def test_catalog(self):
        pm = FeatureCatalogPropertyManager('test')
        catalog = pm.knowledge_catalog
        self.assertCountEqual(['features', 'observations', 'actions', 'journal', 'frames'], catalog)

    def test_manager_name(self):
        pm = FeatureCatalogPropertyManager('test')
        result = pm.manager_name()
        self.assertEqual('feature_catalog', result)


if __name__ == '__main__':
    unittest.main()
