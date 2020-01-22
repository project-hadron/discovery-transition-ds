import os
import shutil
import unittest

from ds_foundation.properties.property_manager import PropertyManager

from ds_discovery import Transition

from ds_discovery.transition.discovery import DataDiscovery as Discovery


class TestDiscovery(unittest.TestCase):
    """Test: """

    def setUp(self):
        # set environment variables
        os.environ['DTU_CONTRACT_PATH'] = os.path.join(os.environ['PWD'], 'work')
        try:
            shutil.copytree('../data', os.path.join(os.environ['PWD'], 'work'))
        except:
            pass
        PropertyManager._remove_all()

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        Discovery()

    def test_find_file(self):
        ds = Discovery()
        df = Transition.from_env('synthetic').load_source_canonical()
        result = ds.data_dictionary(df)
        control = ['Attribute', 'dType', '%_Null', '%_Dom', 'Count', 'Unique', 'Observations']
        self.assertEqual(control, result.columns.to_list())
        result = ds.data_dictionary(df, inc_next_dom=True)
        control = ['Attribute', 'dType', '%_Null', '%_Dom', '%_Nxt', 'Count', 'Unique', 'Observations']
        self.assertEqual(control, result.columns.to_list())

    def test_data_dictionary_filter(self):
        tr = Transition.from_env('synthetic')
        df = tr.load_source_canonical()
        result = tr.canonical_report(df, stylise=False)
        print(result)

if __name__ == '__main__':
    unittest.main()
