import os
import shutil
import unittest

from ds_foundation.properties.property_manager import PropertyManager

from ds_discovery import TransitionAgent

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

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass
        props = PropertyManager().get_all()
        for key in props.keys():
            PropertyManager().remove(key)

    def test_runs(self):
        """Basic smoke test"""
        Discovery()

    def test_find_file(self):
        ds = Discovery()
        df = TransitionAgent.from_env('synthetic').load_source_canonical()
        result = ds.data_dictionary(df)
        control = ['Attribute', 'dType', '%_Null', '%_Dom', 'Count', 'Unique', 'Observations']
        self.assertEqual(control, result.columns.to_list())
        result = ds.data_dictionary(df, inc_next_dom=True)
        control = ['Attribute', 'dType', '%_Null', '%_Dom', '%_Nxt', 'Count', 'Unique', 'Observations']
        self.assertEqual(control, result.columns.to_list())

    def test_data_dictionary_filter(self):
        tr = TransitionAgent.from_env('synthetic')
        df = tr.load_source_canonical()
        result = tr.canonical_report(df, stylise=False)
        print(result)

if __name__ == '__main__':
    unittest.main()
