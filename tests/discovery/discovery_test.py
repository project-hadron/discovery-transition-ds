import os
import shutil
import unittest

from aistac.properties.property_manager import PropertyManager

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

if __name__ == '__main__':
    unittest.main()
