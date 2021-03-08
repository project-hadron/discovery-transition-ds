import unittest
import os
import shutil
from pprint import pprint

import pandas as pd

from aistac.handlers.abstract_handlers import ConnectorContract
from ds_discovery import SyntheticBuilder
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager

from ds_discovery.handlers.event_handlers import EventSourceHandler, EventPersistHandler


class EventHandlerTest(unittest.TestCase):

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

    @property
    def tools(self) -> SyntheticIntentModel:
        return SyntheticBuilder.scratch_pad()

    def test_handler(self):
        """Basic smoke test"""
        cc = ConnectorContract(uri='eb://test_book', module_name='', handler='')
        handler = EventPersistHandler(connector_contract=cc)
        # test persist and load
        df = pd.DataFrame({'A': [1,2,3,4], 'B': [7,2,1,4]})
        handler.persist_canonical(df)
        result = handler.load_canonical()
        self.assertEqual(['A', 'B'], list(result.columns))
        self.assertEqual((4,2), result.shape)

    def test_has_changed(self):
        # test the handler
        cc = ConnectorContract(uri='eb://test_portfolio/test_book', module_name='', handler='')
        handler = EventPersistHandler(connector_contract=cc)
        # Test has changed
        self.assertFalse(handler.has_changed())
        df = pd.DataFrame({'A': [1,2,3,4], 'B': [7,2,1,4]})
        handler.persist_canonical(df)
        self.assertTrue(handler.has_changed())
        result = handler.load_canonical()
        self.assertDictEqual(df.to_dict(), result.to_dict())
        self.assertTrue(handler.has_changed())
        handler.reset_changed()
        self.assertFalse(handler.has_changed())
        df = pd.DataFrame({'C': [9, 8, 7, 3], 'B': [4, 2, 1, 0]})
        handler.persist_canonical(df, reset_state=False)
        result = handler.load_canonical()
        print(result)
        # self.assertCountEqual(list('ABC'), result.columns.to_list())

    def test_from_component(self):
        # EventBook
        os.environ['HADRON_DEFAULT_PATH'] = 'eb://grey_storage/'
        os.environ['HADRON_DEFAULT_MODULE'] = 'ds_engines.handlers.event_handlers'
        os.environ['HADRON_DEFAULT_SOURCE_HANDLER'] = 'EventPersistHandler'
        os.environ['HADRON_DEFAULT_PERSIST_HANDLER'] = 'EventSourceHandler'
        # Portfolio
        builder = SyntheticBuilder.from_env('members', has_contract=False)
        builder.set_outcome(uri_file="synthetic_members")
        builder = SyntheticBuilder.from_env('members')


    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
