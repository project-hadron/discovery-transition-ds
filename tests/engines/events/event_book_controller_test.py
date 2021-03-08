import unittest
import os
import shutil
import pandas as pd
from ds_discovery import SyntheticBuilder
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from aistac.properties.property_manager import PropertyManager
from ds_engines.engines.event_books.event_book_controller import EventBookController


class EventBookControllerTest(unittest.TestCase):

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

    def test_singleton(self):
        """Basic smoke test"""
        eb1 = EventBookController()
        eb1.add_event_book('test_book')
        event = pd.DataFrame(data={'a': [1,2,3]})
        eb1.add_event(book_name='test_book', event=event)
        eb2 = EventBookController()
        eb2.is_event_book(book_name='test_book')
        result1 = eb1.current_state(book_name='test_book')
        result2 = eb2.current_state(book_name='test_book')
        self.assertDictEqual(result1.to_dict(), event.to_dict())
        self.assertDictEqual(result1.to_dict(), result2.to_dict())

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
