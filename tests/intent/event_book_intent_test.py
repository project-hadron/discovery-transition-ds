import unittest
import os
import shutil
from pprint import pprint

from ds_discovery import EventBookPortfolio
from ds_discovery.intent.event_book_intent import EventBookIntentModel


class EventBookIntentModelTest(unittest.TestCase):

    def setUp(self):
        os.environ['HADRON_PM_PATH'] = os.path.join(os.environ['PWD'], 'work')
        os.environ['HADRON_DEFAULT_PATH'] = os.path.join(os.environ['PWD'], 'work')

    def tearDown(self):
        try:
            shutil.rmtree(os.path.join(os.environ['PWD'], 'work'))
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        portfolio = EventBookPortfolio.from_env('localhost', default_save=False, has_contract=False)
        self.assertTrue(isinstance(portfolio.intent_model, EventBookIntentModel))

    def test_add_event_book(self):
        portfolio = EventBookPortfolio.from_env('localhost', default_save=False, has_contract=False)
        portfolio.intent_model.add_event_book(book_name='book_one')
        portfolio.intent_model.add_event_book(book_name='book_two', count_distance=2,
                                              module_name='ds_engines.engines.event_books.pandas_event_book',
                                              event_book_cls='PandasEventBook')
        portfolio.intent_model.add_event_book(book_name='joined', intent_order=-1)
        portfolio.intent_model.add_event_book(book_name='joined', intent_order=-1, count_distance=2,
                                              module_name='ds_engines.engines.event_books.pandas_event_book',
                                              event_book_cls='PandasEventBook')
        result = portfolio.pm.get_intent()
        self.assertCountEqual(['book_one', 'book_two', 'joined'], list(result.keys()))
        self.assertCountEqual(['0', '1'], list(result.get('joined').keys()))

    def test_run_intent_pipeline(self):
        portfolio = EventBookPortfolio.from_env('localhost', default_save=False, has_contract=False)
        portfolio.intent_model.add_event_book(book_name='book_one')
        portfolio.intent_model.add_event_book(book_name='book_two', count_distance=2,
                                              module_name='ds_engines.engines.event_books.pandas_event_book',
                                              event_book_cls='PandasEventBook')
        portfolio.intent_model.add_event_book(book_name='joined', intent_order=-1)
        portfolio.intent_model.add_event_book(book_name='joined', intent_order=-1, count_distance=2,
                                              module_name='ds_engines.engines.event_books.pandas_event_book',
                                              event_book_cls='PandasEventBook')
        result = portfolio.intent_model.run_intent_pipeline()
        self.assertCountEqual(['book_one_0', 'book_two_0', 'joined_0', 'joined_1'], list(result.keys()))

if __name__ == '__main__':
    unittest.main()
