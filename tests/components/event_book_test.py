import shutil
import unittest
import os
import pandas as pd
import numpy as np
from aistac.handlers.abstract_handlers import ConnectorContract
from ds_engines.engines.event_books.pandas_event_book import PandasEventBook


class EventBookTest(unittest.TestCase):

    MODULE = "aistac.handlers.python_handlers"
    HANDLER = "PythonPersistHandler"

    def setUp(self):
        try:
            shutil.rmtree('work')
        except:
            pass
        os.environ['HADRON_PM_PATH'] = os.path.join(os.environ['PWD'], 'work')
        os.makedirs(os.environ['HADRON_PM_PATH'])

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        PandasEventBook(book_name='test')

    def test_events(self):
        event_book = PandasEventBook('test')
        selection = list("ABCD")
        master = pd.DataFrame(columns=selection)
        event_book.add_event(event=master)
        self.assertEqual((0,4), event_book.current_state.shape)
        # add event
        event = pd.DataFrame({'A': [1,1,1], 'E': [1,1,1]})
        event_book.add_event(event=event)
        self.assertEqual((3, 5), event_book.current_state.shape)
        event = pd.DataFrame({'A': [1, 0, 1]})
        event_book.increment_event(event=event)
        control = pd.Series([2,1,2])
        result = event_book.current_state['A']
        self.assertCountEqual(control, result)
        event_book.decrement_event(event=event)
        control = pd.Series([1,1,1])
        result = event_book.current_state['A']
        self.assertCountEqual(control, result)

    def test_parameters(self):
        event_book = PandasEventBook('test')
        self.assertEqual(0, event_book.time_distance)
        self.assertEqual(0, event_book.count_distance)
        self.assertEqual(0, event_book.events_log_distance)
        event_book = PandasEventBook('test', time_distance=1, count_distance=2, events_log_distance=3)
        self.assertEqual(1, event_book.time_distance)
        self.assertEqual(2, event_book.count_distance)
        self.assertEqual(3, event_book.events_log_distance)
        event_book.set_time_distance(distance=11)
        self.assertEqual(11, event_book.time_distance)
        event_book.set_count_distance(distance=12)
        self.assertEqual(12, event_book.count_distance)
        event_book.set_events_log_distance(distance=13)
        self.assertEqual(13, event_book.events_log_distance)

    def test_persist(self):
        state_uri = os.path.join(os.environ['HADRON_PM_PATH'], 'state.pickle')
        events_uri = os.path.join(os.environ['HADRON_PM_PATH'], 'events_log.pickle')
        state_connector = ConnectorContract(uri=state_uri, module_name=self.MODULE, handler=self.HANDLER)
        events_connector = ConnectorContract(uri=events_uri, module_name=self.MODULE, handler=self.HANDLER)
        engine = PandasEventBook('test', state_connector=state_connector, events_log_connector=events_connector)
        self.assertEqual(False, os.path.exists(state_uri))
        self.assertEqual(False, os.path.exists(events_uri))
        for i in range(10):
            engine.increment_event(event=pd.DataFrame(data={'A': [i, i*2, i*3]}))
        self.assertEqual(0, len(engine._current_events_log.keys()), "loop run")
        self.assertEqual(False, os.path.exists(state_uri))
        self.assertEqual(False, os.path.exists(events_uri))
        engine.set_count_distance(3)
        engine.set_events_log_distance(2)
        # add one
        engine.increment_event(event=pd.DataFrame(data={'A': [1,1,1]}))
        self.assertEqual(False, os.path.exists(state_uri))
        self.assertEqual(False, os.path.exists(events_uri))
        self.assertEqual(1, len(engine._current_events_log.keys()), "loop One")
        # add two
        engine.increment_event(event=pd.DataFrame(data={'A': [1,1,1]}))
        self.assertEqual(False, os.path.exists(state_uri))
        self.assertEqual(True, os.path.exists(events_uri))
        self.assertEqual(0, len(engine._current_events_log.keys()), "loop Two")
        # add three
        engine.increment_event(event=pd.DataFrame(data={'A': [1,1,1]}))
        self.assertEqual(True, os.path.exists(state_uri))
        self.assertEqual(True, os.path.exists(events_uri))
        self.assertEqual(0, len(engine._current_events_log.keys()), "loop Three")
        # add four
        engine.increment_event(event=pd.DataFrame(data={'A': [1,1,1]}))
        self.assertEqual(1, len(engine._current_events_log.keys()), "loop Four")

    def test_fillna(self):
        eb = PandasEventBook('test')
        event = pd.DataFrame({'A': [1, 1, 1], 'E': [1.1, 1.5, 2.6]})
        eb.add_event(event)
        eb.add_event(pd.DataFrame({'B': ['A', np.nan, 'C', 'D', np.nan]}))
        eb.add_event(pd.DataFrame({'C': [True, False, np.nan, np.nan, np.nan, False, True]}, dtype=bool))
        eb.add_event(pd.DataFrame({'D': ['M', 'F', np.nan, np.nan, 'F', 'F']}, dtype='category'))
        print(eb.current_state(fillna=True))


if __name__ == '__main__':
    unittest.main()
