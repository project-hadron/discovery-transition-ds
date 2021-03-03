import os
import shutil
import unittest

import pandas as pd

from aistac.handlers.abstract_handlers import ConnectorContract

from ds_engines.components.event_book_portfolio import EventBookPortfolio
from ds_engines.managers.event_book_property_manager import EventBookPropertyManager


class EventBookPortfolioTest(unittest.TestCase):

    def setUp(self):
        os.environ['HADRON_PM_PATH'] = os.path.join(os.environ['PWD'], 'work')
        pass

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_smoke(self):
        """Basic smoke test"""
        engine = EventBookPortfolio.from_env('task', has_contract=False)
        self.assertTrue(isinstance(engine, EventBookPortfolio))


if __name__ == '__main__':
    unittest.main()
