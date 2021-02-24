import os
import shutil
import unittest

import pandas as pd
from ds_discovery import SyntheticBuilder
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel

from ds_discovery.components.commons import Commons


class CommonsTest(unittest.TestCase):

    def setUp(self):
        os.environ['HADRON_PM_PATH'] = os.path.join(os.environ['PWD'], 'work')
        self.tools: SyntheticIntentModel = SyntheticBuilder.from_env('tester', default_save=False,
                                                                     default_save_intent=False,
                                                                     has_contract=False).intent_model

    def tearDown(self):
        try:
            shutil.rmtree(os.path.join(os.environ['PWD'], 'work'))
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        pass

    def test_filter(self):
        tools = self.tools
        sample_size = 1000
        df = pd.DataFrame()
        df['normal_num'] = tools.get_number(1, 10, size=sample_size, seed=31)
        df['single num'] = tools.get_number(1, 2, quantity=0.8, size=sample_size, seed=31)
        df['weight_num'] = tools.get_number(1, 3, relative_freq=[90, 1], size=sample_size, seed=31)
        df['null'] = tools.get_number(1, 100, quantity=0, size=sample_size, seed=31)
        df['single cat'] = tools.get_category(['A'], quantity=0.6, size=sample_size, seed=31)
        df['weight_cat'] = tools.get_category(['A', 'B', 'C'], relative_freq=[80, 1, 1], size=sample_size, seed=31)
        df['normal_cat'] = tools.get_category(['A', 'B', 'C'], size=sample_size, seed=31)
        result = Commons.filter_headers(df, headers=['normal_num', 'single num'])
        control = ['normal_num', 'single num']
        self.assertCountEqual(control, result)
        result = Commons.filter_headers(df, dtype=['number'])
        control = ['weight_num', 'normal_num', 'single num']
        self.assertCountEqual(control, result)

    def test_canonical_formatter(self):
        tools = self.tools
        sample_size = 10
        df = pd.DataFrame()
        df['int'] = tools.get_number(1, 10, size=sample_size)
        df['float'] = tools.get_number(-1, 1, size=sample_size)
        df['bool'] = tools.get_category([1,0], size=sample_size)
        df['cat'] = tools.get_category(list('ABCDEF'), size=sample_size)
        df['object'] = tools.get_string_pattern("cccccccc", size=sample_size)


if __name__ == '__main__':
    unittest.main()
