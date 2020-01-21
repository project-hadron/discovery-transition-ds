import matplotlib
import pandas as pd

matplotlib.use("TkAgg")

import unittest
import warnings

from ds_discovery.intent.engineer import FeatureEngineerTools as fb
from ds_behavioral import DataBuilderTools as tools


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)

    return do_test


class MyTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @ignore_warnings
    def test_runs(self):
        """Basic smoke test"""
        fb()

    def test_scratch(self):
        sample_size = 10
        df = pd.DataFrame()
        df['date'] = tools.get_datetime('01/01/2000', '12/31/2018', quantity=0.7, size=sample_size)
        result = fb.replace_missing(df)
        print(result)


if __name__ == '__main__':
    unittest.main()
