import unittest
import os
import shutil
import uuid

import pandas as pd
import numpy as np
from ds_discovery.components.commons import Commons

from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from ds_discovery import SyntheticBuilder
from aistac.properties.property_manager import PropertyManager


class SyntheticIntentGetTest(unittest.TestCase):

    def setUp(self):
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

    def test_runs(self):
        """Basic smoke test"""
        im = SyntheticBuilder.from_env('tester', default_save=False, default_save_intent=False,
                                       reset_templates=False, has_contract=False).intent_model
        self.assertTrue(SyntheticIntentModel, type(im))

    def test_get_number(self):
        tools = self.tools
        sample_size = 10000
        # from to int
        ref_ids = tools.get_number(from_value=1000, to_value=10000, precision=0, at_most=1, size=10)
        self.assertTrue(all(isinstance(x, np.int64) for x in ref_ids))
        result = tools.get_number(10, 1000, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 10)
        self.assertLessEqual(max(result), 1000)
        self.assertTrue(type)
        result = tools.get_number(10, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 0)
        self.assertLessEqual(max(result), 10)
        # from to float
        result = tools.get_number(0.1, 0.9, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 0.1)
        self.assertLessEqual(max(result), 0.9)
        result = tools.get_number(1.0, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 0)
        self.assertLessEqual(max(result), 1.0)
        # from
        result = tools.get_number(10, 1000, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 10)
        self.assertLessEqual(max(result), 1000)
        # set both
        result = tools.get_number(from_value=10, to_value=1000, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 10)
        self.assertLessEqual(max(result), 1000)
        # set to_value
        result = tools.get_number(to_value=1000, size=sample_size)
        self.assertEqual(sample_size, len(result))
        self.assertGreaterEqual(min(result), 0)
        # self.assertLessEqual(max(result), 1000)

    def test_get_number_environ(self):
        os.environ["HADRON_START_VALUE"] = "2"
        os.environ["HADRON_END_VALUE"] = "6"
        tools = self.tools
        result = tools.get_number("${HADRON_START_VALUE}", "${HADRON_END_VALUE}", precision=0, size=10, seed=31)
        self.assertEqual([3, 2, 4, 4, 3, 4, 3, 4, 2, 5], result)
        os.environ["HADRON_START_VALUE"] = "A"
        with self.assertRaises(ValueError) as context:
            result = tools.get_number("${HADRON_START_VALUE}", "${HADRON_END_VALUE}", size=10, seed=31)
        self.assertTrue("The environment variable 'A' is not set or not numeric." in str(context.exception))
        os.environ["HADRON_START_VALUE"] = "2"
        os.environ["HADRON_END_VALUE"] = "A"
        with self.assertRaises(ValueError) as context:
            result = tools.get_number("${HADRON_START_VALUE}", "${HADRON_END_VALUE}", size=10, seed=31)
        self.assertTrue("The environment variable 'A' is not set or not numeric." in str(context.exception))
        with self.assertRaises(ValueError) as context:
            result = tools.get_number("HADRON_START_VALUE", 10000, size=10, seed=31)
        self.assertTrue("Numeric values can not be a string unless an environment variable wrapped in '${}'." in str(context.exception))

    def test_get_number_at_most(self):
        tools = self.tools
        sample_size = 10000
        result = tools.get_number(100000, 200000, precision=0, at_most=1, relative_freq=[2,3], size=sample_size)
        self.assertEqual(sample_size, pd.Series(result).nunique())
        result = tools.get_number(100000.0, 200000.0, precision=2, at_most=1, relative_freq=[2,3], size=sample_size)
        self.assertEqual(sample_size, pd.Series(result).nunique())
        tools = self.tools
        sample_size = 19
        result = tools.get_number(10, 20, precision=0, at_most=2, size=sample_size)
        self.assertEqual(2, pd.Series(result).value_counts().max())
        self.assertEqual(sample_size, len(result))

    def test_get_number_large(self):
        tools = self.tools
        sample_size = 1000
        result = tools.get_number(1000000, 9999999999, precision=1, size=sample_size)
        self.assertEqual(sample_size,len(result))
        result = tools.get_number(1000000.0, 9999999999.0, precision=1, size=sample_size)
        self.assertEqual(sample_size, len(result))
        # at_most
        result = tools.get_number(1000000, 9999999999, at_most=1, precision=1, size=sample_size)
        self.assertEqual(sample_size, len(result))
        result = tools.get_number(1000000.0, 9999999999.0, at_most=1, precision=1, size=sample_size)
        self.assertEqual(sample_size, len(result))

    def test_get_number_asc(self):
        tools = self.tools
        sample_size = 10
        result = tools.get_number(0, 10, ordered='asc', size=sample_size)
        self.assertTrue(all([result[index] >= result[index + 1] for index in range(len(result) - 1)]))

    def test_get_number_weighting(self):
        tools = self.tools
        sample_size = 100000
        result = tools.get_number(10, 20, relative_freq=[0,1], size=sample_size)
        result = pd.Series(result)
        self.assertGreaterEqual(result.min(), 15)
        self.assertLess(result.max(), 20)
        result = tools.get_number(10.0, 20.0, relative_freq=[0,1], size=sample_size)
        result = pd.Series(result)
        self.assertGreaterEqual(result.min(), 15)
        self.assertLess(result.max(), 20)

    def test_get_datetime_ignore_time(self):
        tools = self.tools
        sample_size = 10
        result = tools.get_datetime('2019/01/01 12:30', '2019/01/02 18:05', size=sample_size)
        self.assertLess(0,pd.Series(result).dt.hour.sum())
        self.assertLess(0,pd.Series(result).dt.minute.sum())
        result = tools.get_datetime('2019/01/01 12:30', '2019/01/02 18:05', ignore_time=True, size=sample_size)
        self.assertEqual(0, pd.Series(result).dt.hour.sum())
        self.assertEqual(0,pd.Series(result).dt.minute.sum())
        result = tools.get_datetime('2019/01/01 12:30', '2019/01/02 18:05', ignore_seconds=True, size=sample_size)
        print(result)

    def test_get_datetime_at_most(self):
        tools = self.tools
        sample_size = 10000
        result = tools.get_datetime('2018/01/01', '2019/01/01', at_most=1, size=sample_size)
        result = pd.Series(result)
        self.assertEqual(sample_size, pd.Series(result).nunique())

    def test_get_datetime_until_delta(self):
        tools = self.tools
        sample_size = 10000
        result = tools.get_datetime(0, {'seconds': 1}, at_most=1, size=sample_size)
        result = pd.Series(result)
        self.assertLess(result.max() - result.min(), pd.Timedelta(seconds=1))
        result = tools.get_datetime(0, {'minutes': 1, 'seconds': -1}, at_most=1, size=sample_size)
        result = pd.Series(result)
        self.assertLess(result.max() - result.min(), pd.Timedelta(minutes=1))

    def test_get_datetime(self):
        tools = self.tools
        sample_size = 3
        result = tools.get_datetime('2019/01/01', '2019/01/02', date_format="%Y-%m-%d", size=sample_size)
        self.assertEqual(['2019-01-01', '2019-01-01', '2019-01-01'], result)
        result = tools.get_datetime(pd.Timestamp('2021-07-01', tz='CET'), pd.Timestamp('2021-07-02', tz='CET'), date_format="%Y-%m-%d %Z", size=sample_size)
        self.assertEqual(['2021-07-01 CEST', '2021-07-01 CEST', '2021-07-01 CEST'], result)

    def test_get_sample(self):
        tools = self.tools
        result = tools.get_sample(sample_name='us_states', size=1000)
        self.assertEqual(1000, len(result))

    def test_get_uuid(self):
        tools = self.tools
        result = tools.get_uuid(size=10, as_hex=False)
        self.assertEqual(10, len(result))
        self.assertTrue(sum([(len(x.split('-'))) for x in result])/10 == 5)
        result = tools.get_uuid(size=10, as_hex=True)
        self.assertTrue(sum([(len(x.split('-'))) for x in result])/10 == 1)
        result = tools.get_uuid(version=3, size=10, namespace=uuid.NAMESPACE_DNS, name='python.org')
        self.assertTrue(sum([(len(x.split('-'))) for x in result])/10 == 5)
        result = tools.get_uuid(version=1, size=10)
        self.assertTrue(sum([(len(x.split('-'))) for x in result])/10 == 5)

    def test_distributions(self):
        tools = self.tools
        result = tools.get_dist_normal(mean=3, std=0.5, precision=1, seed=31, size=4)
        self.assertEqual([2.8, 3.1, 3.3, 2.5], result)
        result = tools.get_dist_bernoulli(probability=0.5, seed=31, size=4)
        self.assertEqual([0, 1, 1, 1], result)

    def test_get_dist_data_types(self):
        tools = self.tools
        result = tools.model_synthetic_data_types(1000, seed=31)
        self.assertTrue(all((v is None) or isinstance(v, str) for v in result['str']))
        self.assertFalse(all((v is None) or isinstance(v, str) for v in result['object']))
        self.assertEqual((1000, 8), result.shape)
        result = tools.model_synthetic_data_types(1000, extended=True, seed=31)
        self.assertEqual((1000, 29), result.shape)

    def test_choice(self):
        os.environ["HADRON_CHOICE_SIZE"] = "4"
        tools = self.tools
        result = tools.get_dist_choice(number=4, size=6, seed=31)
        self.assertEqual(4, sum(result))
        os.environ['HADRON_CHOICE_SIZE'] = "5"
        result = tools.get_dist_choice(number='${HADRON_CHOICE_SIZE}', size=6, seed=31)
        self.assertEqual(5, sum(result))
        os.environ['HADRON_CHOICE_SIZE'] = "0"
        result = tools.get_dist_choice(number='${HADRON_CHOICE_SIZE}', size=6, seed=31)
        self.assertEqual(0, sum(result))
        result = tools.get_dist_choice(number=6, size=6, seed=31)
        self.assertEqual(6, sum(result))
        result = tools.get_dist_choice(number=9, size=6, seed=31)
        self.assertEqual(6, sum(result))

    def test_choice_frac(self):
        tools = self.tools
        result = tools.get_dist_choice(number=0.4, size=10, seed=31)
        self.assertEqual(4, sum(result))
        os.environ['HADRON_CHOICE_SIZE'] = "0.5"
        result = tools.get_dist_choice(number='${HADRON_CHOICE_SIZE}', size=10, seed=31)
        self.assertEqual(5, sum(result))
        os.environ['HADRON_CHOICE_SIZE'] = "0.0"
        result = tools.get_dist_choice(number='${HADRON_CHOICE_SIZE}', size=10, seed=31)
        self.assertEqual(0, sum(result))
        result = tools.get_dist_choice(number=1.0, size=10, seed=31)
        self.assertEqual(10, sum(result))
        result = tools.get_dist_choice(number=1.3, size=10, seed=31)
        self.assertEqual(1, sum(result))

    def test_quality(self):
        sb = SyntheticBuilder.from_memory()
        selection = sb.tools.get_number(100.0, size=10_000)
        result = sb.tools._set_quantity(selection, quantity=0.5)
        s_result = pd.Series(result)
        self.assertEqual(5_000, s_result.dropna().size)
        result = sb.tools._set_quantity(selection, quantity=50)
        s_result = pd.Series(result)
        self.assertEqual(5_000, s_result.dropna().size)
        result = sb.tools._set_quantity(selection, quantity=0.01)
        s_result = pd.Series(result)
        self.assertEqual(100, s_result.dropna().size)

if __name__ == '__main__':
    unittest.main()
