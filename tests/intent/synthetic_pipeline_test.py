import unittest
import os
import shutil
from pprint import pprint

import pandas as pd
import numpy as np
from ds_discovery import SyntheticBuilder
from ds_discovery.components.commons import Commons
from aistac.properties.property_manager import PropertyManager


class SyntheticPipelineTest(unittest.TestCase):

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
    def builder(self) -> SyntheticBuilder:
        return SyntheticBuilder.from_env('tester', has_contract=False)

    def test_run_synthetic_pipeline(self):
        sb = self.builder
        size = 100
        tools = self.builder.intent_model
        tools.get_number(1, 2, size=size, column_name='numbers')
        tools.get_category(selection=['M'], column_name='gender')
        sb.set_persist()
        sb.run_synthetic_pipeline(size=size)
        result = sb.load_synthetic_canonical()
        self.assertEqual((size, 2), result.shape)
        self.assertCountEqual(['numbers', 'gender'], result.columns)
        self.assertEqual('M', result['gender'].value_counts().index[0])
        self.assertEqual(size, result['gender'].value_counts().values[0])
        self.assertEqual(1, result['numbers'].value_counts().index[0])
        self.assertEqual(size, result['numbers'].value_counts().values[0])

    def test_run_intent_pipeline_get(self):
        sb = self.builder
        sb.tools.get_number(1, 2, column_name='numbers')
        result = sb.pm.report_intent()
        self.assertEqual(['numbers'], result.get('level'))
        self.assertEqual(['0'], result.get('order'))
        self.assertEqual(['get_number'], result.get('intent'))
        self.assertEqual([['from_value=1', 'to_value=2', "column_name='numbers'"]], result.get('parameters'))
        sb.tools.get_category(selection=['M'], column_name='gender')
        result = sb.tools.run_intent_pipeline(size=10, columns=['numbers', 'gender', 'jim'])
        self.assertEqual((10, 2), result.shape)
        self.assertCountEqual(['numbers', 'gender'], result.columns)
        self.assertEqual('M', result['gender'].value_counts().index[0])
        self.assertEqual(10, result['gender'].value_counts().values[0])
        self.assertEqual(1, result['numbers'].value_counts().index[0])
        self.assertEqual(10, result['numbers'].value_counts().values[0])

    def test_run_intent_pipeline_correlate(self):
        tools = self.builder.intent_model
        df = pd.DataFrame()
        df['numbers'] = tools.get_number(1, 2, column_name='numbers', intent_order=0)
        df['corr_num'] = tools.correlate_numbers(df, offset=1, header='numbers', column_name='numbers', intent_order=1)
        df['corr_plus'] = tools.correlate_numbers(df, offset=1, header='numbers', column_name='corr_plus')
        result = tools.run_intent_pipeline(size=10)
        self.assertCountEqual(['numbers', 'corr_plus'], result.columns)
        self.assertEqual(1, result['numbers'].value_counts().size)
        self.assertEqual(2, result['numbers'].value_counts().index[0])
        self.assertEqual(1, result['corr_plus'].value_counts().size)
        self.assertEqual(3, result['corr_plus'].value_counts().index[0])

    def test_canonical_run_pipeline_dict(self):
        tools = self.builder.intent_model
        df = pd.DataFrame()
        df['numbers'] = tools.get_number(1, 2, column_name='numbers')
        # create a remote pm contract
        inst = SyntheticBuilder.from_env('sub_set', has_contract=False)
        _ = inst.tools.get_category(selection=['A', 'B'], column_name='value')
        sub_set = Commons.param2dict()
        df['corr_num'] = tools.correlate_numbers(df, offset=1, header='numbers', column_name='numbers', intent_order=1)




if __name__ == '__main__':
    unittest.main()
