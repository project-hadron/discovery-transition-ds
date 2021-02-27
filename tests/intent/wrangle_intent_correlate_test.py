import unittest
import os
import shutil
import numpy as np
import pandas as pd
from aistac import ConnectorContract

from ds_discovery import Wrangle, SyntheticBuilder
from ds_discovery.intent.wrangle_intent import WrangleIntentModel
from aistac.properties.property_manager import PropertyManager


class WrangleIntentCorrelateTest(unittest.TestCase):

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
    def tools(self) -> WrangleIntentModel:
        return Wrangle.scratch_pad()

    def test_runs(self):
        """Basic smoke test"""
        im = Wrangle.from_env('tester', default_save=False, default_save_intent=False,
                              reset_templates=False, has_contract=False).intent_model
        self.assertTrue(WrangleIntentModel, type(im))

    def test_correlate_choice(self):
        tools = self.tools
        df = pd.DataFrame()
        df['A'] = [[1,2,4,6], [1], [2,4,8,1], [2,4]]
        result = tools.correlate_choice(df, header='A', list_size=2)
        control = [[1, 2], [1], [2, 4], [2, 4]]
        self.assertEqual(control, result)
        result = tools.correlate_choice(df, header='A', list_size=1)
        self.assertEqual([1, 1, 2, 2], result)

    def test_correlate_coefficient(self):
        tools = self.tools
        df = pd.DataFrame()
        df['A'] = [1,2,3]
        result = tools.correlate_polynomial(df, header='A', coefficient=[2,1])
        self.assertEqual([3, 4, 5], result)
        result = tools.correlate_polynomial(df, header='A', coefficient=[0, 0, 1])
        self.assertEqual([1, 4, 9], result)

    def test_correlate_join(self):
        tools = self.tools
        df = pd.DataFrame()
        df['A'] = [1,2,3]
        df['B'] = list('XYZ')
        df['C'] = [4.2,7.1,4.1]
        result = tools.correlate_join(df, header='B', action="values", sep='_')
        self.assertEqual(['X_values', 'Y_values', 'Z_values'], result)
        result = tools.correlate_join(df, header='A', action=tools.action2dict(method='correlate_numbers', header='C'))
        self.assertEqual(['14.2', '27.1', '34.1'], result)

    def test_correlate_columns(self):
        tools = self.tools
        df = pd.DataFrame({'A': [1,1,1,1,None], 'B': [1,None,2,3,None], 'C': [2,2,2,2,None], 'D': [5,5,5,5,None]})
        result = tools.correlate_aggregate(df, headers=list('ABC'), agg='sum')
        control = [4.0, 3.0, 5.0, 6.0, 0.0]
        self.assertEqual(result, control)
        for action in ['sum', 'prod', 'count', 'min', 'max', 'mean']:
            print(action)
            result = tools.correlate_aggregate(df, headers=list('ABC'), agg=action)
            self.assertEqual(5, len(result))

    def test_correlate_number(self):
        tools = self.tools
        df = pd.DataFrame(data=[1,2,3,4.0,5,6,7,8,9,0], columns=['numbers'])
        result = tools.correlate_numbers(df, 'numbers', precision=0)
        self.assertCountEqual([1,2,3,4,5,6,7,8,9,0], result)
        # Offset
        df = pd.DataFrame(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], columns=['numbers'])
        result = tools.correlate_numbers(df, 'numbers', offset=1, precision=0)
        self.assertEqual([2,3,4,5,6,7,8,9,10,1], result)
        # str offset
        df = pd.DataFrame(data=[1, 2, 3, 4], columns=['numbers'])
        result = tools.correlate_numbers(df, 'numbers', offset='1-@', precision=0)
        self.assertEqual([0,-1,-2,-3], result)
        # complex str offset
        result = tools.correlate_numbers(df, 'numbers', offset='x + 2 if x <= 2 else x', precision=0)
        self.assertEqual([3, 4, 3, 4], result)
        # jitter
        df = pd.DataFrame(data=[2] * 1000, columns=['numbers'])
        result = tools.correlate_numbers(df, 'numbers', jitter=5, precision=0)
        self.assertLessEqual(max(result), 4)
        self.assertGreaterEqual(min(result), 0)
        df = pd.DataFrame(data=tools._get_number(99999, size=5000), columns=['numbers'])
        result = tools.correlate_numbers(df, 'numbers', jitter=5, precision=1)
        self.assertNotEqual(df['numbers'].to_list(), result)
        self.assertEqual(5000, len(result))
        for index in range(len(result)):
            loss = abs(df['numbers'][index] - result[index])
            self.assertLessEqual(loss, 5)
        df = pd.DataFrame(data=tools._get_number(99999, size=5000), columns=['numbers'])
        result = tools.correlate_numbers(df, 'numbers', jitter=1, precision=1)
        self.assertNotEqual(df['numbers'].to_list(), result)
        self.assertEqual(5000, len(result))
        for index in range(len(result)):
            loss = abs(df['numbers'][index] - result[index])
            self.assertLessEqual(loss, 1)

    def test_correlate_number_to_numeric(self):
        tools = self.tools
        df = pd.DataFrame(data=list("123") + ['4-5'], columns=['numbers'])
        with self.assertRaises(ValueError) as context:
            result = tools.correlate_numbers(df, header='numbers')
        self.assertTrue("The header column is of type" in str(context.exception))
        result = tools.correlate_numbers(df, header='numbers', to_numeric=True)
        self.assertEqual([1.0, 2.0, 3.0], result[:3])
        result = tools.correlate_numbers(df, header='numbers', to_numeric=True, replace_nulls=0, rtn_type='int')
        self.assertEqual([1, 2, 3, 0], result.to_list())

    def test_correlate_number_extras(self):
        tools = self.tools
        # weighting
        df = pd.DataFrame(columns=['numbers'], data=[2] * 1000)
        result = tools.correlate_numbers(df, 'numbers', jitter=5, precision=0, jitter_freq=[0, 0, 1, 1])
        self.assertCountEqual([2,3,4], list(pd.Series(result).value_counts().index))
        result = tools.correlate_numbers(df, 'numbers', jitter=5, precision=0, jitter_freq=[1, 1, 0, 0])
        self.assertCountEqual([0,1,2], list(pd.Series(result).value_counts().index))
        # fill nan
        df = pd.DataFrame(columns=['numbers'], data=[1,1,2,np.nan,3,1,np.nan,3,5,np.nan,7])
        result = tools.correlate_numbers(df, 'numbers', replace_nulls=1, precision=0)
        self.assertEqual([1,1,2,1,3,1,1,3,5,1,7], result)
        df = pd.DataFrame(columns=['numbers'], data=[2] * 1000)
        # jitter, offset and fillna
        result = tools.correlate_numbers(df, 'numbers', offset=2, jitter=5, replace_nulls=2, precision=0)
        self.assertCountEqual([2,3,4,5,6], list(pd.Series(result).value_counts().index))
        # min
        df = pd.DataFrame(columns=['numbers'], data=[2] * 100)
        result = tools.correlate_numbers(df, 'numbers', offset=2, jitter=5, min_value=4, precision=0)
        self.assertCountEqual([4, 5, 6], list(pd.Series(result).value_counts().index))
        result = tools.correlate_numbers(df, 'numbers', offset=2, jitter=5, min_value=6, precision=0)
        self.assertCountEqual([6], list(pd.Series(result).value_counts().index))
        with self.assertRaises(ValueError) as context:
            result = tools.correlate_numbers(df, 'numbers', offset=2, jitter=5, min_value=7, precision=0)
        self.assertTrue("The min value 7 is greater than the max result value" in str(context.exception))
        # max
        result = tools.correlate_numbers(df, 'numbers', offset=2, jitter=5, max_value=4, precision=0)
        self.assertCountEqual([2, 3, 4], list(pd.Series(result).value_counts().index))
        result = tools.correlate_numbers(df, 'numbers', offset=2, jitter=5, max_value=2, precision=0)
        self.assertCountEqual([2], list(pd.Series(result).value_counts().index))
        with self.assertRaises(ValueError) as context:
            result = tools.correlate_numbers(df, 'numbers', offset=2, jitter=5, max_value=1, precision=0)
        self.assertTrue("The max value 1 is less than the min result value" in str(context.exception))

    def test_correlate_categories(self):
        tools = self.tools
        df = pd.DataFrame(columns=['cat'], data=list("ABCDE"))
        correlation = ['A', 'D']
        action = {0: 'F', 1: 'G'}
        result = tools.correlate_categories(df, 'cat', correlations=correlation, actions=action, default_action=tools.action2dict(method='@header', header='cat'))
        self.assertEqual(['F', 'B', 'C', 'G', 'E'], result)
        correlation = ['A', 'D']
        action = {0: {'method': 'get_category', 'selection': list("HIJ")}, 1: {'method': 'get_number', 'to_value': 10}}
        result = tools.correlate_categories(df, 'cat', correlations=correlation, actions=action)
        self.assertIn(result[0], list("HIJ"))
        self.assertTrue(0 <= result[3] < 10)
        df = pd.DataFrame(columns=['cat'], data=tools._get_category(selection=list("ABCDE"), size=5000))
        result = tools.correlate_categories(df, 'cat', correlations=correlation, actions=action)
        self.assertEqual(5000, len(result))

    def test_correlate_categories_builder(self):
        builder = Wrangle.from_env('test', has_contract=False)
        builder.set_persist_contract(ConnectorContract(uri="eb://synthetic_members", module_name='ds_engines.handlers.event_handlers', handler='EventPersistHandler'))
        df = pd.DataFrame()
        df['pcp_tax_id'] = [993406113, 133757370, 260089066, 448512481, 546434723] * 2
        correlations = [993406113, 133757370, 260089066, 448512481, 546434723]
        actions = {0: 'LABCORP OF AMERICA', 1: 'LPCH MEDICAL GROUP', 2: 'ST JOSEPH HERITAGE MEDICAL',
                   3: 'MONARCH HEALTHCARE', 4: 'PRIVIA MEICAL GROUP'}
        df['pcp_name'] = builder.tools.correlate_categories(df, header='pcp_tax_id', correlations=correlations,
                                                            actions=actions, column_name='pcp_name')
        result = builder.tools.run_intent_pipeline(df)
        self.assertEqual((10,2), result.shape)

    def test_correlate_categories_multi(self):
        tools = self.tools
        df = pd.DataFrame(columns=['cat'], data=list("ABCDEFGH"))
        df['cat'] = df['cat'].astype('category')
        correlation = [list("ABC"), list("DEFGH")]
        action = {0: False, 1: True}
        result = tools.correlate_categories(df, 'cat', correlations=correlation, actions=action)
        self.assertEqual([False, False, False, True, True, True, True, True], result)

    def test_correlate_categories_nulls(self):
        tools = self.tools
        builder = SyntheticBuilder.from_memory().tools
        df = pd.DataFrame()
        df['pcp_tax_id'] = builder.get_category(selection=['993406113', '133757370', '260089066', '448512481', '546434723'],
                                                quantity=0.9, size=100, column_name='pcp_tax_id')
        correlations = ['993406113', '133757370', '260089066', '448512481', '546434723']
        actions = {0: 'LABCORP OF AMERICA', 1: 'LPCH MEDICAL GROUP', 2: 'ST JOSEPH HERITAGE MEDICAL',
                   3: 'MONARCH HEALTHCARE', 4: 'PRIVIA MEICAL GROUP'}
        df['pcp_name'] = tools.correlate_categories(df, header='pcp_tax_id', correlations=correlations,
                                                    actions=actions, column_name='pcp_name')
        print(df.head())

    def test_expit(self):
        tools = self.tools
        df = pd.DataFrame(columns=['num'], data=[-2, 1, 0, -2, 2, 0])
        result = tools.correlate_sigmoid(df, header='num')
        self.assertEqual([0.119, 0.731, 0.5, 0.119, 0.881, 0.5], result)

    def test_correlate_date(self):
        tools = self.tools
        df = pd.DataFrame(columns=['dates'], data=['2019/01/30', '2019/02/12', '2019/03/07', '2019/03/07'])
        result = tools.correlate_dates(df, 'dates', date_format='%Y/%m/%d')
        self.assertEqual(df['dates'].to_list(), result)
        # offset
        result = tools.correlate_dates(df, 'dates', offset=2, date_format='%Y/%m/%d')
        self.assertEqual(['2019/02/01', '2019/02/14', '2019/03/09', '2019/03/09'], result)
        result = tools.correlate_dates(df, 'dates', offset=-2, date_format='%Y/%m/%d')
        self.assertEqual(['2019/01/28', '2019/02/10', '2019/03/05', '2019/03/05'], result)
        result = tools.correlate_dates(df, 'dates', offset={'years': 1, 'months': 2}, date_format='%Y/%m/%d')
        self.assertEqual(['2020/03/30', '2020/04/12', '2020/05/07', '2020/05/07'], result)
        result = tools.correlate_dates(df, 'dates', offset={'years': -1, 'months': 2}, date_format='%Y/%m/%d')
        self.assertEqual(['2018/03/30', '2018/04/12', '2018/05/07', '2018/05/07'], result)
        # jitter
        df = pd.DataFrame(columns=['dates'], data=tools._get_datetime("2018/01/01,", '2018/01/02', size=1000))
        result = tools.correlate_dates(df, 'dates', jitter=5, jitter_units='D')
        loss = pd.Series(result) - df['dates']
        self.assertEqual(5, loss.value_counts().size)
        result = tools.correlate_dates(df, 'dates', jitter=5, jitter_units='s')
        loss = pd.Series(result) - df['dates']
        self.assertEqual(5, loss.value_counts().size)
        # jitter weighting
        result = tools.correlate_dates(df, 'dates', jitter=5, jitter_units='D', jitter_freq=[0, 0, 1, 1, 1])
        loss = pd.Series(result) - df['dates']
        self.assertEqual(3, loss.value_counts().size)
        self.assertEqual(0, loss.loc[loss.apply(lambda x: x.days < 0)].size)
        self.assertEqual(1000, loss.loc[loss.apply(lambda x: x.days >= 0)].size)
        # nulls
        df = pd.DataFrame(columns=['dates'], data=['2019/01/30', np.nan, '2019/03/07', '2019/03/07'])
        result = tools.correlate_dates(df, 'dates')
        self.assertEqual('NaT', str(result[1]))

    def test_correlate_date_min_max(self):
        tools = self.tools
        # control
        df = pd.DataFrame(columns=['dates'], data=tools._get_datetime("2018/01/01", '2018/01/02', size=1000))
        result = tools.correlate_dates(df, 'dates', jitter=5, date_format='%Y/%m/%d')
        self.assertEqual("2017/12/30", pd.Series(result).min())
        # min
        result = tools.correlate_dates(df, 'dates', jitter=5, min_date="2018/01/01", date_format='%Y/%m/%d')
        self.assertEqual("2018/01/01", pd.Series(result).min())
        self.assertEqual("2018/01/03", pd.Series(result).max())
        # max
        result = tools.correlate_dates(df, 'dates', jitter=5, max_date="2018/01/01", date_format='%Y/%m/%d')
        self.assertEqual("2018/01/01", pd.Series(result).max())
        self.assertEqual("2017/12/30", pd.Series(result).min())

    def test_correlate_date_as_delta(self):
        tools = self.tools
        # control
        df = pd.DataFrame(columns=['dates'], data=['1964-01-14', '1967-09-28', '1996-05-22', '1997-12-11'])
        result = tools.correlate_dates(df, 'dates', now_delta='Y')
        self.assertEqual([57, 53, 24, 23], result)

    def test_correlate_missing(self):
        tools = self.tools
        df = pd.DataFrame()
        df['cats'] = ['a', 'b', None, 'f', None, 'f', 'b', 'c', 'b', 'a']
        result = tools.correlate_missing(df, header='cats', as_type='category', seed=1973)
        self.assertEqual(['a', 'b', 'b', 'f', 'b', 'f', 'b', 'c', 'b', 'a'], result)


if __name__ == '__main__':
    unittest.main()
