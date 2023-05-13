import datetime
import unittest
import os
import shutil
import numpy as np
import pandas as pd
from aistac import ConnectorContract

from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
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

    def test_correlate_mark_outliers(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame()
        df["number"] = tools.get_dist_normal(2,1,size=1000, seed=99)
        result = tools.correlate_mark_outliers(canonical=df, header="number", measure=1.5, method='quartile')
        df['quartile'] = result
        result = tools.correlate_mark_outliers(canonical=df, header="number", measure=3, method='empirical')
        df['empirical'] = result
        self.assertEqual([992, 8], df['quartile'].value_counts().values.tolist())
        self.assertEqual([995, 5], df['empirical'].value_counts().values.tolist())

    def test_correlate_custom(self):
        tools = self.tools
        df = pd.DataFrame()
        df['A'] = [1, 2, 3]
        result = tools.correlate_custom(df, code_str="[x + 2 for x in @['A']]")
        self.assertEqual([3, 4, 5], result)
        result = tools.correlate_custom(df, code_str="[True if x == $v1 else False for x in @['A']]", v1=2)
        self.assertEqual([False, True, False], result)

    def test_correlate_list_element(self):
        tools = self.tools
        df = pd.DataFrame()
        df['A'] = [[1,2,4,6], [1], [2,4,8,1], [2,4]]
        result = tools.correlate_list_element(df, header='A', list_size=2)
        control = [[1, 2], [1], [2, 4], [2, 4]]
        self.assertEqual(control, result)
        result = tools.correlate_list_element(df, header='A', list_size=1)
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
    def test_correlate_values_from_selection(self):
        sb = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = sb.tools
        sample_size = 10
        df = pd.DataFrame()
        df['creation'] = tools.get_number(10, 99, size=sample_size)
        df['process'] = tools.correlate_values(df, header="creation", offset=1)
        # repeate and overlay this time on 5%
        df['latent_flag'] = tools.get_dist_bernoulli(probability=0.3, size=sample_size)
        print(df)
        # use select to overlay
        selection = [tools.select2dict(column='latent_flag', condition='@==1')]
        action = tools.action2dict(method='correlate_values', header="creation", offset=2)
        default = tools.action2dict(method='@header', header='process')
        df['process'] = tools.correlate_selection(df, selection=selection, action=action, default_action=default)
        print(df.head())

    def test_correlate_values(self):
        tools = self.tools
        df = pd.DataFrame(data=[1,2,3,4.0,5,6,7,8,9,0], columns=['numbers'])
        result = tools.correlate_values(df, 'numbers', precision=0)
        self.assertCountEqual([1,2,3,4,5,6,7,8,9,0], result)
        # Offset
        df = pd.DataFrame(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], columns=['numbers'])
        result = tools.correlate_values(df, 'numbers', offset=1, precision=0)
        self.assertEqual([2,3,4,5,6,7,8,9,10,1], result)
        # str offset
        os.environ['CORR_VAL_TEST'] = '3'
        df = pd.DataFrame(data=[1, 2, 3, 4], columns=['numbers'])
        result = tools.correlate_values(df, 'numbers', offset='${CORR_VAL_TEST}', precision=0)
        self.assertEqual([4,5,6,7], result)
        # set transform
        df = pd.DataFrame(data=[1, 2, 3, 4], columns=['numbers'])
        result = tools.correlate_values(df, 'numbers', transform='lambda x: x + 2 if x <= 2 else x', precision=0)
        self.assertEqual([3, 4, 3, 4], result)

    def test_correlate_values_jitter(self):
        tools = self.tools
        # jitter
        df = pd.DataFrame(data=([2] * 5) + ([1] * 5), columns=['numbers'])
        result = tools.correlate_values(df, 'numbers', jitter=1, precision=1, choice=3, seed=31)
        self.assertEqual([2.0, 2.0, 2.0, 2.0, 2.4, 1.0, 1.0, 1.0, 1.9, 0.7], result)
        # jitter is zero
        result = tools.correlate_values(df, 'numbers', jitter=0, precision=1, seed=31)
        self.assertEqual([2, 2, 2, 2, 2, 1, 1, 1, 1, 1], result)
        # data has zero std
        df = pd.DataFrame(data=[2] * 10, columns=['numbers'])
        result = tools.correlate_values(df, 'numbers', precision=3, choice=5, jitter=1, seed=31)
        self.assertEqual(20, pd.Series(result).sum())
        # str jitter
        os.environ['CORR_VAL_TEST'] = '1'
        df = pd.DataFrame(data=[1, 2, 3, 4], columns=['numbers'])
        result = tools.correlate_values(df, 'numbers', jitter='${CORR_VAL_TEST}', precision=0, seed=31)
        self.assertEqual([0, 4, 4, 7], result)
        # loss
        df = pd.DataFrame(data=[1, 1, 2, 3, 5, 8, 13,] * 100, columns=['numbers'])
        result = tools.correlate_values(df, 'numbers', jitter=1, precision=1, seed=31)
        result = pd.Series(result)
        std = round(result.std())
        loss = abs(df['numbers'] - result)
        # loss is less than 4 stds
        std1 = [tools.s2d(column='default', condition=f'@<={std}')]
        std2 = [tools.s2d(column='default', condition=f'@<={std * 2}'), tools.s2d(column='default', condition=f'@>{std}', logic='AND')]
        std3 = [tools.s2d(column='default', condition=f'@>{std * 2}', logic='AND')]
        diff = tools.frame_selection(loss, selection=std1)
        # print(np.round(diff.shape[0]/loss.shape[0],5))
        self.assertTrue(np.round(diff.shape[0]/loss.shape[0],2) < 0.87)
        diff = tools.frame_selection(loss, selection=std2)
        # print(np.round(diff.shape[0]/loss.shape[0],5))
        self.assertTrue(np.round(diff.shape[0]/loss.shape[0],2) < 0.14)
        diff = tools.frame_selection(loss, selection=std3)
        # print(np.round(diff.shape[0]/loss.shape[0],5))
        self.assertTrue(np.round(diff.shape[0]/loss.shape[0], 2) < 0.01)

    def test_correlate_values_transform(self):
        tools = self.tools
        df = pd.DataFrame(data=[1,2,3,4.0,5,6,7,8,9,0], columns=['numbers'])
        result = tools.correlate_values(df, 'numbers', transform="lambda x: x+1 if x %2==0 else x", precision=0)
        self.assertEqual([1, 3, 3, 5, 5, 7, 7, 9, 9, 1],result)
        result = tools.correlate_values(df, 'numbers', transform="@ + 7 - @", precision=0)
        self.assertEqual([7]*10,result)

    def test_correlate_normalize(self):
        tools = self.tools
        df = pd.DataFrame(data=[1,2,2,3,3,2,2,1], columns=['numbers'])
        result = tools.correlate_numbers(df, header='numbers', normalize=True)
        self.assertEqual([0.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.0], result)

    def test_correlate_standardise(self):
        tools = self.tools
        df = pd.DataFrame(data=[1,2,2,3,3,2,2,1], columns=['numbers'])
        result = tools.correlate_numbers(df, header='numbers', standardize=True, precision=1)
        self.assertEqual([-1.4, 0.0, 0.0, 1.4, 1.4, 0.0, 0.0, -1.4], result)

    def test_correlate_scalarize(self):
        tools = self.tools
        df = pd.DataFrame(data=[1,2,2,3,3,2,2,1], columns=['numbers'])
        result = tools.correlate_numbers(df, header='numbers', scalar=(-1, 1))
        self.assertEqual([-1.0, 0, 0, 1.0, 1.0, 0, 0, -1.0], result)

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

    def test_correlate_categories_selection(self):
        tools = self.tools
        df = pd.DataFrame(columns=['cat'], data=list("ABACDBA"))
        correlation = [[tools.select2dict(column='cat', condition="@=='A'")], [tools.select2dict(column='cat', condition="@=='B'")]]
        action = {0: 'F', 1: 'G'}
        default = 'H'
        result = tools.correlate_categories(df, 'cat', correlations=correlation, actions=action, default_action=default)
        self.assertEqual(['F', 'G', 'F', 'H', 'H', 'G', 'F'], result)
        correlation = [[tools.select2dict(column='cat', condition="@=='A'")], ['B', 'C'], 'D']
        result = tools.correlate_categories(df, 'cat', correlations=correlation, actions=action, default_action=default)
        self.assertEqual(['F', 'G', 'F', 'G', 'H', 'G', 'F'], result)
        # use with numbers
        df = pd.DataFrame(columns=['cat'], data=[1,2,3,4,2,1])
        correlation = [[tools.select2dict(column='cat', condition="@<=2")],
                       [tools.select2dict(column='cat', condition="@==3")]]
        result = tools.correlate_categories(df, 'cat', correlations=correlation, actions=action, default_action=default)
        self.assertEqual(['F', 'F', 'G', 'H', 'F', 'F'], result)

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
        self.assertEqual((10, 2), result.shape)

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

    def test_correlate_activation(self):
        tools = self.tools
        df = pd.DataFrame(columns=['num'], data=[-2, -1, 0, 2, 3, 4])
        result = tools._correlate_activation(df, header='num', activation='sigmoid', precision=3)
        self.assertEqual([0.119, 0.269, 0.5, 0.881, 0.953, 0.982], result)
        result = tools._correlate_activation(df, header='num', activation='tanh', precision=3)
        self.assertEqual([-0.964, -0.762, 0.0, 0.964, 0.995, 0.999], result)
        result = tools._correlate_activation(df, header='num', activation='ReLU')
        self.assertEqual([0, 0, 0, 2, 3, 4], result)
        with self.assertRaises(ValueError) as context:
            result = tools._correlate_activation(df, header='num', activation='Softmax')
        self.assertTrue("The activation function 'softmax' is not supported." in str(context.exception))

    def test_correlate_dates_from_selection(self):
        sb = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = sb.tools
        sample_size = 10
        df = pd.DataFrame()
        df['creationDate'] = tools.get_datetime(start=-30, until=-14, relative_freq=[0.001, 0.1, 1, 3, 5, 3, 2, 2, 2], size=sample_size, ignore_time=True)
        df['processDate'] = tools.correlate_dates(df, header="creationDate", offset={'days': 1}, jitter=3, ignore_time=True)
        # repeate and overlay this time on 5%
        df['latent_flag'] = tools.get_dist_choice(3, size=sample_size)

        # use select to overlay
        selection = [tools.select2dict(column='latent_flag', condition='@==1')]
        action = tools.action2dict(method='correlate_dates', header="creationDate", offset={'minutes': 20}, jitter=3, ignore_time=True)
        default = tools.action2dict(method='@header', header='processDate')
        df['processDate'] = tools.correlate_selection(df, selection=selection, action=action, default_action=default)
        print((df['processDate'] - df['creationDate']).sum().days)


    def test_correlate_dates_jitter(self):
        sb = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = sb.tools
        sample_size = 10
        df = pd.DataFrame()
        df['creationDate'] = tools.get_datetime(start=-30, until=-14, ordered=True, ignore_time=True, size=sample_size)
        df['processDate'] = tools.correlate_dates(df, header="creationDate", ignore_time=True, offset={'days': 10},
                                                  jitter=1, jitter_units='D')
        print((df['processDate'] - df['creationDate']))

    def test_correlate_dates_choice(self):
        sb = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = sb.tools
        sample_size = 10
        df = pd.DataFrame()
        df['creationDate'] = tools.get_datetime(start=-30, until=-14, ordered=True, ignore_time=True, size=sample_size)
        df['processDate'] = tools.correlate_dates(df, header="creationDate", ignore_time=True, offset={'days': 10},
                                                 choice=4, jitter=1, jitter_units='D')
        print(df['processDate'] - df['creationDate'])

    def test_correlate_dates(self):
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
        now = datetime.datetime.now()
        df = pd.DataFrame(columns=['dates'], data=tools._get_datetime(now, now + datetime.timedelta(days=1), size=1000, seed=31))
        df['result'] = tools.correlate_dates(df, 'dates', jitter=1, jitter_units='D', seed=31)
        loss = tools.correlate_dates(df, header='result', now_delta='D')
        self.assertEqual([579, 329, 83, 9], pd.Series(loss).value_counts().to_list())
        # nulls
        df = pd.DataFrame(columns=['dates'], data=['2019/01/30', np.nan, '2019/03/07', '2019/03/07'])
        result = tools.correlate_dates(df, 'dates')
        self.assertEqual('NaT', str(result[1]))

    def test_correlate_date_min_max(self):
        tools = self.tools
        # control
        df = pd.DataFrame(columns=['dates'], data=tools._get_datetime("2018/01/01", '2018/01/02', size=1000, seed=31))
        result = tools.correlate_dates(df, 'dates', jitter=5, jitter_units='D', date_format='%Y/%m/%d', seed=31)
        self.assertEqual("2017/12/14", pd.Series(result).min())
        self.assertEqual("2018/01/18", pd.Series(result).max())
        # min
        result = tools.correlate_dates(df, 'dates', jitter=5, jitter_units='D', min_date="2018/01/01", date_format='%Y/%m/%d', seed=31)
        self.assertEqual("2018/01/01", pd.Series(result).min())
        self.assertEqual("2018/01/18", pd.Series(result).max())
        # max
        result = tools.correlate_dates(df, 'dates', jitter=5, jitter_units='D', max_date="2018/01/01", date_format='%Y/%m/%d', seed=31)
        self.assertEqual("2017/12/14", pd.Series(result).min())
        self.assertEqual("2018/01/01", pd.Series(result).max())

    def test_correlate_date_as_delta(self):
        tools = self.tools
        # control
        now = pd.Timestamp.now()
        df = pd.DataFrame(columns=['dates'], data=[now - pd.DateOffset(years=52), now - pd.DateOffset(years=20)])
        result = tools.correlate_dates(df, 'dates', now_delta='Y')
        self.assertEqual([52, 20], result)

    def test_correlate_missing(self):
        tools = self.tools
        df = pd.DataFrame()
        df['age'] = [2, 1, 2, 2, 9, 1, None, None]
        result = tools.correlate_missing(df, header='age', method='mean')
        self.assertEqual([2.0, 1.0, 2.0, 2.0, 9.0, 1.0, 2.833, 2.833], result)
        result = tools.correlate_missing(df, header='age', method='mode')
        self.assertEqual([2.0, 1.0, 2.0, 2.0, 9.0, 1.0, 2.0, 2.0], result)
        result = tools.correlate_missing(df, header='age', method='median')
        self.assertEqual([2.0, 1.0, 2.0, 2.0, 9.0, 1.0, 2.0, 2.0], result)
        result = tools.correlate_missing(df, header='age', method='mean', precision=0)
        self.assertEqual([2.0, 1.0, 2.0, 2.0, 9.0, 1.0, 3.0, 3.0], result)
        result = tools.correlate_missing(df, header='age', method='random', seed=0)
        self.assertEqual([2.0, 1.0, 2.0, 2.0, 9.0, 1.0, 1.0, 2.0], result)
        result = tools.correlate_missing(df, header='age', method='neighbour', seed=0)
        self.assertEqual([2.0, 1.0, 2.0, 2.0, 9.0, 1.0, 2.833, 2.833], result)
        result = tools.correlate_missing(df, header='age', method='indicator')
        self.assertEqual([0,0,0,0,0,0,1,1], result)
        df['cat'] = ['A', 'C', 'A', 'B', 'A', 'C', None, None]
        result = tools.correlate_missing(df, header='cat', method='mode')
        self.assertEqual(['A', 'C', 'A', 'B', 'A', 'C', 'A', 'A'], result)
        result = tools.correlate_missing(df, header='cat', method='random', seed=31)
        self.assertEqual(['A', 'C', 'A', 'B', 'A', 'C', 'B', 'C'], result)
        result = tools.correlate_missing(df, header='cat', method='indicator')
        self.assertEqual([0,0,0,0,0,0,1,1], result)
        with self.assertRaises(ValueError) as context:
            result = tools.correlate_missing(df, header='cat', method='mean')
        self.assertTrue("The header 'cat' is not numeric and thus not compatible" in str(context.exception))

    def test_correlate_missing_constant(self):
        tools = self.tools
        df = pd.DataFrame()
        df['age'] = [2, 1, 2, 2, 9, 1, None, None]
        df['cat'] = ['A', 'C', 'A', 'B', 'A', 'C', None, None]
        result = tools.correlate_missing(df, header='cat', method='constant', constant='U')
        self.assertEqual(['A', 'C', 'A', 'B', 'A', 'C', 'U', 'U'], result)
        result = tools.correlate_missing(df, header='age', method='constant', constant=-1)
        self.assertEqual([2.0, 1.0, 2.0, 2.0, 9.0, 1.0, -1.0, -1.0], result)
        with self.assertRaises(ValueError) as context:
            result = tools.correlate_missing(df, header='age', method='constant', constant='U')
        self.assertTrue("The value 'U' is a string and column 'age' expects a numeric value" in str(context.exception))
        with self.assertRaises(ValueError) as context:
            result = tools.correlate_missing(df, header='age', method='constant')
        self.assertTrue("When using the 'constant' method a constant value must be provided" in str(context.exception))

    def test_correlate_discrete(self):
        tools = self.tools
        df = pd.DataFrame()
        df['age'] = [0, 1, 2, 0]
        result = tools.correlate_discrete_intervals(df, header='age', categories=['low', 'mid', 'high'])
        self.assertEqual(['low', 'mid', 'high', 'low'], result)

    def test_correlate_continuous(self):
        tools = self.tools
        df = pd.DataFrame()
        df['num'] = [0, 1, 2, 0, 6, 4, 2, 2]
        df['letter'] = list('asdfghcv')
        result = tools._correlate_values(df, header='num', seed=31)
        self.assertEqual(df['num'].to_list(), result)
        result = tools._correlate_values(df, header='num', offset=10, seed=31)
        self.assertEqual([10, 11, 12, 10, 16, 14, 12, 12], result)
        result = tools._correlate_values(df, header='num', jitter=1, precision=1, seed=73)
        self.assertEqual([0.4, 1.1, 2.0, 0.3, 6.0, 3.2, 1.3, 2.2], result)
        result = tools._correlate_values(df, header='num', jitter=3, precision=0, seed=73)
        self.assertTrue(isinstance(all(result), int))
        self.assertEqual([1, 1, 2, 1, 6, 2, 0, 3], result)
        result = tools._correlate_values(df, header='num', choice=3, offset=10, seed=31)
        self.assertEqual([0, 1, 2, 10, 6, 4, 12, 12], result)
        result = tools._correlate_values(df, header='num', jitter=1, keep_zero=True, precision=2, seed=73)
        self.assertEqual([0.0, 1.1, 2.03, 0.0, 5.98, 3.21, 1.31, 2.18], result)
        df['num'] = [0, 1, 2, 0, 6, 4, None, 2]
        result = tools._correlate_values(df, header='num', jitter=1, precision=0, seed=73)
        self.assertTrue(pd.Series(result).isna()[6])
        self.assertEqual(7, pd.Series(result).dropna().size)
        self.assertEqual(8, pd.Series(result).size)


if __name__ == '__main__':
    unittest.main()
