from pprint import pprint

import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import seaborn as sns
import unittest
import warnings

from ds_discovery.transition.discovery import DataDiscovery as Discover, DataAnalytics
from ds_discovery.intent.pandas_transition_intent import PandasTransitionIntent as Cleaner
from ds_behavioral.generator.data_builder_tools import DataBuilderTools

def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)

    return do_test


class DiscoveryAnalysisMethod(unittest.TestCase):

    def setUp(self):
         pass

    def tearDown(self):
        pass

    def test_runs(self):
        """Basic smoke test"""
        Discover()

    def test_discovery_analytics_class(self):
        tools = DataBuilderTools()
        dataset = tools.get_category(list('ABCDE')+[np.nan], weight_pattern=[1,3,2,7,4], size=694)
        result = Discover.analyse_category(dataset)
        analytics = DataAnalytics('tester', result)
        print(analytics.dominance_map)


    def test_analyse_category(self):
        tools = DataBuilderTools()
        dataset = tools.get_category(list('ABCDE')+[np.nan], weight_pattern=[1,3,2,7,4], size=694)
        result = Discover.analyse_category(dataset)
        control = ['intent', 'patterns', 'stats']
        self.assertCountEqual(control, list(result.keys()))
        control = ['dtype', 'selection', 'upper', 'lower', 'granularity', 'weighting_precision']
        self.assertCountEqual(control, list(result.get('intent').keys()))
        control = ['weight_pattern']
        self.assertCountEqual(control, list(result.get('patterns').keys()))
        control = ['outlier_percent', 'nulls_percent', 'sample']
        self.assertCountEqual(control, list(result.get('stats').keys()))

    def test_analyse_category_limits(self):
        top = 2
        dataset = ['A']*8 + ['B']*6 + ['C']*4 + ['D']*2
        result = Discover.analyse_category(dataset, top=top, weighting_precision=0)
        control = ['dtype', 'selection', 'top', 'upper', 'lower', 'granularity', 'weighting_precision']
        self.assertCountEqual(control, list(result.get('intent').keys()))
        self.assertEqual(top, result.get('intent').get('top'))
        self.assertEqual(top, len(result.get('intent').get('selection')))
        self.assertCountEqual(['A', 'B'], result.get('intent').get('selection'))
        self.assertCountEqual([40, 30], result.get('patterns').get('weight_pattern'))
        self.assertEqual(30, result.get('stats').get('outlier_percent'))
        self.assertEqual(14, result.get('stats').get('sample'))
        lower = 0.2
        upper = 7
        result = Discover.analyse_category(dataset, lower=lower, upper=upper, weighting_precision=0)
        control = ['dtype', 'selection', 'upper', 'lower', 'granularity', 'weighting_precision']
        self.assertCountEqual(control, list(result.get('intent').keys()))
        self.assertEqual(lower, result.get('intent').get('lower'))
        self.assertEqual(upper, result.get('intent').get('upper'))
        self.assertCountEqual(['C', 'B'], result.get('intent').get('selection'))
        self.assertCountEqual([33, 50], result.get('patterns').get('weight_pattern'))
        self.assertEqual(50, result.get('stats').get('outlier_percent'))
        self.assertEqual(10, result.get('stats').get('sample'))

    def test_analyse_number_empty(self):
         dataset = []
         result = Discover.analyse_number(dataset)

         pprint(result)

         # control = [100.0]
        # self.assertEqual(control, result.get('weighting'))
        # self.assertEqual((1,1,10), (result.get('lower'), result.get('upper'), result.get('granularity')))
        # self.assertEqual([10], result.get('sample'))
        # dataset = [1,2,3,4,5,6,7,8,9,10]

        # result = Discover.analyse_number(dataset, granularity=2)
        # control = [50.0, 50.0]
        # self.assertEqual(control, result.get('weighting'))
        # self.assertEqual((1,10), (result.get('lower'), result.get('upper')))
        # control = [(1.0, 5.5, 'both'), (5.5, 10.0, 'right')]
        # self.assertEqual(control, result.get('granularity'))
        #
        # result = Discover.analyse_number(dataset, granularity=3.0)
        # control = [30.0, 30.0, 40.0]
        # self.assertEqual(control, result.get('weighting'))
        # self.assertEqual((1,10), (result.get('lower'), result.get('upper')))
        # control = [(1.0, 4.0, 'left'), (4.0, 7.0, 'left'), (7.0, 10.0, 'both')]
        # self.assertEqual(control, result.get('granularity'))
        #
        # result = Discover.analyse_number(dataset, granularity=2.0)
        # control = [20.0, 20.0, 20.0, 20.0, 20.0]
        # self.assertEqual(control, result.get('weighting'))
        # self.assertEqual((1,10), (result.get('lower'), result.get('upper')))
        # control = [(1.0, 3.0, 'left'), (3.0, 5.0, 'left'), (5.0, 7.0, 'left'), (7.0, 9.0, 'left'), (9.0, 11.0, 'both')]
        # self.assertEqual(control, result.get('granularity'))
        #
        # result = Discover.analyse_number(dataset, granularity=1.0, lower=0, upper=5)
        # control = [0.0, 20.0, 20.0, 20.0, 20.0, 20.0]
        # self.assertEqual(control, result.get('weighting'))
        # self.assertEqual((0,5), (result.get('lower'), result.get('upper')))
        # control = [(0.0, 1.0, 'left'), (1.0, 2.0, 'left'), (2.0, 3.0, 'left'), (3.0, 4.0, 'left'), (4.0, 5.0, 'left'), (5.0, 6.0, 'both')]
        # self.assertEqual(control, result.get('granularity'))

    def test_number_zero_count(self):
        dataset = [1,0,0,1,1,1,0,0,1,0]
        result = Discover.analyse_number(dataset)
        self.assertEqual([50.0], result.get('zero_count'))
        dataset = [1,0,0,2,5,4,0,4,1,0]
        result = Discover.analyse_number(dataset, lower=0, granularity=3)
        self.assertEqual([40.0], result.get('zero_count'))
        self.assertEqual([10], result.get('sample'))
        result = Discover.analyse_number(dataset, lower=1, granularity=3)
        self.assertEqual([40.0], result.get('zero_count'))
        self.assertEqual([6], result.get('sample'))

    def test_analysis_granularity_list(self):
        dataset = [0,1,2,3]
        intervals = [(0,1,'both'),(1,2),(2,3)]
        result = Discover.analyse_number(dataset, granularity=intervals)
        control = [50.0, 25.0, 25.0]
        self.assertEqual(control, result.get('weighting'))
        intervals = [(0,1,'both'),(1,2,'both'),(2,3)]
        result = Discover.analyse_number(dataset, granularity=intervals)
        control = [50.0, 50.0, 25.0]
        self.assertEqual(control, result.get('weighting'))
        # percentile
        dataset = [0, 0, 2, 2, 1, 2, 3]
        percentile = [.25, .5, .75]
        result = Discover.analyse_number(dataset, granularity=percentile)
        pprint(result)

    def test_analyse_number_lower_upper(self):
        # Default
        dataset = list(range(1,10))
        result = Discover.analyse_number(dataset)
        control = {'dropped': [0],
                   'dtype': 'number',
                   'granularity': 3,
                   'lower': 1,
                   'null_values': [0.0],
                   'sample': [9],
                   'selection': [(1.0, 3.667), (3.667, 6.333), (6.333, 9.0)],
                   'upper': 9,
                   'weighting': [33.33, 33.33, 33.33]}
        self.assertEqual(control, result)
        # Outer Boundaries
        result = Discover.analyse_number(dataset, lower=0, upper=10)
        control = {'dropped': [0],
                   'dtype': 'number',
                   'granularity': 3,
                   'lower': 0,
                   'null_values': [0.0],
                   'sample': [9],
                   'selection': [(0.0, 3.333), (3.333, 6.667), (6.667, 10.0)],
                   'upper': 10,
                   'weighting': [33.33, 33.33, 33.33]}
        self.assertEqual(control, result)
        # Inner Boundaries
        result = Discover.analyse_number(dataset, lower=2, upper=8)
        control = {'dropped': [2],
                   'dtype': 'number',
                    'granularity': 3,
                    'lower': 2,
                    'null_values': [0.0],
                    'sample': [7],
                    'selection': [(2, 4), (4, 6), (6, 8)],
                    'upper': 8,
                    'weighting': [28.57, 28.57, 42.86]}
        self.assertEqual(control, result)

    @ignore_warnings
    def test_analyse_date(self):
        tools = DataBuilderTools()
        str_dates = tools.get_datetime('12/01/2016', '12/01/2018', date_format='%d-%m-%Y', size=10, seed=31)
        ts_dates = tools.get_datetime('12/01/2016', '12/01/2018', size=10, seed=31)
        result = Discover.analyse_date(str_dates, granularity=3, date_format='%Y-%m-%d')
        pprint(result)


    @ignore_warnings
    def test_analyse_associate_single(self):
        tools = DataBuilderTools()
        size = 50
        df = tools.get_profiles(size=size, mf_weighting=[60, 40], seed=31, quantity=90.0)
        # category
        columns_list = [{'gender': {}}]
        result = Discover.analyse_association(df, columns_list)
        control = {'gender': {'analysis': {'dtype': 'category',
                                           'null_values': [14.0],
                                           'sample': [43],
                                           'selection': ['F', 'M'],
                                           'weighting': [30.23, 69.77]},
                              'associate': 'gender'}}
        self.assertEqual(control, result)
        columns_list = [{'gender': {'chunk_size': 1, 'replace_zero': 0}}]
        result = Discover.analyse_association(df, columns_list)
        self.assertEqual(control, result)
        # number
        df['numbers'] = tools.get_number(from_value=1000, weight_pattern=[5,0,2], size=size, quantity=0.9, seed=31)
        columns_list = [{'numbers': {'type': 'number', 'granularity': 3}}]
        result = Discover.analyse_association(df, columns_list)
        control = {'numbers': {'analysis': {'dropped': [0],
                                            'dtype': 'number',
                                            'granularity': 3,
                                            'lower': 10.0,
                                            'null_values': [14.0],
                                            'sample': [43],
                                            'selection': [(10.0, 325.667),
                                                          (325.667, 641.333),
                                                          (641.333, 957.0)],
                                            'upper': 957.0,
                                            'weighting': [64.0, 0.0, 22.0]},
                               'associate': 'numbers'}}
        self.assertEqual(control, result)
        #dates
        df['dates'] = tools.get_datetime('10/10/2000', '31/12/2018', date_pattern=[1,9,4], size=size, quantity=0.9, seed=31)
        columns_list = [{'dates': {'dtype': 'datetime', 'granularity': 3, 'date_format': '%d-%m-%Y'}}]
        result = Discover.analyse_association(df, columns_list)
        control = {'dates': {'analysis': {'dropped': [0],
                                          'dtype': 'date',
                                          'granularity': 3,
                                          'lower': '30-09-2003',
                                          'null_values': [14.0],
                                          'sample': [43],
                                          'selection': [('30-09-2003', '27-07-2008'),
                                                        ('27-07-2008', '25-05-2013'),
                                                        ('25-05-2013', '22-03-2018')],
                                          'upper': '22-03-2018',
                                          'weighting': [28.0, 42.0, 16.0]},
                             'associate': 'dates'}}
        self.assertEqual(control, result)

    @ignore_warnings
    def test_multi_analyse_associate(self):
        tools = DataBuilderTools()
        size = 50
        df = tools.get_profiles(size=size, mf_weighting=[60, 40], seed=31, quantity=90.0)
        df['numbers'] = tools.get_number(from_value=1000, weight_pattern=[5,0,2], size=size, quantity=0.9, seed=31)
        df['dates'] = tools.get_datetime('10/10/2000', '31/12/2018', date_pattern=[1,9,4], size=size, quantity=0.9, seed=31)
        columns_list = [{'gender': {}}, {'numbers': {}}]
        result = Discover.analyse_association(df, columns_list)
        control = control_01()
        self.assertEqual(control, result)
        df = sns.load_dataset('titanic')
        columns_list = [{'sex': {}}, {'age': {}}, {'survived': {'dtype': 'category'}}]
        result = Discover.analyse_association(df, columns_list)
        control = control_02()
        self.assertEqual(control, result)

    def test_levels_analyse_associate(self):
        tools = DataBuilderTools()
        size = 50
        df = tools.get_profiles(mf_weighting=[60, 40], quantity=90.0, seed=31, size=size)
        df['lived'] = tools.get_category(selection=['yes', 'no'], quantity=80.0, seed=31, size=size)
        df['age'] = tools.get_number(from_value=20,to_value=80, weight_pattern=[1,2,5,6,2,1,0.5], seed=31, size=size)
        df['fare'] = tools.get_number(from_value=1000, weight_pattern=[5,0,2], size=size, quantity=0.9, seed=31)
        columns_list = [{'gender': {}, 'age':  {}}, {'lived': {}}]
        exclude = ['age.lived']
        result = Discover.analyse_association(df, columns_list, exclude)
        control = {'age': {'analysis': {'dropped': [0],'dtype': 'number',
                                        'granularity': 3,
                                        'lower': 24,
                                        'null_values': [0.0],
                                        'sample': [50],
                                        'selection': [(24.0, 41.333),
                                                      (41.333, 58.667),
                                                      (58.667, 76.0)],
                                        'upper': 76,
                                        'weighting': [42.0, 44.0, 14.0]},
                           'associate': 'age'},
                   'gender': {'analysis': {'dtype': 'category',
                                           'null_values': [14.0],
                                           'sample': [43],
                                           'selection': ['F', 'M'],
                                           'weighting': [30.23, 69.77]},
                              'associate': 'gender',
                              'sub_category': {'F': {'lived': {'analysis': {'dtype': 'category',
                                                                            'null_values': [7.69],
                                                                            'sample': [12],
                                                                            'selection': ['no'],
                                                                            'weighting': [100.0]},
                                                               'associate': 'gender.lived'}},
                                               'M': {'lived': {'analysis': {'dtype': 'category',
                                                                            'null_values': [6.67],
                                                                            'sample': [28],
                                                                            'selection': ['yes',
                                                                                          'no'],
                                                                            'weighting': [78.57,
                                                                                          21.43]},
                                                               'associate': 'gender.lived'}}}}}
        self.assertEqual(control, result)

    def get_weights(self, df, columns: list, index: int, weighting: dict):
        col = columns[index]
        weighting.update({col: Discover.analyse_category(df[col])})
        if index == len(columns)-1:
            return
        for category in weighting.get(col).get('selection'):
            if weighting.get(col).get('sub_category') is None:
                weighting[col].update({'sub_category': {}})
            weighting.get(col).get('sub_category').update({category: {}})
            sub_category = weighting.get(col).get('sub_category').get(category)
            self.get_weights(df[df[col] == category], columns, index + 1, sub_category)
        return

if __name__ == '__main__':
    unittest.main()


def control_01():
    return {'gender': {'analysis': {'dtype': 'category',
                                    'null_values': [14.0],
                                    'sample': [43],
                                    'selection': ['F', 'M'],
                                    'weighting': [30.23, 69.77]},
                       'associate': 'gender',
                       'sub_category': {'F': {'numbers': {'analysis': {'dropped': [0],'dtype': 'number',
                                                                       'granularity': 3,
                                                                       'lower': 14.0,
                                                                       'null_values': [0.0],
                                                                       'sample': [13],
                                                                       'selection': [(14.0,
                                                                                      328.333),
                                                                                     (328.333,
                                                                                      642.667),
                                                                                     (642.667,
                                                                                      957.0)],
                                                                       'upper': 957.0,
                                                                       'weighting': [15.38,
                                                                                     0.0,
                                                                                     84.62]},
                                                          'associate': 'gender.numbers'}},
                                        'M': {'numbers': {'analysis': {'dropped': [0],'dtype': 'number',
                                                                       'granularity': 3,
                                                                       'lower': 10.0,
                                                                       'null_values': [0.0],
                                                                       'sample': [30],
                                                                       'selection': [(10.0,
                                                                                      114.667),
                                                                                     (114.667,
                                                                                      219.333),
                                                                                     (219.333,
                                                                                      324.0)],
                                                                       'upper': 324.0,
                                                                       'weighting': [26.67,
                                                                                     30.0,
                                                                                     43.33]},
                                                          'associate': 'gender.numbers'}}}}}


def control_02():
    return {'sex': {'associate': 'sex',
                    'analysis': {'selection': ['male', 'female'], 'weighting': [64.76, 35.24], 'dtype': 'category',
                                 'null_values': [0.0], 'sample': [891]},
                    'sub_category': {
                        'male': {'age': {'associate': 'sex.age',
                                         'analysis': {'selection': [(0.42, 26.947), (26.947, 53.473), (53.473, 80.0)],
                                                      'weighting': [33.28, 38.82, 6.41], 'lower': 0.42, 'upper': 80.0,
                                                      'granularity': 3, 'dtype': 'number', 'null_values': [21.49],
                                                      'sample': [453],
                                                      'dropped': [0]},
                                         'sub_category': {(0.42, 26.947): {
                                             'survived': {'associate': 'sex.age.survived',
                                                          'analysis': {'selection': [0, 1], 'weighting': [79.58, 20.42],
                                                                       'dtype': 'category',
                                                                       'null_values': [0.0], 'sample': [191]}}},
                                             (26.947, 53.473): {
                                                 'survived': {'associate': 'sex.age.survived',
                                                              'analysis': {'selection': [0, 1],
                                                                           'weighting': [78.12, 21.88],
                                                                           'dtype': 'category',
                                                                           'null_values': [0.0], 'sample': [224]}}},
                                             (53.473, 80.0): {
                                                 'survived': {'associate': 'sex.age.survived',
                                                              'analysis': {'selection': [0, 1],
                                                                           'weighting': [89.19, 10.81],
                                                                           'dtype': 'category',
                                                                           'null_values': [0.0], 'sample': [37]}}}}}},
                        'female': {'age': {'associate': 'sex.age',
                                           'analysis': {'selection': [(0.75, 21.5), (21.5, 42.25), (42.25, 63.0)],
                                                        'weighting': [26.75, 43.31, 13.06], 'lower': 0.75,
                                                        'upper': 63.0,
                                                        'granularity': 3, 'dtype': 'number', 'null_values': [16.88],
                                                        'sample': [261], 'dropped': [0]},
                                           'sub_category': {(0.75, 21.5): {
                                               'survived': {'associate': 'sex.age.survived',
                                                            'analysis': {'selection': [1, 0],
                                                                         'weighting': [67.07, 32.93],
                                                                         'dtype': 'category',
                                                                         'null_values': [0.0], 'sample': [82]}}},
                                               (21.5, 42.25): {
                                                   'survived': {'associate': 'sex.age.survived',
                                                                'analysis': {'selection': [1, 0],
                                                                             'weighting': [79.41, 20.59],
                                                                             'dtype': 'category',
                                                                             'null_values': [0.0], 'sample': [136]}}},
                                               (42.25, 63.0): {
                                                   'survived': {'associate': 'sex.age.survived',
                                                                'analysis': {'selection': [1, 0],
                                                                             'weighting': [78.05, 21.95],
                                                                             'dtype': 'category',
                                                                             'null_values': [0.0],
                                                                             'sample': [41]}}}}}}}}}
