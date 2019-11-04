from pprint import pprint

import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import seaborn as sns
import unittest
import warnings

from ds_discovery.transition.discovery import DataDiscovery as Discover
from ds_discovery.cleaners.pandas_cleaners import PandasCleaners as Cleaner
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

    @ignore_warnings
    def test_analyse_date(self):
        tools = DataBuilderTools()
        str_dates = tools.get_datetime('12/01/2016', '12/01/2018', date_format='%d-%m-%Y', size=10, seed=31)
        ts_dates = tools.get_datetime('12/01/2016', '12/01/2018', size=10, seed=31)
        result = Discover.analyse_date(str_dates, granularity=6, chunk_size=1)
        control =  [10.0, 20.0, 30.0, 10.0, 10.0, 20.0]
        self.assertEqual(control, result.get('weighting'))
        result = Discover.analyse_date(ts_dates, granularity=2, chunk_size=1, date_format='%d-%m-%Y')
        control = [60.0, 40.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual(('09-02-2016','02-12-2017',2), (result.get('lower'), result.get('upper'), result.get('granularity')))
        result = Discover.analyse_date(ts_dates, granularity=pd.Timedelta(days=365), chunk_size=1, date_format='%d-%m-%Y')
        control = [60.0, 40.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual(('09-02-2016','02-12-2017',365.0), (result.get('lower'), result.get('upper'), result.get('granularity')))
        control = {'dropped': [0],
                   'dtype': 'date',
                   'granularity': 365.0,
                   'lower': '09-02-2016',
                   'null_values': [0.0],
                   'sample': [10],
                   'selection': [('09-02-2016', '08-02-2017'), ('08-02-2017', '08-02-2018')],
                   'upper': '02-12-2017',
                   'weighting': [60.0, 40.0]}
        self.assertEqual(control, result)
        result = Discover.analyse_date(['12/12/2015'], granularity=2, date_format='%d-%m-%Y')
        control = {'dropped': [0],
                   'dtype': 'date',
                   'granularity': 2,
                   'lower': '12-12-2015',
                   'quantity': 0.0,
                   'sample': [1],
                   'selection': [('12-12-2015', '12-12-2015')],
                   'upper': '12-12-2015',
                   'weighting': [100.0]}
        self.assertEqual(control, result)

    def test_analysis_dictionary(self):
        tools = DataBuilderTools()
        df = tools.get_profiles(size=10, mf_weighting=[60, 40], seed=31)
        df['str_dates'] = tools.get_datetime('12/01/2016', '12/01/2018', date_format='%d-%m-%Y', size=10, seed=31)
        df['ts_dates'] = tools.get_datetime('12/01/2016', '12/01/2018', size=10, seed=31)
        df['age'] = tools.get_number(20, 90, weight_pattern=[2,3,6,3,2,6,7,4,2,1,0.5], size=10, seed=31)
        df['fare'] = tools.get_number(5.0, 300.0, weight_pattern=[2,7,3,1,0,0,0,0,0,0,1], precision=2, size=10, seed=31)
        result = Discover.analysis_dictionary(df, granularity=4, col_kwargs={'age': {'granularity': 2}})
        print(Cleaner.filter_columns(result, headers=['Attribute', 'Type', 'Granularity']))

    def test_analyse_value(self):
        dataset = [1,1,1,1,1,1,1,1,1,1]
        result = Discover.analyse_number(dataset, granularity=10)
        control = [100.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual((1,1,10), (result.get('lower'), result.get('upper'), result.get('granularity')))
        self.assertEqual([10], result.get('sample'))
        dataset = [1,2,3,4,5,6,7,8,9,10]

        result = Discover.analyse_number(dataset, granularity=2)
        control = [50.0, 50.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual((1,10), (result.get('lower'), result.get('upper')))
        control = [(1.0, 5.5, 'both'), (5.5, 10.0, 'right')]
        self.assertEqual(control, result.get('granularity'))

        result = Discover.analyse_number(dataset, granularity=3.0)
        control = [30.0, 30.0, 40.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual((1,10), (result.get('lower'), result.get('upper')))
        control = [(1.0, 4.0, 'left'), (4.0, 7.0, 'left'), (7.0, 10.0, 'both')]
        self.assertEqual(control, result.get('granularity'))

        result = Discover.analyse_number(dataset, granularity=2.0)
        control = [20.0, 20.0, 20.0, 20.0, 20.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual((1,10), (result.get('lower'), result.get('upper')))
        control = [(1.0, 3.0, 'left'), (3.0, 5.0, 'left'), (5.0, 7.0, 'left'), (7.0, 9.0, 'left'), (9.0, 11.0, 'both')]
        self.assertEqual(control, result.get('granularity'))

        result = Discover.analyse_number(dataset, granularity=1.0, lower=0, upper=5)
        control = [0.0, 20.0, 20.0, 20.0, 20.0, 20.0]
        self.assertEqual(control, result.get('weighting'))
        self.assertEqual((0,5), (result.get('lower'), result.get('upper')))
        control = [(0.0, 1.0, 'left'), (1.0, 2.0, 'left'), (2.0, 3.0, 'left'), (3.0, 4.0, 'left'), (4.0, 5.0, 'left'), (5.0, 6.0, 'both')]
        self.assertEqual(control, result.get('granularity'))

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
        # Test nulls and chunks
        dataset = [1, np.nan, 3,4,5,6,7,8,np.nan]
        result = Discover.analyse_number(dataset, lower=2, upper=8, chunk_size=2)
        control = {'dropped': [1, 0],
                   'dtype': 'number',
                   'granularity': 3,
                   'lower': 2,
                   'null_values': [25.0, 25.0],
                   'sample': [3, 3],
                   'selection': [(2, 4), (4, 6), (6, 8)],
                   'upper': 8,
                   'weighting': [[25.0, 50.0, 0.0], [0.0, 0.0, 75.0]]}
        self.assertEqual(control, result)


    def test_analyse_value_chunk(self):
        tools = Discover()
        values = [1,2,2,1,np.nan,4,2,1,3,9,9,18,21, np.nan]
        result = tools.analyse_number(values, granularity=3, chunk_size=2, replace_weight_zero=0.1)
        control = [[71.43, 0.1, 14.29], [28.57, 28.57, 28.57]]
        self.assertEqual(control, result.get('weighting'))
        control = [14.29, 14.29]
        self.assertEqual(control, result.get('null_values'))
        self.assertEqual([6,6], result.get('sample'))

    def test_analyse_cat(self):
        tools = DataBuilderTools()
        df = sns.load_dataset('titanic')
        df.loc[0, ('sex')] = 'unknown'
        df.loc[600:620, ('sex')] = 'unknown'
        result = Discover.analyse_category(df['sex'], chunk_size=2)
        control = {'dtype': 'category', 'selection': ['unknown', 'female', 'male'], 'sample': [446, 445],
                   'weighting': [[0.22, 38.34, 61.43], [4.72, 30.34, 64.94]], 'null_values': [0.0, 0.0]}
        self.assertEqual(control, result)
        df = tools.get_profiles(size=100, mf_weighting=[60, 40], seed=31, quantity=90.0)
        result = Discover.analyse_category(df['gender'])
        control = {'selection': ['F', 'M'], 'weighting': [34.44, 65.56], 'sample': [90], 'dtype': 'category', 'null_values': [10.0]}
        self.assertEqual(control, result)

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
