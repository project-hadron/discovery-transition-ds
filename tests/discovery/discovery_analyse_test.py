import shutil
from pprint import pprint
import os
import pandas as pd
import numpy as np
import seaborn as sns
import unittest

from ds_foundation.properties.property_manager import PropertyManager

from ds_discovery import Transition
from ds_discovery.transition.discovery import DataDiscovery as Discover, DataAnalytics
from ds_behavioral.component.synthetic_component import SyntheticBuilder


class DiscoveryAnalysisMethod(unittest.TestCase):

    def setUp(self):
        # set environment variables
        os.environ['AISTAC_PM_PATH'] = os.path.join(os.environ['PWD'], 'work', 'config')
        os.environ['AISTAC_DATA_PATH'] = os.path.join(os.environ['PWD'], 'work', 'data', '0_raw')
        PropertyManager._remove_all()
        try:
            shutil.rmtree('work')
        except:
            pass
        try:
            shutil.copytree('../data', os.path.join(os.environ['PWD'], 'work'))
        except:
            pass

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        Discover()

    def test_filter_univariate_roc_auc(self):
        data = pd.read_csv('../../../data/raw/paribas.csv', nrows=50000)
        result = Discover.filter_univariate_roc_auc(data, target='target', threshold=0.55)
        self.assertCountEqual(['v10', 'v129', 'v14', 'v62', 'v50'], result)
        classifier_kwargs = {'iterations': 2, 'learning_rate': 1, 'depth': 2}
        result = Discover.filter_univariate_roc_auc(data, target='target', threshold=0.55, package='catboost' ,
                                                    model='CatBoostClassifier', classifier_kwargs=classifier_kwargs,
                                                    fit_kwargs={'verbose': False})
        self.assertCountEqual(['v50', 'v10', 'v14', 'v12', 'v129', 'v62', 'v21', 'v34'], result)


    def test_filter_univariate_mse(self):
        data = pd.read_csv('../../../data/raw/ames_housing.csv', nrows=50000)
        result = Discover.filter_univariate_mse(data, target='SalePrice', as_series=True, )
        print(result)
        regressor_kwargs = {'iterations': 2, 'learning_rate': 1, 'depth': 2}
        result = Discover.filter_univariate_mse(data, target='SalePrice', as_series=True, package='catboost', model='CatBoostRegressor',
                                                regressor_kwargs=regressor_kwargs, fit_kwargs={'verbose': False})
        print(result)

    def test_filter_fisher_score(self):
        df = sns.load_dataset('titanic')
        result = Discover.filter_fisher_score(df, target='survived')
        self.assertEqual(['class', 'pclass', 'deck', 'parch', 'sibsp', 'age'], result)
        result = Discover.filter_fisher_score(df, target='survived', inc_zero_score=True)
        self.assertEqual(['class', 'pclass', 'deck', 'parch', 'sibsp', 'age', 'fare'], result)
        result = Discover.filter_fisher_score(df, target='survived', inc_zero_score=True, top=3)
        self.assertEqual(['class', 'pclass', 'deck'], result)
        result = Discover.filter_fisher_score(df, target='survived', top=0)
        self.assertEqual(['class', 'pclass', 'deck', 'parch', 'sibsp', 'age'], result)
        result = Discover.filter_fisher_score(df, target='survived', top=20)
        self.assertEqual(['class', 'pclass', 'deck', 'parch', 'sibsp', 'age'], result)
        result = Discover.filter_fisher_score(df, target='survived', inc_zero_score=True, top=20)
        self.assertEqual(['class', 'pclass', 'deck', 'parch', 'sibsp', 'age', 'fare'], result)
        result = Discover.filter_fisher_score(df, target='survived', inc_zero_score=True, top=0.3)
        self.assertEqual(['class', 'pclass'], result)
        result = Discover.filter_fisher_score(df, target='survived', inc_zero_score=True, top=0.999)
        self.assertEqual(['class', 'pclass', 'deck', 'parch', 'sibsp'], result)

    def test_discovery_analytics_class(self):
        tools = SyntheticBuilder.from_env('test', default_save=False).intent_model
        dataset = tools.get_category(list('ABCDE')+[np.nan], weight_pattern=[1,3,2,7,4], size=694)
        result = Discover.analyse_category(dataset)
        analytics = DataAnalytics(label='tester', analysis=result)
        self.assertEqual(analytics.selection, analytics.sample_map.index.to_list())
        self.assertEqual(analytics.sample_distribution, analytics.sample_map.to_list())

    def test_analyse_category(self):
        tools = SyntheticBuilder.from_env('test', default_save=False).intent_model
        dataset = tools.get_category(list('ABCDE')+[np.nan], weight_pattern=[1,3,2,7,4], size=694)
        result = Discover.analyse_category(dataset)
        control = ['intent', 'patterns', 'stats']
        self.assertCountEqual(control, list(result.keys()))
        control = ['dtype', 'selection', 'upper', 'lower', 'granularity', 'weighting_precision']
        self.assertCountEqual(control, list(result.get('intent').keys()))
        control = ['weight_pattern', 'sample_distribution']
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
        result = DataAnalytics(Discover.analyse_number(dataset, granularity=intervals))
        control = [50.0, 25.0, 25.0]
        self.assertEqual(control, result.weight_pattern)
        intervals = [(0,1,'both'),(1,2,'both'),(2,3)]
        result = DataAnalytics(Discover.analyse_number(dataset, granularity=intervals))
        control = [50.0, 50.0, 25.0]
        self.assertEqual(control, result.weight_pattern)
        # percentile
        dataset = [0, 0, 2, 2, 1, 2, 3]
        percentile = [.25, .5, .75]
        result = DataAnalytics(Discover.analyse_number(dataset, granularity=percentile))
        control = [28.57, 57.14, 0.0, 14.29]
        self.assertEqual(control, result.weight_pattern)

    def test_analyse_number_lower_upper(self):
        # Default
        dataset = list(range(1,10))
        result = DataAnalytics(Discover.analyse_number(dataset))
        self.assertEqual(1, result.lower)
        self.assertEqual(9, result.upper)
        # Outer Boundaries
        result = DataAnalytics(Discover.analyse_number(dataset, lower=0, upper=10))
        self.assertEqual(0, result.lower)
        self.assertEqual(10, result.upper)
        # Inner Boundaries
        result = DataAnalytics(Discover.analyse_number(dataset, lower=2, upper=8))
        self.assertEqual(2, result.lower)
        self.assertEqual(8, result.upper)

    def test_analyse_date(self):
        tools = SyntheticBuilder.from_env('test', default_save=False).intent_model
        str_dates = tools.get_datetime('12/01/2016', '12/01/2018', date_format='%d-%m-%Y', size=10, seed=31)
        ts_dates = tools.get_datetime('12/01/2016', '12/01/2018', size=10, seed=31)
        result = Discover.analyse_date(str_dates, granularity=3, date_format='%Y-%m-%d')
        control = {'intent': {'date_format': '%Y-%m-%d',
                              'day_first': False,
                              'dtype': 'date',
                              'granularity': 3,
                              'lower': '2017-12-02',
                              'selection': [('2017-12-02', '2018-03-13', 'both'),
                                            ('2018-03-13', '2018-06-22', 'right'),
                                            ('2018-06-22', '2018-10-01', 'right')],
                              'upper': '2018-10-01',
                              'weighting_precision': 2,
                              'year_first': False},
                   'patterns': {'sample_distribution': [1, 0, 3],
                                'weight_pattern': [25.0, 0.0, 75.0]},
                   'stats': {'kurtosis': 3.86,
                             'mean': '2018-07-04',
                             'nulls_percent': 60.0,
                             'outlier_percent': 0.0,
                             'sample': 4,
                             'skew': -1.96}}
        self.assertEqual(control, result)
        result = Discover.analyse_date(ts_dates, granularity=3)
        control = {'intent': {'day_first': False,
                              'dtype': 'date',
                              'granularity': 3,
                              'lower': pd.Timestamp('2017-02-12 19:02:11.531780+0000', tz='UTC'),
                              'selection': [(pd.Timestamp('2017-02-12 19:02:11.531780+0000', tz='UTC'),
                                             pd.Timestamp('2017-09-08 17:43:30.973860+0000', tz='UTC'),
                                             'both'),
                                            (pd.Timestamp('2017-09-08 17:43:30.973860+0000', tz='UTC'),
                                             pd.Timestamp('2018-04-04 16:24:50.415940+0000', tz='UTC'),
                                             'right'),
                                            (pd.Timestamp('2018-04-04 16:24:50.415940+0000', tz='UTC'),
                                             pd.Timestamp('2018-10-29 15:06:09.858020+0000', tz='UTC'),
                                             'right')],
                              'upper': pd.Timestamp('2018-10-29 15:06:09.858020+0000', tz='UTC'),
                              'weighting_precision': 2,
                              'year_first': False},
                   'patterns': {'sample_distribution': [2, 3, 5],
                                'weight_pattern': [20.0, 30.0, 50.0]},
                   'stats': {'kurtosis': 0.64,
                             'mean': pd.Timestamp('2018-03-22 17:31:12+0000', tz='UTC'),
                             'nulls_percent': 0.0,
                             'outlier_percent': 0.0,
                             'sample': 10,
                             'skew': -0.94}}
        self.assertEqual(control, result)

    def test_analyse_associate_single(self):
        tools = SyntheticBuilder.from_env('test', default_save=False).intent_model
        size = 50
        df = tools.create_profiles(size=size, dominance=0.6, seed=31)
        # category
        columns_list = [{'gender': {}}]
        result = Discover.analyse_association(df, columns_list)
        control = {'gender': {'analysis': {'intent': {'dtype': 'category',
                                                      'granularity': 2,
                                                      'lower': 40.0,
                                                      'selection': ['M', 'F'],
                                                      'upper': 60.0,
                                                      'weighting_precision': 2},
                                           'patterns': {'sample_distribution': [30, 20],
                                                        'weight_pattern': [60.0, 40.0]},
                                           'stats': {'nulls_percent': 0.0,
                                                     'outlier_percent': 0.0,
                                                     'sample': 50}},
                              'associate': 'gender'}}
        self.assertEqual(control, result)
        columns_list = [{'gender': {'chunk_size': 1, 'replace_zero': 0}}]
        result = Discover.analyse_association(df, columns_list)
        self.assertEqual(control, result)
        # number
        df['numbers'] = tools.get_number(from_value=1000, weight_pattern=[5,0,2], size=size, quantity=0.9, seed=31)
        columns_list = [{'numbers': {'type': 'number', 'granularity': 3}}]
        result = Discover.analyse_association(df, columns_list)
        control = {'numbers': {'analysis': {'intent': {'dtype': 'number',
                                                       'granularity': 3,
                                                       'lower': 9.0,
                                                       'precision': 3,
                                                       'selection': [(9.0, 330.0, 'both'),
                                                                     (330.0, 651.0, 'right'),
                                                                     (651.0, 972.0, 'right')],
                                                       'upper': 972.0,
                                                       'weighting_precision': 2},
                                            'patterns': {'dominance_weighting': [50.0, 50.0],
                                                         'dominant_percent': 9.09,
                                                         'dominant_values': [100.0, 139.0],
                                                         'sample_distribution': [31, 0, 13],
                                                         'weight_mean': [140.484, 0.0, 827.231],
                                                         'weight_pattern': [70.45, 0.0, 29.55],
                                                         'weight_std': [7568.791, 0.0, 7760.859]},
                                            'stats': {'kurtosis': -1.03,
                                                      'mean': 343.39,
                                                      'nulls_percent': 12.0,
                                                      'outlier_percent': 0.0,
                                                      'sample': 44,
                                                      'skew': 0.84,
                                                      'var': 107902.71}},
                               'associate': 'numbers'}}
        self.assertEqual(control, result)
        #dates
        df['dates'] = tools.get_datetime('10/10/2000', '31/12/2018', weight_pattern=[1, 9, 4], size=size, quantity=0.9, seed=31)
        columns_list = [{'dates': {'dtype': 'datetime', 'granularity': 3, 'date_format': '%d-%m-%Y'}}]
        result = Discover.analyse_association(df, columns_list)
        control = {'dates': {'analysis': {'intent': {'date_format': '%d-%m-%Y',
                                                     'day_first': False,
                                                     'dtype': 'date',
                                                     'granularity': 3,
                                                     'lower': '14-01-2003',
                                                     'selection': [('14-01-2003', '29-04-2008', 'both'),
                                                                   ('29-04-2008', '13-08-2013', 'right'),
                                                                   ('13-08-2013', '27-11-2018', 'right')],
                                                     'upper': '27-11-2018',
                                                     'weighting_precision': 2,
                                                     'year_first': False},
                                          'patterns': {'sample_distribution': [12, 21, 11],
                                                       'weight_pattern': [27.27, 47.73, 25.0]},
                                          'stats': {'kurtosis': -0.5,
                                                    'mean': '14-03-2011',
                                                    'nulls_percent': 12.0,
                                                    'outlier_percent': 0.0,
                                                    'sample': 44,
                                                    'skew': 0.25}},
                             'associate': 'dates'}}
        self.assertEqual(control, result)

    def test_analyse_associate_multi(self):
        tools = SyntheticBuilder.from_env('test', default_save=False).intent_model
        size = 50
        tools = SyntheticBuilder.from_env('test', default_save=False).intent_model
        size = 50
        df = tools.create_profiles(dominance=0.6, seed=31, size=size)
        df['lived'] = tools.get_category(selection=['yes', 'no'], quantity=80.0, seed=31, size=size)
        df['age'] = tools.get_number(from_value=20,to_value=80, weight_pattern=[1,2,5,6,2,1,0.5], seed=31, size=size)
        df['fare'] = tools.get_number(from_value=1000, weight_pattern=[5,0,2], size=size, quantity=0.9, seed=31)
        df['numbers'] = tools.get_number(from_value=1000, weight_pattern=[5,0,2], size=size, quantity=0.9, seed=31)
        df['dates'] = tools.get_datetime('10/10/2000', '31/12/2018', weight_pattern=[1, 9, 4], size=size, quantity=0.9, seed=31)
        columns_list = ['numbers']
        result = Discover.analyse_association(df, columns_list)
        print(result)

    def test_analyse_associate_levels(self):
        tools = SyntheticBuilder.from_env('test', default_save=False).intent_model
        size = 50
        df = tools.create_profiles(dominance=0.6, seed=31, size=size)
        df['lived'] = tools.get_category(selection=['yes', 'no'], quantity=80.0, seed=31, size=size)
        df['age'] = tools.get_number(from_value=20,to_value=80, weight_pattern=[1,2,5,6,2,1,0.5], seed=31, size=size)
        df['fare'] = tools.get_number(from_value=1000, weight_pattern=[5,0,2], size=size, quantity=0.9, seed=31)
        columns_list = [{'gender': {}, 'age':  {}}, {'lived': {}}]
        exclude = ['age.lived']
        result = Discover.analyse_association(df, columns_list, exclude)
        self.assertCountEqual(['age', 'gender'], list(result.keys()))
        self.assertNotIn('sub_category', result.get('age').keys())
        self.assertIn('sub_category', result.get('gender').keys())
        self.assertCountEqual(['M', 'F'], list(result.get('gender').get('sub_category').keys()))
        self.assertCountEqual(['lived'], list(result.get('gender').get('sub_category').get('M').keys()))
        self.assertCountEqual(['lived'], list(result.get('gender').get('sub_category').get('F').keys()))

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


