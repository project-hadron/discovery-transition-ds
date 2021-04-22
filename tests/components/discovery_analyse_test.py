import os
import shutil
import unittest
from pprint import pprint

import numpy as np
import pandas as pd
from aistac.components.aistac_commons import DataAnalytics
from aistac.properties.property_manager import PropertyManager
from ds_discovery import SyntheticBuilder

from ds_discovery import Transition
from ds_discovery.components.discovery import DataDiscovery as Discover, DataDiscovery


class DiscoveryAnalysisTest(unittest.TestCase):

    def setUp(self):
        # set environment variables
        os.environ['HADRON_PM_PATH'] = os.path.join(os.environ['PWD'], 'work', 'config')
        os.environ['HADRON_DEFAULT_SOURCE_PATH'] = os.path.join(os.environ['HOME'], 'code', 'projects', 'prod', 'data', 'raw')
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
        self.assertEqual(DataDiscovery, type(Discover()))

    def test_discovery_associate(self):
        tools = SyntheticBuilder.scratch_pad()
        df = pd.DataFrame()
        df['cat'] = tools.get_category(list('AB'), relative_freq=[1,3], size=1000)
        df['gender'] = tools.get_category(list('MF'), relative_freq=[1,3], size=1000)
        result = Discover.analyse_association(df, columns_list=['cat', 'gender'])
        self.assertEqual(['cat', 'gender'], list(result))

    def test_discovery_analytics_class(self):
        tools = SyntheticBuilder.scratch_pad()
        dataset = tools.get_category(list('ABCDE')+[np.nan], relative_freq=[1,3,2,7,4], size=694)
        result = Discover.analyse_category(dataset)
        analytics = DataAnalytics(analysis=result)
        self.assertEqual(analytics.intent.selection, analytics.sample_map.index.to_list())
        self.assertEqual(analytics.patterns.sample_distribution, analytics.sample_map.to_list())

    def test_analyse_category(self):
        builer = SyntheticBuilder.from_memory()
        tools = builer.tools
        dataset = tools.get_category(list('ABCDE')+[np.nan], relative_freq=[1,3,2,7,4], size=694)
        result = Discover.analyse_category(dataset)
        control = ['intent', 'patterns', 'stats', 'params']
        self.assertCountEqual(control, list(result.keys()))
        control = ['dtype', 'categories', 'highest_unique', 'lowest_unique', 'category_count']
        self.assertCountEqual(control, list(result.get('intent').keys()))
        control = ['relative_freq', 'sample_distribution']
        self.assertCountEqual(control, list(result.get('patterns').keys()))
        control = ['nulls_percent', 'sample_size', 'excluded_percent']
        self.assertCountEqual(control, list(result.get('stats').keys()))
        control = ['freq_precision']
        self.assertCountEqual(control, list(result.get('params').keys()))

    def test_analyse_category_limits(self):
        top = 2
        dataset = ['A']*8 + ['B']*6 + ['C']*4 + ['D']*2
        result = Discover.analyse_category(dataset, top=top, freq_precision=0)
        control = ['dtype', 'categories', 'top', 'highest_unique', 'lowest_unique', 'category_count']
        self.assertCountEqual(control, list(result.get('top')))
        self.assertEqual(top, len(result.get('intent').get('categories')))
        self.assertCountEqual(['A', 'B'], result.get('intent').get('categories'))
        self.assertCountEqual([40, 30], result.get('patterns').get('relative_freq'))
        self.assertEqual(30, result.get('stats').get('excluded_percent'))
        self.assertEqual(14, result.get('stats').get('sample_size'))
        lower = 0.2
        upper = 7
        result = Discover.analyse_category(dataset, lower=lower, upper=upper, freq_precision=0)
        control = ['dtype', 'categories', 'highest_unique', 'lowest_unique', 'granularity']
        self.assertCountEqual(control, list(result.get('intent').keys()))
        self.assertEqual(lower, result.get('intent').get('lowest_unique'))
        self.assertEqual(upper, result.get('intent').get('highest_unique'))
        self.assertCountEqual(['C', 'B'], result.get('intent').get('categories'))
        self.assertCountEqual([33, 50], result.get('patterns').get('relative_freq'))
        self.assertEqual(50, result.get('stats').get('excluded_percent'))
        self.assertEqual(10, result.get('stats').get('sample_size'))

    def test_analyse_number(self):
        dataset = []
        result = Discover.analyse_number(dataset)
        control = [1]
        self.assertEqual(control, result.get('patterns').get('relative_freq'))
        self.assertEqual((0,0,3), (result.get('intent').get('lowest'), result.get('intent').get('highest'), result.get('intent').get('granularity')))
        self.assertEqual(0, result.get('stats').get('sample_size'))
        dataset = [1,2,3,4,5,6,7,8,9,10]
        result = Discover.analyse_number(dataset, granularity=2)
        control = [50.0, 50.0]
        self.assertEqual(control, result.get('patterns').get('relative_freq'))
        self.assertEqual((1,10), (result.get('intent').get('lowest'), result.get('intent').get('highest')))
        control = [(1.0, 5.5, 'both'), (5.5, 10.0, 'right')]
        self.assertEqual(control, result.get('intent').get('intervals'))
        self.assertEqual(2, result.get('intent').get('granularity'))

        result = Discover.analyse_number(dataset, granularity=3.0)
        control = [30.0, 30.0, 40.0]
        self.assertEqual(control, result.get('patterns').get('relative_freq'))
        self.assertEqual((1,10), (result.get('intent').get('lowest'), result.get('intent').get('highest')))
        control = [(1.0, 4.0, 'left'), (4.0, 7.0, 'left'), (7.0, 10.0, 'both')]
        self.assertEqual(control, result.get('intent').get('intervals'))
        self.assertEqual(3.0, result.get('intent').get('granularity'))

        result = Discover.analyse_number(dataset, granularity=2.0)
        control = [20.0, 20.0, 20.0, 20.0, 20.0]
        self.assertEqual(control, result.get('patterns').get('relative_freq'))
        self.assertEqual((1,10), (result.get('intent').get('lowest'), result.get('intent').get('highest')))
        control = [(1.0, 3.0, 'left'), (3.0, 5.0, 'left'), (5.0, 7.0, 'left'), (7.0, 9.0, 'left'), (9.0, 11.0, 'both')]
        self.assertEqual(control, result.get('intent').get('intervals'))
        self.assertEqual(2.0, result.get('intent').get('granularity'))

        result = Discover.analyse_number(dataset, granularity=1.0, lower=0, upper=5)
        control = [0.0, 20.0, 20.0, 20.0, 20.0, 20.0]
        self.assertEqual(control, result.get('patterns').get('relative_freq'))
        self.assertEqual((0,5), (result.get('intent').get('lowest'), result.get('intent').get('highest')))
        control = [(0.0, 1.0, 'left'), (1.0, 2.0, 'left'), (2.0, 3.0, 'left'), (3.0, 4.0, 'left'), (4.0, 5.0, 'left'), (5.0, 6.0, 'both')]
        self.assertEqual(control, result.get('intent').get('intervals'))
        self.assertEqual(1.0, result.get('intent').get('granularity'))
        pprint(result)

    def test_number_zero_count(self):
        dataset = [1,0,0,1,1,1,0,0,1,0]
        result = DataAnalytics(analysis=Discover.analyse_number(dataset))
        self.assertEqual([50.0], result.get('dominance_weighting'))
        dataset = [1,0,0,2,5,4,0,4,1,0]
        result = Discover.analyse_number(dataset, lower=0, granularity=3)
        self.assertEqual([40.0], result.get('dominance_weighting'))
        self.assertEqual([10], result.get('sample'))
        result = Discover.analyse_number(dataset, lower=1, granularity=3)
        self.assertEqual([40.0], result.get('dominance_weighting'))
        self.assertEqual([6], result.get('sample'))

    def test_analysis_granularity_list(self):
        dataset = [0,1,2,3]
        intervals = [(0,1,'both'),(1,2),(2,3)]
        result = DataAnalytics(Discover.analyse_number(dataset, granularity=intervals))
        control = [50.0, 25.0, 25.0]
        self.assertEqual(control, result.patterns.relative_freq)
        intervals = [(0,1,'both'),(1,2,'both'),(2,3)]
        result = DataAnalytics(Discover.analyse_number(dataset, granularity=intervals))
        control = [50.0, 50.0, 25.0]
        self.assertEqual(control, result.patterns.relative_freq)
        # percentile
        dataset = [0, 0, 2, 2, 1, 2, 3]
        percentile = [.25, .5, .75]
        result = DataAnalytics(Discover.analyse_number(dataset, granularity=percentile))
        control = [28.57, 57.14, 0.0, 14.29]
        self.assertEqual(control, result.patterns.relative_freq)

    def test_analyse_number_lower_upper(self):
        # Default
        dataset = list(range(1,10))
        result = DataAnalytics(Discover.analyse_number(dataset))
        self.assertEqual(1, result.stats.lowest)
        self.assertEqual(9, result.stats.highest)
        # Outer Boundaries
        result = DataAnalytics(Discover.analyse_number(dataset, lower=0, upper=10))
        self.assertEqual(0, result.params.lower)
        self.assertEqual(10, result.params.upper)
        self.assertEqual(0, result.stats.lowest)
        self.assertEqual(10, result.stats.highest)
        # Inner Boundaries
        result = DataAnalytics(Discover.analyse_number(dataset, lower=2, upper=8))
        self.assertEqual(2, result.params.lower)
        self.assertEqual(8, result.params.upper)
        self.assertEqual(2, result.stats.lowest)
        self.assertEqual(8, result.stats.highest)

    def test_analyse_date(self):
        tools = SyntheticBuilder.scratch_pad()
        str_dates = tools.get_datetime('12/01/2016', '12/01/2018', date_format='%d-%m-%Y', size=10, seed=31)
        ts_dates = tools.get_datetime('12/01/2016', '12/01/2018', size=10, seed=31)
        result = Discover.analyse_date(str_dates, granularity=3, date_format='%Y-%m-%d')
        control = {'intent': {'date_format': '%Y-%m-%d',
                              'day_first': False,
                              'dtype': 'date',
                              'granularity': 3,
                              'lowest': '2017-12-02',
                              'selection': [('2017-12-02', '2018-03-13', 'both'),
                                            ('2018-03-13', '2018-06-22', 'right'),
                                            ('2018-06-22', '2018-10-01', 'right')],
                              'highest': '2018-10-01',
                              'freq_precision': 2,
                              'year_first': False},
                   'patterns': {'sample_distribution': [1, 0, 3],
                                'relative_freq': [25.0, 0.0, 75.0]},
                   'stats': {'bootstrap_bci': (17572.5, 17797.75),
                             'emp_outliers': [0, 0],
                             'excluded_percent': 0.0,
                             'irq_outliers': [1, 0],
                             'kurtosis': 3.86,
                             'mean': '2018-07-04',
                             'nulls_percent': 60.0,
                             'sample': 4,
                             'skew': -1.96}}
        self.assertEqual(control, result)
        result = Discover.analyse_date(ts_dates, granularity=3)
        control = {'intent': {'day_first': False,
                              'dtype': 'date',
                              'granularity': 3,
                              'lowest': pd.Timestamp('2017-02-12 19:02:11.531780+0000', tz='UTC'),
                              'selection': [(pd.Timestamp('2017-02-12 19:02:11.531780+0000', tz='UTC'),
                                             pd.Timestamp('2017-09-08 17:43:30.973860+0000', tz='UTC'),
                                             'both'),
                                            (pd.Timestamp('2017-09-08 17:43:30.973860+0000', tz='UTC'),
                                             pd.Timestamp('2018-04-04 16:24:50.415940+0000', tz='UTC'),
                                             'right'),
                                            (pd.Timestamp('2018-04-04 16:24:50.415940+0000', tz='UTC'),
                                             pd.Timestamp('2018-10-29 15:06:09.858020+0000', tz='UTC'),
                                             'right')],
                              'highest': pd.Timestamp('2018-10-29 15:06:09.858020+0000', tz='UTC'),
                              'freq_precision': 2,
                              'year_first': False},
                   'patterns': {'sample_distribution': [2, 3, 5],
                                'relative_freq': [20.0, 30.0, 50.0]},
                   'stats': {'bootstrap_bci': (17493.5054775573, 17724.4628926684),
                             'emp_outliers': [0, 0],
                             'excluded_percent': 0.0,
                             'irq_outliers': [1, 0], 'kurtosis': 0.64,
                             'mean': pd.Timestamp('2018-03-22 17:31:12+0000', tz='UTC'),
                             'nulls_percent': 0.0,
                             'excluded_percent': 0.0,
                             'sample': 10,
                             'skew': -0.94}}
        self.assertEqual(control, result)

    def test_analyse_associate_single(self):
        tools = SyntheticBuilder.scratch_pad()
        size = 50
        df = pd.DataFrame()
        df['gender'] = tools.get_category(selection=['M', 'F'], relative_freq=[6, 4], bounded_weighting=True, size=size)
        # category
        columns_list = [{'gender': {}}]
        result = Discover.analyse_association(df, columns_list)
        control = {'gender': {'analysis': {'intent': {'dtype': 'category',
                                                      'granularity': 2,
                                                      'lowest': 40.0,
                                                      'selection': ['M', 'F'],
                                                      'highest': 60.0,
                                                      'freq_precision': 2},
                                           'patterns': {'sample_distribution': [30, 20],
                                                        'relative_freq': [60.0, 40.0]},
                                           'stats': {'excluded_percent': 0.0,'nulls_percent': 0.0,
                                                     'sample': 50}},
                              'associate': 'gender'}}
        self.assertEqual(control, result)
        columns_list = [{'gender': {'chunk_size': 1, 'replace_zero': 0}}]
        result = Discover.analyse_association(df, columns_list)
        self.assertEqual(control, result)
        # number
        df['numbers'] = tools.get_number(from_value=1000, relative_freq=[5,0,2], size=size, quantity=0.9, seed=31)
        columns_list = [{'numbers': {'type': 'number', 'granularity': 3}}]
        result = Discover.analyse_association(df, columns_list)
        control = {'numbers': {'analysis': {'intent': {'dtype': 'number',
                                                       'granularity': 3,
                                                       'lowest': 9.0,
                                                       'precision': 3,
                                                       'selection': [(9.0, 330.0, 'both'),
                                                                     (330.0, 651.0, 'right'),
                                                                     (651.0, 972.0, 'right')],
                                                       'highest': 972.0,
                                                       'freq_precision': 2},
                                            'patterns': {'dominance_weighting': [50.0, 50.0],
                                                         'dominant_percent': 9.09,
                                                         'dominant_values': [100.0, 139.0],
                                                         'sample_distribution': [31, 0, 13],
                                                         'freq_mean': [140.484, 0.0, 827.231],
                                                         'relative_freq': [70.45, 0.0, 29.55],
                                                         'freq_std': [7568.791, 0.0, 7760.859]},
                                            'stats': {'bootstrap_bci': (253.857, 445.214),
                                                      'emp_outliers': [0, 0],
                                                      'excluded_percent': 0.0,
                                                      'irq_outliers': [0, 0], 'kurtosis': -1.03,
                                                      'mad': 285.91,
                                                      'mean': 343.39,
                                                      'excluded_percent': 0.0,
                                                      'nulls_percent': 12.0,
                                                      'sample': 44,
                                                      'sem': 49.52,
                                                      'skew': 0.84,
                                                      'var': 107902.71}},
                               'associate': 'numbers'}}
        self.assertEqual(control, result)
        #dates
        df['dates'] = tools.get_datetime('10/10/2000', '31/12/2018', relative_freq=[1, 9, 4], size=size, quantity=0.9, seed=31)
        columns_list = [{'dates': {'dtype': 'datetime', 'granularity': 3, 'date_format': '%d-%m-%Y'}}]
        control = {'dates': {'analysis': {'intent': {'date_format': '%d-%m-%Y',
                                                     'day_first': False,
                                                     'dtype': 'date',
                                                     'granularity': 3,
                                                     'lowest': '14-01-2003',
                                                     'selection': [('14-01-2003', '29-04-2008', 'both'),
                                                                   ('29-04-2008', '13-08-2013', 'right'),
                                                                   ('13-08-2013', '27-11-2018', 'right')],
                                                     'highest': '27-11-2018',
                                                     'freq_precision': 2,
                                                     'year_first': False},
                                          'patterns': {'sample_distribution': [12, 21, 11],
                                                       'relative_freq': [27.27, 47.73, 25.0]},
                                          'stats': {'bootstrap_bci': (14622.3654489759,
                                                                     15435.002697157),
                                                    'emp_outliers': [0, 0],
                                                    'excluded_percent': 0.0,
                                                    'irq_outliers': [0, 0], 'kurtosis': -0.5,
                                                    'mean': '14-03-2011',
                                                    'nulls_percent': 12.0,
                                                    'sample': 44,
                                                    'skew': 0.25}},
                             'associate': 'dates'}}
        result = Discover.analyse_association(df, columns_list)
        self.assertEqual(control, result)

    def test_analyse_associate_multi(self):
        tools = SyntheticBuilder.scratch_pad()
        size = 50
        df = pd.DataFrame()
        df['gender'] = tools.get_category(selection=['M', 'F'], relative_freq=[6, 4], size=size)
        df['lived'] = tools.get_category(selection=['yes', 'no'], quantity=80.0, seed=31, size=size)
        df['age'] = tools.get_number(from_value=20,to_value=80, relative_freq=[1,2,5,6,2,1,0.5], seed=31, size=size)
        df['fare'] = tools.get_number(from_value=1000, relative_freq=[5,0,2], size=size, quantity=0.9, seed=31)
        df['numbers'] = tools.get_number(from_value=1000, relative_freq=[5,0,2], size=size, quantity=0.9, seed=31)
        df['dates'] = tools.get_datetime('10/10/2000', '31/12/2018', relative_freq=[1, 9, 4], size=size, quantity=0.9, seed=31)
        columns_list = ['numbers', 'age', 'fare']
        result = Discover.analyse_association(df, columns_list)
        pprint(result)
        # self.assertCountEqual(['numbers'], list(result.keys()))
        # self.assertNotIn('sub_category', result.get('numbers').keys())
        # data_analysis = DataAnalytics(result)
        # self.assertCountEqual(['M', 'F'], list(result.get('numbers').get('analysis').get('intent').get(dtype)))
        # self.assertCountEqual(['lived'], list(result.get('gender').get('sub_category').get('M').keys()))
        # self.assertCountEqual(['lived'], list(result.get('gender').get('sub_category').get('F').keys()))

    def test_analyse_associate_levels_nums(self):
        clinical_health = 'https://assets.datacamp.com/production/repositories/628/datasets/444cdbf175d5fbf564b564bd36ac21740627a834/diabetes.csv'
        tr = Transition.from_memory()
        tr.set_source_uri(clinical_health)
        # columns_list = [{'diabetes': {'dtype': 'category'}},
        #                 {'age': {'dtype': 'int', 'granularity': 10.0, 'lower': 21, 'upper': 90, }},
        #                 {'pregnancies': {}, 'glucose': {}, 'diastolic': {}, 'triceps': {}, 'insulin': {}, 'bmi': {},
        #                  'dpf': {}}]
        columns_list = [{'age': {'dtype': 'int', 'granularity': 1}},
                        {'glucose': {}}]
        df_clinical = tr.load_source_canonical()
        discover: DataDiscovery = tr.discover
        analysis_blob = discover.analyse_association(df_clinical, columns_list=columns_list)
        age = DataAnalytics.from_branch(analytics_blob=analysis_blob, branch="age")
        glucose = DataAnalytics.from_branch(analytics_blob=analysis_blob, branch="age.0.glucose")
        pprint(age.intent.to_dict())
        pprint(glucose.intent.to_dict())


    def test_analyse_associate_levels(self):
        tools = SyntheticBuilder.scratch_pad()
        size = 50
        df = pd.DataFrame()
        df['gender'] = tools.get_category(selection=['M', 'F'], relative_freq=[6, 4], size=size)
        df['lived'] = tools.get_category(selection=['yes', 'no'], quantity=80.0, size=size)
        df['age'] = tools.get_number(from_value=20,to_value=80, relative_freq=[1,2,5,6,2,1,0.5], size=size)
        df['fare'] = tools.get_number(from_value=1000, relative_freq=[5,0,2], size=size, quantity=0.9)
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

    def test_sandbox(self):
        dates = pd.Series([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"),
                           pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-04")])




if __name__ == '__main__':
    unittest.main()


