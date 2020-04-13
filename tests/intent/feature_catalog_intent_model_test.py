import os
import shutil
import unittest
from pprint import pprint

import pandas as pd
from aistac.properties.property_manager import PropertyManager

from ds_behavioral import SyntheticBuilder
from ds_discovery.transition.feature_catalog import FeatureCatalog
from ds_engines.engines.event_books.pandas_event_book import PandasEventBook

from ds_discovery.intent.feature_catalog_intent import FeatureCatalogIntentModel
from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager
from aistac.handlers.abstract_handlers import ConnectorContract


class FeatureCatalogIntentTest(unittest.TestCase):
    """Test: """

    def setUp(self):
        os.environ['AISTAC_PM_PATH'] = os.path.join('work', 'config')
        os.environ['AISTAC_DEFAULT_PATH'] = os.path.join('work', 'data')
        try:
            os.makedirs(os.environ['AISTAC_PM_PATH'])
            os.makedirs(os.environ['AISTAC_DEFAULT_PATH'])
        except:
            pass
        PropertyManager._remove_all()
        self.tools = SyntheticBuilder.scratch_pad()
        self.fc = FeatureCatalog.scratch_pad()

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        FeatureCatalogIntentModel(property_manager=FeatureCatalogPropertyManager('test'), default_save_intent=False)

    def test_run_pipeline(self):
        local_fc = FeatureCatalog.from_env('tester', default_save=False)
        df = pd.DataFrame()
        df['cu_id'] = self.tools.get_number(100000, 1000000, at_most=1, size=1000)
        df['age'] = self.tools.get_number(20, 90, weight_pattern=[5, 2, 4, 3, 2, 0.5, 0.1], size=1000)
        df['salary'] = self.tools.get_number(0, 100.0, weight_pattern=[10, 5, 3, 10], size=1000)
        _ = local_fc.intent_model.interval_categorical(df, key='cu_id', column='salary', granularity=[(0, 20), (80, 100)], precision=2, feature_name='First')
        local_fc.intent_model.group_features(df, headers=['age', 'salary'], aggregator='sum', group_by=['cu_id'], feature_name='First', unindex= True, intent_order=1)
        _ = local_fc.intent_model.interval_categorical(df, key='cu_id', column='age', granularity=[0.9, 0.1], categories=['younger', 'average', 'older'], precision=0, feature_name='Second')
        result = local_fc.intent_model.run_intent_pipeline(df, feature_name='First')
        result = local_fc.intent_model.run_intent_pipeline(df, feature_name='Second')

    def test_date_diff(self):
        df = pd.DataFrame()
        df['key'] = range(1000)
        df['primary'] = self.tools.get_datetime(start='2000/01/01', until='2000/01/2', year_first=True, size=1000)
        df['secondary'] = self.tools.get_datetime(start='2000/02/01', until='2000/02/2', year_first=True, size=1000)
        result = self.fc.apply_date_diff(df, key='key', first_date='primary', second_date='secondary', units='D', save_intent=False)
        self.assertEqual([30], result.min().values)
        self.assertEqual([32], result.max().values)
        result = self.fc.apply_date_diff(df, key='key', first_date='primary', second_date='secondary', units='W', save_intent=False)
        self.assertEqual([4], result.min().values)
        self.assertEqual([5], result.max().values)
        # check nulls work
        df['primary'] = self.tools.get_datetime(start='2000/01/01', until='2000/01/2', year_first=True, quantity=0.1, size=1000)
        df['secondary'] = self.tools.get_datetime(start='2000/02/01', until='2000/02/2', year_first=True, quantity=0.5, size=1000)
        result = self.fc.apply_date_diff(df, key='key', first_date='primary', second_date='secondary', units='D', save_intent=False)
        self.assertEqual(1000, result.size)

    def test_interval_categorical(self):
        df = pd.DataFrame()
        df['cu_id'] = self.tools.get_number(100000, 1000000, at_most=1, size=1000)
        df['age'] = self.tools.get_number(20, 90, weight_pattern=[5, 2, 4, 3, 2, 0.5, 0.1], size=1000)
        df['salary'] = self.tools.get_number(0, 100.0, weight_pattern=[10, 5, 3, 10], size=1000)
        result = self.fc.interval_categorical(df, key='cu_id', column='salary', granularity=[(0, 20), (80, 100)],
                                                  precision=2)
        self.assertEqual('category', result['salary_cat'].dtype.name)
        self.assertCountEqual(['<NA>', '80->100', '0->20'], result['salary_cat'].value_counts().index.to_list())
        result = self.fc.interval_categorical(df, key='cu_id', column='age', granularity=[0.9, 0.1],
                                                  categories=['younger', 'average', 'older'], precision=0)
        self.assertEqual('category', result['age_cat'].dtype.name)
        self.assertCountEqual(['younger', 'average', 'older'], result['age_cat'].value_counts().index.to_list())
        result = self.fc.interval_categorical(df, key='cu_id', column='age', rename='age_gap', lower=0, upper=100)
        self.assertCountEqual(['33->66', '0->33', '66->100'], result['age_gap'].value_counts().index.to_list())

    def test_select_where(self):
        df = pd.DataFrame()
        df['cid'] = [1,2,3,4,5,6]
        df['starts'] = list('ABABCD')
        df['ends'] = list("XZZXYZ")
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='==')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([2,4], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='=='),
                           self.fc.select2dict(column='ends', condition=".str.contains('Z')")]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([2], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='=='),
                           self.fc.select2dict(column='ends', condition=".str.contains('Z')", logic='AND')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([2], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='=='),
                           self.fc.select2dict(column='ends', condition=".str.contains('Z')", logic='NOT')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([4], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='=='),
                           self.fc.select2dict(column='ends', condition=".str.contains('Z')", logic='OR')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([2,3,4,6], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='starts', condition="'B'", operator='=='),
                           self.fc.select2dict(column='ends', condition=".str.contains('Z')", logic='XOR')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([3,4,6], result.index.tolist())
        dates = []
        for offset in range (-2, 4):
            dates.append(pd.Timestamp.now()-pd.Timedelta(days=offset))
        df['dates'] = dates
        conditions_list = [self.fc.select2dict(column='dates', condition="date.now", operator='>')]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([1,2,3], result.index.tolist())
        conditions_list = [self.fc.select2dict(column='dates', condition="date.now", operator='<=', offset=-2)]
        result = self.fc.select_where(df, key='cid', selection=conditions_list, save_intent=False)
        self.assertEqual([6], result.index.tolist())

    def test_group_features(self):
        df = pd.DataFrame()
        df['genre'] = ['Comedy', 'Drama', 'Drama', 'Drama', 'Undefined']
        df['end_type'] = ['Ad End', 'Ad Start', 'Ad End', 'Ad Start', 'Ad End']
        df['spend'] = [1, 3, 2, 4, 0]
        df['viewed'] = [1, 2, 1, 3, 1]
        result = self.fc.group_features(df, headers=['viewed', 'spend'], aggregator='nunique', group_by=['genre'],
                                            drop_group_by=True)
        print(result)

    def test_remove_outliers(self):
        df = pd.DataFrame()
        df['key'] = [1, 2, 3, 4, 5, 6]
        df['values'] = [10, 3, 1, 5, 6, 10]
        result = self.fc.remove_outliers(df, key='key', column='values', lower_quantile=0.1, upper_quantile=0.9)
        self.assertCountEqual([2, 4, 5], result.index.to_list())
        self.assertCountEqual([3, 5, 6], result['values'].to_list())

    def test_date_elements(self):
        df = pd.DataFrame()
        df['cu_id'] = self.tools.get_number(1000, 10000, at_most=1, size=20)
        df['primary'] = self.tools.get_datetime(start='2000/01/01', until='2000/02/2', year_first=True, size=20)
        result = self.fc.select_date_elements(df, key='cu_id', header='primary', matrix=[], rtn_columns=['primary'])
        self.assertEqual(1, result.shape[1])
        result = self.fc.select_date_elements(df, key='cu_id', header='primary', matrix=['day', 'dow', 'woy', 'doy'])
        self.assertCountEqual(['primary_woy', 'primary_doy', 'primary_dow', 'primary_day'], result.columns)
        result = self.fc.select_date_elements(df, key='cu_id', header='primary', matrix=['day', 'dow'], rename='cat')
        self.assertCountEqual(['cat_dow', 'cat_day'], result.columns)

    def test_apply_condition(self):
        df = pd.DataFrame()
        df['key'] = [1, 2, 3, 4, 5, 6, 3, 5, 1]
        df['values'] = [10, 3, 1, 5, 6, 10, 2, 4, 5]
        conditions = [('< 5', '-1'), ('> 5', '1')]
        result = self.fc.apply_condition(df, key='key', header='values', conditions=conditions, default=0, save_intent=False)
        self.assertEqual(['1', '-1', '-1', '0', '1', '1', '-1', '-1', '0'], result['values'].to_list())

    def test_apply_map(self):
        df = pd.DataFrame()
        df['key'] = list(range(10))
        df['values'] = [1,3,2,5,4,1,3,2,1,6]
        result = self.fc.apply_map(df, key='key', header='values', value_map={1: "A", 2: "B", 3: "C"})
        self.assertEqual(['A', 'C', 'B', 'A', 'C', 'B', 'A'], result['values'].to_list())
        result = self.fc.apply_map(df, key='key', header='values', value_map={1: "A", 2: "B", 3: "C"}, default_to='Z')
        self.assertEqual(['A', 'C', 'B', 'Z', 'Z', 'A', 'C', 'B', 'A', 'Z'], result['values'].to_list())

    def test_apply_replace(self):
        df = pd.DataFrame()
        df['key'] = list(range(10))
        df['values'] = [1,3,2,5,4,1,3,2,1,6]
        result = self.fc.apply_replace(df, key='key', header='values', to_replace={1: 10, 2: 11})
        self.assertEqual([10, 3, 11, 5, 4, 10, 3, 11, 10, 6], result['values'].to_list())


if __name__ == '__main__':
    unittest.main()
