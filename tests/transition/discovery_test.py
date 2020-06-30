import os
import shutil
import unittest

import numpy as np
import pandas as pd
import seaborn as sns
from aistac.properties.property_manager import PropertyManager

from ds_discovery import Transition, FeatureCatalog
from ds_discovery.transition.discovery import DataDiscovery as Discover, DataDiscovery

from ds_behavioral import SyntheticBuilder


class TestDiscovery(unittest.TestCase):
    """Test: """

    def setUp(self):
        # set environment variables
        os.environ['HADRON_PM_PATH'] = os.path.join("${PWD}", 'work', 'config')
        os.environ['HADRON_DEFAULT_SOURCE_PATH'] = os.path.join("${HOME}", 'code', 'projects',  'data', 'sample')
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

    def test_interquartile_outliers(self):
        expo = pd.Series(np.random.exponential(size=10000))
        result = DataDiscovery.interquartile_outliers(expo)
        print(result)

    def test_bootstrap_confidence_interval(self):
        normal = pd.Series(np.random.normal(size=10000))
        expo = pd.Series(np.random.exponential(size=10000))
        result = DataDiscovery.bootstrap_confidence_interval(normal)
        print(result)
        result = DataDiscovery.bootstrap_confidence_interval(expo)
        print(result)

    def test_jensen_shannon_distance(self):
        p = [0.20, 0.60, 0.20]
        q = [0.20, 0.599, 0.201]
        result = Discover.jensen_shannon_distance(p, q, precision=5)
        print(result)


    def test_filter_univariate_roc_auc(self):
        tr = Transition.from_env('test', default_save=False, default_save_intent=False)
        tr.set_source('paribas.csv', nrows=5000)
        data = tr.load_source_canonical()
        result = Discover.filter_univariate_roc_auc(data, target='target', threshold=0.55)
        self.assertCountEqual(['v10', 'v129', 'v14', 'v62', 'v50'], result)
        # Custom classifier
        classifier_kwargs = {'iterations': 2, 'learning_rate': 1, 'depth': 2}
        result = Discover.filter_univariate_roc_auc(data, target='target', threshold=0.55, package='catboost',
                                                    model='CatBoostClassifier', classifier_kwargs=classifier_kwargs,
                                                    fit_kwargs={'verbose': False})
        self.assertCountEqual(['v50', 'v10', 'v14', 'v12', 'v129', 'v62', 'v21', 'v34'], result)

    def test_filter_univariate_mse(self):
        tr = Transition.from_env('test', default_save=False, default_save_intent=False)
        tr.set_source('ames_housing.csv', nrows=5000)
        data = tr.load_source_canonical()
        result = Discover.filter_univariate_mse(data, target='SalePrice', as_series=False, )
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

    def test_filter_correlated(self):
        tools = SyntheticBuilder.scratch_pad()
        df = pd.DataFrame()
        df['col1'] = [1,2,3,4,5,6,7]
        df['col2'] = [1,2,3,4,5,6,7]
        df['col3'] = [2,2,3,2,2,2,3]
        df['col4'] = [2,2,3,2,2,2,3]
        df['col5'] = [2,2,3,2,2,2,3]
        df['col4'] = [7,2,4,2,1,6,4]
        df['target'] = [1,0,1,1,0,0,1]
        result = DataDiscovery.filter_correlated(df, target='target')
        print(result)



if __name__ == '__main__':
    unittest.main()
