import shutil
from pprint import pprint
import os
import pandas as pd
import numpy as np
import seaborn as sns
import unittest

from aistac.properties.property_manager import PropertyManager

from ds_discovery import Transition
from ds_discovery.transition.discovery import DataDiscovery as Discover, DataDiscovery


class TestDiscovery(unittest.TestCase):
    """Test: """

    def setUp(self):
        # set environment variables
        os.environ['AISTAC_PM_PATH'] = os.path.join(os.environ['PWD'], 'work', 'config')
        os.environ['AISTAC_DEFAULT_SOURCE_PATH'] = os.path.join(os.environ['HOME'], 'code', 'projects', 'prod', 'data',
                                                                'raw')
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

    def test_filter_univariate_roc_auc(self):
        tr = Transition.from_env('test')
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
        data = pd.read_csv('../../../data/raw/ames_housing.csv', nrows=50000)
        result = Discover.filter_univariate_mse(data, target='SalePrice', as_series=True, )
        print(result)
        # customer
        regressor_kwargs = {'iterations': 2, 'learning_rate': 1, 'depth': 2}
        result = Discover.filter_univariate_mse(data, target='SalePrice', as_series=True, package='catboost',
                                                model='CatBoostRegressor',
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


if __name__ == '__main__':
    unittest.main()
