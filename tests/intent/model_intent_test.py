import unittest
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from build.lib.ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from ds_discovery import ModelsBuilder as Model, SyntheticBuilder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from aistac.properties.property_manager import PropertyManager


class SyntheticTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # clean out any old environments
        for key in os.environ.keys():
            if key.startswith('HADRON'):
                del os.environ[key]
        # Local Domain Contract
        os.environ['HADRON_PM_PATH'] = os.path.join('working', 'contracts')
        os.environ['HADRON_PM_TYPE'] = 'json'
        # Local Connectivity
        os.environ['HADRON_DEFAULT_PATH'] = Path('working/data').as_posix()
        # Specialist Component
        try:
            os.makedirs(os.environ['HADRON_PM_PATH'])
        except:
            pass
        try:
            os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        except:
            pass
        PropertyManager._remove_all()

    def tearDown(self):
        try:
            shutil.rmtree('working')
        except:
            pass

    def test_label_predict_log(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = tools.model_synthetic_classification(1000, n_features=4, seed=31)
        df['cust_id'] = tools.get_number(from_value=100_000, to_value=999_999, at_most=1, size=df.shape[0], seed=31)
        X = df.drop(['target', 'cust_id'], axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        df_cust = df['cust_id'].iloc[X_test.index].to_frame()
        log_reg = LogisticRegression(solver='liblinear')
        log_reg.fit(X_train.values, y_train.values)
        ml = Model.from_env('tester', has_contract=False)
        ml.add_trained_model(log_reg)
        result = ml.intent_model.label_predict(X_test)
        self.assertEqual((300,1), result.shape)
        X_test = pd.concat([df_cust, X_test], axis=1)
        result = ml.intent_model.label_predict(X_test, id_header='cust_id')
        self.assertEqual((300,2), result.shape)

    def test_label_predict_lin(self):
        builder = SyntheticBuilder.from_memory()
        df = builder.tools.model_synthetic_regression(1000, n_features=4)
        X = df.drop('target', axis=1)
        y = df['target']
        ml = Model.from_env('tester', has_contract=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        log_reg = LinearRegression()
        log_reg.fit(X_train, y_train)
        ml.add_trained_model(log_reg)
        result = ml.intent_model.label_predict(X_test)
        # self.assertEqual((1000, 1), result.shape)
        print(result.head())


    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))



if __name__ == '__main__':
    unittest.main()
