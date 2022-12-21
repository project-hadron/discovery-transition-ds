import unittest
import os
from pathlib import Path
import shutil
from ds_discovery import ModelsBuilder as Model, SyntheticBuilder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

    def test_set_trained_model(self):
        builder = SyntheticBuilder.from_memory()
        df = builder.tools.model_synthetic_classification(1000, n_features=10)
        X = df.drop('target', axis=1)
        y = df['target']
        ml = Model.from_env('tester', has_contract=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        log_reg = LogisticRegression(solver='liblinear')
        log_reg.fit(X_train, y_train)
        ml.set_trained_model(log_reg)
        result = ml.intent_model.get_prediction(X)
        print(result.shape)


    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))



if __name__ == '__main__':
    unittest.main()
