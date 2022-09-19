import os
import shutil
from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np
import unittest
from aistac.properties.abstract_properties import AbstractPropertyManager
from aistac.properties.property_manager import PropertyManager

from ds_discovery import Transition
from ds_discovery.components.discovery import DataDiscovery as Discovery, DataDiscovery

from ds_discovery import SyntheticBuilder
from ds_discovery.components.commons import Commons
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel


class SyntheticIntentAnalysisTest(unittest.TestCase):

    def setUp(self):
        # clean out any old environments
        for key in os.environ.keys():
            if key.startswith('HADRON'):
                del os.environ[key]
        # Local Domain Contract
        os.environ['HADRON_PM_PATH'] = Path('working/contracts').as_posix()
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

    def test_associate_analysis_from_discovery(self):
        builder = SyntheticBuilder.from_env('tester', has_contract=False)
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame()
        df['cat'] = tools.get_category(selection=list('ABC'), quantity=0.9, size=100, column_name='cat')
        df['values'] = tools.get_number(from_value=20, size=100, column_name='values')
        builder.run_component_pipeline()
        # discover
        associate = [{'cat': {'dtype': 'category'}, 'values': {'dtype': 'category','granularity': 5, 'precision': 3}}]
        # build
        sample_size=173
        result = tools.model_analysis(pd.DataFrame(index=range(sample_size)), other='primary_persist',
                                      columns_list=associate, column_name='analysis')
        self.assertCountEqual(['cat', 'values'], result.keys())
        for key in result.keys():
            self.assertEqual(sample_size, len(result.get(key)))

    def test_associate_analysis_complex(self):
        builder = SyntheticBuilder.from_memory()
        clinical_health = 'https://assets.datacamp.com/production/repositories/628/datasets/444cdbf175d5fbf564b564bd36ac21740627a834/diabetes.csv'
        builder.add_connector_uri('clinical_health', uri=clinical_health)
        discover: DataDiscovery = Transition.from_memory().discover
        A = discover.analysis2dict(header='age', dtype='int', granularity=10.0, lower=21, upper=90)
        B = discover.analysis2dict(header='pregnancies')
        columns_list = [A, B]
        df_clinical = builder.load_canonical('clinical_health')
        canonical = pd.DataFrame(index=range(1973))
        df = builder.tools.model_analysis(canonical, other='clinical_health', columns_list=columns_list, column_name='clinical')
        self.assertEqual((1973, 2), df.shape)
        pregnancies = Commons.list_standardize(Commons.list_formatter(df_clinical.pregnancies))
        low, high = discover.bootstrap_confidence_interval(pd.Series(pregnancies), func=np.mean)
        pregnancies = Commons.list_standardize(Commons.list_formatter(df.pregnancies))
        self.assertTrue(low <= np.mean(pregnancies) <= high)
