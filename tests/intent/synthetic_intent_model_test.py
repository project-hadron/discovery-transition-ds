import unittest
import os
import shutil
import pandas as pd
from ds_discovery import SyntheticBuilder, Wrangle
from aistac.properties.property_manager import PropertyManager

from ds_discovery.intent.synthetic_intent import SyntheticIntentModel


class SyntheticIntentModelTest(unittest.TestCase):

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

        os.environ['HADRON_PM_PATH'] = os.path.join('work', 'config')
        os.environ['HADRON_DEFAULT_PATH'] = os.path.join('work', 'data')
        try:
            os.makedirs(os.environ['HADRON_PM_PATH'])
            os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        except:
            pass
        PropertyManager._remove_all()

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_complex_sample_modelling(self):
        tools = SyntheticBuilder.from_memory().tools
        state_code = ['CA', 'NY', 'LA', 'NJ', 'VA', 'CO', 'NV', 'GA', 'IN', 'OH', 'KY', 'ME', 'MO', 'WI']
        df = pd.DataFrame(index=range(100))
        df = tools.model_sample_map(canonical=df, sample_map='us_zipcode',
                                    state_filter=state_code, column_name='zipcodes')
        sample_data = tools.action2dict(method='model_sample_map', canonical=tools.action2dict(method='@empty'),
                                        sample_map='us_healthcare_practitioner', headers=['city', 'pcp_tax_id'],
                                        shuffle=False)
        merge_data = tools.action2dict(method='model_group', canonical=sample_data, headers='pcp_tax_id',
                                       group_by='city', aggregator='list')
        df = tools.model_merge(df, merge_data, how='left', left_on='city', right_on='city', column_name='pcp_tax_id')
        self.assertCountEqual(['city', 'state_abbr', 'state', 'county_fips', 'county', 'zipcode', 'pcp_tax_id'], df.columns.to_list())

    def test_model_columns_headers(self):
        builder = SyntheticBuilder.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        tools: SyntheticIntentModel = builder.tools
        builder.set_source_uri(uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        df = pd.DataFrame(index=range(300))
        result = tools.model_concat(df, other=builder.CONNECTOR_SOURCE, as_rows=False, headers=['survived', 'sex', 'fare'])
        self.assertCountEqual(['survived', 'sex', 'fare'], list(result.columns))
        self.assertEqual(300, result.shape[0])

    def test_model_merge(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={'A': [1,2,3,4,5], 'B': list('ABCDE')})
        other = {'A': [5,2,3,1,4], 'X': list('VWXYZ'), 'Y': list('VWXYZ')}
        # using left_on and right_on
        result = tools.model_merge(canonical=df, other=other, right_on='A', left_on='A')
        self.assertEqual((5, 4), result.shape)
        self.assertEqual(['A', 'B', 'X', 'Y'], result.columns.to_list())
        # using on
        result = tools.model_merge(canonical=df, other=other, on='A')
        self.assertEqual((5, 4), result.shape)
        self.assertEqual(['A', 'B', 'X', 'Y'], result.columns.to_list())
        # filter headers
        result = tools.model_merge(canonical=df, other=other, on='A', headers=['X'])
        self.assertEqual((5, 3), result.shape)
        self.assertEqual(['A', 'B', 'X'], result.columns.to_list())

    def test_remove_unwanted_headers(self):
        builder = SyntheticBuilder.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        builder.set_source_uri(uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        selection = [builder.tools.select2dict(column='survived', condition='@==1')]
        result = builder.tools.frame_selection(canonical=builder.CONNECTOR_SOURCE, selection=selection, headers=['survived', 'sex', 'fare'])
        self.assertCountEqual(['survived', 'sex', 'fare'], list(result.columns))
        self.assertEqual(1, result['survived'].min())
        result = builder.tools.frame_selection(canonical=builder.CONNECTOR_SOURCE, headers=['survived', 'sex', 'fare'])
        self.assertEqual((891, 3), result.shape)

    def test_remove_unwanted_rows(self):
        builder = SyntheticBuilder.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        builder.set_source_uri(uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        selection = [builder.tools.select2dict(column='survived', condition='@==1')]
        result = builder.tools.frame_selection(canonical=builder.CONNECTOR_SOURCE, selection=selection)
        self.assertEqual(1, result['survived'].min())

    def test_model_sample_map(self):
        builder = SyntheticBuilder.from_memory(default_save_intent=False)
        result = builder.tools.model_sample_map(pd.DataFrame(), sample_map='us_healthcare_practitioner')
        self.assertEqual((192865, 6), result.shape)
        result = builder.tools.model_sample_map(pd.DataFrame(index=range(50)), sample_map='us_healthcare_practitioner')
        self.assertEqual((50, 6), result.shape)
        result = builder.tools.model_sample_map(pd.DataFrame(index=range(50)), sample_map='us_healthcare_practitioner',
                                                headers=['pcp_tax_id'])
        self.assertEqual((50, 1), result.shape)

    def test_model_us_person(self):
        builder = SyntheticBuilder.from_memory(default_save_intent=False)
        df = pd.DataFrame(index=range(300))
        result = builder.tools.model_sample_map(canonical=df, sample_map='us_persona')
        self.assertCountEqual(['first_name', 'middle_name', 'gender', 'family_name', 'email'], result.columns.to_list())
        self.assertEqual(300, result.shape[0])
        df = pd.DataFrame(index=range(1000))
        df = builder.tools.model_sample_map(canonical=df, sample_map='us_persona', female_bias=0.3)
        self.assertEqual((1000, 5), df.shape)
        print(df['gender'].value_counts().loc['F'])

    def test_model_iterator(self):
        builder = SyntheticBuilder.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        tools: SyntheticIntentModel = builder.tools
        builder.add_connector_uri('titanic', uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        # do nothing
        result = tools.model_iterator(canonical='titanic')
        self.assertEqual(builder.load_canonical('titanic').shape, result.shape)
        # add marker
        result = tools.model_iterator(canonical='titanic', marker_col='marker')
        self.assertEqual(builder.load_canonical('titanic').shape[1]+1, result.shape[1])
        # with selection
        selection = [tools.select2dict(column='survived', condition="@==1")]
        control = tools.frame_selection(canonical='titanic', selection=selection)
        result = tools.model_iterator(canonical='titanic', marker_col='marker', selection=selection)
        self.assertEqual(control.shape[0], result.shape[0])
        # with iteration
        result = tools.model_iterator(canonical='titanic', marker_col='marker', iter_stop=3)
        self.assertCountEqual([0,1,2], result['marker'].value_counts().index.to_list())
        # with actions
        actions = {2: (tools.action2dict(method='get_category', selection=[4,5]))}
        result = tools.model_iterator(canonical='titanic', marker_col='marker', iter_stop=3, iteration_actions=actions)
        self.assertCountEqual([0,1,4,5], result['marker'].value_counts().index.to_list())

    def test_model_group(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        builder.add_connector_uri('titanic', uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        df = tools.model_group('titanic', headers='fare', group_by=['survived', 'sex'], aggregator='sum')
        self.assertEqual((4, 3), df.shape)
        df = tools.model_group('titanic', headers=['class', 'embark_town'], group_by=['survived', 'sex'],
                               aggregator='set', list_choice=2)
        # print(df.loc[:, ['class', 'embark_town']])
        self.assertEqual((4, 4), df.shape)
        self.assertCountEqual(['class', 'embark_town', 'survived', 'sex'], df.columns.to_list())
        df = tools.model_group('titanic', headers=['fare', 'survived'], group_by='sex', aggregator='sum', include_weighting=True)
        self.assertEqual((2, 4), df.shape)
        self.assertCountEqual(['survived', 'sex', 'fare', 'weighting'], df.columns.to_list())

    def test_model_explode(self):
        df = pd.DataFrame({'A': [1,2,3], 'B': [[2,2], [3], [7,8,9]]})
        wr = Wrangle.from_memory(default_save_intent=False)
        df = wr.tools.model_explode(df, header='B')
        self.assertEqual([1, 1, 2, 3, 3, 3], df['A'].to_list())
        self.assertEqual([2, 2, 3, 7, 8, 9], df['B'].to_list())

    def test_model_sample(self):
        builder = SyntheticBuilder.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        tools: SyntheticIntentModel = builder.tools
        builder.add_connector_uri(connector_name='titanic', uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        df = pd.DataFrame(index=range(300))
        df = tools.model_sample(df, sample='titanic')
        self.assertEqual((300, 15), df.shape)

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
