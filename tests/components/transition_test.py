import os
import shutil
import unittest
from pathlib import Path
from pprint import pprint

import pandas as pd
from aistac.handlers.abstract_handlers import ConnectorContract
from aistac.properties.property_manager import PropertyManager

from ds_discovery import Transition


class TransitionTest(unittest.TestCase):

    """Test: """
    def setUp(self):
        # set environment variables
        os.environ['HADRON_PM_PATH'] = os.path.join(os.environ['PWD'], 'work', 'config')
        os.environ['HADRON_DEFAULT_PATH'] = os.path.join(os.environ['PWD'], 'work', 'data', '0_raw')
        try:
            shutil.copytree('../data', os.path.join(os.environ['PWD'], 'work'))
        except:
            pass
        PropertyManager._remove_all()

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        Transition.from_env('TestAgent', has_contract=False)

    def test_provenance_report(self):
        tr: Transition = Transition.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        tr.set_provenance(title='new_title', domain='Healthcare', author_name='Joe Bloggs')
        result = tr.report_provenance(stylise=False)
        self.assertEqual((6,1), result.shape)
        self.assertCountEqual(['title', 'domain', 'license_type', 'license_name', 'license_uri', 'author_name'], list(result.index))

    def test_provenance_from_bootstrap(self):
        tr: Transition = Transition.from_memory()
        tr.setup_bootstrap(domain='heathcare', project_name='datalake_gen')
        tr.set_provenance(provider_name="Project Hadron", author_name='doatridge', cost_price="$0.00")
        report = tr.report_provenance(stylise=False).index.to_list()
        control = ['title', 'domain', 'description', 'license_type', 'license_name', 'license_uri', 'cost_price', 'provider_name', 'author_name']
        self.assertCountEqual(control, report)

    def test_dictionary_report(self):
        df = pd.DataFrame({'A': [1,2,3], 'B': [1,2,3], 'C': [1,2,3], 'D': [1,2,3]})
        notes = pd.DataFrame()
        notes['label'] = ['A', 'B', 'D', 'F']
        notes['text'] = ['This is the Alpha', 'Beta follows it closely', 'D is the last', 'F is out of place']
        cp = Transition.from_env('task', has_contract=False)
        cp.upload_attributes(canonical=notes, label_key='label', text_key='text')
        result = cp.report_attributes(df, stylise=False)
        self.assertEqual((4, 2), result.shape)

    def test_from_env(self):
        os.environ['HADRON_PM_PATH'] = Path(os.environ['PWD'], 'work').as_posix()
        os.environ['HADRON_PM_TYPE'] = 'pickle'
        os.environ['HADRON_PM_MODULE'] = 'aistac.handlers.python_handlers'
        os.environ['HADRON_PM_HANDLER'] = 'PythonPersistHandler'
        tr = Transition.from_env('task', has_contract=False)
        self.assertEqual( os.environ['HADRON_PM_PATH'] + "/hadron_pm_transition_task.pickle", tr.pm.get_connector_contract(tr.pm.CONNECTOR_PM_CONTRACT).uri)
        self.assertEqual( os.environ['HADRON_PM_MODULE'], tr.pm.get_connector_contract(tr.pm.CONNECTOR_PM_CONTRACT).module_name)
        self.assertEqual(os.environ['HADRON_PM_HANDLER'], tr.pm.get_connector_contract(tr.pm.CONNECTOR_PM_CONTRACT).handler)

        os.environ['HADRON_PM_MODULE'] = 'ds_discovery.handlers.pandas_handlers'
        os.environ['HADRON_PM_HANDLER'] = 'PandasPersistHandler'
        tr = Transition.from_env('task', has_contract=False)
        self.assertEqual( os.environ['HADRON_PM_PATH'] + "/hadron_pm_transition_task.pickle", tr.pm.get_connector_contract(tr.pm.CONNECTOR_PM_CONTRACT).uri)
        self.assertEqual( os.environ['HADRON_PM_MODULE'], tr.pm.get_connector_contract(tr.pm.CONNECTOR_PM_CONTRACT).module_name)
        self.assertEqual(os.environ['HADRON_PM_HANDLER'], tr.pm.get_connector_contract(tr.pm.CONNECTOR_PM_CONTRACT).handler)

        os.unsetenv('HADRON_PM_PATH')
        os.unsetenv('HADRON_PM_TYPE')
        os.unsetenv('HADRON_PM_MODULE')
        os.unsetenv('HADRON_PM_HANDLER')

    def test_transition_summary_report(self):
        tr: Transition = Transition.from_env('test', default_save=False, default_save_intent=False, has_contract=False)
        cc = ConnectorContract(uri=os.path.join(os.environ['HOME'], 'code', 'projects', 'data', 'sample', 'synthetic_customer.csv'),
                               module_name=tr.DEFAULT_MODULE, handler=tr.DEFAULT_SOURCE_HANDLER)
        tr.set_source_contract(connector_contract=cc)
        report = tr.report_quality_summary(as_dict=True)
        self.assertEqual(['score', 'data_shape', 'data_type', 'usability', 'cost'], list(report.keys()))

    def test_set_report_persist(self):
        tr: Transition = Transition.from_env('tester', default_save=False, has_contract=False)
        tr.setup_bootstrap(domain='domain', project_name='project_name', path=None)
        report = tr.report_connectors(stylise=False)
        _, file = os.path.split(report.uri.iloc[0])
        self.assertTrue(file.startswith('project_name'))

    def test_repo_load(self):
        os.environ['HADRON_PM_REPO'] = "https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/bundles/samples/hk_income_sample/contracts/"
        tr: Transition = Transition.from_env('hk_income', has_contract=False)

    def test_run_transition_pipeline(self):
        os.environ['HADRON_PM_REPO'] = "https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/bundles/samples/hk_income_sample/contracts/"
        tr: Transition = Transition.from_env('hk_income', default_save=False, default_save_intent=False, has_contract=False)
        pprint(tr.pm.report_connectors())
        # builder: SyntheticBuilder = SyntheticBuilder.from_env('hk_income', default_save=False, default_save_intent=False, has_contract=False)
        # builder.run_synthetic_pipeline(size=1000)
        # tr.run_transition_pipeline()



    """
        OLD STUFF BELLOW
    """

    # def test_report_connectors(self):
    #     pm = TransitionPropertyManager('task')
    #     im = TransitionIntentModel(pm)
    #     instance = Transition(pm, im)
    #     instance._init_properties(pm, os.environ['HADRON_PM_PATH'])
    #     sc = ConnectorContract(uri='synthetic_customer.csv',
    #                            module_name=instance.PYTHON_MODULE_NAME, handler=instance.PYTHON_HANDLER, sep=',', encoding='latin1')
    #     instance.set_source_contract(connector_contract=sc)
    #     pc = ConnectorContract(uri= instance.file_pattern(),
    #                            module_name=instance.PYTHON_MODULE_NAME,
    #                            handler=instance.PYTHON_HANDLER)
    #     instance.set_persist_contract(connector_contract=pc)
    #     report_canonical = instance.report_connectors(stylise=True)
    #     print(report_canonical.columns)
    #
    # def test_keys(self):
    #     tr = Transition.from_env('Example01', has_contract=False)
    #     join = tr.pm.join
    #     self.assertEqual('data.Example01.connectors', tr.pm.KEY.connectors_key)
    #     self.assertEqual('data.Example01.cleaners', tr.pm.KEY.cleaners_key)
    #     self.assertEqual('data.Example01.connectors.resource', join(tr.pm.KEY.connectors_key, 'resource'))
    #     self.assertEqual('data.Example01.connectors.type', join(tr.pm.KEY.connectors_key, 'type'))
    #     self.assertEqual('data.Example01.connectors.location', join(tr.pm.KEY.connectors_key, 'location'))
    #     self.assertEqual('data.Example01.connectors.module', join(tr.pm.KEY.connectors_key, 'module'))
    #     self.assertEqual('data.Example01.connectors.handler', join(tr.pm.KEY.connectors_key, 'handler'))
    #     self.assertEqual('data.Example01.connectors.kwargs', join(tr.pm.KEY.connectors_key, 'kwargs'))
    #
    # def test_is_contract_empty(self):
    #     tr = Transition.from_env('synthetic', has_contract=False)
    #     self.assertTrue(tr.is_contract_empty())
    #     tr.set_source_contract(uri='synthetic.csv', encoding='latin1', load=False)
    #     self.assertFalse(tr.is_contract_empty())
    #     tr.remove_source_contract()
    #     self.assertTrue(tr.is_contract_empty())
    #     section = {'auto_clean_header': {'case': 'title', 'rename_map': {'forename': 'first_name'}, 'replace_spaces': '_'}}
    #     tr.set_cleaner(section)
    #     self.assertFalse(tr.is_contract_empty())
    #     tr.remove_cleaner('auto_clean_header')
    #     self.assertTrue(tr.is_contract_empty())
    #     tr.create_snapshot()
    #     self.assertFalse(tr.is_contract_empty())
    #     tr.delete_snapshot(tr.snapshots[0])
    #     self.assertTrue(tr.is_contract_empty())
    #
#     def test_source_report(self):
#         tr = Transition.from_env('synthetic', has_contract=False)
#         tr.set_source_contract(uri='synthetic.csv', encoding='latin1', load=False)
#         report_canonical = tr.report_connectors(stylise=False)
#         self.assertEqual(['param', 'Property Source', 'Data Source'], list(report_canonical.columns))
#
#     def test_load_clean_file(self):
#         tr = Transition.from_env('Example01', has_contract=False)
#         tr.set_version('0.01')
#         tr.set_source_contract(uri='example01.csv', sep=',', encoding='latin1', load=False)
#         df = tr.load_source_canonical()
#
#         tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True))
#
#         tr.set_cleaner(tr.clean.to_remove(df, headers=['status', 'type'], inplace=True))
#         tr.set_cleaner(tr.clean.to_date_type(df, headers=['date', 'datetime'], inplace=True))
#         tr.set_cleaner(tr.clean.to_category_type(df, dtype=['object'], headers=['active', 'agent_id', 'postcode'],
#                                                  drop=True, inplace=True), save=False)
#
#         bm = {1: True, 'Y': True}
#         tr.set_cleaner(tr.clean.to_bool_type(df, bool_map=bm, headers=['active', 'cortex'], inplace=True), save=False)
#
#         tr.set_cleaner(tr.clean.to_float_type(df, headers=[''], inplace=True), save=False)
#         tr.set_cleaner(tr.clean.to_int_type(df, headers=['age'], inplace=True), save=False)
#         tr.set_cleaner(tr.clean.to_str_type(df, headers=['policy_id', 'agent_id', 'pid'], inplace=True), save=False)
#
#         tr.persist_contract()
#         return
#
    # def test_load_clean_layers(self):
    #     tr = Transition.from_env('Example01')
    #     tr.set_version('0.01')
    #     tr.set_source('example01.csv', sep=',', encoding='latin1', load=False)
    #     df = tr.load_source_canonical()
    #
    #     tr.intent_model.clean.auto_clean_header(df, inplace=True))
    #     control = {'0': {'auto_clean_header': {'replace_spaces': '_'}}}
    #     self.assertEqual(control, tr.pm.cleaners)
    #     tr.set_cleaner(tr.clean.auto_clean_header(df, case='Title', inplace=True))
    #     control = {'0': {'auto_clean_header': {'case': 'Title', 'replace_spaces': '_'}}}
    #     self.assertEqual(control, tr.pm.cleaners)
    #
    #     tr.set_cleaner(tr.clean.auto_clean_header(df, replace_spaces='#', inplace=True), level=-1)
    #     tr.set_cleaner(tr.clean.auto_clean_header(df, replace_spaces='$', inplace=True), level=-1)
    #     tr.set_cleaner(tr.clean.auto_clean_header(df, case='Title', inplace=True))
    #     control = {'0': {'auto_clean_header': {'case': 'Title', 'replace_spaces': '_'}},
    #                '1': {'auto_clean_header': {'replace_spaces': '#'}},
    #                '2': {'auto_clean_header': {'replace_spaces': '$'}}}
    #     self.assertEqual(control, tr.pm.cleaners)
    #
    #     tr.set_cleaner(tr.clean.auto_clean_header(df, case='lower', inplace=True), level=1)
    #     control = {'0': {'auto_clean_header': {'case': 'Title', 'replace_spaces': '_'}},
    #                '1': {'auto_clean_header': {'case': 'lower', 'replace_spaces': '_'}},
    #                '2': {'auto_clean_header': {'replace_spaces': '$'}}}
    #     self.assertEqual(control, tr.pm.cleaners)
    #
    #     tr.remove_cleaner()
    #     control = {}
    #     self.assertEqual(control, tr.pm.cleaners)
    #     return
    #
#     def test_load_clean_remove(self):
#         tr = Transition.from_env('Example01')
#         tr.set_version('0.01')
#         tr.set_source('example01.csv', sep=',', encoding='latin1')
#         df = tr.load_source_canonical()
#
#         tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=0)
#         tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=2)
#         tr.remove_cleaner(cleaner='auto_clean_header')
#         control = {}
#         self.assertEqual(control, tr.pm.cleaners)
#
#         tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=0)
#         tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=2)
#         tr.set_cleaner(tr.clean.auto_drop_duplicates(df, inplace=True), level=1)
#         tr.remove_cleaner(cleaner='auto_clean_header')
#         control = {'1': {'auto_drop_duplicates': {}}}
#         self.assertEqual(control, tr.pm.cleaners)
#         tr.set_cleaner(tr.clean.auto_drop_duplicates(df, inplace=True), level=-1)
#         control = {'0': {'auto_drop_duplicates': {}}}
#         self.assertEqual(control, tr.pm.cleaners)
#         tr.remove_cleaner(level=0)
#         control = {}
#         self.assertEqual(control, tr.pm.cleaners)
#
#         tr.remove_cleaner()
#         control = {}
#         self.assertEqual(control, tr.pm.cleaners)
#
#         tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=0)
#         tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=2)
#         tr.set_cleaner(tr.clean.auto_drop_duplicates(df, inplace=True), level=1)
#         tr.set_cleaner(tr.clean.auto_drop_duplicates(df, inplace=True), level=0)
#         tr.remove_cleaner(cleaner='auto_clean_header')
#         control = {'0': {'auto_drop_duplicates': {}}}
#         self.assertEqual(control, tr.pm.cleaners)
#         return
#
#     def test_create_contract(self):
#         name = 'TestContract'
#         tr = Transition.from_env(name)
#         dpm = tr.pm
#         self.assertTrue(name in dpm.contract_name)
#         control = {'TestContract': {'cleaners': {},'snapshot': {},
#                   'connectors': {'pm_data_testcontract': {'handler': 'PandasPersistHandler',
#                                                       'modified': 0,
#                                                       'module_name': 'ds_discovery.handlers.pandas_handlers',
#                                                       'uri': '/tmp/aistac/contracts/config_transition_data_TestContract.yaml'}},
#                   'version': 'v0.00'}}
#         result = dpm.get(dpm.KEY.manager_key)
#         self.assertEqual(control, result)
#
#     def test_snapshot(self):
#         tr = Transition.from_env('synthetic')
#         tr.create_snapshot(suffix='Test')
#         print(tr.pm.get(tr.pm.KEY.snapshot_key).keys())
#         self.assertTrue('synthetic_#Test' in tr.pm.get(tr.pm.KEY.snapshot_key).keys())
#         self.assertTrue('synthetic_#Test' in tr.snapshots)
#
#     def test_refresh_canonical(self):
#         tr = Transition.from_env('synthetic')
#         df = tr.set_source_contract(uri='synthetic.csv', sep=',', encoding='latin1', load=True)
#         tr.set_feature_contract()
#         self.assertEqual((5000, 14), df.shape)
#         tr.set_cleaner(tr.clean.auto_clean_header(df, rename_map={'start': 'start_date'}, inplace=True))
#         tr.set_cleaner(tr.clean.auto_remove_columns(df, null_min=0.99, predominant_max=0.90, inplace=True, nulls_list=['']))
#         tr.set_cleaner(tr.clean.auto_to_category(df, unique_max=20, null_max=0.7, inplace=True))
#         self.assertEqual((5000, 9), df.shape)
#         df = tr.refresh_clean_canonical()
#         self.assertEqual((5000, 9), df.shape)
#
#     def test_multi_instance(self):
#         tr = Transition.from_env('synthetic')
#         structure = tr.pm.get(tr.pm.KEY.contract_key)
#         tr2 = Transition.from_env('control')
#         self.assertEqual(structure, tr.pm.get(tr.pm.KEY.contract_key))
#         control = {'cleaners': {}, 'snapshot': {}, 'connectors':
#                    {'pm_data_control': {'handler': 'PandasPersistHandler',
#                                  'location': '/Users/doatridge/code/projects/prod/components-components-ds/tests/components/work/config/control',
#                                  'modified': 0,
#                                  'module_name': 'ds_discovery.handlers.pandas_handlers',
#                                  'resource': 'config_data_control.yaml',
#                                  'connector_type': 'yaml'}},
#                    'version': 'v0.00'}
#         self.assertEqual(control, tr2.pm.get(tr2.pm.KEY.contract_key))
#         tr2 = Transition.from_env('control')
#         self.assertEqual(structure, tr.pm.get(tr.pm.KEY.contract_key))
#         self.assertEqual(control, tr2.pm.get(tr2.pm.KEY.contract_key))
#
#     def test_notes_add_remove(self):
#         tr = Transition.from_env('synthetic')
#         control = ['overview', 'notes', 'observations', 'attribute', 'dictionary', 'tor', 'general']
#         self.assertEqual(control, tr.augment_pm.catalogue)
#         tr.add_attribute_notes(text='Text for Attribute A', attribute='attrA')
#         tr.add_notes('add note for label A', label='labelA')
#         tr.add_notes('include note type', label='scale', note_type=tr.augment_pm.catalogue[0])
#         tr.remove_notes()
#
#     def test_notes_report(self):
#         tr = Transition.from_env('synthetic')
#         tr.add_notes(text='The file is a synthetic customer data file created for this demonstration')
#         tr.add_notes(label='connector', text='This was generated using the Discovery Behavioral Synthetic Data Generator')
#         tr.add_notes(label='connector', text='The script to rerun the data generation can be found in the synthetic scripts folder')
#         tr.add_attribute_notes(attribute='null', text="Here for demo of removal of nulls")
#         tr.add_attribute_notes(attribute='weight_cat', text="Demonstration of removal of columns with predominant values")
#         tr.add_attribute_notes(attribute='weight_cat', text="the value 'A' is over 95% predominant")
#         tr.add_attribute_notes(attribute='start', text="changing this to start_date so it being a date is obvious")
#         report_canonical = tr.report_notes(regex="'A'", stylise=False, drop_dates=True)
#         control = {'section': ['attribute'], 'label': ['weight_cat'], 'text': ["the value 'A' is over 95% predominant"]}
#         self.assertEqual(control, report_canonical.to_dict(orient='list'))
#
#     def test_is_raw_modified(self):
#         tr = Transition.from_env('synthetic')
#         with self.assertRaises(ModuleNotFoundError) as context:
#             tr.is_source_modified()
#         self.assertTrue("The connector 'origin_connector' has not been set" in str(context.exception))
#         tr.set_source_contract(resource='synthetic.csv', connector_type='csv', location=os.environ['DTU_ORIGIN_PATH'],
#                                     module_name=tr.MODULE_NAME, handler=tr.SOURCE_HANDLER, sep=',', encoding='latin1', load=False)
#         tr.set_feature_contract()
#         tr.load_source_canonical()
#         self.assertFalse(tr.is_source_modified())
#         connector_contract = tr.pm.get_connector_contract(tr.ORIGIN_CONNECTOR)
#         raw_file = os.path.join(connector_contract.location, connector_contract.resource)
#         Path(raw_file).touch()
#         self.assertTrue(tr.is_source_modified())
#
#     def test_source_load_change(self):
#         tr = Transition.from_env('synthetic')
#         tr.set_source_contract(uri='synthetic.csv', encoding='latin1', load=False)
#         tr.set_feature_contract()
#         df = tr.load_source_canonical()
#         self.assertEqual((5000, 14), df.shape)
#         Customer.generate(extra=True)
#         self.assertTrue(tr.is_source_modified())
#         df = tr.load_source_canonical()
#         self.assertEqual((100, 16), df.shape)
#
#     def test_reload_instance(self):
#         control = Transition.from_env('synthetic_test')
#         control.set_source_contract(uri='synthetic.csv', encoding='latin1', load=False)
#         control.set_feature_contract()
#         control.set_version('v_test')
#         control.persist_contract()
#         contract = control.pm.get(control.pm.KEY.contract_key)
#         control.pm.reset_contract_properties()
#         reset = control.pm.get(control.pm.KEY.contract_key)
#         tr = Transition.from_env('synthetic_test')
#         result = tr.pm.get(tr.pm.KEY.contract_key)
#         self.assertEqual(contract, result)
#
#
# class Customer(object):
#     """ Builds a synthetic customer"""
#
#     @staticmethod
#     def generate(noise: bool=False, extra: bool=False, sample_size: int=None):
#         builder = SyntheticBuilder.from_env('synthetic_data_customer')
#         tools = builder.tools
#
#         # main build
#         sample_size = sample_size if isinstance(sample_size, int) and 1 <= sample_size <= 10000 else 100
#         df = tools.get_profiles(size=sample_size, mf_weighting=[5, 3])
#         df['id'] = tools.unique_identifiers(from_value=1000000, to_value=9999999, prefix='CU_', size=sample_size)
#         value_distribution = [0.01, 0.8, 1, 3, 9, 8, 3, 2, 1] + list(np.flip(np.exp(np.arange(-5, 0.0, 0.2)).round(2)))
#         df['balance'] = tools.get_number(0.0, 1000, precision=2, relative_freq=value_distribution, size=sample_size)
#         age_pattern = [3, 5, 6, 10, 6, 5, 7, 15, 5, 2, 1, 0.5, 0.2, 0.1]
#         df['age'] = tools.get_number(20.0, 90.0, relative_freq=age_pattern, quantity=0.85, size=sample_size)
#         df['start'] = tools._get_datetime(start='01/01/2018', until='31/12/2018', date_format='%m-%d-%y',
#                                           size=sample_size)
#         prof_pattern = [10, 8, 5, 4, 3, 2] + [1] * 9
#         profession = ProfileSample.professions(size=15)
#         df['profession'] = tools.get_category(selection=profession, relative_freq=prof_pattern, quantity=0.90,
#                                               size=sample_size)
#         df['online'] = tools.get_category(selection=[1, 0], relative_freq=[1, 4], size=sample_size)
#
#         # Selective Noise
#         df['single num'] = tools.get_number(1, 1, quantity=0.8, size=sample_size, seed=31)
#         df['weight_num'] = tools.get_number(1, 2, relative_freq=[90, 1], size=sample_size, seed=31)
#         df['null'] = tools.get_number(1, 100, quantity=0, size=sample_size, seed=31)
#         df['single cat'] = tools.get_category(['A'], quantity=0.6, size=sample_size, seed=31)
#         df['weight_cat'] = tools.get_category(['A', 'B', 'C'], relative_freq=[80, 1, 1], size=sample_size, seed=31)
#
#         # Optional extra fields
#         if extra:
#             df['last_login'] = tools._get_datetime(start='01/01/2019', until='01/05/2019',
#                                                    date_pattern=[1, 2, 3, 5, 9, 20], date_format='%m-%d-%y %H:%M',
#                                                    size=sample_size)
#             df['status'] = tools.get_category(selection=['Active', 'Closed', 'Suspended'], relative_freq=[50, 5, 2],
#                                               size=sample_size)
#         # Optional extra noise
#         if noise:
#             for i in range(40):
#                 quantity = tools.get_number(0.005, 0.03, relative_freq=[5, 2, 1, 0.5])[0]
#                 col = "noise_{}".format(i)
#                 df[col] = tools.get_number(0, 1, relative_freq=[20, 1], quantity=quantity, size=sample_size)
#
#         # save
#         filename = os.path.join(os.environ['PWD'], 'work', 'data', '0_raw', 'synthetic.csv')
#         builder.save_to_disk(df, filename=filename)
#         return


if __name__ == '__main__':
    unittest.main()
