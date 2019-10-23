import os
import shutil
import unittest
import numpy as np

from pprint import pprint
from pathlib import Path

from ds_behavioral import DataBuilder
from ds_behavioral.sample.sample_data import ProfileSample

from ds_discovery import TransitionAgent
from ds_foundation.properties.property_manager import PropertyManager


class TransitionTest(unittest.TestCase):
    """Test: """
    def setUp(self):
        # set environment variables
        os.environ['DTU_CONTRACT_PATH'] = os.path.join(os.environ['PWD'], 'work', 'config')
        os.environ['DTU_ORIGIN_PATH'] = os.path.join(os.environ['PWD'], 'work', 'data', '0_raw')
        try:
            shutil.copytree('../data', os.path.join(os.environ['PWD'], 'work'))
        except:
            pass

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass
        props = PropertyManager().get_all()
        for key in props.keys():
            PropertyManager().remove(key)

    def test_runs(self):
        """Basic smoke test"""
        TransitionAgent.from_env('TestAgent')

    def test_factory_remote(self):
        tr = TransitionAgent.from_remote('test_factory')
        tr.set_source_contract(resource='scratch/source/synthetic_customer.csv', connector_type='csv',
                               location='discovery-persistence', module_name='ds_connectors.handlers.aws_s3_handlers',
                               handler='AwsS3SourceHandler')
        df = tr.load_source_canonical()
        self.assertEqual((1000, 15), df.shape)

    def test_keys(self):
        tr = TransitionAgent.from_env('Example01')
        join = tr.data_pm.join
        self.assertEqual('data.Example01.connectors', tr.data_pm.KEY.connectors_key)
        self.assertEqual('data.Example01.cleaners', tr.data_pm.KEY.cleaners_key)
        self.assertEqual('data.Example01.connectors.resource', join(tr.data_pm.KEY.connectors_key, 'resource'))
        self.assertEqual('data.Example01.connectors.type', join(tr.data_pm.KEY.connectors_key, 'type'))
        self.assertEqual('data.Example01.connectors.location', join(tr.data_pm.KEY.connectors_key, 'location'))
        self.assertEqual('data.Example01.connectors.module', join(tr.data_pm.KEY.connectors_key, 'module'))
        self.assertEqual('data.Example01.connectors.handler', join(tr.data_pm.KEY.connectors_key, 'handler'))
        self.assertEqual('data.Example01.connectors.kwargs', join(tr.data_pm.KEY.connectors_key, 'kwargs'))

    def test_is_contract_empty(self):
        tr = TransitionAgent.from_env('synthetic')
        self.assertTrue(tr.is_contract_empty())
        tr.set_source_contract(resource='synthetic.csv', connector_type='csv', sep=',', encoding='latin1', load=False)
        self.assertFalse(tr.is_contract_empty())
        tr.remove_source_contract()
        self.assertTrue(tr.is_contract_empty())
        section = {'auto_clean_header': {'case': 'title', 'rename_map': {'forename': 'first_name'}, 'replace_spaces': '_'}}
        tr.set_cleaner(section)
        self.assertFalse(tr.is_contract_empty())
        tr.remove_cleaner('auto_clean_header')
        self.assertTrue(tr.is_contract_empty())
        tr.create_snapshot()
        self.assertFalse(tr.is_contract_empty())
        tr.delete_snapshot(tr.snapshots[0])
        self.assertTrue(tr.is_contract_empty())

    def test_source_report(self):
        tr = TransitionAgent.from_env('synthetic')
        tr.set_source_contract(resource='synthetic.csv', connector_type='csv', sep=',', encoding='latin1', load=False)
        report = tr.report_source(stylise=False)
        self.assertEqual(['param', 'Property Source', 'Data Source'], list(report.columns))

    def test_load_clean_file(self):
        tr = TransitionAgent.from_env('Example01')
        tr.set_version('0.01')
        tr.set_source_contract('example01.csv', connector_type='csv', sep=',', encoding='latin1', load=False)
        df = tr.load_source_canonical()

        tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True))

        tr.set_cleaner(tr.clean.to_remove(df, headers=['status', 'type'], inplace=True))
        tr.set_cleaner(tr.clean.to_date_type(df, headers=['date', 'datetime'], inplace=True))
        tr.set_cleaner(tr.clean.to_category_type(df, dtype=['object'], headers=['active', 'agent_id', 'postcode'],
                                                 drop=True, inplace=True), save=False)

        bm = {1: True, 'Y': True}
        tr.set_cleaner(tr.clean.to_bool_type(df, bool_map=bm, headers=['active', 'cortex'], inplace=True), save=False)

        tr.set_cleaner(tr.clean.to_float_type(df, headers=[''], inplace=True), save=False)
        tr.set_cleaner(tr.clean.to_int_type(df, headers=['age'], inplace=True), save=False)
        tr.set_cleaner(tr.clean.to_str_type(df, headers=['policy_id', 'agent_id', 'pid'], inplace=True), save=False)

        tr.persist_contract()
        return

    def test_load_clean_layers(self):
        tr = TransitionAgent.from_env('Example01')
        tr.set_version('0.01')
        tr.set_source_contract('example01.csv', connector_type='csv', sep=',', encoding='latin1', load=False)
        df = tr.load_source_canonical()

        tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True))
        control = {'0': {'auto_clean_header': {'replace_spaces': '_'}}}
        self.assertEqual(control, tr.data_pm.cleaners)
        tr.set_cleaner(tr.clean.auto_clean_header(df, case='Title', inplace=True))
        control = {'0': {'auto_clean_header': {'case': 'Title', 'replace_spaces': '_'}}}
        self.assertEqual(control, tr.data_pm.cleaners)

        tr.set_cleaner(tr.clean.auto_clean_header(df, replace_spaces='#', inplace=True), level=-1)
        tr.set_cleaner(tr.clean.auto_clean_header(df, replace_spaces='$', inplace=True), level=-1)
        tr.set_cleaner(tr.clean.auto_clean_header(df, case='Title', inplace=True))
        control = {'0': {'auto_clean_header': {'case': 'Title', 'replace_spaces': '_'}},
                   '1': {'auto_clean_header': {'replace_spaces': '#'}},
                   '2': {'auto_clean_header': {'replace_spaces': '$'}}}
        self.assertEqual(control, tr.data_pm.cleaners)

        tr.set_cleaner(tr.clean.auto_clean_header(df, case='lower', inplace=True), level=1)
        control = {'0': {'auto_clean_header': {'case': 'Title', 'replace_spaces': '_'}},
                   '1': {'auto_clean_header': {'case': 'lower', 'replace_spaces': '_'}},
                   '2': {'auto_clean_header': {'replace_spaces': '$'}}}
        self.assertEqual(control, tr.data_pm.cleaners)

        tr.remove_cleaner()
        control = {}
        self.assertEqual(control, tr.data_pm.cleaners)
        return

    def test_load_clean_remove(self):
        tr = TransitionAgent.from_env('Example01')
        tr.set_version('0.01')
        tr.set_source_contract('example01.csv', connector_type='csv', sep=',', encoding='latin1', load=False)
        df = tr.load_source_canonical()

        tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=0)
        tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=2)
        tr.remove_cleaner(cleaner='auto_clean_header')
        control = {}
        self.assertEqual(control, tr.data_pm.cleaners)

        tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=0)
        tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=2)
        tr.set_cleaner(tr.clean.auto_drop_duplicates(df, inplace=True), level=1)
        tr.remove_cleaner(cleaner='auto_clean_header')
        control = {'1': {'auto_drop_duplicates': {}}}
        self.assertEqual(control, tr.data_pm.cleaners)
        tr.set_cleaner(tr.clean.auto_drop_duplicates(df, inplace=True), level=-1)
        control = {'0': {'auto_drop_duplicates': {}}}
        self.assertEqual(control, tr.data_pm.cleaners)
        tr.remove_cleaner(level=0)
        control = {}
        self.assertEqual(control, tr.data_pm.cleaners)

        tr.remove_cleaner()
        control = {}
        self.assertEqual(control, tr.data_pm.cleaners)

        tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=0)
        tr.set_cleaner(tr.clean.auto_clean_header(df, inplace=True), level=2)
        tr.set_cleaner(tr.clean.auto_drop_duplicates(df, inplace=True), level=1)
        tr.set_cleaner(tr.clean.auto_drop_duplicates(df, inplace=True), level=0)
        tr.remove_cleaner(cleaner='auto_clean_header')
        control = {'0': {'auto_drop_duplicates': {}}}
        self.assertEqual(control, tr.data_pm.cleaners)
        return

    def test_create_contract(self):
        name = 'TestContract'
        tr = TransitionAgent.from_env(name)
        dpm = tr.data_pm
        self.assertTrue(name in dpm.contract_name)
        control = {'TestContract': {'cleaners': {},'snapshot': {},
                  'connectors': {'pm_data_testcontract': {'handler': 'PandasPersistHandler',
                                                      'location': '/Users/doatridge/code/projects/prod/discovery-transition-ds/tests/discovery/work/config/TestContract',
                                                      'modified': 0,
                                                      'module_name': 'ds_discovery.handlers.pandas_handlers',
                                                      'resource': 'config_data_TestContract.yaml',
                                                      'connector_type': 'yaml'}},
                  'version': 'v0.00'}}
        result = dpm.get(dpm.KEY.manager_key)
        self.assertEqual(control, result)

    def test_snapshot(self):
        tr = TransitionAgent.from_env('synthetic')
        tr.create_snapshot(suffix='Test')
        print(tr.data_pm.get(tr.data_pm.KEY.snapshot_key).keys())
        self.assertTrue('synthetic_#Test' in tr.data_pm.get(tr.data_pm.KEY.snapshot_key).keys())
        self.assertTrue('synthetic_#Test' in tr.snapshots)

    def test_refresh_canonical(self):
        tr = TransitionAgent.from_env('synthetic')
        df = tr.set_source_contract(resource='synthetic.csv', connector_type='csv', sep=',', encoding='latin1', load=True)
        tr.set_persist_contract()
        self.assertEqual((5000, 14), df.shape)
        tr.set_cleaner(tr.clean.auto_clean_header(df, rename_map={'start': 'start_date'}, inplace=True))
        tr.set_cleaner(tr.clean.auto_remove_columns(df, null_min=0.99, predominant_max=0.90, inplace=True, nulls_list=['']))
        tr.set_cleaner(tr.clean.auto_to_category(df, unique_max=20, null_max=0.7, inplace=True))
        self.assertEqual((5000, 9), df.shape)
        df = tr.refresh_clean_canonical()
        self.assertEqual((5000, 9), df.shape)

    def test_multi_instance(self):
        tr = TransitionAgent.from_env('synthetic')
        structure = tr.data_pm.get(tr.data_pm.KEY.contract_key)
        tr2 = TransitionAgent.from_env('control')
        self.assertEqual(structure, tr.data_pm.get(tr.data_pm.KEY.contract_key))
        control = {'cleaners': {}, 'snapshot': {}, 'connectors':
                   {'pm_data_control': {'handler': 'PandasPersistHandler',
                                 'location': '/Users/doatridge/code/projects/prod/discovery-transition-ds/tests/discovery/work/config/control',
                                 'modified': 0,
                                 'module_name': 'ds_discovery.handlers.pandas_handlers',
                                 'resource': 'config_data_control.yaml',
                                 'connector_type': 'yaml'}},
                   'version': 'v0.00'}
        self.assertEqual(control, tr2.data_pm.get(tr2.data_pm.KEY.contract_key))
        tr2 = TransitionAgent.from_env('control')
        self.assertEqual(structure, tr.data_pm.get(tr.data_pm.KEY.contract_key))
        self.assertEqual(control, tr2.data_pm.get(tr2.data_pm.KEY.contract_key))

    def test_is_backup(self):
        tr = TransitionAgent.from_env('synthetic')
        tr.backup_contract()
        self.assertTrue(os.path.exists("./work/config/synthetic/config_data_synthetic_00.yaml"))
        tr.set_version('v0.01')
        tr.backup_contract()
        self.assertTrue(os.path.exists("./work/config/synthetic/config_data_synthetic_01.yaml"))
        result = tr.create_snapshot(suffix='test')
        self.assertEqual('synthetic_#test', result)
        self.assertEqual(['synthetic_#test'], tr.snapshots)

    def test_notes_report(self):
        tr = TransitionAgent.from_env('synthetic')
        df = tr.set_source_contract(resource='synthetic.csv', sep=',', encoding='latin1', load=True)
        tr.add_notes(text='The file is a synthetic customer data file created for this demonstration')
        tr.add_notes(label='connector', text='This was generated using the Discovery Behavioral Synthetic Data Generator')
        tr.add_notes(label='connector', text='The script to rerun the data generation can be found in the synthetic scripts folder')
        tr.add_attribute_notes(attribute='null', text="Here for demo of removal of nulls")
        tr.add_attribute_notes(attribute='weight_cat', text="Demonstration of removal of columns with predominant values")
        tr.add_attribute_notes(attribute='weight_cat', text="the value 'A' is over 95% predominant")
        tr.add_attribute_notes(attribute='start', text="changing this to start_date so it being a date is obvious")
        report = tr.report_notes(regex="'A'", stylise=False, drop_dates=True)
        control = {'section': ['attribute'], 'label': ['weight_cat'], 'text': ["the value 'A' is over 95% predominant"]}
        self.assertEqual(control, report.to_dict(orient='list'))

    def test_is_raw_modified(self):
        tr = TransitionAgent.from_env('synthetic')
        with self.assertRaises(ModuleNotFoundError) as context:
            tr.is_source_modified()
        self.assertTrue("The connector 'origin_connector' has not been set" in str(context.exception))
        tr.set_source_contract(resource='synthetic.csv', connector_type='csv', sep=',', encoding='latin1', load=False)
        tr.set_persist_contract()
        tr.load_source_canonical()
        self.assertFalse(tr.is_source_modified())
        connector_contract = tr.data_pm.get_connector_contract(tr.ORIGIN_CONNECTOR)
        raw_file = os.path.join(connector_contract.location, connector_contract.resource)
        Path(raw_file).touch()
        self.assertTrue(tr.is_source_modified())

    def test_source_load_change(self):
        tr = TransitionAgent.from_env('synthetic')
        tr.set_source_contract(resource='synthetic.csv', connector_type='csv', sep=',', encoding='latin1', load=False)
        tr.set_persist_contract()
        df = tr.load_source_canonical()
        self.assertEqual((5000, 14), df.shape)
        Customer.generate(extra=True)
        self.assertTrue(tr.is_source_modified())
        df = tr.load_source_canonical()
        self.assertEqual((100, 16), df.shape)

    def test_reload_instance(self):
        control = TransitionAgent.from_env('synthetic_test')
        control.set_source_contract(resource='synthetic.csv', connector_type='csv', sep=',', encoding='latin1', load=False)
        control.set_persist_contract()
        control.set_version('v_test')
        control.persist_contract()
        contract = control.data_pm.get(control.data_pm.KEY.contract_key)
        control.data_pm.reset_contract_properties()
        reset = control.data_pm.get(control.data_pm.KEY.contract_key)
        tr = TransitionAgent.from_env('synthetic_test')
        result = tr.data_pm.get(tr.data_pm.KEY.contract_key)
        self.assertEqual(contract, result)


class Customer(object):
    """ Builds a synthetic customer"""

    @staticmethod
    def generate(noise: bool=False, extra: bool=False, sample_size: int=None):
        builder = DataBuilder('synthetic_data_customer')
        tools = builder.tools

        # main build
        sample_size = sample_size if isinstance(sample_size, int) and 1 <= sample_size <= 10000 else 100
        df = tools.get_profiles(size=sample_size, mf_weighting=[5, 3])
        df['id'] = tools.unique_identifiers(from_value=1000000, to_value=9999999, prefix='CU_', size=sample_size)
        value_distribution = [0.01, 0.8, 1, 3, 9, 8, 3, 2, 1] + list(np.flip(np.exp(np.arange(-5, 0.0, 0.2)).round(2)))
        df['balance'] = tools.get_number(0.0, 1000, precision=2, weight_pattern=value_distribution, size=sample_size)
        age_pattern = [3, 5, 6, 10, 6, 5, 7, 15, 5, 2, 1, 0.5, 0.2, 0.1]
        df['age'] = tools.get_number(20.0, 90.0, weight_pattern=age_pattern, quantity=0.85, size=sample_size)
        df['start'] = tools.get_datetime(start='01/01/2018', until='31/12/2018', date_format='%m-%d-%y',
                                         size=sample_size)
        prof_pattern = [10, 8, 5, 4, 3, 2] + [1] * 9
        profession = ProfileSample.professions(size=15)
        df['profession'] = tools.get_category(selection=profession, weight_pattern=prof_pattern, quantity=0.90,
                                              size=sample_size)
        df['online'] = tools.get_category(selection=[1, 0], weight_pattern=[1, 4], size=sample_size)

        # Selective Noise
        df['single num'] = tools.get_number(1, 1, quantity=0.8, size=sample_size, seed=31)
        df['weight_num'] = tools.get_number(1, 2, weight_pattern=[90, 1], size=sample_size, seed=31)
        df['null'] = tools.get_number(1, 100, quantity=0, size=sample_size, seed=31)
        df['single cat'] = tools.get_category(['A'], quantity=0.6, size=sample_size, seed=31)
        df['weight_cat'] = tools.get_category(['A', 'B', 'C'], weight_pattern=[80, 1, 1], size=sample_size, seed=31)

        # Optional extra fields
        if extra:
            df['last_login'] = tools.get_datetime(start='01/01/2019', until='01/05/2019',
                                                  date_pattern=[1, 2, 3, 5, 9, 20], date_format='%m-%d-%y %H:%M',
                                                  size=sample_size)
            df['status'] = tools.get_category(selection=['Active', 'Closed', 'Suspended'], weight_pattern=[50, 5, 2],
                                              size=sample_size)
        # Optional extra noise
        if noise:
            for i in range(40):
                quantity = tools.get_number(0.005, 0.03, weight_pattern=[5, 2, 1, 0.5])[0]
                col = "noise_{}".format(i)
                df[col] = tools.get_number(0, 1, weight_pattern=[20, 1], quantity=quantity, size=sample_size)

        # save
        filename = os.path.join(os.environ['PWD'], 'work', 'data', '0_raw', 'synthetic.csv')
        builder.save_to_disk(df, filename=filename)
        return


if __name__ == '__main__':
    unittest.main()
