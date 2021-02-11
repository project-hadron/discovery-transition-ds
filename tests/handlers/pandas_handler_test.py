import os
import shutil
import unittest
import pandas as pd

from aistac.handlers.abstract_handlers import ConnectorContract
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel

from ds_discovery.handlers.pandas_handlers import PandasSourceHandler, PandasPersistHandler
from ds_discovery import SyntheticBuilder


class PandasHandlerTest(unittest.TestCase):

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
        scratch: SyntheticIntentModel = SyntheticBuilder.scratch_pad()
        df = pd.DataFrame(index=range(1000))
        df = scratch.model_noise(df, num_columns=10)
        file = os.path.join(os.environ['HADRON_DEFAULT_PATH'], 'example01.csv')
        df.to_csv(file)
        file = os.path.join(os.environ['HADRON_DEFAULT_PATH'], 'example01.dat')
        df.to_csv(file)

    def tearDown(self):
        try:
            shutil.rmtree('work')
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        PandasSourceHandler(ConnectorContract('work/data/0_raw/example01.csv', '', '', file_type='csv'))
        PandasPersistHandler(ConnectorContract('work/data/2_transition/example01.pkl', '', '', file_type='pickle'))

    def test_source_handler(self):
        file = os.path.join(os.environ['HADRON_DEFAULT_PATH'], 'example01.csv')
        handler = PandasSourceHandler(ConnectorContract(file, '', ''))
        self.assertTrue(isinstance(handler.supported_types(), list) and len(handler.supported_types()) > 0)
        df = handler.load_canonical()
        self.assertEqual((1000,11), df.shape)
        self.assertTrue(handler.has_changed())
        file = os.path.join(os.environ['HADRON_DEFAULT_PATH'], 'example01.dat')
        handler = PandasSourceHandler(ConnectorContract(file, '', '', file_type='csv'))
        df = handler.load_canonical()
        self.assertEqual((1000,11), df.shape)

    def test_persist_handler(self):
        handler = PandasPersistHandler(ConnectorContract('work/data/2_transition/example01.pkl', '', '', file_type='pickle'))
        self.assertTrue(isinstance(handler.supported_types(), list) and len(handler.supported_types()) > 0)
        self.assertFalse(handler.has_changed())
        self.assertFalse(handler.exists())
        # create the file and persist
        df = SyntheticBuilder.scratch_pad().model_noise(pd.DataFrame(index=range(1000)), num_columns=10)
        self.assertTrue(handler.persist_canonical(df))
        self.assertTrue(handler.exists())
        self.assertTrue(handler.has_changed())
        df = handler.load_canonical()
        self.assertEqual((1000, 10), df.shape)
        # write again to check modified
        df['value'] = [0] * df.shape[0]
        self.assertTrue(handler.persist_canonical(df))
        df = handler.load_canonical()
        self.assertEqual((1000, 11), df.shape)
        self.assertTrue(handler.has_changed())
        df = df.drop('value', axis='columns')
        self.assertTrue(handler.persist_canonical(df))
        self.assertEqual((1000, 10), df.shape)
        self.assertTrue(handler.has_changed())

    def test_persist_backup(self):
        handler = PandasPersistHandler(ConnectorContract('work/data/2_transition/example01.json', '', ''))
        df = SyntheticBuilder.scratch_pad().model_noise(pd.DataFrame(index=range(1000)), num_columns=10)
        self.assertTrue(handler.persist_canonical(df))
        df = pd.DataFrame(data=handler.load_canonical())
        self.assertEqual((1000, 10), df.shape)
        handler.remove_canonical()
        self.assertFalse(handler.exists())
        # Backup
        uri = 'work/data/2_transition/example01.pq.bak?file_type=parquet'
        self.assertFalse(os.path.exists(uri))
        handler.backup_canonical(canonical=df, uri=uri)
        self.assertTrue(os.path.exists(ConnectorContract.parse_address(uri)))

    def test_json(self):
        df = SyntheticBuilder.scratch_pad().model_noise(pd.DataFrame(index=range(1000)), num_columns=10)
        handler = PandasPersistHandler(ConnectorContract('work/data/2_transition/handler_test.json',
                                                         '', '', file_type='json'))
        handler.persist_canonical(df)
        self.assertTrue(handler.exists())
        result = pd.DataFrame(data=handler.load_canonical())
        self.assertEqual(df.shape, result.shape)
        self.assertCountEqual(df.columns, result.columns)
        handler.remove_canonical()
        self.assertFalse(handler.exists())

    def test_csv(self):
        df = SyntheticBuilder.scratch_pad().model_noise(pd.DataFrame(index=range(1000)), num_columns=10)
        handler = PandasPersistHandler(ConnectorContract('work/data/2_transition/handler_test.csv',
                                                        '', '', file_type='csv'))
        handler.persist_canonical(df)
        self.assertTrue(handler.exists())
        result = handler.load_canonical()
        self.assertEqual(df.shape, result.shape)
        self.assertCountEqual(df.columns, result.columns)
        handler.remove_canonical()
        self.assertFalse(handler.exists())

    def test_pickle(self):
        df = SyntheticBuilder.scratch_pad().model_noise(pd.DataFrame(index=range(1000)), num_columns=10)
        handler = PandasPersistHandler(ConnectorContract('work/data/2_transition/handler_test.pkl',
                                                        '', '', file_type='pickle'))
        handler.persist_canonical(df)
        self.assertTrue(handler.exists())
        result = handler.load_canonical()
        self.assertEqual(df.shape, result.shape)
        self.assertCountEqual(df.columns, result.columns)
        handler.remove_canonical()
        self.assertFalse(handler.exists())

    def test_parquet(self):
        df = SyntheticBuilder.scratch_pad().model_noise(pd.DataFrame(index=range(1000)), num_columns=10)
        handler = PandasPersistHandler(ConnectorContract('work/data/2_transition/handler_test.pq',
                                                        '', '', file_type='parquet'))
        handler.persist_canonical(df)
        self.assertTrue(handler.exists())
        result = handler.load_canonical()
        self.assertEqual(df.shape, result.shape)
        self.assertCountEqual(df.columns, result.columns)
        handler.remove_canonical()
        self.assertFalse(handler.exists())
        
    def test_change_flags(self):
        """Basic smoke test"""
        file = os.path.join(os.environ['HADRON_DEFAULT_PATH'], 'example01.csv')
        open(file, 'a').close()
        cc = ConnectorContract(uri=file, module_name='', handler='')
        source = PandasSourceHandler(cc)
        self.assertTrue(source.has_changed())
        _ = source.load_canonical()
        self.assertFalse(source.has_changed())
        source.reset_changed(True)
        self.assertTrue(source.has_changed())
        source.reset_changed()
        self.assertFalse(source.has_changed())
        # touch the file
        os.remove(file)
        with open(file, 'a'):
            os.utime(file, None)
        self.assertTrue(source.has_changed())

if __name__ == '__main__':
    unittest.main()
