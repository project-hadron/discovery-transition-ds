import unittest
import os
import shutil
from datetime import datetime

from ds_behavioral import DataBuilderTools
from ds_foundation.handlers.abstract_handlers import ConnectorContract

from ds_connectors.handlers.aws_s3_handlers import AwsS3SourceHandler, AwsS3PersistHandler


class AwsS3HandlerTest(unittest.TestCase):

    AWS = 's3://aistac-discovery-persist/data/synthetic/'

    @classmethod
    def setUpClass(cls):
        os.environ['CONTRACT_PATH'] = os.path.join(os.environ['PWD'], 'work')
        try:
            shutil.copytree('../data', os.path.join(os.environ['PWD'], 'work'))
        except:
            pass

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(os.path.join(os.environ['PWD'], 'work'))
        except:
            pass

    def setUp(self):
        try:
            shutil.rmtree(os.path.join(os.environ['PWD'], 'work/data/2_transition'))
        except:
            pass
        os.mkdir(os.path.join(os.environ['PWD'], 'work/data/2_transition'))

    def tearDown(self):
        pass

    def test_runs(self):
        """Basic smoke test"""
        AwsS3SourceHandler(ConnectorContract('s3://aistac-discovery-persist/data/synthetic/synthetic_customer.csv',
                                             '', '', file_type='csv'))
        AwsS3PersistHandler(ConnectorContract('s3://aistac-discovery-persist/data/synthetic/synthetic_customer..pkl',
                                              '', '', file_type='pickle'))
    def test_source_handler(self):
        uri = 's3://aistac-discovery-persist/data/synthetic/synthetic_customer.csv'
        handler = AwsS3SourceHandler(ConnectorContract(uri, '', ''))
        self.assertTrue(isinstance(handler.supported_types(), list) and len(handler.supported_types()) > 0)
        df = handler.load_canonical()
        self.assertEqual((500, 16), df.shape)
        isinstance(handler.get_modified(), datetime)
        uri = 's3://aistac-discovery-persist/data/synthetic/synthetic_agent.dat'
        handler = AwsS3SourceHandler(ConnectorContract(uri, '', '', file_type='csv'))
        df = handler.load_canonical()
        self.assertEqual((10000, 49), df.shape)

    def test_persist_handler(self):
        handler = AwsS3PersistHandler(ConnectorContract('s3://aistac-discovery-persist/data/synthetic/example01.pkl', '', '', file_type='pickle'))
        self.assertTrue(isinstance(handler.supported_types(), list) and len(handler.supported_types()) > 0)
        # create the file and persist
        df = DataBuilderTools.get_profiles(include_id=True, size=1000)
        self.assertTrue(handler.persist_canonical(df))
        modified = handler.get_modified()
        df = handler.load_canonical()
        self.assertEqual((1000, 5), df.shape)
        # write again to check modified
        df['value'] = [0] * df.shape[0]
        self.assertTrue(handler.persist_canonical(df))
        df = handler.load_canonical()
        self.assertEqual((1000, 6), df.shape)
        self.assertGreater(handler.get_modified(), modified)
        handler.remove_canonical()
        self.assertFalse(handler.exists())

    def test_persist_backup(self):
        handler = AwsS3PersistHandler(ConnectorContract('s3://aistac-discovery-persist/data/synthetic/example01.json', '', ''))
        df = DataBuilderTools.get_profiles(include_id=True, size=1000)
        self.assertTrue(handler.persist_canonical(df))
        df = handler.load_canonical()
        self.assertEqual((1000, 5), df.shape)
        handler.remove_canonical()
        # Backup
        uri = 's3://aistac-discovery-persist/data/synthetic/example01.pq.bak?file_type=parquet'
        self.assertFalse(os.path.exists(uri))
        handler.backup_canonical(canonical=df, uri=uri)

    def test_json(self):
        df = DataBuilderTools.get_profiles(include_id=True, size=1000)
        handler = AwsS3PersistHandler(ConnectorContract('s3://aistac-discovery-persist/data/synthetic/handler_test.json',
                                                        '', '', file_type='json'))
        handler.persist_canonical(df)
        self.assertTrue(handler.exists())
        result = handler.load_canonical()
        self.assertEqual(df.shape, result.shape)
        self.assertCountEqual(df.columns, result.columns)
        handler.remove_canonical()
        self.assertFalse(handler.exists())

    def test_csv(self):
        df = DataBuilderTools.get_profiles(include_id=True, size=1000)
        handler = AwsS3PersistHandler(ConnectorContract('s3://aistac-discovery-persist/data/synthetic/handler_test.csv',
                                                        '', '', file_type='csv'))
        handler.persist_canonical(df)
        self.assertTrue(handler.exists())
        result = handler.load_canonical()
        self.assertEqual(df.shape, result.shape)
        self.assertCountEqual(df.columns, result.columns)
        handler.remove_canonical()
        self.assertFalse(handler.exists())

    def test_pickle(self):
        df = DataBuilderTools.get_profiles(include_id=True, size=1000)
        handler = AwsS3PersistHandler(ConnectorContract('s3://aistac-discovery-persist/data/synthetic/handler_test.pkl',
                                                        '', '', file_type='pickle'))
        handler.persist_canonical(df)
        self.assertTrue(handler.exists())
        result = handler.load_canonical()
        self.assertEqual(df.shape, result.shape)
        self.assertCountEqual(df.columns, result.columns)
        handler.remove_canonical()
        self.assertFalse(handler.exists())

    def test_parquet(self):
        df = DataBuilderTools.get_profiles(include_id=True, size=1000)
        handler = AwsS3PersistHandler(ConnectorContract('s3://aistac-discovery-persist/data/synthetic/handler_test.pq',
                                                        '', '', file_type='parquet'))
        handler.persist_canonical(df)
        self.assertTrue(handler.exists())
        result = handler.load_canonical()
        self.assertEqual(df.shape, result.shape)
        self.assertCountEqual(df.columns, result.columns)
        handler.remove_canonical()
        self.assertFalse(handler.exists())


if __name__ == '__main__':
    unittest.main()
