import filecmp
import io
import os
import pickle
import shutil
from contextlib import closing
import threading

import boto3
import pandas as pd
import yaml

from ds_foundation.handlers.abstract_handlers import AbstractSourceHandler, ConnectorContract, AbstractPersistHandler

__author__ = 'Darryl Oatridge'


class PandasSourceHandler(AbstractSourceHandler):

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Hander passing the connector_contract dictionary """
        super().__init__(connector_contract)
        self._modified = 0

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['parquet', 'csv', 'tsv', 'txt', 'json', 'pickle', 'xlsx']

    def load_canonical(self) -> pd.DataFrame:
        """ returns the canoonical dataset based on the source contract"""
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The PandasSource Connector Contract has not been set")
        resource = self.connector_contract.resource
        connector_type = self.connector_contract.connector_type
        location = self.connector_contract.location
        _filepath = os.path.join(location, resource)
        if not os.path.exists(_filepath):
            raise FileNotFoundError("The file {} can't be found".format(_filepath))
        _kwargs = self.connector_contract.kwargs
        if _kwargs is None:
            _kwargs = {}
        if connector_type.lower() in ['parquet', 'pa']:
            df = self._read_parquet(_filepath, **_kwargs)
        elif connector_type.lower() in ['csv', 'tsv', 'txt']:
            df = self._read_csv(_filepath, **_kwargs)
        elif connector_type.lower() in ['json']:
            df = self._read_json(_filepath, **_kwargs)
        elif connector_type.lower() in ['p', 'pickle']:
            df = self._read_pickle(_filepath, **_kwargs)
        elif connector_type.lower() in ['xls', 'xlsx']:
            df = self._read_excel(_filepath, **_kwargs)
        else:
            raise LookupError('The source format {} is not currently supported'.format(connector_type))
        self._modified = os.stat(_filepath)[8] if os.path.exists(_filepath) else 0
        return df

    def get_modified(self) -> [int, float, str]:
        """ """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        resource = self.connector_contract.resource
        location = self.connector_contract.location
        _filepath = os.path.join(location, resource)
        return os.stat(_filepath)[8] if os.path.exists(_filepath) else 0

    @staticmethod
    def _read_csv(file, **kwargs) -> pd.DataFrame:
        """ Loads a csv file based on configuration parameters from the source reference

        :param contract_name: the name of the contract where the file properties are held
        """
        with threading.Lock():
            if os.path.exists(file):
                return pd.read_csv(file, **kwargs)
            raise FileNotFoundError("The file {} does not exist".format(file))

    @staticmethod
    def _read_pickle(file, **kwargs) -> pd.DataFrame:
        """ Loads a pickle file based on configuration parameters from the source reference

        :param contract_name: the name of the contract where the file properties are held
        """
        with threading.Lock():
            if os.path.exists(file):
                return pd.read_pickle(file, **kwargs)
            raise FileNotFoundError("The file {} does not exist".format(file))

    @staticmethod
    def _read_parquet(file, **kwargs) -> pd.DataFrame:
        """ Loads a parquet file based on configuration parameters from the source reference

        :param contract_name: the name of the contract where the file properties are held
        """
        with threading.Lock():
            if os.path.exists(file):
                return pd.read_parquet(file, **kwargs)
            raise FileNotFoundError("The file {} does not exist".format(file))

    @staticmethod
    def _read_excel(file, **kwargs) -> pd.DataFrame:
        """ Loads a excel file based on configuration parameters from the source reference

        :param contract_name: the name of the contract where the file properties are held
        """
        with threading.Lock():
            if os.path.exists(file):
                return pd.read_excel(file, **kwargs)
            raise FileNotFoundError("The file {} does not exist".format(file))

    @staticmethod
    def _read_json(file, **kwargs) -> pd.DataFrame:
        """ Loads a json file based on configuration parameters for the source reference

        :param contract_name: the name of the contract where the file properties are held
        """
        with threading.Lock():
            if os.path.exists(file):
                return pd.read_json(file, **kwargs)
            raise FileNotFoundError("The file {} does not exist".format(file))


class PandasPersistHandler(AbstractPersistHandler):

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Handler passing the connector_contract dictionary """
        super().__init__(connector_contract)
        self._modified = 0

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['pickle', 'yaml']

    def get_modified(self) -> [int, float, str]:
        """ """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        resource = self.connector_contract.resource
        connector_type = self.connector_contract.connector_type
        location = self.connector_contract.location
        _filepath = os.path.join(location, resource)
        return os.stat(_filepath)[8] if os.path.exists(_filepath) else 0

    def exists(self) -> bool:
        """ Returns True is the file exists """
        resource = self.connector_contract.resource
        location = self.connector_contract.location
        _filepath = os.path.join(location, resource)
        if os.path.exists(_filepath):
            return True
        return False

    def load_canonical(self) -> [pd.DataFrame, dict]:
        """ returns either the canonical dataset or the Yaml configuration dictionary"""
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError('The PandasHandler ConnectorContract has not been set')
        resource = self.connector_contract.resource
        connector_type = self.connector_contract.connector_type
        location = self.connector_contract.location
        _filepath = os.path.join(location, resource)
        if not os.path.exists(_filepath):
            raise FileNotFoundError("The file '{}' does not exist".format(_filepath))
        if connector_type.lower() in ['p', 'pickle']:
            rtn_data = self._pickle_load(path_file=_filepath)
        elif connector_type.lower() in ['y', 'yaml']:
            rtn_data = self._yaml_load(path_file=_filepath)
        else:
            raise ValueError("PandasPersistHandler only supports 'pickle' and 'yaml' source type,"
                             " '{}' found in Source Contract source-type".format(connector_type))
        self._modified = os.stat(_filepath)[8] if os.path.exists(_filepath) else 0
        return rtn_data

    def persist_canonical(self, canonical: [pd.DataFrame, dict]) -> bool:
        """ persists either the canonical dataset or the YAML contract dictionary"""
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        resource = self.connector_contract.resource
        connector_type = self.connector_contract.connector_type
        location = self.connector_contract.location
        _filepath = os.path.join(location, resource)
        if not os.path.exists(location):
            os.makedirs(location)
        _kwargs = self.connector_contract.kwargs
        # pickle
        if connector_type.lower() in ['p', 'pickle']:
            if isinstance(_kwargs, dict) and _kwargs.get('protocol') is not None:
                protocol = _kwargs.get('protocol')
            else:
                protocol = pickle.HIGHEST_PROTOCOL
            self._pickle_dump(df=canonical, path_file=_filepath, protocol=protocol)
            return True
        # yaml
        if connector_type.lower() in ['y', 'yaml']:
            if isinstance(_kwargs, dict) and _kwargs.get('default_flow_style') is not None:
                default_flow_style = _kwargs.get('default_flow_style')
            else:
                default_flow_style = False
            self._yaml_dump(data=canonical, path_file=_filepath, default_flow_style=default_flow_style)
            return True
        # not found
        raise ValueError("PandasPersistHandler only supports 'pickle' and 'yaml' source type,"
                         " '{}' found in Source Contract source-type".format(connector_type))

    def remove_canonical(self) -> bool:
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        resource = self.connector_contract.resource
        location = self.connector_contract.location
        _filepath = os.path.join(location, resource)
        if os.path.exists(_filepath):
            os.remove(_filepath)
            return True
        return False

    def backup_canonical(self, max_backups=None):
        """ creates a backup of the current source contract resource"""
        if not isinstance(self.connector_contract, ConnectorContract):
            return
        max_backups = max_backups if isinstance(max_backups, int) else 10
        resource = self.connector_contract.resource
        location = self.connector_contract.location
        _filepath = os.path.join(location, resource)
        # Check existence of previous versions
        name, _, ext = _filepath.rpartition('.')
        for index in range(max_backups):
            backup = '%s_%2.2d.%s' % (name, index, ext)
            if index > 0:
                # No need to backup if file and last version
                # are identical
                old_backup = '%s_%2.2d.%s' % (name, index - 1, ext)
                if not os.path.exists(old_backup):
                    break
                abspath = os.path.abspath(old_backup)

                try:
                    if os.path.isfile(abspath) and filecmp.cmp(abspath, _filepath, shallow=False):
                        continue
                except OSError:
                    pass
            try:
                if not os.path.exists(backup):
                    shutil.copy(_filepath, backup)
            except (OSError, IOError):
                pass
        return

    @staticmethod
    def _yaml_dump(data, path_file, default_flow_style=False) -> None:
        """

        :param data: the data to persist
        :param path_file: the name and path of the file
        :param default_flow_style: (optional) if to include the default YAML flow style
        """
        _path, _file = os.path.split(path_file)
        if _path is not None and len(_path) > 0 and isinstance(_path, str) and not os.path.exists(_path):
            os.makedirs(_path, exist_ok=True)
        _path_file = path_file
        with threading.Lock():
            # make sure the dump is clean
            try:
                with closing(open(_path_file, 'w')) as ymlfile:
                    yaml.safe_dump(data=data, stream=ymlfile, default_flow_style=default_flow_style)
            except IOError as e:
                raise IOError("The yaml file {} failed to open with: {}".format(path_file, e))
        # check the file was created
        if not os.path.exists(_path_file):
            raise IOError("Failed to save yaml file {}. Check the disk is writable".format(path_file))
        return

    @staticmethod
    def _yaml_load(path_file) -> dict:
        """ loads the YAML file

        :param path_file: the name and path of the file
        :return: a dictionary
        """
        _path_file = path_file
        if not os.path.exists(_path_file):
            raise FileNotFoundError("The yaml file {} does not exist".format(path_file))
        with threading.Lock():
            try:
                with closing(open(_path_file, 'r')) as ymlfile:
                    rtn_dict = yaml.safe_load(ymlfile)
            except IOError as e:
                raise IOError("The yaml file {} failed to open with: {}".format(path_file, e))
            if not isinstance(rtn_dict, dict) or not rtn_dict:
                raise TypeError("The yaml file {} could not be loaded as a dict type".format(path_file))
            return rtn_dict

    @staticmethod
    def _pickle_dump(df: pd.DataFrame, path_file: str, protocol: int=None) -> None:
        """ dumps a pickle file

        :param df: the Dataframe to write
        :param path_file: the name and path of the file
        :param protocol: the pickle protocol. Default is pickle.DEFAULT_PROTOCOL
        """
        _path, _file = os.path.split(path_file)
        if _path is not None and len(_path) > 0 and isinstance(_path, str) and not os.path.exists(_path):
            os.makedirs(_path, exist_ok=True)
        _path_file = path_file
        if protocol is None:
            protocol = pickle.HIGHEST_PROTOCOL
        with threading.Lock():
            with closing(open(_path_file, 'wb')) as f:
                pickle.dump(df, f, protocol=protocol)

    @staticmethod
    def _pickle_load(path_file: str) -> pd.DataFrame:
        """ loads a pickle file

        :param path_file: the name and path of the file
        :return: a pandas DataFrame
        """
        _path_file = path_file
        if not os.path.exists(_path_file):
            raise FileNotFoundError("The pickle file {} does not exist".format(path_file))
        with threading.Lock():
            with closing(open(_path_file, 'rb')) as f:
                return pickle.load(f)


class FileHandlers(object):

    @staticmethod
    def _pickle_dump(df: pd.DataFrame, file: str, protocol: int=None) -> None:
        """ dumps a pickle file

        :param df: the Dataframe to write
        :param file: the name and path of the file
        :param protocol: the pickle protocol. Default is pickle.DEFAULT_PROTOCOL
        """
        _path = file
        if protocol is None:
            protocol = pickle.HIGHEST_PROTOCOL
        with closing(open(_path, 'wb')) as f:
            pickle.dump(df, f, protocol=protocol)

    @staticmethod
    def _pickle_load(file: str) -> pd.DataFrame:
        """ loads a pickle file

        :param file: the name and path of the file
        :return:
        """
        _path = file
        with closing(open(_path, 'rb')) as f:
            return pickle.load(f)

    @staticmethod
    def _aws_s3_dump(file_name: str, bucket: str, key: str) -> None:
        """ dumps a file to s3 bucket

        :param file_name: the name of the file to dump
        :param bucket: the name of the bucket in s#
        :param key: the key
        """
        s3 = boto3.client('s3')
        bucket = s3.Bucket(bucket)
        with open(file_name, 'rb') as f:
            object_data = f.read()
            s3.put_object(Body=object_data, Bucket=bucket, Key=key)
        return

    @staticmethod
    def _aws_s3_load(file_name: str, bucket: str, key: str) -> None:
        """ loads a file from s3 bucket

        :param file_name: the name of the file to dump
        :param bucket: the name of the bucket in s#
        :param key: the key
        """
        buffer = io.BytesIO()
        s3 = boto3.resource('s3')
        s3object = s3.Object(bucket, key)
        s3object.download_fileobj(buffer)
        with open(file_name, 'w+') as f:
            f.write(s3object)
        return

    @staticmethod
    def _parquet_dump(df: pd.DataFrame, file: str, engine: str=None, compression: str=None, index: bool=None,
                      partition_cols: list=None, **kwargs) -> None:
        """ Write a DataFrame to the binary parquet format.

        :param df: the dataframe to write
        :param file: File path or Root Directory path
        :param engine: Parquet library to use. {‘auto’, ‘pyarrow’, ‘fastparquet’}
        :param compression: Name of the compression to use. Use None for no compression. {‘snappy’, ‘gzip’, ‘brotli’}
        :param index: If True, include the dataframe’s index(es) in the file output.
        :param partition_cols: Column names by which to partition the dataset.
        :param kwargs: Additional arguments passed to the parquet library.
        """
        df.to_parquet(file, engine=engine, compression=compression, index=index, partition_cols=partition_cols,
                      **kwargs)

    @staticmethod
    def _parquet_load(file, engine: str=None, columns: list=None, **kwargs) -> pd.DataFrame:
        """Load a parquet object from the file path, returning a DataFrame.

        :param file: file path
        :param engine: Parquet library to use. {‘auto’, ‘pyarrow’, ‘fastparquet’}
        :param columns: If not None, only these columns will be read from the file.
        :param kwargs: Additional arguments passed to the parquet library.
        :return: pandas Dataframe
        """
        return pd.read_parquet(file, engine=engine, columns=columns, **kwargs)

    @staticmethod
    def _yaml_dump(data, file) -> None:
        _path = file
        try:
            with closing(open(_path, 'w')) as ymlfile:
                yaml.safe_dump(data, ymlfile, default_flow_style=False)
        except IOError as e:
            raise IOError("The configuration file {} failed to open with: {}".format(_path, e))

    @staticmethod
    def _yaml_load(file) -> dict:
        _path = file
        if os.path.exists(_path) and os.path.isfile(_path):
            try:
                with closing(open(_path, 'r')) as ymlfile:
                    rtn_dict = yaml.safe_load(ymlfile)
            except IOError as e:
                raise IOError("The configuration file {} failed to open with: {}".format(_path, e))
            if not isinstance(rtn_dict, dict) or not rtn_dict:
                raise TypeError("The configuration file {} could not be loaded as a dict type".format(_path))
            return rtn_dict
        else:
            raise FileNotFoundError("The yaml file {} does not exist".format(_path))
