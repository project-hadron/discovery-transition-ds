import filecmp
import os
import pickle
import shutil
from contextlib import closing
import threading

import pandas as pd
import yaml

from ds_foundation.handlers.abstract_handlers import AbstractSourceHandler, ConnectorContract, AbstractPersistHandler

__author__ = 'Darryl Oatridge'


class PandasSourceHandler(AbstractSourceHandler):
    """ Pandas read only Source Handler. The format of the uri should be as a minimum:
                    uri = '/path/filename.ext'
        but can be a full url
                    uri = scheme://netloc/path/filename.ext
        or with query (name value pairs)
                    uri = scheme://netloc/path/filename.ext?file_type=csv&encoding=Latin1

        if the target file ext is not a recognised type use 'file_type' as a key with the extension type as the value.
        key/value pairs can be added as either part of the URI query string or as kwargs in the Connector Contract

    """

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Hander passing the connector_contract dictionary """
        super().__init__(connector_contract)
        self._modified = 0

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['parquet', 'csv', 'tsv', 'txt', 'json', 'pickle', 'xlsx']

    def load_canonical(self) -> pd.DataFrame:
        """ returns the canonical dataset based on the connector contract"""
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The PandasSource Connector Contract has not been set")
        _cc = self.connector_contract
        _, _ext = os.path.splitext(_cc.address)
        file_type = _cc.pop_key_value('file_type', _ext if len(_ext) > 0 else 'csv')
        if file_type.lower() in ['parquet', 'pq', 'pqt']:
            df = self._read_parquet(_cc.address, **_cc.key_values)
        elif file_type.lower() in ['csv', 'tsv', 'txt']:
            df = self._read_csv(_cc.address, **_cc.key_values)
        elif file_type.lower() in ['json']:
            df = self._read_json(_cc.address, **_cc.key_values)
        elif file_type.lower() in ['pkl ', 'pickle']:
            df = self._read_pickle(_cc.address, **_cc.key_values)
        elif file_type.lower() in ['xls', 'xlsx']:
            df = self._read_excel(_cc.address, **_cc.key_values)
        else:
            raise LookupError('The source format {} is not currently supported'.format(file_type))
        self._modified = os.stat(_cc.address)[8] if os.path.exists(_cc.address) else 0
        return df

    def get_modified(self) -> [int, float, str]:
        """ returns the modified state of the connector resource"""
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The PandasSource Connector Contract has not been set")
        return os.stat(self.connector_contract.address)[8] if os.path.exists(self.connector_contract.address) else 0

    @staticmethod
    def _read_csv(file, **kwargs) -> pd.DataFrame:
        """ Loads a csv file based on configuration parameters from the source reference

        :param contract_name: the name of the contract where the file properties are held
        """
        with threading.Lock():
            return pd.read_csv(file, **kwargs)

    @staticmethod
    def _read_pickle(file, **kwargs) -> pd.DataFrame:
        """ Loads a pickle file based on configuration parameters from the source reference

        :param contract_name: the name of the contract where the file properties are held
        """
        with threading.Lock():
            return pd.read_pickle(file, **kwargs)

    @staticmethod
    def _read_parquet(file, **kwargs) -> pd.DataFrame:
        """ Loads a parquet file based on configuration parameters from the source reference

        :param contract_name: the name of the contract where the file properties are held
        """
        with threading.Lock():
            return pd.read_parquet(file, **kwargs)

    @staticmethod
    def _read_excel(file, **kwargs) -> pd.DataFrame:
        """ Loads a excel file based on configuration parameters from the source reference

        :param contract_name: the name of the contract where the file properties are held
        """
        with threading.Lock():
            return pd.read_excel(file, **kwargs)

    @staticmethod
    def _read_json(file, **kwargs) -> pd.DataFrame:
        """ Loads a json file based on configuration parameters for the source reference

        :param contract_name: the name of the contract where the file properties are held
        """
        with threading.Lock():
            return pd.read_json(file, **kwargs)


class PandasPersistHandler(AbstractPersistHandler):
    """ Pandas read/write Persist Handler. The format of the uri should be as a minimum:
                    uri = '/path/filename.ext'
        but can be a full url
                    uri = scheme://netloc/path/filename.ext
        or with query (name value pairs)
                    uri = scheme://netloc/path/filename.ext?file_type=csv&encoding=Latin1

        if the target file ext is not a recognised type use 'file_type' as a key with the extension type as the value.
        key/value pairs can be added as either part of the URI query string or as kwargs in the Connector Contract

    """

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Handler passing the connector_contract dictionary """
        super().__init__(connector_contract)
        self._modified = 0

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['pickle', 'yaml']

    def get_modified(self) -> [int, float, str]:
        """ returns True if the modified state of the connector resource has changed"""
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        return os.stat(self.connector_contract.address)[8] if os.path.exists(self.connector_contract.address) else 0

    def exists(self) -> bool:
        """ Returns True is the file exists """
        if os.path.exists(self.connector_contract.address):
            return True
        return False

    def load_canonical(self) -> [pd.DataFrame, dict]:
        """ returns either the canonical dataset or the Yaml configuration dictionary"""
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError('The PandasHandler ConnectorContract has not been set')
        _cc = self.connector_contract
        _, _ext = os.path.splitext(_cc.address)
        file_type = _cc.pop_key_value('file_type', _ext if len(_ext) > 0 else 'yaml')
        if file_type.lower() in ['pkl', 'pickle']:
            rtn_data = self._pickle_load(path_file=_cc.address, **_cc.key_values)
        elif file_type.lower() in ['yml', 'yaml']:
            rtn_data = self._yaml_load(path_file=_cc.address)
        elif file_type.lower() in ['pq', 'parquet']:
            engine = _cc.pop_key_value('engine', None)
            columns = _cc.pop_key_value('columns', None)
            rtn_data = self._parquet_load(path_file=_cc.address, engine=engine, columns=columns, **_cc.key_values)
        else:
            raise ValueError("PandasPersistHandler only supports 'pickle', 'parquet' and 'yaml' source type,"
                             " '{}' found. Set 'file_type' as a kwarg to specify file type".format(file_type))
        self._modified = os.stat(_cc.address)[8] if os.path.exists(_cc.address) else 0
        return rtn_data

    def persist_canonical(self, canonical: [pd.DataFrame, dict]) -> bool:
        """ persists either the canonical dataset or the YAML contract dictionary"""
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _cc = self.connector_contract
        _, _ext = os.path.splitext(_cc.address)
        file_type = _cc.pop_key_value('file_type', _ext if len(_ext) > 0 else 'yaml')
        # parquet
        if file_type.lower() in ['pq', 'pqt', 'parquet']:
            engine = _cc.pop_key_value('engine', None)
            compression = _cc.pop_key_value('compression', None)
            index = _cc.pop_key_value('index', None)
            partition_cols = _cc.pop_key_value('partition_cols', None)
            self._parquet_dump(df=canonical, path_file=_cc.address, engine=engine, compression=compression, index=index,
                               partition_cols=partition_cols, **_cc.key_values)
            return True
        # pickle
        if file_type.lower() in ['pkl', 'pickle']:
            protocol = _cc.pop_key_value('protocol', pickle.HIGHEST_PROTOCOL)
            self._pickle_dump(df=canonical, path_file=_cc.address, protocol=protocol, **_cc.key_values)
            return True
        # yaml
        if file_type.lower() in ['yml', 'yaml']:
            default_flow_style = _cc.pop_key_value('default_flow_style', False)
            self._yaml_dump(data=canonical, path_file=_cc.address, default_flow_style=default_flow_style,
                            **_cc.key_values)
            return True
        # not found
        raise ValueError("PandasPersistHandler only supports 'pickle', 'parquet' and 'yaml' source type,"
                         " '{}' found. Set 'file_type' as a kwarg to specify file type".format(file_type))

    def remove_canonical(self) -> bool:
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        if os.path.exists(self.connector_contract.address):
            os.remove(self.connector_contract.address)
            return True
        return False

    def backup_canonical(self, max_backups=None):
        """ creates a backup of the current source contract resource"""
        if not isinstance(self.connector_contract, ConnectorContract):
            return
        _cc = self.connector_contract
        max_backups = max_backups if isinstance(max_backups, int) else 10
        # Check existence of previous versions
        name, _, ext = _cc.address.rpartition('.')
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
                    if os.path.isfile(abspath) and filecmp.cmp(abspath, _cc.address, shallow=False):
                        continue
                except OSError:
                    pass
            try:
                if not os.path.exists(backup):
                    shutil.copy(_cc.address, backup)
            except (OSError, IOError):
                pass
        return

    @staticmethod
    def _yaml_dump(data, path_file, default_flow_style=False, **kwargs) -> None:
        """

        :param data: the data to persist
        :param path_file: the name and path of the file
        :param default_flow_style: (optional) if to include the default YAML flow style
        """
        with threading.Lock():
            # make sure the dump is clean
            try:
                with closing(open(path_file, 'w')) as ymlfile:
                    yaml.safe_dump(data=data, stream=ymlfile, default_flow_style=default_flow_style, **kwargs)
            except IOError as e:
                raise IOError("The yaml file {} failed to open with: {}".format(path_file, e))
        # check the file was created
        return

    @staticmethod
    def _yaml_load(path_file) -> dict:
        """ loads the YAML file

        :param path_file: the name and path of the file
        :return: a dictionary
        """
        with threading.Lock():
            try:
                with closing(open(path_file, 'r')) as ymlfile:
                    rtn_dict = yaml.safe_load(ymlfile)
            except IOError as e:
                raise IOError("The yaml file {} failed to open with: {}".format(path_file, e))
            if not isinstance(rtn_dict, dict) or not rtn_dict:
                raise TypeError("The yaml file {} could not be loaded as a dict type".format(path_file))
            return rtn_dict

    @staticmethod
    def _pickle_dump(df: pd.DataFrame, path_file: str, protocol: int=None, **kwargs) -> None:
        """ dumps a pickle file

        :param df: the DataFrame to write
        :param path_file: the name and path of the file
        :param protocol: the pickle protocol. Default is pickle.DEFAULT_PROTOCOL
        """
        if protocol is None:
            protocol = pickle.HIGHEST_PROTOCOL
        with threading.Lock():
            with closing(open(path_file, 'wb')) as f:
                pickle.dump(df, f, protocol=protocol)

    @staticmethod
    def _pickle_load(path_file: str, **kwargs) -> pd.DataFrame:
        """ loads a pickle file

        :param path_file: the name and path of the file
        :return: a pandas DataFrame
        """
        with threading.Lock():
            with closing(open(path_file, 'rb')) as f:
                return pickle.load(f, )

    @staticmethod
    def _parquet_dump(df: pd.DataFrame, path_file: str, **kwargs) -> None:
        """ Write a DataFrame to the binary parquet format.

        :param df: the dataframe to write
        :param path_file: File path or Root Directory path
        :param engine: Parquet library to use. {‘auto’, ‘pyarrow’, ‘fastparquet’}
        :param compression: Name of the compression to use. Use None for no compression. {‘snappy’, ‘gzip’, ‘brotli’}
        :param index: If True, include the dataframe’s index(es) in the file output.
        :param partition_cols: Column names by which to partition the dataset.
        """
        with threading.Lock():
            df.to_parquet(path_file, **kwargs)

    @staticmethod
    def _parquet_load(path_file, engine: str=None, **kwargs) -> pd.DataFrame:
        """Load a parquet object from the file path, returning a DataFrame.

        :param path_file: file path
        :param engine: Parquet library to use. {‘auto’, ‘pyarrow’, ‘fastparquet’}
        :param columns: If not None, only these columns will be read from the file.
        :param kwargs: Additional arguments passed to the parquet library.
        :return: pandas Dataframe
        """
        with threading.Lock():
            return pd.read_parquet(path_file, engine=engine, **kwargs)
