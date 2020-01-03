import os
from contextlib import closing
import threading

import pandas as pd
import yaml
try:
    import cPickel as pickle
except ImportError:
    import pickle

from ds_foundation.handlers.abstract_handlers import AbstractSourceHandler, ConnectorContract, AbstractPersistHandler

__author__ = 'Darryl Oatridge'


class PandasSourceHandler(AbstractSourceHandler):
    """ Pandas read only Source Handler. The format of the uri should be as a minimum:
                    uri = '[/<path>/]<filename.ext>'
        but can be a full url
                    uri = <scheme>://<netloc>/[<path>/]<filename.ext>

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - read_kw: (optional) value pair dictionary of parameters to pass to the pandas read methods
    """

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Hander passing the connector_contract dictionary """
        super().__init__(connector_contract)
        self._modified = 0

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['parquet', 'csv', 'tsv', 'txt', 'json', 'pickle', 'xlsx']

    def load_canonical(self) -> pd.DataFrame:
        """ returns the canonical dataset based on the connector contract

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - read_kw: (optional) value pair dictionary of parameters to pass to the pandas read methods
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The PandasSource Connector Contract has not been set")
        _cc = self.connector_contract
        _key_values = _cc.kwargs.get('read_kw', {})
        _, _ext = os.path.splitext(_cc.address)
        file_type = _cc.kwargs.get('file_type', _ext if len(_ext) > 0 else 'csv')
        with threading.Lock():
            if file_type.lower() in ['parquet', 'pq', 'pqt']:
                df = pd.read_parquet(_cc.address, **_key_values)
            elif file_type.lower() in ['csv', 'tsv', 'txt']:
                df = pd.read_csv(_cc.address, **_key_values)
            elif file_type.lower() in ['json']:
                df = pd.read_json(_cc.address, **_key_values)
            elif file_type.lower() in ['pkl ', 'pickle']:
                df = pd.read_pickle(_cc.address, **_key_values)
            elif file_type.lower() in ['xls', 'xlsx']:
                df = pd.read_excel(_cc.address, **_key_values)
            else:
                raise LookupError('The source format {} is not currently supported'.format(file_type))
        self._modified = os.stat(_cc.address)[8] if os.path.exists(_cc.address) else 0
        return df

    def get_modified(self) -> [int, float, str]:
        """ returns the modified state of the connector resource"""
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The PandasSource Connector Contract has not been set")
        return os.stat(self.connector_contract.address)[8] if os.path.exists(self.connector_contract.address) else 0


class PandasPersistHandler(AbstractPersistHandler):
    """ Pandas read/write Persist Handler. The format of the uri should be as a minimum:
                    uri = '[/<path>/]<filename.ext>'
        but can be a full url
                    uri = <scheme>://<netloc>/[<path>/]<filename.ext>

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - read_kw: (optional) value pair dictionary of parameters to pass to the pandas read methods
            - write_kw: (optional) value pair dictionary of parameters to pass to the pandas read methods
    """

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Handler passing the connector_contract dictionary """
        super().__init__(connector_contract)
        self._modified = 0

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['pickle', 'yaml', 'parquet', 'json', 'xlsx']

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

    def load_canonical(self) -> pd.DataFrame:
        """ returns the canonical dataset based on the connector contract

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - read_kw: (optional) value pair dictionary of parameters to pass to the pandas read methods
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The PandasSource Connector Contract has not been set")
        _cc = self.connector_contract
        _key_values = _cc.kwargs.get('read_kw', {})
        _, _ext = os.path.splitext(_cc.address)
        file_type = _cc.kwargs.get('file_type', _ext if len(_ext) > 0 else 'csv')
        with threading.Lock():
            if file_type.lower() in ['parquet', 'pq', 'pqt']:
                df = pd.read_parquet(_cc.address, **_key_values)
            elif file_type.lower() in ['csv', 'tsv', 'txt']:
                df = pd.read_csv(_cc.address, **_key_values)
            elif file_type.lower() in ['json']:
                df = pd.read_json(_cc.address, **_key_values)
            elif file_type.lower() in ['pkl ', 'pickle']:
                df = pd.read_pickle(_cc.address, **_key_values)
            elif file_type.lower() in ['xls', 'xlsx']:
                df = pd.read_excel(_cc.address, **_key_values)
            else:
                raise LookupError('The file format {} is not currently supported for read'.format(file_type))
        self._modified = os.stat(_cc.address)[8] if os.path.exists(_cc.address) else 0
        return df

    def persist_canonical(self, canonical: pd.DataFrame) -> bool:
        """ persists the canonical dataset

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - write_kw: (optional) value pair dictionary of parameters to pass to the pandas read methods
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _uri = self.connector_contract.address
        return self.backup_canonical(uri=_uri, canonical=canonical)

    def backup_canonical(self, canonical: pd.DataFrame, uri: str) -> bool:
        """ creates a backup of the canonical to an alternative URI

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - write_kw: (optional) value pair dictionary of parameters to pass to the pandas read methods
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _cc = self.connector_contract
        _address = _cc.parse_address(uri=uri)
        _key_values = _cc.kwargs.get('write_kw', {})
        _, _ext = os.path.splitext(_address)
        file_type = _key_values.pop('file_type', _ext if len(_ext) > 0 else 'yaml')
        # parquet
        if file_type.lower() in ['pq', 'pqt', 'parquet']:
            with threading.Lock():
                canonical.to_parquet(_address, **_key_values)
            return True
        # pickle
        if file_type.lower() in ['pkl', 'pickle']:
            _compression = _key_values.pop('compression', None)
            _protocol = _key_values.pop('protocol', pickle.HIGHEST_PROTOCOL)
            with threading.Lock():
                canonical.to_pickle(path=_address, compression=_compression, protocol=_protocol)
            return True
        if file_type.lower() in ['json']:
            with threading.Lock():
                canonical.to_json(_address, **_key_values)
            return True
        if file_type.lower() in ['csv', 'tsv', 'txt']:
            with threading.Lock():
                canonical.to_csv(_address, **_key_values)
            return True
        if file_type.lower() in ['pkl', 'pickle']:
            _compression = _key_values.pop('compression', None)
            _protocol = _key_values.pop('protocol', pickle.HIGHEST_PROTOCOL)
            with threading.Lock():
                canonical.to_pickle(path=_address, compression=_compression, protocol=_protocol)
            return True
        # yaml
        if file_type.lower() in ['yml', 'yaml']:
            default_flow_style = _key_values.pop('default_flow_style', False)
            self._yaml_dump(data=canonical, path_file=_address, default_flow_style=default_flow_style,
                            **_key_values)
            return True
        # not found
        raise LookupError('The file format {} is not currently supported for write'.format(file_type))

    def remove_canonical(self) -> bool:
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _cc = self.connector_contract
        if len(_cc.schema) > 0:
            raise NotImplemented("Remove Canonical does not support {} schema based URIs".format(_cc.schema))
        if os.path.exists(_cc.address):
            os.remove(_cc.address)
            return True
        return False

    @staticmethod
    def _yaml_dump(data, path_file, default_flow_style=False, **kwargs) -> None:
        """ dump YAML file

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
