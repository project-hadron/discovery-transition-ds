import os
from contextlib import closing
import threading
import pandas as pd
import requests
import yaml
import pickle
import json


from aistac.handlers.abstract_handlers import AbstractSourceHandler, ConnectorContract, AbstractPersistHandler

__author__ = 'Darryl Oatridge'


class PandasSourceHandler(AbstractSourceHandler):
    """ Pandas read only Source Handler. The format of the uri should be as a minimum:
                    uri = '[/<path>/]<filename.ext>'
        but can be a full url
                    uri = <scheme>://<netloc>/[<path>/]<filename.ext>
    """

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Hander passing the connector_contract dictionary """
        super().__init__(connector_contract)

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['parquet', 'csv', 'tsv', 'txt', 'json', 'pickle', 'xlsx', 'yaml']

    def load_canonical(self, **kwargs) -> [pd.DataFrame, dict]:
        """ returns the canonical dataset based on the connector contract. This method utilises the pandas
        'pd.read_' methods and directly passes the kwargs to these methods.

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - read_params: (optional) value pair dict of parameters to pass to the read methods. Underlying
                           read methods the parameters are passed to are all pandas 'read_*', e.g. pd.read_csv
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The Pandas Connector Contract has not been set")
        _cc = self.connector_contract
        load_params = _cc.kwargs
        load_params.update(_cc.query)  # Update kwargs with those in the uri query
        load_params.update(kwargs)     # Update with any passed though the call
        _, _, _ext = _cc.address.rpartition('.')
        file_type = load_params.pop('file_type', _ext if len(_ext) > 0 else 'csv')
        with threading.Lock():
            if file_type.lower() in ['parquet', 'pq']:
                rtn_data = pd.read_parquet(_cc.address, **load_params)
            elif file_type.lower() in ['zip', 'csv', 'tsv', 'txt']:
                rtn_data = pd.read_csv(_cc.address, **load_params)
            elif file_type.lower() in ['json']:
                rtn_data = self._json_load(path_file=_cc.address, **load_params)
            elif file_type.lower() in ['xls', 'xlsx']:
                rtn_data = pd.read_excel(_cc.address, **load_params)
            elif file_type.lower() in ['pkl ', 'pickle']:
                rtn_data = self._pickle_load(path_file=_cc.address, **load_params)
            elif file_type.lower() in ['yml', 'yaml']:
                rtn_data = self._yaml_load(path_file=_cc.address, **load_params)
            else:
                raise LookupError('The source format {} is not currently supported'.format(file_type))
        return rtn_data

    def exists(self) -> bool:
        """ Returns True is the file exists """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The Pandas Connector Contract has not been set")
        _cc = self.connector_contract
        if _cc.schema.startswith('http'):
            return requests.get(_cc.address).status_code == 200
        if os.path.exists(_cc.address):
            return True
        return False

    def get_modified(self) -> [int, float, str]:
        """ returns the modified state of the connector resource"""
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The Pandas Connector Contract has not been set")
        _cc = self.connector_contract
        if _cc.schema.startswith('http'):
            return requests.head(_cc.address).headers.get('last-modified', 0)
        return os.path.getmtime(_cc.address) if os.path.exists(_cc.address) else 0

    @staticmethod
    def _yaml_load(path_file, **kwargs) -> dict:
        """ loads the YAML file

        :param path_file: the name and path of the file
        :return: a dictionary
        """
        encoding = kwargs.pop('encoding', 'utf-8')
        with threading.Lock():
            try:
                with closing(open(path_file, mode='r', encoding=encoding)) as ymlfile:
                    rtn_dict = yaml.safe_load(ymlfile)
            except IOError as e:
                raise IOError("The yaml file {} failed to open with: {}".format(path_file, e))
            if not isinstance(rtn_dict, dict) or not rtn_dict:
                raise TypeError("The yaml file {} could not be loaded as a dict type".format(path_file))
            return rtn_dict

    @staticmethod
    def _pickle_load(path_file: str, **kwargs) -> [dict, pd.DataFrame]:
        """ loads a pickle file """
        fix_imports = kwargs.pop('fix_imports', True)
        encoding = kwargs.pop('encoding', 'ASCII')
        errors = kwargs.pop('errors', 'strict')
        with threading.Lock():
            with closing(open(path_file, mode='rb')) as f:
                return pickle.load(f, fix_imports=fix_imports, encoding=encoding, errors=errors)

    @staticmethod
    def _json_load(path_file: str, **kwargs) -> [dict, pd.DataFrame]:
        """ loads a pickle file """
        with threading.Lock():
            with closing(open(path_file, mode='r')) as f:
                return json.load(f, **kwargs)


class PandasPersistHandler(PandasSourceHandler, AbstractPersistHandler):
    """ Pandas read/write Persist Handler. The format of the uri should be as a minimum:
                    uri = '[/<path>/]<filename.ext>'
        but can be a full url
                    uri = <scheme>://<netloc>/[<path>/]<filename.ext>
    """

    def persist_canonical(self, canonical: pd.DataFrame, **kwargs) -> bool:
        """ persists the canonical dataset

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _uri = self.connector_contract.uri
        return self.backup_canonical(uri=_uri, canonical=canonical, **kwargs)

    def backup_canonical(self, canonical: pd.DataFrame, uri: str, **kwargs) -> bool:
        """ creates a backup of the canonical to an alternative URI

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - write_params (optional) a dictionary of additional write parameters directly passed to 'write_' methods
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _cc = self.connector_contract
        _address = _cc.parse_address(uri=uri)
        persist_params = kwargs if isinstance(kwargs, dict) else _cc.kwargs
        persist_params.update(_cc.parse_query(uri=uri))
        _, _, _ext = _address.rpartition('.')
        if not self.connector_contract.schema.startswith('http'):
            _path, _ = os.path.split(_address)
            if not os.path.exists(_path):
                os.makedirs(_path)
        file_type = persist_params.pop('file_type', _ext if len(_ext) > 0 else 'pkl')
        write_params = persist_params.pop('write_params', {})
        # parquet
        if file_type.lower() in ['pq', 'pqt', 'parquet']:
            with threading.Lock():
                canonical.to_parquet(_address, **write_params)
            return True
        # csv
        if file_type.lower() in ['csv', 'tsv', 'txt']:
            _index = write_params.pop('index', False)
            with threading.Lock():
                canonical.to_csv(_address, index=_index, **write_params)
            return True
        # json
        if file_type.lower() in ['json']:
            self._json_dump(data=canonical, path_file=_address, **write_params)
            return True
        # pickle
        if file_type.lower() in ['pkl', 'pickle']:
            self._pickle_dump(data=canonical, path_file=_address, **write_params)
            return True
        # yaml
        if file_type.lower() in ['yml', 'yaml']:
            self._yaml_dump(data=canonical, path_file=_address, **write_params)
            return True
        # not found
        raise LookupError('The file format {} is not currently supported for write'.format(file_type))

    def remove_canonical(self) -> bool:
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _cc = self.connector_contract
        if self.connector_contract.schema.startswith('http'):
            raise NotImplemented("Remove Canonical does not support {} schema based URIs".format(_cc.schema))
        if os.path.exists(_cc.address):
            os.remove(_cc.address)
            return True
        return False

    @staticmethod
    def _yaml_dump(data, path_file, **kwargs) -> None:
        """ dump YAML file

        :param data: the data to persist
        :param path_file: the name and path of the file
        :param default_flow_style: (optional) if to include the default YAML flow style
        """
        encoding = kwargs.pop('encoding', 'utf-8')
        default_flow_style = kwargs.pop('default_flow_style', False)
        with threading.Lock():
            # make sure the dump is clean
            try:
                with closing(open(path_file, mode='w', encoding=encoding)) as ymlfile:
                    yaml.safe_dump(data=data, stream=ymlfile, default_flow_style=default_flow_style, **kwargs)
            except IOError as e:
                raise IOError("The yaml file {} failed to open with: {}".format(path_file, e))
        # check the file was created
        return

    @staticmethod
    def _pickle_dump(data, path_file: str, **kwargs) -> None:
        """ dumps a pickle file"""
        protocol = kwargs.pop('protocol', pickle.HIGHEST_PROTOCOL)
        fix_imports = kwargs.pop('fix_imports', True)
        with threading.Lock():
            with closing(open(path_file, mode='wb')) as f:
                pickle.dump(data, f, protocol=protocol, fix_imports=fix_imports)

    @staticmethod
    def _json_dump(data, path_file: str, **kwargs) -> None:
        """ dumps a pickle file"""
        with threading.Lock():
            with closing(open(path_file, mode='w')) as f:
                json.dump(data, f, **kwargs)
