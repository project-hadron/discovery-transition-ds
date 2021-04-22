from contextlib import closing
import os
import threading
import pandas as pd
import numpy as np
import pickle
import json

from aistac.handlers.abstract_handlers import AbstractSourceHandler, AbstractPersistHandler
from aistac.handlers.abstract_handlers import ConnectorContract, HandlerFactory

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
        self._lock = threading.Lock()
        self._file_state = 0
        self._changed_flag = True

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['parquet', 'csv', 'tsv', 'txt', 'json', 'pickle', 'xlsx', 'yaml']

    def load_canonical(self, **kwargs) -> [pd.DataFrame, dict]:
        """ returns the canonical dataset based on the connector contract. This method utilises the pandas
        'pd.read_' methods and directly passes the kwargs to these methods.

        if reading large CSV file you can use the lambda functions in skiprows by passing it as a string at setup
        example:
                # 0.01 probability (1% of the rows)
                skiprows=lambda i: i>0 and random.random() > 0.01

                # or every nth row, in this case 100th
                lambda i: i % 100 != 0

        Extra Parameters in the ConnectorContract kwargs:
            - use_full_uri: (optional) use the uri in full and don't remove the query parameters
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - read_params: (optional) value pair dict of parameters to pass to the read methods. Underlying
                           read methods the parameters are passed to are all pandas 'read_*', e.g. pd.read_csv
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The Pandas Connector Contract has not been set")
        _cc = self.connector_contract
        load_params = kwargs
        load_params.update(_cc.kwargs)  # Update with any kwargs in the
        if load_params.pop('use_full_uri', False):
            file_type = load_params.pop('file_type', 'csv')
            address = _cc.uri
        else:
            load_params.update(_cc.query)  # Update kwargs with those in the uri query
            _, _, _ext = _cc.address.rpartition('.')
            address = _cc.address
            file_type = load_params.pop('file_type', _ext if len(_ext) > 0 else 'csv')
        if file_type.lower() in ['parquet', 'pq']:
            rtn_data = pd.read_parquet(address, **load_params)
        elif file_type.lower() in ['zip', 'csv', 'tsv', 'txt']:
            if 'skiprows' in load_params and load_params.get('skiprows').startswith("lambda"):
                load_params['skiprows'] = eval(load_params.get('skiprows'))
            rtn_data = pd.read_csv(address, **load_params)
        elif file_type.lower() in ['json']:
            rtn_data = self._json_load(path_file=address, **load_params)
        elif file_type.lower() in ['xls', 'xlsx']:
            rtn_data = pd.read_excel(address, **load_params)
        elif file_type.lower() in ['pkl ', 'pickle']:
            rtn_data = self._pickle_load(path_file=address, **load_params)
        elif file_type.lower() in ['yml', 'yaml']:
            rtn_data = self._yaml_load(path_file=address, **load_params)
        else:
            raise LookupError('The source format {} is not currently supported'.format(file_type))
        # set the change flag to read
        self.reset_changed()
        return rtn_data

    def exists(self) -> bool:
        """ Returns True is the file exists """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The Pandas Connector Contract has not been set")
        _cc = self.connector_contract
        if _cc.schema.startswith('http') or _cc.schema.startswith('git'):
            module_name = 'requests'
            _address = _cc.address.replace("git://", "https://")
            if HandlerFactory.check_module(module_name=module_name):
                module = HandlerFactory.get_module(module_name=module_name)
                return module.get(_address).status_code == 200
            raise ModuleNotFoundError(f"The required module {module_name} has not been installed. "
                                      f"Please pip install the appropriate package in order to complete this action")
        if os.path.exists(_cc.address):
            return True
        return False

    def has_changed(self) -> bool:
        """ returns the status of the change_flag indicating if the file has changed since last load or reset"""
        if not self.exists():
            return False
        # maintain the change flag
        _cc = self.connector_contract
        if _cc.schema.startswith('http') or _cc.schema.startswith('git'):
            if not isinstance(self.connector_contract, ConnectorContract):
                raise ValueError("The Pandas Connector Contract has not been set")
            module_name = 'requests'
            _address = _cc.address.replace("git://", "https://")
            if HandlerFactory.check_module(module_name=module_name):
                module = HandlerFactory.get_module(module_name=module_name)
                state = module.head(_address).headers.get('last-modified', 0)
            else:
                raise ModuleNotFoundError(f"The required module {module_name} has not been installed. Please pip "
                                          f"install the appropriate package in order to complete this action")
        else:
            state = os.stat(_cc.address).st_mtime_ns
        if state != self._file_state:
            self._changed_flag = True
            self._file_state = state
        return self._changed_flag

    def reset_changed(self, changed: bool = False):
        """ manual reset to say the file has been seen. This is automatically called if the file is loaded"""
        changed = changed if isinstance(changed, bool) else False
        self._changed_flag = changed

    @staticmethod
    def _yaml_load(path_file, **kwargs) -> dict:
        """ loads the YAML file

        :param path_file: the name and path of the file
        :return: a dictionary
        """
        module_name = 'yaml'
        if HandlerFactory.check_module(module_name=module_name):
            module = HandlerFactory.get_module(module_name=module_name)
        else:
            raise ModuleNotFoundError(f"The required module {module_name} has not been installed. "
                                      f"Please pip install the appropriate package in order to complete this action")
        encoding = kwargs.pop('encoding', 'utf-8')
        try:
            with closing(open(path_file, mode='r', encoding=encoding)) as ymlfile:
                rtn_dict = module.safe_load(ymlfile)
        except IOError as e:
            raise IOError(f"The yaml file {path_file} failed to open with: {e}")
        if not isinstance(rtn_dict, dict) or not rtn_dict:
            raise TypeError(f"The yaml file {path_file} could not be loaded as a dict type")
        return rtn_dict

    @staticmethod
    def _pickle_load(path_file: str, **kwargs) -> [dict, pd.DataFrame]:
        """ loads a pickle file """
        fix_imports = kwargs.pop('fix_imports', True)
        encoding = kwargs.pop('encoding', 'ASCII')
        errors = kwargs.pop('errors', 'strict')
        if path_file.startswith('http'):
            module_name = 'requests'
            if HandlerFactory.check_module(module_name=module_name):
                module = HandlerFactory.get_module(module_name=module_name)
                username = kwargs.get('username', None)
                password = kwargs.get('password', None)
                auth = (username, password) if username and password else None
                r = module.get(path_file, auth=auth)
                return r.content
        with closing(open(path_file, mode='rb')) as f:
            return pickle.load(f, fix_imports=fix_imports, encoding=encoding, errors=errors)

    @staticmethod
    def _json_load(path_file: str, **kwargs) -> [dict, pd.DataFrame]:
        """ loads a pickle file """
        if path_file.startswith('http'):
            module_name = 'requests'
            if HandlerFactory.check_module(module_name=module_name):
                module = HandlerFactory.get_module(module_name=module_name)
                username = kwargs.get('username', None)
                password = kwargs.get('password', None)
                auth = (username, password) if username and password else None
                r = module.get(path_file, auth=auth)
                return r.json()
        with closing(open(path_file, mode='r')) as f:
            return json.load(f, **kwargs)


class PandasPersistHandler(PandasSourceHandler, AbstractPersistHandler):
    """ Pandas read/write Persist Handler. The format of the uri should be as a minimum:
                    uri = '[/<path>/]<filename.ext>'
        but can be a full url
                    uri = <scheme>://<netloc>/[<path>/]<filename.ext>
    """

    def persist_canonical(self, canonical: [pd.DataFrame, dict], **kwargs) -> bool:
        """ persists the canonical dataset

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _uri = self.connector_contract.uri
        return self.backup_canonical(uri=_uri, canonical=canonical, **kwargs)

    def backup_canonical(self, canonical: [pd.DataFrame, dict], uri: str, **kwargs) -> bool:
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
            if len(_path) > 0 and not os.path.exists(_path):
                os.makedirs(_path)
        file_type = persist_params.pop('file_type', _ext if len(_ext) > 0 else 'pkl')
        write_params = persist_params.pop('write_params', {})
        # parquet
        if file_type.lower() in ['pq', 'pqt', 'parquet']:
            _index = write_params.pop('index', False)
            with self._lock:
                canonical.to_parquet(_address, index=_index, **write_params)
            return True
        # csv
        if file_type.lower() in ['csv', 'tsv', 'txt']:
            _index = write_params.pop('index', False)
            with self._lock:
                canonical.to_csv(_address, index=_index, **write_params)
            return True
        # json
        if file_type.lower() in ['json']:
            if isinstance(canonical, pd.DataFrame):
                with self._lock:
                    canonical.to_json(_address, **write_params)
                return True
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

    def _yaml_dump(self, data, path_file, **kwargs) -> None:
        """ dump YAML file

        :param data: the data to persist
        :param path_file: the name and path of the file
        :param default_flow_style: (optional) if to include the default YAML flow style
        """
        module_name = 'yaml'
        if HandlerFactory.check_module(module_name=module_name):
            module = HandlerFactory.get_module(module_name=module_name)
        else:
            raise ModuleNotFoundError(f"The required module {module_name} has not been installed. "
                                      f"Please pip install the appropriate package in order to complete this action")
        encoding = kwargs.pop('encoding', 'utf-8')
        default_flow_style = kwargs.pop('default_flow_style', False)
        with self._lock:
            # make sure the dump is clean
            try:
                with closing(open(path_file, mode='w', encoding=encoding)) as ymlfile:
                    module.safe_dump(data=data, stream=ymlfile, default_flow_style=default_flow_style, **kwargs)
            except IOError as e:
                raise IOError(f"The yaml file {path_file} failed to open with: {e}")
        # check the file was created
        return

    def _pickle_dump(self, data, path_file: str, **kwargs) -> None:
        """ dumps a pickle file"""
        protocol = kwargs.pop('protocol', pickle.HIGHEST_PROTOCOL)
        fix_imports = kwargs.pop('fix_imports', True)
        with self._lock:
            with closing(open(path_file, mode='wb')) as f:
                pickle.dump(data, f, protocol=protocol, fix_imports=fix_imports)

    def _json_dump(self, data, path_file: str, **kwargs) -> None:
        """ dumps a pickle file"""
        with self._lock:
            with closing(open(path_file, mode='w')) as f:
                json.dump(data, f, cls=NpEncoder, **kwargs)


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.datetime64):
            return np.datetime_as_string(obj, unit='s')
        elif isinstance(obj, pd.Timestamp):
            return np.datetime_as_string(obj.to_datetime64(), unit='s')
        else:
            return super(NpEncoder, self).default(obj)
