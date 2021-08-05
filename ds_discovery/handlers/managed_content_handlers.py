import threading
import pandas as pd
import numpy as np
import json
import os,io
from cortex.content import ManagedContentClient
from aistac.handlers.abstract_handlers import AbstractSourceHandler, AbstractPersistHandler
from aistac.handlers.abstract_handlers import ConnectorContract

__author__ = ''


class ManagedContentSourceHandler(AbstractSourceHandler):
    """ Cortex Managed Content source handler.
    """

    def __init__(self, connector_contract: ConnectorContract):

        super().__init__(connector_contract)
        cc_params = connector_contract.kwargs
        self.project = cc_params.pop('project', os.environ.get('PROJECT'))
        self.mission = cc_params.pop('mission', os.environ.get('MISSION'))
        self.token = cc_params.pop('token', os.environ.get('TOKEN'))
        self.url = cc_params.pop('url', os.environ.get('URL'))
        self._file_state = 0
        self._changed_flag = True
        self._lock = threading.Lock()

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['parquet', 'csv']

    def exists(self) -> bool:
        """ Returns True is the file exists

        Extra Parameters in the ConnectorContract kwargs:
            - s3_list_params: (optional) a dictionary of additional s3 parameters directly passed to 'list_objects_v2'

        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The Managed Content Contract has not been set correctly")
        _cc = self.connector_contract
        cc_params = _cc.kwargs
        cc_params.update(_cc.query)  # Update kwargs with those in the uri query
        key=cc_params.pop("key",os.environ.get('KEY'))
        managedcontent_client = ManagedContentClient(url=self.url,token=self.token)
        if managedcontent_client.exists(key, self.project):
            return True
        return False

    def load_canonical(self, **kwargs) -> [pd.DataFrame, dict]:

        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The Managed Content Contract has not been set correctly")
        _cc = self.connector_contract
        _, _, _ext = _cc.address.rpartition('.')
        cc_params = _cc.kwargs
        cc_params.update(_cc.query)  # Update kwargs with those in the uri query
        cc_params.update(kwargs)     # Update with any passed though the call
        key = cc_params.pop("key",os.environ.get('KEY'))
        managedcontent_client = ManagedContentClient(url=self.url, token=self.token)
        response = managedcontent_client.download(key,self.project)
        content = response.data
        bytes_file = io.BytesIO(content)
        #NOTE: info on how the managed content source handler will change
        
        # we will recieve the key from the raw_uri and the token needs to be in the environment
        # use the key to pull out data
        with self._lock:
            if ".parquet" in key:
                return pd.read_parquet(bytes_file)
            elif ".csv" in key:
                return pd.read_csv(bytes_file)
            else:
                raise LookupError('The source format {} is not currently supported')

    def reset_changed(self, changed: bool = False):
        """ manual reset to say the file has been seen. This is automatically called if the file is loaded"""
        changed = changed if isinstance(changed, bool) else False
        self._changed_flag = changed
    
    def has_changed(self) -> bool:
        """ returns if the file has been modified

            - s3_get_params: (optional) a dictionary of additional s3 client parameters directly passed to 'get_object'
        """
        
        return True


class ManagedContentPersistHandler(ManagedContentSourceHandler, AbstractPersistHandler):
    """ Cortex Managed Content Persist handler.

        URI Format:
            uri = 's3://<bucket>[/<path>]/<filename.ext>'

        Restrictions:
            - This does not use the AWS S3 Multipart Upload and is limited to 5GB files
    """

    def persist_canonical(self, canonical: [pd.DataFrame, dict], **kwargs) -> bool:
        """ persists either the canonical dataset.

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - s3_put_params: (optional) a dictionary of additional s3 client parameters directly passed to 'get_object'
            - write_params: (optional) value pair dict of parameters to pass to the write methods - pandas.to_csv,
                              pandas.to_json, pickle.dump and parquet.Table.from_pandas
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        return self.backup_canonical(canonical=canonical, **kwargs)

    def backup_canonical(self, canonical: [pd.DataFrame, dict], **kwargs) -> bool:
        """ persists the canonical dataset as a backup to the specified URI resource. Note that only the
        address is taken from the URI and all other attributes are taken from the ConnectorContract

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - s3_put_params: (optional) value pair dict of parameters to pass to the Boto3 put_object method
            - write_params: (optional) value pair dict of parameters to pass to the write methods - pandas.to_csv,
                              pandas.to_json, pickle.dump and parquet.Table.from_pandas
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        cc_params = _cc.kwargs
        cc_params.update(_cc.query)
        managedcontent_client = ManagedContentClient(url=self.url, token=self.token)
        key = cc_params.pop("key", os.environ.get('KEY_FEEDBACK'))


        if ".parquet" in key:

            canonical.to_parquet(key)
            f_obj = open(key, mode="rb")
            managedcontent_client.upload_streaming(key=key, project=self.project, stream=f_obj,
                                                    content_type="application/octet-stream")

        elif ".csv" in key:
            canonical.to_csv(key)
            f_obj = open(key, mode="rb")
            managedcontent_client.upload_streaming(key=key, project=self.project, stream=f_obj,
                                                    content_type="application/octet-stream")

            raise LookupError('The source format is not currently supported for write')
        return True

    def remove_canonical(self) -> bool:
        """ removes the URI named resource

        Extra Parameters in the ConnectorContract kwargs:
            - s3_del_params: (optional) value pair dict of parameters to pass to the Boto3 delete_object method
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        cc_params = _cc.kwargs
        cc_params.update(_cc.query)  # Update kwargs with those in the uri query

        key = cc_params.pop("key",os.environ.get('KEY'))
        if not self.exists():
            return False
        else:
            pass
           #TODO
        return True


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