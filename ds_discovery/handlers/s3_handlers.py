import threading
from io import StringIO, BytesIO
import pandas as pd
import numpy as np
import pickle
import json
import os
import boto3
from botocore.exceptions import ClientError
from aistac.handlers.abstract_handlers import AbstractSourceHandler, AbstractPersistHandler
from aistac.handlers.abstract_handlers import ConnectorContract

__author__ = 'Darryl Oatridge'


class S3SourceHandler(AbstractSourceHandler):
    """ An Amazon AWS S3 source handler.

        URI Format:
            uri = 's3://<bucket>[/<path>]/<filename.ext>'

        Restrictions:
            - This does not use the AWS S3 Multipart Upload and is limited to 5GB files
    """

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Handler passing the connector_contract dictionary

        Extra Parameters in the ConnectorContract kwargs:
            - region_name (optional) session region name
            - profile_name (optional) session shared credentials file profile name
        """
        super().__init__(connector_contract)
        cc_params = connector_contract.kwargs
        cc_params.update(connector_contract.query)  # Update kwargs with those in the uri query
        region_name = cc_params.pop('region_name', 'us-east-2')
        aws_access_key_id = cc_params.pop('aws_access_key_id', os.environ.get('AWS_ACCESS_KEY_ID'))
        aws_secret_access_key = cc_params.pop('aws_secret_access_key', os.environ.get('AWS_SECRET_ACCESS_KEY'))
        aws_session_token = cc_params.pop('aws_session_token', os.environ.get('AWS_SESSION_TOKEN'))
        profile_name = cc_params.pop('profile_name', None)
        self._session = boto3.Session(region_name=region_name, aws_access_key_id=aws_access_key_id,
                                      aws_secret_access_key=aws_secret_access_key, profile_name=profile_name,
                                      aws_session_token=aws_session_token)
        self._file_state = 0
        self._changed_flag = True
        self._lock = threading.Lock()

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['parquet', 'csv', 'tsv', 'txt', 'json', 'pickle']

    def exists(self) -> bool:
        """ Returns True is the file exists

        Extra Parameters in the ConnectorContract kwargs:
            - s3_list_params: (optional) a dictionary of additional s3 parameters directly passed to 'list_objects_v2'

        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        cc_params = _cc.kwargs
        cc_params.update(_cc.query)  # Update kwargs with those in the uri query
        s3_list_params = cc_params.pop('s3_list_params', {})
        if _cc.schema not in ['s3']:
            raise ValueError("The Connector Contract Schema has not been set correctly.")
        s3_client = self._session.client(_cc.schema)
        response = s3_client.list_objects_v2(Bucket=_cc.netloc, **s3_list_params)
        for obj in response.get('Contents', []):
            if obj['Key'] == _cc.path[1:]:
                return True
        return False

    def has_changed(self) -> bool:
        """ returns if the file has been modified

            - s3_get_params: (optional) a dictionary of additional s3 client parameters directly passed to 'get_object'
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        cc_params = _cc.kwargs
        cc_params.update(_cc.query)  # Update kwargs with those in the uri query
        # pop all the extra params
        s3_get_params = cc_params.pop('s3_get_params', {})
        if _cc.schema not in ['s3']:
            raise ValueError("The Connector Contract Schema has not been set correctly.")
        s3_client = self._session.client(_cc.schema)
        try:
            s3_object = s3_client.get_object(Bucket=_cc.netloc, Key=_cc.path[1:], **s3_get_params)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            raise ConnectionError("Failed to retrieve the object from region '{}', bucket '{}' "
                                  "Key '{}' with error code '{}'".format(self._session.region_name, _cc.netloc,
                                                                         _cc.path[1:], code))
        state = s3_object.get('LastModified', 0)
        if state != self._file_state:
            self._changed_flag = True
            self._file_state = state
        return self._changed_flag

    def reset_changed(self, changed: bool = False):
        """ manual reset to say the file has been seen. This is automatically called if the file is loaded"""
        changed = changed if isinstance(changed, bool) else False
        self._changed_flag = changed

    def load_canonical(self, **kwargs) -> [pd.DataFrame, dict]:
        """Loads the canonical dataset, returning a Pandas DataFrame. This method utilises the pandas
        'pd.read_' methods and directly passes the kwargs to these methods.

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
                            by default json files load as dict, to load as pandas use read_params '{as_dataframe: True}
            - encoding: (optional) the encoding of the s3 object body. Default 'utf-8'
            - s3_get_params: (optional) a dictionary of additional s3 client parameters directly passed to 'get_object'
            - read_params: (optional) value pair dict of parameters to pass to the read methods. Underlying
                           read methods the parameters are passed to are all pandas 'read_*', e.g. pd.read_csv

        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        _, _, _ext = _cc.address.rpartition('.')
        cc_params = _cc.kwargs
        cc_params.update(_cc.query)  # Update kwargs with those in the uri query
        cc_params.update(kwargs)     # Update with any passed though the call
        # pop all the extra params
        encoding = cc_params.pop('encoding', 'utf-8')
        file_type = cc_params.pop('file_type', _ext if len(_ext) > 0 else 'pickle')
        s3_get_params = cc_params.pop('s3_get_params', {})
        read_params = cc_params.pop('read_params', {})
        if file_type.lower() not in self.supported_types():
            raise ValueError("The file type {} is not recognised. "
                             "Set file_type parameter to a recognised source type".format(file_type))
        # session
        if _cc.schema not in ['s3']:
            raise ValueError("The Connector Contract Schema has not been set correctly.")
        s3_client = self._session.client(_cc.schema)
        try:
            s3_object = s3_client.get_object(Bucket=_cc.netloc, Key=_cc.path[1:], **s3_get_params)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            raise ConnectionError("Failed to retrieve the object from region '{}', bucket '{}' "
                                  "Key '{}' with error code '{}'".format(self._session.region_name, _cc.netloc,
                                                                         _cc.path[1:], code))
        resource_body = s3_object['Body'].read()
        with self._lock:
            if file_type.lower() in ['parquet', 'pq', 'pqt']:
                return pd.read_parquet(BytesIO(resource_body), **read_params)
            if file_type.lower() in ['csv', 'tsv', 'txt']:
                return pd.read_csv(StringIO(resource_body.decode(encoding)), **read_params)
            if file_type.lower() in ['json']:
                as_dataframe = read_params.pop('as_dataframe', False)
                if as_dataframe:
                    return pd.read_json(StringIO(resource_body.decode(encoding)), **read_params)
                return json.load(StringIO(resource_body.decode(encoding)), **read_params)
            if file_type.lower() in ['pkl ', 'pickle']:
                fix_imports = read_params.pop('fix_imports', True)
                encoding = read_params.pop('encoding', 'ASCII')
                errors = read_params.pop('errors', 'strict')
                return pickle.loads(resource_body, fix_imports=fix_imports, encoding=encoding, errors=errors)
        raise LookupError('The source format {} is not currently supported'.format(file_type))


class S3PersistHandler(S3SourceHandler, AbstractPersistHandler):
    """ An Amazon AWS S3 source handler.

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
        _uri = self.connector_contract.address
        return self.backup_canonical(uri=_uri, canonical=canonical, **kwargs)

    def backup_canonical(self, canonical: [pd.DataFrame, dict], uri: str, **kwargs) -> bool:
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
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        schema, bucket, path = _cc.parse_address_elements(uri=uri)
        _, _, _ext = path.rpartition('.')
        cc_params = kwargs if isinstance(kwargs, dict) else _cc.kwargs
        cc_params.update(_cc.parse_query(uri=uri))
        # pop all the extra params
        s3_put_params = cc_params.pop('s3_put_params', _cc.kwargs.get('put_object_kw', {}))
        write_params = cc_params.pop('write_params', _cc.kwargs.get('write_kw', {}))
        file_type = cc_params.pop('file_type', _cc.kwargs.get('file_type', _ext if len(_ext) > 0 else 'pkl'))
        if _cc.schema not in ['s3']:
            raise ValueError("The Connector Contract Schema has not been set correctly.")
        s3_client = self._session.client(_cc.schema)
        # csv
        if file_type.lower() in ['csv', 'tsv', 'txt']:
            byte_obj = StringIO()
            _mode = write_params.pop('mode', 'w')
            _index = write_params.pop('index', False)
            with self._lock:

                canonical.to_csv(byte_obj, mode=_mode, index=_index, **write_params)
                s3_client.put_object(Bucket=bucket, Key=path[1:], Body=byte_obj.getvalue(), **s3_put_params)
        # pickle
        elif file_type.lower() in ['pkl ', 'pickle']:
            _protocol = write_params.pop('protocol', pickle.HIGHEST_PROTOCOL)
            _fix_imports = write_params.pop('fix_imports', True)
            with self._lock:
                byte_obj = pickle.dumps(canonical, protocol=_protocol, fix_imports=_fix_imports)
                s3_client.put_object(Bucket=bucket, Key=path[1:], Body=byte_obj, **s3_put_params)
        # json
        elif file_type.lower() in ['json']:
            byte_obj = StringIO()
            with self._lock:
                if isinstance(canonical, pd.DataFrame):
                    canonical.to_json(byte_obj, **write_params)
                    body = byte_obj.getvalue()
                else:
                    encode = write_params.pop('encode', 'UTF-8')
                    body = (bytes(json.dumps(canonical, cls=NpEncoder, **s3_put_params).encode(encode)))
                s3_client.put_object(Bucket=bucket, Key=path[1:], Body=body, **s3_put_params)
        # parquet
        elif file_type.lower() in ['parquet', 'pq', 'pqt']:
            _index = write_params.pop('index', False)
            byte_obj = BytesIO()
            with self._lock:
                # table = pa.Table.from_pandas(df=canonical, **write_params)
                canonical.to_parquet(byte_obj, index=_index, **write_params)
                s3_client.put_object(Bucket=bucket, Key=path[1:], Body=byte_obj.getvalue(), **s3_put_params)
        else:
            raise LookupError('The source format {} is not currently supported for write'.format(file_type))
        return True

    def remove_canonical(self) -> bool:
        """ removes the URI named resource

        Extra Parameters in the ConnectorContract kwargs:
            - s3_del_params: (optional) value pair dict of parameters to pass to the Boto3 delete_object method
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        cc_params = _cc.kwargs
        cc_params.update(_cc.query)  # Update kwargs with those in the uri query
        # pop all the extra params
        s3_del_params = cc_params.pop('s3_put_params', _cc.kwargs.get('put_object_kw', {}))
        if _cc.schema not in ['s3']:
            raise ValueError("The Connector Contract Schema has not been set correctly.")
        s3_client = self._session.client(_cc.schema)
        response = s3_client.response = s3_client.delete_object(Bucket=_cc.netloc, Key=_cc.path[1:], **s3_del_params)
        if response.get('RequestCharged') is None:
            return False
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
