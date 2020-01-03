import os
import threading
from io import BytesIO
import pandas as pd
from botocore.exceptions import ClientError
import boto3
try:
    import pyarrow as pa
except ImportError:
    import fastparquet as pa
try:
    import cPickel as pickle
except ImportError:
    import pickle


from ds_foundation.handlers.abstract_handlers import AbstractSourceHandler, ConnectorContract, AbstractPersistHandler

__author__ = 'Darryl Oatridge'


class AwsS3SourceHandler(AbstractSourceHandler):
    """ An Amazon AWS S3 source handler.

        URI Format:
            uri = 's3://<bucket>[/<path>]/<filename.ext>'

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - get_object_kw: (optional) value pair dictionary of parameters to pass to the Boto3 get_client method
            - read_kw: (optional) value pair dictionary of parameters to pass to the pandas read methods

        Restrictions:
            - This does not use the AWS S3 Multipart Upload and is limited to 5GB files
    """

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Hander passing the connector_contract dictionary """
        super().__init__(connector_contract)
        self._modified = 0

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['parquet', 'csv', 'json', 'pickle', 'xlsx']

    def load_canonical(self) -> pd.DataFrame:
        """Loads the canonical dataset, returning a Pandas DataFrame.

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - get_object_kw: (optional) value pair dictionary of parameters to pass to the Boto3 get_client method
            - read_kw: (optional) value pair dictionary of parameters to pass to the pandas read methods
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        _client_params = _cc.kwargs.get('get_object_kw', {})
        _read_params = _cc.kwargs.get('read_kw', {})
        _, _ext = os.path.splitext(_cc.address)
        file_type = _cc.get_key_value('file_type', _ext if len(_ext) > 0 else 'dsv')
        if file_type.lower() not in self.supported_types():
            raise ValueError("The file type {} is not recognised. "
                             "Set file_type parameter to a recognised source type".format(file_type))
        s3_client = boto3.client(_cc.schema)
        # take the encoding out before passing the query value pairs
        encoding = _client_params.pop('encoding', 'utf-8')
        try:
            s3_object = s3_client.get_object(Bucket=_cc.netloc, Key=_cc.path, **_client_params)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            raise ConnectionError(f"Failed to retrieve the object from S3 client with error code '{code}")
        resource_body = s3_object['Body'].read().decode(encoding)
        with threading.Lock():
            if file_type.lower() in ['parquet', 'pq', 'pqt']:
                df = pd.read_parquet(resource_body, **_read_params)
            elif file_type.lower() in ['csv', 'tsv', 'txt']:
                df = pd.read_csv(resource_body, **_read_params)
            elif file_type.lower() in ['json']:
                df = pd.read_json(resource_body, **_read_params)
            elif file_type.lower() in ['pkl ', 'pickle']:
                df = pd.read_pickle(resource_body, **_read_params)
            elif file_type.lower() in ['xls', 'xlsx']:
                df = pd.read_excel(resource_body, **_read_params)
            else:
                raise LookupError('The source format {} is not currently supported'.format(file_type))
        self._modified = s3_object.get('LastModified', 0)
        return df

    def get_modified(self) -> [int, float, str]:
        """ returns if the file has been modified """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        _client_params = _cc.kwargs.get('get_object_kw')
        _ = _client_params.pop('encoding', 'utf-8')
        s3_client = boto3.client(_cc.schema)
        try:
            s3_object = s3_client.get_object(Bucket=_cc.netloc, Key=_cc.path, **_client_params)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            raise ConnectionError(f"Failed to retrieve the object from S3 client with error code '{code}")
        self._modified = s3_object.get('LastModified', 0)
        return self._modified


class AwsS3PersistHandler(AbstractPersistHandler):
    """ An Amazon AWS S3 source handler.

        URI Format:
            uri = 's3://<bucket>[/<path>]/<filename.ext>'

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - get_object_kw: (optional) value pair dict of parameters to pass to the Boto3 get_object method
            - put_object_kw: (optional) value pair dict of parameters to pass to the Boto3 put_object method
            - del_object_kw: (optional) value pair dict of parameters to pass to the Boto3 delete_object method
            - read_kw: (optional) value pair dict of parameters to pass to the read methods
            - write_kw: (optional) value pair dict of parameters to pass to the write methods

        Restrictions:
            - This does not use the AWS S3 Multipart Upload and is limited to 5GB files
    """

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Hander passing the connector_contract dictionary """
        super().__init__(connector_contract)
        self._modified = 0

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['pickle', 'parquet', 'csv', 'json', 'xlsx']

    def get_modified(self) -> [int, float, str]:
        """ returns if the file has been modified """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        _client_params = _cc.kwargs.get('get_object_kw')
        _ = _client_params.pop('encoding', 'utf-8')
        s3_client = boto3.client(_cc.schema)
        try:
            s3_object = s3_client.get_object(Bucket=_cc.netloc, Key=_cc.path, **_client_params)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            raise ConnectionError(f"Failed to retrieve the object from S3 client with error code '{code}")
        self._modified = s3_object.get('LastModified', 0)
        return self._modified

    def exists(self) -> bool:
        """ Returns True is the file exists """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        s3_client = boto3.client(_cc.schema)
        response = s3_client.list_objects_v2(Bucket=_cc.netloc)
        for obj in response.get('Contents', []):
            if obj['Key'] == _cc.path:
                return True
        return False

    def load_canonical(self) -> pd.DataFrame:
        """Loads the canonical dataset, returning a Pandas DataFrame.

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - get_object_kw: (optional) value pair dictionary of parameters to pass to the Boto3 get_client method
            - read_kw: (optional) value pair dictionary of parameters to pass to the pandas read methods
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        _client_params = _cc.kwargs.get('get_object_kw', {})
        _read_params = _cc.kwargs.get('read_kw', {})
        _, _ext = os.path.splitext(_cc.address)
        file_type = _cc.get_key_value('file_type', _ext if len(_ext) > 0 else 'dsv')
        if file_type.lower() not in self.supported_types():
            raise ValueError("The file type {} is not recognised. "
                             "Set file_type parameter to a recognised source type".format(file_type))
        s3_client = boto3.client(_cc.schema)
        # take the encoding out before passing the query value pairs
        encoding = _client_params.pop('encoding', 'utf-8')
        try:
            s3_object = s3_client.get_object(Bucket=_cc.netloc, Key=_cc.path, **_client_params)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            raise ConnectionError(f"Failed to retrieve the object from S3 client with error code '{code}")
        resource_body = s3_object['Body'].read().decode(encoding)
        with threading.Lock():
            if file_type.lower() in ['parquet', 'pq', 'pqt']:
                df = pd.read_parquet(resource_body, **_read_params)
            elif file_type.lower() in ['csv', 'tsv', 'txt']:
                df = pd.read_csv(resource_body, **_read_params)
            elif file_type.lower() in ['json']:
                df = pd.read_json(resource_body, **_read_params)
            elif file_type.lower() in ['pkl ', 'pickle']:
                df = pd.read_pickle(resource_body, **_read_params)
            elif file_type.lower() in ['xls', 'xlsx']:
                df = pd.read_excel(resource_body, **_read_params)
            else:
                raise LookupError('The source format {} is not currently supported for read'.format(file_type))
        self._modified = s3_object.get('LastModified', 0)
        return df

    def persist_canonical(self, canonical: pd.DataFrame) -> bool:
        """ persists either the canonical dataset. if the file extension does
        not match any supported source types then pass 'file_type=' as a method parameter.

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - put_object_kw: (optional) value pair dict of parameters to pass to the Boto3 put_object method
            - write_kw: (optional) value pair dict of parameters to pass to the write methods
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _uri = self.connector_contract.address
        return self.backup_canonical(uri=_uri, canonical=canonical)

    def backup_canonical(self, canonical: pd.DataFrame, uri: str) -> bool:
        """ persists the canonical dataset as a backup to the specified URI resource. Note that only the
        address is taken from the URI and all other attributes are taken from the ConnectorContract

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - put_object_kw: (optional) value pair dict of parameters to pass to the Boto3 put_object method
            - write_kw: (optional) value pair dict of parameters to pass to the write methods
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        schema, bucket, path = _cc.parse_address_elements(uri=uri)
        _client_params = _cc.kwargs.get('put_object_kw', {})
        _write_params = _cc.kwargs.get('write_kw', {})
        _, _ext = os.path.splitext(path)
        file_type = _cc.get_key_value('file_type', _ext if len(_ext) > 0 else 'dsv')
        s3_client = boto3.client(schema)
        # csv
        if file_type.lower() in ['csv', 'tsv', 'txt']:
            byte_obj = BytesIO()
            _mode = _write_params.pop('mode', 'wb')
            with threading.Lock():
                canonical.to_csv(byte_obj, mode=_mode, **_write_params)
                s3_client.put_object(Bucket=bucket, Key=path, Body=byte_obj, **_client_params)
        # pickle
        elif file_type.lower() in ['pkl ', 'pickle']:
            with threading.Lock():
                byte_obj = pickle.dump(canonical, **_write_params)
                s3_client.put_object(Bucket=bucket, Key=path, Body=byte_obj, **_client_params)
        # json
        elif file_type.lower() in ['json']:
            byte_obj = BytesIO()
            with threading.Lock():
                canonical.to_json(byte_obj, **_write_params)
                s3_client.put_object(Bucket=bucket, Key=path, Body=byte_obj, **_client_params)
        # parquet
        elif file_type.lower() in ['parquet', 'pq', 'pqt']:
            with threading.Lock():
                table = pa.Table.from_pandas(df=canonical, **_write_params)
                s3_client.put_object(Bucket=bucket, Key=path, Body=table, **_client_params)
        else:
            raise LookupError('The source format {} is not currently supported for write'.format(file_type))
        return True

    def remove_canonical(self) -> bool:
        """ removes the URI named resource

        Extra Parameters in the ConnectorContract kwargs:
            - del_object_kw: (optional) value pair dict of parameters to pass to the Boto3 delete_object method
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The S3 Source Connector Contract has not been set correctly")
        _cc = self.connector_contract
        if not isinstance(_cc, ConnectorContract):
            raise ValueError("The Python Source Connector Contract has not been set correctly")
        _client_params = _cc.kwargs.get('del_object_kw', {})
        s3_client = boto3.client(_cc.schema)
        response = s3_client.response = s3_client.delete_object(Bucket=_cc.netloc, Key=_cc.path, **_client_params)
        if response.get('RequestCharged') is None:
            return False
        return True
