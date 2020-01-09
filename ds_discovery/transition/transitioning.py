import os
import pandas as pd
from datetime import datetime

from ds_foundation.managers.augment_properties import AugmentedPropertyManager
from ds_foundation.managers.data_properties import DataPropertyManager
from ds_foundation.handlers.abstract_handlers import ConnectorContract
from ds_discovery.intent.pandas_cleaners import PandasCleaners
from ds_discovery.transition.discovery import DataDiscovery, Visualisation

__author__ = 'Darryl Oatridge'


class Transition(object):

    CONNECTOR_SOURCE = 'read_only_connector'
    CONNECTOR_PERSIST = 'persist_connector'
    CONNECTOR_DATA_INTENT: str
    CONNECTOR_AUGMENT_INTENT: str
    MODULE_NAME: str
    HANDLER_SOURCE: str
    HANDLER_PERSIST: str

    def __init__(self, contract_name: str, data_properties: [ConnectorContract],
                 augment_properties: [ConnectorContract], default_save=None):
        """ Encapsulation class for the discovery set of classes

        :param contract_name: The name of the contract
        :param data_properties: The persist handler for the data properties
        :param augment_properties: The persist handler for the augmented knowledge properties
        :param default_save: The default behaviour of persisting the contracts:
                    if True: all contract properties are persisted
                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        """
        if not isinstance(contract_name, str) or len(contract_name) < 1:
            raise ValueError("The contract name must be a valid string")
        self._contract_name = contract_name
        self._default_save = default_save if isinstance(default_save, bool) else True
        # set property managers
        self._data_pm = DataPropertyManager.from_properties(contract_name=contract_name,
                                                            connector_contract=data_properties)
        self.CONNECTOR_DATA_INTENT = self._data_pm.CONNECTOR_INTENT
        if self._data_pm.has_persisted_properties():
            self._data_pm.load_properties()
        self._knowledge_catalogue = ['overview', 'notes', 'observations', 'attribute', 'dictionary', 'tor']
        self._augment_pm = AugmentedPropertyManager.from_properties(self._contract_name,
                                                                    connector_contract=augment_properties,
                                                                    knowledge_catalogue=self._knowledge_catalogue)
        self.CONNECTOR_AUGMENT_INTENT = self.augment_pm.CONNECTOR_INTENT
        if self._augment_pm.has_persisted_properties():
            self._augment_pm.load_properties()
        # initialise the values
        self.persist_contract(save=self._default_save)
        self._raw_attribute_list = []

    @classmethod
    def from_uri(cls, contract_name: str, properties_uri: str, default_save=None):
        """ Class Factory Method that builds the connector handlers for the properties contract. The method uses
        the schema of the URI to determine if it is remote or local. s3:// schema denotes remote, empty schema denotes
        local.
        Note: the 'properties_uri' only provides a URI up to and including the path but not the properties file names.

         :param contract_name: The reference name of the properties contract
         :param properties_uri: A URI that identifies the resource path. The syntax should be either
                          s3://<bucket>/<path>/ for remote or <path> for local
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :return: the initialised class instance
         """
        _uri = properties_uri
        if not isinstance(_uri, str) or len(_uri) == 0:
            raise ValueError("the URI must take the form 's3://<bucket>/<path>/' for remote or '<path>/' for local")
        _schema, _netloc, _path = ConnectorContract.parse_address_elements(uri=_uri)
        if str(_schema).lower().startswith('s3'):
            return cls._from_remote(contract_name=contract_name, properties_uri=_uri, default_save=default_save)
        _uri = _path
        if not os.path.exists(_path):
            os.makedirs(_path, exist_ok=True)
        return cls._from_local(contract_name=contract_name, properties_uri=_uri, default_save=default_save)

    @classmethod
    def from_env(cls, contract_name: str,  default_save=None):
        """ Class Factory Method that builds the connector handlers taking the property contract path from
        the os.envon['AISTAC_TR_URI'] or locally from the current working directory './' if
        no environment variable is found. This assumes the use of the pandas handler module and yaml persisted file.

         :param contract_name: The reference name of the properties contract
         :param default_save: (optional) if the configuration should be persisted
         :return: the initialised class instance
         """
        if 'AISTAC_INTENT' in os.environ.keys():
            properties_uri = os.environ['AISTAC_INTENT']
        else:
            properties_uri = "/tmp/aistac/transition/contracts"
        return cls.from_uri(contract_name=contract_name, properties_uri=properties_uri, default_save=default_save)

    @classmethod
    def _from_remote(cls, contract_name: str, properties_uri: str, default_save=None):
        """ Class Factory Method that builds the connector handlers an Amazon AWS s3 remote store.
        Note: the 'properties_uri' only provides a URI up to and including the path but not the properties file names.

         :param contract_name: The reference name of the properties contract
         :param properties_uri: A URI that identifies the S3 properties resource path. The syntax should be:
                          s3://<bucket>/<path>/
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :return: the initialised class instance
         """
        if not isinstance(contract_name, str) or len(contract_name) == 0:
            raise ValueError("A contract_name must be provided")
        _default_save = default_save if isinstance(default_save, bool) else True
        _module_name = 'ds_discovery.handlers.aws_s3_handlers'
        _handler = 'AwsS3PersistHandler'
        _address = ConnectorContract.parse_address(uri=properties_uri)
        _query_kw = ConnectorContract.parse_query(uri=properties_uri)
        _data_uri = os.path.join(_address, "config_transition_data_{}.pickle".format(contract_name))
        _data_connector = ConnectorContract(uri=_data_uri, module_name=_module_name, handler=_handler, **_query_kw)
        _aug_uri = os.path.join(_address, "config_transition_augment_{}.pickle".format(contract_name))
        _aug_connector = ConnectorContract(uri=_aug_uri, module_name=_module_name, handler=_handler, **_query_kw)
        rtn_cls = cls(contract_name=contract_name, data_properties=_data_connector,
                      augment_properties=_aug_connector, default_save=default_save)
        rtn_cls.MODULE_NAME = _module_name
        rtn_cls.HANDLER_SOURCE = 'AwsS3SourceHandler'
        rtn_cls.HANDLER_PERSIST = _handler
        return rtn_cls

    @classmethod
    def _from_local(cls, contract_name: str,  properties_uri: str, default_save=None):
        """ Class Factory Method that builds the connector handlers from a local resource path.
        This assumes the use of the pandas handler module and yaml persisted file.

        :param contract_name: The reference name of the properties contract
        :param properties_uri: (optional) A URI that identifies the properties resource path.
                            by default is '/tmp/aistac/contracts'
        :param default_save: (optional) if the configuration should be persisted
        :return: the initialised class instance
        """
        if not isinstance(contract_name, str) or len(contract_name) == 0:
            raise ValueError("A contract_name must be provided")
        _properties_uri = properties_uri if isinstance(properties_uri, str) else "/tmp/aistac/contracts"
        _default_save = default_save if isinstance(default_save, bool) else True
        _module_name = 'ds_discovery.handlers.pandas_handlers'
        _handler = 'PandasPersistHandler'
        _data_uri = os.path.join(properties_uri, "config_transition_data_{}.yaml".format(contract_name))
        _data_connector = ConnectorContract(uri=_data_uri, module_name=_module_name, handler=_handler)
        _augment_uri = os.path.join(properties_uri, "config_transition_augment_{}.yaml".format(contract_name))
        _augment_connector = ConnectorContract(uri=_augment_uri, module_name=_module_name, handler=_handler)
        rtn_cls = cls(contract_name=contract_name, data_properties=_data_connector,
                      augment_properties=_augment_connector, default_save=default_save)
        rtn_cls.MODULE_NAME = _module_name
        rtn_cls.HANDLER_SOURCE = 'PandasSourceHandler'
        rtn_cls.HANDLER_PERSIST = _handler
        return rtn_cls

    @property
    def contract_name(self) -> str:
        """The contract name of this transition instance"""
        return self._contract_name

    @property
    def version(self):
        """The version number of the contracts"""
        return self.data_pm.version

    @property
    def data_pm(self) -> DataPropertyManager:
        """The data properties manager instance"""
        if self._data_pm is None or self._data_pm.contract_name != self.contract_name:
            self._data_pm = DataPropertyManager(self._contract_name)
        return self._data_pm

    @property
    def augment_pm(self) -> AugmentedPropertyManager:
        """The augmented properties manager instance"""
        if self._augment_pm is None or self._augment_pm.contract_name != self.contract_name:
            self._augment_pm = AugmentedPropertyManager(self._contract_name, self._knowledge_catalogue)
        return self._augment_pm

    @property
    def clean(self) -> PandasCleaners:
        """The cleaner instance"""
        return PandasCleaners()

    @property
    def discover(self) -> DataDiscovery:
        """The discovery instance"""
        return DataDiscovery()

    @property
    def visual(self) -> Visualisation:
        """The visualisation instance"""
        return Visualisation()

    @property
    def snapshots(self):
        return self.data_pm.snapshots

    def is_contract_empty(self):
        """Test if the transitioning contract is empty but excludes the contract connector"""
        connector_list = self.data_pm.connector_contract_list
        if self.data_pm.CONNECTOR_INTENT in connector_list:
            connector_list.remove(self.data_pm.CONNECTOR_INTENT)
        if len(connector_list) > 0 or self.data_pm.has_cleaners() or len(self.data_pm.snapshots) > 0:
            return False
        return True

    def is_source_modified(self):
        """Test if the source file is modified since last load"""
        return self.data_pm.is_modified(self.CONNECTOR_SOURCE)

    def reset_transition_contracts(self, save: bool=None):
        """ resets the contract back to a default.

        :param save: if True, save to file. Default is True
        """
        if not isinstance(save, bool):
            save = self._default_save
        self.data_pm.reset_contract_properties()
        self._raw_attribute_list = []
        self.persist_contract(save)

    def remove_source_contract(self, save: bool=None):
        """removes the source contract

        :param save: if True, save to file. Default is True
        """
        if not isinstance(save, bool):
            save = self._default_save
        self.data_pm.remove_connector_contract(self.CONNECTOR_SOURCE)
        self._raw_attribute_list = []
        self.persist_contract(save)

    def set_source_contract(self, uri: str, module_name: str=None, handler: str=None, save: bool=None, **kwargs):
        """ Sets the source contract, returning the source data as a DataFrame if load=True. If the connection
        module_name and/or handler is not provided the the default properties connection setting are used

        :param uri: A Uniform Resource Identifier that unambiguously identifies a particular resource
        :param module_name: (optional) a module name with full package path. Default MODULE_NAME constant
        :param handler: (optional) the name of the Handler Class within the module. Default tr.HANDLER_SOURCE constant
        :param save: (optional) if True, save to file. Default is True
        :param kwargs: (optional) a list of key additional word argument properties associated with the resource
        :return: if load is True, returns a Pandas.DataFrame else None
        """
        save = save if isinstance(save, bool) else self._default_save
        if not isinstance(module_name, str):
            module_name = self.MODULE_NAME
        if not isinstance(handler, str):
            handler = self.HANDLER_SOURCE
        if self.data_pm.has_connector(self.CONNECTOR_SOURCE):
            self.data_pm.remove_connector_contract(self.CONNECTOR_SOURCE)
        self.data_pm.set_connector_contract(self.CONNECTOR_SOURCE, uri=uri, module_name=module_name, handler=handler,
                                            **kwargs)
        self.persist_contract(save)
        return

    def set_persist_contract(self, uri: str, module_name: str=None, handler: str=None, save: bool=None, **kwargs):
        """ Sets the persist contract. For parameters not provided the default resource name and data properties
        connector contract module and handler are used.

        :param uri: A Uniform Resource Identifier that unambiguously identifies a particular resource
        :param module_name: (optional) a module name with full package path. Default MODULE_NAME constant
        :param handler: (optional) the name of the Handler Class within the module. Default tr.HANDLER_PERSIST constant
        :param handler: the name of the Handler Class. Must be
        :param save: if True, save to file. Default is True
        :param kwargs: (optional) a list of key additional word argument properties associated with the resource
        :return: if load is True, returns a Pandas.DataFrame else None
        """
        save = save if isinstance(save, bool) else self._default_save
        if not isinstance(module_name, str):
            module_name = self.MODULE_NAME
        if not isinstance(handler, str):
            handler = self.HANDLER_PERSIST
        if self.data_pm.has_connector(self.CONNECTOR_PERSIST):
            self.data_pm.remove_connector_contract(self.CONNECTOR_PERSIST)
        self.data_pm.set_connector_contract(self.CONNECTOR_PERSIST, uri=uri, module_name=module_name, handler=handler,
                                            **kwargs)
        self.persist_contract(save)
        return

    def load_source_canonical(self) -> pd.DataFrame:
        """returns the contracted source data as a DataFrame"""
        if self.data_pm.has_connector(self.CONNECTOR_SOURCE):
            handler = self.data_pm.get_connector_handler(self.CONNECTOR_SOURCE)
            df = handler.load_canonical()
            if isinstance(df, dict):
                df = pd.DataFrame(df)
            if len(df.columns) > 0:
                self._raw_attribute_list = df.columns.to_list()
            self.data_pm.set_modified(self.CONNECTOR_SOURCE, handler.get_modified())
            return df
        return pd.DataFrame()

    def report_connectors(self, connector_name: str=None, stylise: bool=True):
        """ generates a report on the source contract

        :param connector_name: (optional) filters on the connector name. Aliases can be used instead of the default
                raw data source (self.ORIGIN_SOURCE):  'source', 'data', 'dataset', 'origin', 'raw'
                persisted source (self.PERSIST_SOURCE): 'persist', 'canonical', 'transition'
                properties source (self.data_pm.PROPERTY_SOURCE): 'properties', 'property', 'props', 'config'
        :param stylise: (optional) returns a stylised dataframe with formatting
        :return: pd.DataFrame
        """
        if connector_name in ['source', 'data', 'dataset', 'origin', 'raw']:
            connector_name = self.CONNECTOR_SOURCE
        if connector_name in ['persist', 'canonical', 'transition']:
            connector_name = self.CONNECTOR_PERSIST
        if connector_name in ['properties', 'property', 'props', 'config']:
            connector_name = self.data_pm.CONNECTOR_INTENT
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame()
        join = self.data_pm.join
        dpm = self.data_pm
        df['param'] = ['connector_name', 'uri', 'module_name', 'handler', 'modified', 'kwargs', 'query', 'params']
        for name_key in dpm.get(join(dpm.KEY.connectors_key)).keys():
            if isinstance(connector_name, str) and connector_name != name_key:
                continue
            connector_contract = dpm.get_connector_contract(name_key)
            if isinstance(connector_contract, ConnectorContract):
                if name_key == self.CONNECTOR_SOURCE:
                    label = 'Data Source'
                elif name_key == self.CONNECTOR_PERSIST:
                    label = 'Persist Source'
                elif name_key == self.data_pm.CONNECTOR_INTENT:
                    label = 'Property Source'
                else:
                    label = name_key
                kwargs = ''
                if isinstance(connector_contract.kwargs, dict):
                    for k, v in connector_contract.kwargs.items():
                        if len(kwargs) > 0:
                            kwargs += "  "
                        kwargs += "{}='{}'".format(k, v)
                query = ''
                if isinstance(connector_contract.query, dict):
                    for k, v in connector_contract.query.items():
                        if len(query) > 0:
                            query += "  "
                        query += "{}='{}'".format(k, v)
                df[label] = [
                    name_key,
                    connector_contract.address,
                    connector_contract.module_name,
                    connector_contract.handler,
                    kwargs,
                    query,
                    connector_contract.params,
                    dpm.get(join(dpm.KEY.connectors_key, name_key, 'modified')) if dpm.is_key(
                        join(dpm.KEY.connectors_key, name_key, 'modified')) else '',
                ]
        if stylise:
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['param'], **{'font-weight': 'bold'})
            return df_style
        return df

    def remove_cleaner(self, cleaner: [str, dict]=None, level: int=None, save: bool=None):
        """ removes part or all the cleaner contract.
                if no parameters then the full cleaner contract is removed
                if only cleaner then all references in all levels of that named cleaner will be removed
                if only level then that level is removed
                if both level and cleaner then that specific cleaner on that level is removed

        :param cleaner: (optional) removes the method contract
        :param level: (optional) removes the level contract
        :param save: if True, save to file. Default is True
        :return True if removed, False if not
        """
        if not isinstance(save, bool):
            save = self._default_save
        self.data_pm.remove_cleaner(cleaner=cleaner, level=level)
        self.persist_contract(save)
        return

    def set_cleaner(self, cleaner_section: dict, level: int=None, save: bool=None):
        """ sets the cleaner section in the yaml configuration file. Note: by default any identical intent, e.g.
        intent with the same intent (name) and the same parameter values, are removed from any level.

        :param cleaner_section: a dictionary type set of configuration representing a cleaner section contract
        :param level: (optional) the level of the cleaner,
                        If None: defualt's to -1
                        if -1: added to a level above any current instance of the cleaner section, level 0 if not found
                        if int: added to the level specified, overwiting any that already exist
        :param save: if True, save to file. Default is True
        """
        if not isinstance(cleaner_section, dict):
            raise ValueError("The cleaner section must be a dictionary. Ensure inplace=True in the passing method")
        if not isinstance(save, bool):
            save = self._default_save
        methods = self.clean.__dir__()
        for cleaner in cleaner_section.keys():
            if cleaner not in methods:
                raise ValueError("The cleaner {} is not recognised as a method".format(cleaner))
        self.data_pm.set_cleaner(cleaner=cleaner_section, level=level)
        self.persist_contract(save)
        return

    def report_cleaners(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        join = self.data_pm.join
        df = pd.DataFrame(columns=['level', 'intent', 'parameters'])
        if isinstance(self.data_pm.cleaners, dict):
            for level in sorted(self.data_pm.cleaners.keys()):
                for cleaner in sorted(self.data_pm.get(join(self.data_pm.KEY.cleaners_key, level))):
                    params = []
                    for key, value in self.data_pm.get(join(self.data_pm.KEY.cleaners_key, level, cleaner)).items():
                        params.append("{}={}".format(key, value))
                    df = df.append({'level': level, 'intent': cleaner, 'parameters': ", ".join(params)},
                                   ignore_index=True)
            df = df.sort_values(['level', 'intent'])
            if stylise:
                index = df[df['level'].duplicated()].index.to_list()
                df.loc[index, 'level'] = ''
                df = df.reset_index(drop=True)
                df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
                _ = df_style.set_properties(subset=['level'],  **{'font-weight': 'bold', 'font-size': "120%"})
                return df_style
        return df

    def set_version(self, version, save=None):
        """ sets the version
        :param version: the version to be set
        :param save: if True, save to file. Default is True
        """
        if not isinstance(save, bool):
            save = self._default_save
            self.data_pm.set_version(version=version)
        self.persist_contract(save)
        return

    def remove_notes(self, note_type: str, label: str=None, save=None):
        """ removes a all entries for a labeled note

        :param note_type: the type of note to delete, if left empyt all notes removed
        :param label: (Optional) the name of the label to be removed
        :param save: (Optional) if True, save to file. Default is True
        :return: True is successful, False if not
        """
        if not isinstance(save, bool):
            save = self._default_save
        self.augment_pm.remove_knowledge(catalogue=note_type, label=label)
        self.persist_contract(save)

    def add_attribute_notes(self, text: str, attribute: str, save=None):
        """ add's information to the contract

        :param attribute: a sub key label to separate different information strands
        :param text: the text to add to the contract info
        :param save: if True, save to file. Default is True
        """
        if not isinstance(save, bool):
            save = self._default_save
        self.augment_pm.add_knowledge(text=text, label=attribute, catalogue='attribute',
                                      only_labels=self._raw_attribute_list)
        self.persist_contract(save)

    def add_notes(self, text: str, label: [str, list]=None, selection: list=None, note_type: str=None, save=None):
        """ add's an overview notes to the contract

        :param text: the text to add to the contract info
        :param label: (optional) a sub key label or list of labels to separate different information strands
        :param selection: (optional) a list of allowed label values, if None then any value allowed
        :param note_type: (optional) the type of note, options are 'attribute' or 'overview', overview if None
        :param save: if True, save to file. Default is True
        """
        if not isinstance(save, bool):
            save = self._default_save

        note_type = 'notes' if not isinstance(note_type, str) else note_type
        label = 'comment' if not isinstance(label, str) else label
        self.augment_pm.add_knowledge(text=text, label=label, catalogue=note_type, only_labels=selection)
        self.persist_contract(save)

    def upload_notes(self, df: pd.DataFrame, label_header: str, text_header: str, note_type: str=None,
                     selection: list=None, save=None):
        """ Allows bulk upload of notes

        :param df: the DataFrame to take the data from
        :param label_header: the column header name for the labels
        :param text_header: the column header name for the text
        :param note_type: the section these notes should be put in
        :param selection: (optional) the limited list of acceptable labels. If not in list then ignored
        :param save: if True, save to file. Default is True
        """
        if label_header not in df.columns:
            raise ValueError("The label header {} can't be found in the DataFrame".format(label_header))
        if text_header not in df.columns:
            raise ValueError("The text header {} can't be found in the DataFrame".format(text_header))
        note_type = 'overview' if not isinstance(note_type, str) else note_type
        data = {}
        for _, row in df.iterrows():
            label = row[label_header]
            text = row[text_header]
            data[label] = text
        self.augment_pm.bulk_upload_knowledge(data, catalogue=note_type, only_labels=selection)
        self.persist_contract(save)

    def report_notes(self, note_type: [str, list]=None, labels: [str, list]=None, regex: [str, list]=None,
                     re_ignore_case: bool=False, stylise: bool=True, date_format: str=None, drop_dates: bool=False):
        """ generates a report on the notes

        :param note_type: (optional) the type of note to filter on, options are 'attribute', 'overview', 'dictionary'
        :param labels: (optional) s label or list of labels to filter on
        :param date_format: (optional) the display date format
        :param regex: a regular expression on the notes
        :param re_ignore_case: if the regular expression should be case sensitive
        :param stylise: (optional) returns a stylised dataframe with formatting
        :param drop_dates: (optional) excludes the 'date' column from the report
        :return: pd.Dataframe
        """
        if isinstance(note_type, (str, list)):
            note_type = self.augment_pm.list_formatter(note_type)
        else:
            note_type = self.augment_pm.catalogue
        if isinstance(labels, (list, str)):
            labels = self.augment_pm.list_formatter(labels)
        for t in note_type:
            if t not in self.augment_pm.catalogue:
                raise ValueError("The note_type {} is not recognised as a Augmented Knowledge type".format(t))
        stylise = True if not isinstance(stylise, bool) else stylise
        drop_dates = False if not isinstance(drop_dates, bool) else drop_dates
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df_final = pd.DataFrame()
        report_data = self.augment_pm.knowledge_filter(catalogue=note_type, label=labels, regex=regex,
                                                       re_ignore_case=re_ignore_case)
        if not isinstance(report_data, dict):
            return pd.DataFrame(columns=['section', 'label', 'date', 'text'])
        for section in report_data.keys():
            df = pd.DataFrame(columns=['section', 'label', 'date', 'text'])
            if report_data.get(section) is not None:
                for label, values in report_data.get(section).items():
                    if labels is not None and label not in labels:
                        continue
                    if isinstance(values, dict):
                        for date, text in values.items():
                            df = df.append({'section': section, 'label': label, 'date': date, 'text': text},
                                           ignore_index=True)
                df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True, yearfirst=True)
                if isinstance(date_format, str):
                    df['date'] = df['date'].dt.strftime(date_format)
                elif stylise:
                    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M')
                df = df.sort_values('label')
                if drop_dates:
                    df.drop('date', axis=1, inplace=True)
                if stylise:
                    index = df[df['section'].duplicated()].index.to_list()
                    df.loc[index, 'section'] = ''
                    df = df.reset_index(drop=True)
                    index = df[df['label'].duplicated()].index.to_list()
                    df.loc[index, 'label'] = ''
                    df = df.reset_index(drop=True)
                df_final = df_final.append(df)
        df_final = df_final.reset_index(drop=True)
        if stylise:
            df_style = df_final.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['section'], **{'font-weight': 'bold'})
            _ = df_style.set_properties(subset=['label', 'section'], **{'font-size': "120%"})
            return df_style
        return df_final

    def load_clean_canonical(self) -> pd.DataFrame:
        """loads the clean pandas.DataFrame from the clean folder for this contract"""
        if self.data_pm.has_connector(self.CONNECTOR_PERSIST):
            handler = self.data_pm.get_connector_handler(self.CONNECTOR_PERSIST)
            df = handler.load_canonical()
            return df
        return pd.DataFrame()

    def refresh_clean_canonical(self) -> pd.DataFrame:
        """loads the clean pandas.DataFrame from the clean folder for this contract"""
        join = self.data_pm.join
        df = self.load_source_canonical()
        for level in sorted(self.data_pm.cleaners.keys()):
            clean_contract = self.data_pm.get(join(self.data_pm.KEY.cleaners_key, level))
            df = self.clean.run_contract_pipeline(df, cleaner_contract=clean_contract)
        handler = self.data_pm.get_connector_handler(self.CONNECTOR_PERSIST)
        handler.persist_canonical(df)
        return df

    def save_clean_canonical(self, df):
        """Saves the pandas.DataFrame to the clean files folder"""
        if self.data_pm.has_connector(self.CONNECTOR_PERSIST):
            handler = self.data_pm.get_connector_handler(self.CONNECTOR_PERSIST)
            handler.persist_canonical(df)
        return

    def remove_clean_canonical(self):
        """removes the current persisted canonical"""
        if self.data_pm.has_connector(self.CONNECTOR_PERSIST):
            handler = self.data_pm.get_connector_handler(self.CONNECTOR_PERSIST)
            handler.remove_canonical()
        return

    def canonical_report(self, df, stylise: bool=True, inc_next_dom: bool=False, report_header: str=None,
                         condition: str=None):
        """The Canonical Report is a data dictionary of the canonical providing a reference view of the dataset's
        attribute properties

        :param df: the DataFrame to view
        :param stylise: if True present the report stylised.
        :param inc_next_dom: (optional) if to include the next dominate element column
        :param report_header: (optional) filter on a header where the condition is true. Condition must exist
        :param condition: (optional) the condition to apply to the header. Header must exist. examples:
                ' > 0.95', ".str.contains('shed')"
        :return:
        """
        return self.discover.data_dictionary(df=df, stylise=stylise, inc_next_dom=inc_next_dom,
                                             report_header=report_header, condition=condition)

    def create_snapshot(self, suffix: str=None, version: str=None, note: str=None, save: bool=None):
        """ creates a snapshot of contracts configuration. The name format will be <contract_name>_#<suffix>.
        To find all snapshot versions use self.data_pm.contract_snapshots
        To get a list of contract snapshots under this root contract use self.data_pm.contract_snapshots
        To recover a snapshot and replace the current contract use self.data_pm.recover_snapshot(snapshot_name)

        :param suffix: (optional) adds the suffix to the end of the contract name. if None then date & time used
        :param version: (optional) changes the version number of the current contract
        :param note: (optional) adds a note to the archived snapshot
        :param save: if True, save to file. Default is True
        :return: a list of current contract snapshots
        """
        if not isinstance(save, bool):
            save = self._default_save
        if note is not None:
            self.add_notes(text=note)
        result = self.data_pm.set_snapshot(suffix)
        if version is not None:
            self.set_version(version=version)
        self.persist_contract(save)
        return result

    def recover_snapshot(self, snapshot_name: str, overwrite: bool=None, save: bool=None) -> bool:
        """ recovers a snapshot back to the current. The snapshot must be from this root contract.
        by default the original root contract will be overwitten unless the overwrite is set to False.
        if overwrite is False a timestamped snapshot is created

        :param snapshot_name:the name of the snapshot (use self.contract_snapshots to get list of names)
        :param overwrite: (optional) if the original contract should be overwritten. Default to True
        :param save: if True, save to file. Default is True
        :return: True if the contract was recovered, else False
        """
        if not isinstance(save, bool):
            save = self._default_save
        result = self.data_pm.recover_snapshot(snapshot_name=snapshot_name, overwrite=overwrite)
        self.persist_contract(save)
        return result

    def delete_snapshot(self, snapshot_name: str, save: bool=None):
        """ deletes a snapshot

        :param snapshot_name: the name of the snapshot
        :param save: if True, save to file. Default is True
        :return: True if successful, False is not found or not deleted
        """
        if not isinstance(save, bool):
            save = self._default_save
        self.data_pm.delete_snapshot(snapshot_name=snapshot_name)
        self.persist_contract(save)
        return

    def backup_canonical(self, connector_name: str, df: [pd.DataFrame, dict]):
        """backup of the contract properties with optional maximum backups, defaults to 10"""
        if self.data_pm.has_connector(connector_name):
            _handler = self._data_pm.get_connector_handler(connector_name)
            _cc = self._data_pm.get_connector_contract(connector_name)
            _address = _cc.parse_address(uri=_cc.uri)
            _path, _, _ext = _address.rpartition('.')
            _new_uri = "{}_{}.{}".format(_path, str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')), _ext)
            _handler.backup_canonical(canonical=df, uri=_new_uri)
        return

    def persist_contract(self, save=None):
        """Saves the current configuration to file"""
        if not isinstance(save, bool):
            save = self._default_save
        if save:
            self.data_pm.persist_properties()
            self.augment_pm.persist_properties()
        return

    def get_persist_file_name(self):
        """ Returns a persist pattern based on name"""
        _pattern = "transition_{}_{}.pickle"
        return _pattern.format(self.contract_name, self.version)

    def get_report_file_name(self, prefix: str):
        """ Returns a report pattern based on name"""
        _pattern = "{}_{}_{}.xlsx"
        return _pattern.format(prefix, self.contract_name, self.version)

    def get_visual_file_name(self, prefix: str, ref_name: str) -> str:
        """ The visual file name with a reference name
        :param prefix: the prefix name of the file
        :param ref_name: the name of the specific visualisation
        """
        _pattern = "{}}_{}_{}_{}.png"
        return _pattern.format(prefix, self.contract_name, ref_name, self.version)

    @staticmethod
    def list_formatter(value) -> [list, None]:
        """ Useful utility method to convert any type of str, list, tuple or pd.Series into a list"""
        if isinstance(value, (int, float, str, pd.Timestamp)):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, dict):
            return list(value.items())
        return None
