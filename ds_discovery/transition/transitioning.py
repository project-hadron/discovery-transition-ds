import os
import pandas as pd

from ds_foundation.managers.augment_properties import AugmentedPropertyManager
from ds_foundation.managers.data_properties import DataPropertyManager
from ds_foundation.handlers.abstract_handlers import ConnectorContract
from ds_discovery.intent.pandas_cleaners import PandasCleaners
from ds_discovery.transition.discovery import DataDiscovery, Visualisation

__author__ = 'Darryl Oatridge'


class TransitionAgent(object):

    ORIGIN_CONNECTOR = 'origin_connector'
    PERSIST_CONNECTOR = 'persist_connector'
    PM_DATA_CONNECTOR: str
    PM_AUGMENT_CONNECTOR: str
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
        self.PM_DATA_CONNECTOR = self._data_pm.CONTRACT_CONNECTOR
        if self._data_pm.has_persisted_properties():
            self._data_pm.load_properties()
        self._knowledge_catalogue = ['overview', 'notes', 'observations', 'attribute', 'dictionary', 'tor']
        self._augment_pm = AugmentedPropertyManager.from_properties(self._contract_name,
                                                                    connector_contract=augment_properties,
                                                                    knowledge_catalogue=self._knowledge_catalogue)
        self.PM_AUGMENT_CONNECTOR = self.augment_pm.CONTRACT_CONNECTOR
        if self._augment_pm.has_persisted_properties():
            self._augment_pm.load_properties()
        # initialise the values
        self.persist_contract(save=self._default_save)
        self._raw_attribute_list = []

    @classmethod
    def from_remote(cls, contract_name: str, location: str=None, vertical: str=None, default_save=None):
        """ Class Factory Method that builds the connector handlers from the default remote.
        This assumes the use of the pandas handler module and pickle persistence on a remote default.

         :param contract_name: The reference name of the properties contract
         :param location: (optional) the location or bucket where the data resource can be found
         :param vertical: (optional) the name of the discovery vertical. default to 'scratch'
                    Options include 'client', 'discovery', 'synthetic' or 'scratch'
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :return: the initialised class instance
         """
        for param in ['contract_name']:
            if not isinstance(eval(param), str) or len(eval(param)) == 0:
                raise ValueError("a {} must be provided".format(param))
        _default_save = default_save if isinstance(default_save, bool) else True
        _vertical = 'scratch' if not isinstance(vertical, str) else vertical
        _module_name = 'ds_connectors.handlers.aws_s3_handlers'
        _location = 'discovery-persistence' if not isinstance(location, str) else location
        _resource_prefix = "{v}/contract/{n}/config_transition_data_{n}.pickle".format(v=_vertical, n=contract_name)
        _data_connector = ConnectorContract(resource=_resource_prefix, connector_type='pickle',
                                            location=_location, module_name=_module_name,
                                            handler='AwsS3PersistHandler')

        _resource_prefix = "{v}/contract/{n}/config_transition_augment_{n}.pickle".format(v=_vertical, n=contract_name)
        _augment_connector = ConnectorContract(resource=_resource_prefix, connector_type='pickle',
                                               location=_location, module_name=_module_name,
                                               handler='AwsS3PersistHandler')
        rtn_cls = cls(contract_name=contract_name, data_properties=_data_connector,
                      augment_properties=_augment_connector, default_save=default_save)
        rtn_cls.MODULE_NAME = _module_name
        rtn_cls.HANDLER_SOURCE = 'AwsS3SourceHandler'
        rtn_cls.HANDLER_PERSIST = 'AwsS3PersistHandler'
        if not rtn_cls.data_pm.has_connector(rtn_cls.PERSIST_CONNECTOR):
            _resource = "{v}/persist/transition/{f}".format(v=_vertical, f=rtn_cls.get_persist_file_name('transition'))
            rtn_cls.set_persist_contract(resource=_resource, connector_type='pickle')
        return rtn_cls

    @classmethod
    def from_remote_client(cls, contract_name: str, location: str=None, default_save=None):
        """ Class Factory Method that builds the connector handlers from the default remote.
        This assumes the use of the pandas handler module and pickle persistence on a remote default.

         :param contract_name: The reference name of the properties contract
         :param location: (optional) the location or bucket where the data resource can be found
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :return: the initialised class instance
         """
        return cls.from_remote(contract_name=contract_name, location=location, vertical='client',
                               default_save=default_save)

    @classmethod
    def from_remote_discovery(cls, contract_name: str, location: str=None, default_save=None):
        """ Class Factory Method that builds the connector handlers from the default remote.
        This assumes the use of the pandas handler module and pickle persistence on a remote default.

         :param contract_name: The reference name of the properties contract
         :param location: (optional) the location or bucket where the data resource can be found
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :return: the initialised class instance
         """
        return cls.from_remote(contract_name=contract_name, location=location, vertical='discovery',
                               default_save=default_save)

    @classmethod
    def from_remote_synthetic(cls, contract_name: str, location: str=None, default_save=None):
        """ Class Factory Method that builds the connector handlers from the default remote.
        This assumes the use of the pandas handler module and pickle persistence on a remote default.

         :param contract_name: The reference name of the properties contract
         :param location: (optional) the location or bucket where the data resource can be found
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :return: the initialised class instance
         """
        return cls.from_remote(contract_name=contract_name, location=location, vertical='synthetic',
                               default_save=default_save)

    @classmethod
    def from_path(cls, contract_name: str,  contract_path: str, default_save=None):
        """ Class Factory Method that builds the connector handlers from the data paths.
        This assumes the use of the pandas handler module and yaml persisted file.

        :param contract_name: The reference name of the properties contract
        :param contract_path: (optional) the path of the properties contracts
        :param default_save: (optional) if the configuration should be persisted
        :return: the initialised class instance
        """
        for param in ['contract_name', 'contract_path']:
            if not isinstance(eval(param), str) or len(eval(param)) == 0:
                raise ValueError("a {} must be provided".format(param))
        _default_save = default_save if isinstance(default_save, bool) else True
        _module_name = 'ds_discovery.handlers.pandas_handlers'
        _location = os.path.join(contract_path, contract_name)
        _data_connector = ConnectorContract(resource="config_transition_data_{}.yaml".format(contract_name),
                                            connector_type='yaml', location=_location, module_name=_module_name,
                                            handler='PandasPersistHandler')
        _augment_connector = ConnectorContract(resource="config_transition_augment_{}.yaml".format(contract_name),
                                               connector_type='yaml', location=_location, module_name=_module_name,
                                               handler='PandasPersistHandler')
        rtn_cls = cls(contract_name=contract_name, data_properties=_data_connector,
                      augment_properties=_augment_connector, default_save=default_save)
        rtn_cls.MODULE_NAME = _module_name
        rtn_cls.HANDLER_SOURCE = 'PandasSourceHandler'
        rtn_cls.HANDLER_PERSIST = 'PandasPersistHandler'
        if not rtn_cls.data_pm.has_connector(rtn_cls.PERSIST_CONNECTOR):
            _resource = rtn_cls.get_persist_file_name('transition')
            rtn_cls.set_persist_contract(resource=_resource, connector_type='pickle')
        return rtn_cls

    @classmethod
    def from_env(cls, contract_name: str,  default_save=None):
        """ Class Factory Method that builds the connector handlers taking the property contract path from
        the os.envon['TR_CONTRACT_PATH'] or locally from the current working directory 'dtu/contracts' if
        no environment variable is found. This assumes the use of the pandas handler module and yaml persisted file.

         :param contract_name: The reference name of the properties contract
         :param default_save: (optional) if the configuration should be persisted
         :return: the initialised class instance
         """
        if 'TR_CONTRACT_PATH' in os.environ.keys():
            contract_path = os.environ['TR_CONTRACT_PATH']
        elif 'DTU_CONTRACT_PATH' in os.environ.keys(): # Legacy
            contract_path = os.environ['DTU_CONTRACT_PATH']
        else:
            contract_path = os.path.join(os.getcwd(), 'dtu', 'contracts')
        return cls.from_path(contract_name=contract_name, contract_path=contract_path, default_save=default_save)

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
        if self.data_pm.CONTRACT_CONNECTOR in connector_list:
            connector_list.remove(self.data_pm.CONTRACT_CONNECTOR)
        if len(connector_list) > 0 or self.data_pm.has_cleaners() or len(self.data_pm.snapshots) > 0:
            return False
        return True

    def is_source_modified(self):
        """Test if the source file is modified since last load"""
        return self.data_pm.is_modified(self.ORIGIN_CONNECTOR)

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
        self.data_pm.remove_connector_contract(self.ORIGIN_CONNECTOR)
        self._raw_attribute_list = []
        self.persist_contract(save)

    def set_source_contract(self, resource: str, connector_type: str, location: str, module_name: str, handler: str,
                            load: bool=False, save: bool=None, **kwargs) -> [pd.DataFrame, None]:
        """ Sets the source contract, returning the source data as a DataFrame if load=True. If the connection
        module_name and/or handler is not provided the the default properties connection setting are used

        :param resource: a local file, connector, URI or URL
        :param location: a path, region or uri reference that can be used to identify location of resource
        :param connector_type:  a reference to the type of resource. if None then csv file assumed
        :param module_name: a module name with full package path
        :param handler: the name of the Handler Class within the module.
        :param load: (optional) if True,` attempts to read the given file or source and returns a pandas.DataFrame
        :param save: (optional) if True, save to file. Default is True
        :param kwargs: (optional) a list of key additional word argument properties associated with the resource
        :return: if load is True, returns a Pandas.DataFrame else None
        """
        save = save if isinstance(save, bool) else self._default_save
        load = load if isinstance(load, bool) else False
        if not isinstance(location, str):
            location = self.data_pm.get_connector_contract(self.PM_DATA_CONNECTOR).location
        if not isinstance(module_name, str):
            module_name = self.data_pm.get_connector_contract(self.PM_DATA_CONNECTOR).module_name
        if not isinstance(handler, str):
            handler = self.data_pm.get_connector_contract(self.PM_DATA_CONNECTOR).handler
        self.data_pm.set_connector_contract(self.ORIGIN_CONNECTOR, resource=resource, connector_type=connector_type,
                                            location=location, module_name=module_name, handler=handler, **kwargs)
        self.persist_contract(save)
        if load:
            return self.load_source_canonical()
        return

    def set_persist_contract(self, resource: str, connector_type: str, location: str=None,
                             module_name: str=None, handler: str=None, save: bool=None, **kwargs):
        """ Sets the persist contract. For parameters not provided the default resource name and data properties
        connector contract module and handler are used.

        :param resource: a local file, connector, URI or URL
        :param connector_type: a reference to the type of resource. if None then csv file assumed
        :param location: (optional) a path, region or uri reference that can be used to identify location of resource
        :param module_name: a module name with full package path e.g 'ds_discovery.handlers.pandas_handlers
        :param handler: the name of the Handler Class. Must be
        :param save: if True, save to file. Default is True
        :param kwargs: (optional) a list of key additional word argument properties associated with the resource
        :return: if load is True, returns a Pandas.DataFrame else None
        """
        save = save if isinstance(save, bool) else self._default_save
        if not isinstance(location, str):
            location = self.data_pm.get_connector_contract(self.PM_DATA_CONNECTOR).location
        if not isinstance(module_name, str):
            module_name = self.data_pm.get_connector_contract(self.PM_DATA_CONNECTOR).module_name
        if not isinstance(handler, str):
            handler = self.data_pm.get_connector_contract(self.PM_DATA_CONNECTOR).handler
        if self.data_pm.has_connector(self.PERSIST_CONNECTOR):
            self.data_pm.remove_connector_contract(self.PERSIST_CONNECTOR)
        self.data_pm.set_connector_contract(self.PERSIST_CONNECTOR, resource=resource, connector_type=connector_type,
                                            location=location, module_name=module_name, handler=handler, **kwargs)
        self.persist_contract(save)
        return

    def load_source_canonical(self) -> pd.DataFrame:
        """returns the contracted source data as a DataFrame"""
        if self.data_pm.has_connector(self.ORIGIN_CONNECTOR):
            handler = self.data_pm.get_connector_handler(self.ORIGIN_CONNECTOR)
            df = handler.load_canonical()
            if isinstance(df, dict):
                df = pd.DataFrame(df)
            if len(df.columns) > 0:
                self._raw_attribute_list = df.columns.to_list()
            self.data_pm.set_modified(self.ORIGIN_CONNECTOR, handler.get_modified())
            return df
        return pd.DataFrame()

    def report_source(self, connector_name: str=None, stylise: bool=True):
        """ generates a report on the source contract

        :param connector_name: (optional) filters on the connector name. Aliases can be used instead of the default
                raw data source (self.ORIGIN_SOURCE):  'source', 'data', 'dataset', 'origin', 'raw'
                persisted source (self.PERSIST_SOURCE): 'persist', 'canonical', 'transition'
                properties source (self.data_pm.PROPERTY_SOURCE): 'properties', 'property', 'props', 'config'
        :param stylise: (optional) returns a stylised dataframe with formatting
        :return: pd.DataFrame
        """
        if connector_name in ['source', 'data', 'dataset', 'origin', 'raw']:
            connector_name = self.ORIGIN_CONNECTOR
        if connector_name in ['persist', 'canonical', 'transition']:
            connector_name = self.PERSIST_CONNECTOR
        if connector_name in ['properties', 'property', 'props', 'config']:
            connector_name = self.data_pm.CONTRACT_CONNECTOR
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame()
        join = self.data_pm.join
        dpm = self.data_pm
        df['param'] = ['connector_name', 'resource', 'connector_type', 'location', 'module_name',
                       'handler', 'modified', 'kwargs']
        for name_key in dpm.get(join(dpm.KEY.connectors_key)).keys():
            if isinstance(connector_name, str) and connector_name != name_key:
                continue
            connector_contract = dpm.get_connector_contract(name_key)
            if isinstance(connector_contract, ConnectorContract):
                if name_key == self.ORIGIN_CONNECTOR:
                    label = 'Data Source'
                elif name_key == self.PERSIST_CONNECTOR:
                    label = 'Persist Source'
                elif name_key == self.data_pm.CONTRACT_CONNECTOR:
                    label = 'Property Source'
                else:
                    label = name_key
                kwargs = ''
                if isinstance(connector_contract.kwargs, dict):
                    for k, v in connector_contract.kwargs.items():
                        if len(kwargs) > 0:
                            kwargs += "  "
                        kwargs += "{}='{}'".format(k, v)
                df[label] = [
                    name_key,
                    connector_contract.resource,
                    connector_contract.connector_type,
                    connector_contract.location,
                    connector_contract.module_name,
                    connector_contract.handler,
                    dpm.get(join(dpm.KEY.connectors_key, name_key, 'modified')) if dpm.is_key(
                        join(dpm.KEY.connectors_key, name_key, 'modified')) else '',
                    kwargs
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
        if self.data_pm.has_connector(self.PERSIST_CONNECTOR):
            handler = self.data_pm.get_connector_handler(self.PERSIST_CONNECTOR)
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
        handler = self.data_pm.get_connector_handler(self.PERSIST_CONNECTOR)
        handler.persist_canonical(df)
        return df

    def save_clean_canonical(self, df):
        """Saves the pandas.DataFrame to the clean files folder"""
        if self.data_pm.has_connector(self.PERSIST_CONNECTOR):
            handler = self.data_pm.get_connector_handler(self.PERSIST_CONNECTOR)
            handler.persist_canonical(df)
        return

    def remove_clean_canonical(self):
        """removes the current persisted canonical"""
        if self.data_pm.has_connector(self.PERSIST_CONNECTOR):
            handler = self.data_pm.get_connector_handler(self.PERSIST_CONNECTOR)
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

    def backup_contract(self, max_backups=None):
        """backup of the contract properties with optional maximum backups, defaults to 10"""
        if self.data_pm.has_connector(self.data_pm.CONTRACT_CONNECTOR):
            _handler = self.data_pm.get_connector_handler(connector_name=self.data_pm.CONTRACT_CONNECTOR)
            _handler.backup_canonical(max_backups=max_backups)
        return

    def persist_contract(self, save=None):
        """Saves the current configuration to file"""
        if not isinstance(save, bool):
            save = self._default_save
        if save:
            self.data_pm.persist_properties()
            self.augment_pm.persist_properties()
        return

    def get_persist_file_name(self, prefix: str):
        """ Returns a persist pattern based on name"""
        _pattern = "{}_{}_{}.pickle"
        return _pattern.format(prefix, self.contract_name, self.version)

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
