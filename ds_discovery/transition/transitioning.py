import pandas as pd
import uuid
import ds_discovery
from aistac.handlers.abstract_handlers import ConnectorContract
from aistac.components.abstract_component import AbstractComponent
from ds_discovery.intent.transition_intent import TransitionIntentModel
from ds_discovery.managers.transition_property_manager import TransitionPropertyManager
from ds_discovery.transition.commons import Commons
from ds_discovery.transition.discovery import DataDiscovery, Visualisation

__author__ = 'Darryl Oatridge'


class Transition(AbstractComponent):

    CONNECTOR_SOURCE = 'primary_source'
    CONNECTOR_PERSIST = 'primary_persist'
    CONNECTOR_DICTIONARY = 'dictionary'
    CONNECTOR_REPORT = 'report'

    def __init__(self, property_manager: TransitionPropertyManager, intent_model: TransitionIntentModel,
                 default_save=None, reset_templates: bool=None, align_connectors: bool=None):
        """ Encapsulation class for the transition set of classes

        :param property_manager: The contract property manager instance for this component
        :param intent_model: the model codebase containing the parameterizable intent
        :param default_save: The default behaviour of persisting the contracts:
                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        :param reset_templates: (optional) reset connector templates from environ variables (see `report_environ()`)
        :param align_connectors: (optional) resets aligned connectors to the template
        """
        super().__init__(property_manager=property_manager, intent_model=intent_model, default_save=default_save,
                         reset_templates=reset_templates, align_connectors=align_connectors)
        self._raw_attribute_list = []

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, pm_file_type: str=None, pm_module: str=None,
                 pm_handler: str=None, pm_kwargs: dict=None, default_save=None, reset_templates: bool=None,
                 align_connectors: bool=None, default_save_intent: bool=None, default_intent_level: bool=None,
                 order_next_available: bool=None, default_replace_intent: bool=None):
        """ Class Factory Method to instantiates the components application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.
        See class inline docs for an example method

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param pm_file_type: (optional) defines a specific file type for the property manager
         :param pm_module: (optional) the module or package name where the handler can be found
         :param pm_handler: (optional) the handler for retrieving the resource
         :param pm_kwargs: (optional) a dictionary of kwargs to pass to the property manager
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :param reset_templates: (optional) reset connector templates from environ variables. Default True
                                (see `report_environ()`)
         :param align_connectors: (optional) resets aligned connectors to the template. default Default True
         :param default_save_intent: (optional) The default action for saving intent in the property manager
         :param default_intent_level: (optional) the default level intent should be saved at
         :param order_next_available: (optional) if the default behaviour for the order should be next available order
         :param default_replace_intent: (optional) the default replace existing intent behaviour
         :return: the initialised class instance
         """
        pm_file_type = pm_file_type if isinstance(pm_file_type, str) else 'json'
        pm_module = pm_module if isinstance(pm_module, str) else 'ds_discovery.handlers.pandas_handlers'
        pm_handler = pm_handler if isinstance(pm_handler, str) else 'PandasPersistHandler'
        _pm = TransitionPropertyManager(task_name=task_name)
        _intent_model = TransitionIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
                                              default_intent_level=default_intent_level,
                                              order_next_available=order_next_available,
                                              default_replace_intent=default_replace_intent)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, pm_file_type=pm_file_type,
                                 pm_module=pm_module, pm_handler=pm_handler, pm_kwargs=pm_kwargs)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save,
                   reset_templates=reset_templates, align_connectors=align_connectors)

    @classmethod
    def _from_remote_s3(cls) -> (str, str):
        """ Class Factory Method that builds the connector handlers for Amazon AWS S3 """
        _module_name = 'ds_connectors.handlers.aws_s3_handlers'
        _handler = 'AwsS3PersistHandler'
        return _module_name, _handler

    @classmethod
    def _from_remote_redis(cls) -> (str, str):
        """ Class Factory Method that builds the connector handlers for Redis."""
        _module_name = 'ds_connectors.handlers.redis_handlers'
        _handler = 'RedisPersistHandler'
        return _module_name, _handler

    @classmethod
    def discovery_pad(cls):
        """ A class method to use the Components discovery methods as a scratch pad"""
        return DataDiscovery()

    @classmethod
    def visual_pad(cls):
        """ A class method to use the Components visualisation methods as a scratch pad"""
        return Visualisation()

    @property
    def intent_model(self) -> TransitionIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> TransitionPropertyManager:
        """The properties manager instance"""
        return self._component_pm

    @property
    def discover(self) -> DataDiscovery:
        """The transition instance"""
        return DataDiscovery()

    @property
    def visual(self) -> Visualisation:
        """The visualisation instance"""
        return Visualisation()

    def set_provenance(self, title: str=None, domain: str=None, description: str=None, save: bool=None):
        """sets some fixed transition knowledge """

        def _set_item(name: str, value: str):
            value = value if isinstance(value, str) else ""
            if self.pm.has_knowledge(catalog='transition', label=name):
                self.pm.remove_knowledge(catalog='transition', label=name)
            self.pm.set_knowledge(catalog='transition', label=name, text=value)

        _set_item('title', title)
        _set_item('domain', domain)
        _set_item('description', description)
        self.pm_persist(save=save)

    def is_source_modified(self):
        """Test if the source file is modified since last load"""
        return self.pm.is_connector_modified(self.CONNECTOR_SOURCE)

    def get_persist_contract(self):
        """ gets the persist connector contract that can be used as the next chain source"""
        return self.pm.get_connector_contract(self.CONNECTOR_PERSIST)

    def set_source_contract(self, connector_contract: ConnectorContract, save: bool=None):
        """ Sets the source contract

        :param connector_contract: a Connector Contract for the source data
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_contract(self.CONNECTOR_SOURCE, connector_contract=connector_contract, save=save)
        return

    def set_persist_contract(self, connector_contract: ConnectorContract, save: bool=None):
        """ Sets the persist contract.

        :param connector_contract: a Connector Contract for the persisted data
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_contract(self.CONNECTOR_PERSIST, connector_contract=connector_contract, save=save)
        return

    def set_dictionary_contract(self, connector_contract: ConnectorContract, save: bool=None):
        """ Sets the dictionary contract

        :param connector_contract: a Connector Contract for the persisted data
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_contract(self.CONNECTOR_DICTIONARY, connector_contract=connector_contract, save=save)
        return

    def set_source(self, uri_file: str, save: bool=None, **kwargs):
        """sets the source contract CONNECTOR_SOURCE using the TEMPLATE_SOURCE connector contract,

        :param uri_file: the uri_file is appended to the template path
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_from_template(connector_name=self.CONNECTOR_SOURCE, uri_file=uri_file,
                                         template_name=self.TEMPLATE_SOURCE, save=save, **kwargs)

    def set_persist(self, uri_file: str=None, save: bool=None, **kwargs):
        """sets the persist contract CONNECTOR_PERSIST using the TEMPLATE_PERSIST connector contract

        :param uri_file: (optional) the uri_file is appended to the template path
        :param save: (optional) if True, save to file. Default is True
        """
        file_pattern = self.pm.file_pattern(connector_name=self.CONNECTOR_PERSIST)
        uri_file = uri_file if isinstance(uri_file, str) else file_pattern
        self.add_connector_from_template(connector_name=self.CONNECTOR_PERSIST, uri_file=uri_file,
                                         template_name=self.TEMPLATE_PERSIST, save=save, **kwargs)

    def set_dictionary(self, uri_file: str=None, save: bool=None, **kwargs):
        """sets the persist contract CONNECTOR_PERSIST using the TEMPLATE_PERSIST connector contract

        :param uri_file: (optional) the uri_file is appended to the template path
        :param save: (optional) if True, save to file. Default is True
        """
        file_pattern = self.pm.file_pattern(connector_name=self.CONNECTOR_DICTIONARY, file_type='csv', versioned=True)
        uri_file = uri_file if isinstance(uri_file, str) else file_pattern
        self.add_connector_from_template(connector_name=self.CONNECTOR_DICTIONARY, uri_file=uri_file,
                                         template_name=self.TEMPLATE_PERSIST, save=save, **kwargs)

    def load_source_canonical(self, **kwargs) -> pd.DataFrame:
        """returns the contracted source data as a DataFrame """
        return self.load_canonical(self.CONNECTOR_SOURCE, **kwargs)

    def load_clean_canonical(self, **kwargs) -> pd.DataFrame:
        """loads the clean pandas.DataFrame from the clean folder for this contract"""
        return self.load_canonical(self.CONNECTOR_PERSIST, **kwargs)

    def load_dictionary(self, **kwargs) -> pd.DataFrame:
        """loads the clean pandas.DataFrame from the dictionary folder for this contract"""
        return self.load_canonical(self.CONNECTOR_DICTIONARY, **kwargs)

    def load_canonical(self, connector_name: str, **kwargs) -> pd.DataFrame:
        """returns the canonical of the referenced connector

        :param connector_name: the name or label to identify and reference the connector
        """
        canonical = super().load_canonical(connector_name=connector_name, **kwargs)
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical, orient='columns')
        return canonical

    def save_clean_canonical(self, df, **kwargs):
        """Saves the pandas.DataFrame to the clean files folder"""
        self.persist_canonical(connector_name=self.CONNECTOR_PERSIST, canonical=df, **kwargs)

    def save_dictionary(self, df, **kwargs):
        """Saves the pandas.DataFrame to the dictionary folder"""
        self.persist_canonical(connector_name=self.CONNECTOR_DICTIONARY, canonical=df, **kwargs)

    def run_transition_pipeline(self, intent_levels: [str, int, list]=None):
        """Runs the transition pipeline from source to persist"""
        canonical = self.load_source_canonical()
        result = self.intent_model.run_intent_pipeline(canonical, intent_levels=intent_levels, inplace=False)
        self.save_clean_canonical(result)

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

    def report_attributes(self, df, stylise: bool=True):
        labels = [f'Attributes ({len(df.columns)})', 'dType', 'Description']
        file = []
        for c in df.columns.sort_values().values:
            line = [c, str(df[c].dtype),
                    ". ".join(self.pm.report_notes(catalog='attributes', labels=c, drop_dates=True).get('text', []))]
            file.append(line)
        df_dd = pd.DataFrame(file, columns=labels)
        if stylise:
            style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                     {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
            df_style = df_dd.style.set_table_styles(style)
            _ = df_style.applymap(self._dtype_color, subset=['dType'])
            _ = df_style.set_properties(subset=['Description'],  **{"text-align": "left"})
            _ = df_style.set_properties(subset=[f'Attributes ({len(df.columns)})'],  **{'font-weight': 'bold',
                                                                                        'font-size': "120%"})
            return df_style
        df_dd.set_index(keys=f'Attributes ({len(df.columns)})', inplace=True)
        return df_dd

    def report_connectors(self, connector_filter: [str, list]=None, stylise: bool=True):
        """ generates a report on the source contract

        :param connector_filter: (optional) filters on the connector name.
        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_connectors(connector_filter=connector_filter), orient='columns')
        if stylise:
            Commons.report(df, index_header='connector_name')
        df.set_index(keys='connector_name', inplace=True)
        return df

    def report_run_book(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_run_book(), orient='columns')
        if stylise:
            Commons.report(df, index_header='name')
        df.set_index(keys='name', inplace=True)
        return df

    def report_intent(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(), orient='columns')
        if stylise:
            Commons.report(df, index_header='level')
        df.set_index(keys='level', inplace=True)
        return df

    def report_notes(self, catalog: [str, list]=None, labels: [str, list]=None, regex: [str, list]=None,
                     re_ignore_case: bool=False, stylise: bool=True, drop_dates: bool=False):
        """ generates a report on the notes

        :param catalog: (optional) the catalog to filter on
        :param labels: (optional) s label or list of labels to filter on
        :param regex: (optional) a regular expression on the notes
        :param re_ignore_case: (optional) if the regular expression should be case sensitive
        :param stylise: (optional) returns a stylised dataframe with formatting
        :param drop_dates: (optional) excludes the 'date' column from the report
        :return: pd.Dataframe
        """
        report = self.pm.report_notes(catalog=catalog, labels=labels, regex=regex, re_ignore_case=re_ignore_case,
                                      drop_dates=drop_dates)
        df = pd.DataFrame.from_dict(data=report, orient='columns')
        if stylise:
            Commons.report(df, index_header='section', bold='label')
        df.set_index(keys='section', inplace=True)
        return df

    def transition_report(self, df: pd.DataFrame, analytics: dict=None) -> dict:
        """A complete report of the transition"""
        report = {'meta-data': {}, 'transition': {}, 'provenance': {}, 'dictionary': {}, 'fields': {}, 'analysis': {}}
        # meta
        report['meta-data'].update({'uid': str(uuid.uuid4()),
                                    'component': self.pm.manager_name(),
                                    'task': self.pm.task_name,
                                    'created': str(pd.Timestamp.now()),
                                    'version': ds_discovery.__version__})
        # dataset-facts
        _source_connector = self.pm.get_connector_contract(connector_name=self.CONNECTOR_SOURCE)
        _source_dict = {}
        if isinstance(_source_connector, ConnectorContract):
            kwargs = ''
            if isinstance(_source_connector.raw_kwargs, dict):
                for k, v in _source_connector.raw_kwargs.items():
                    if len(kwargs) > 0:
                        kwargs += "  "
                    kwargs += f"{k}='{v}'"
            query = ''
            if isinstance(_source_connector.query, dict):
                for k, v in _source_connector.query.items():
                    if len(query) > 0:
                        query += "  "
                    query += f"{k}='{v}'"
            _source_dict['uri'] = _source_connector.raw_uri
            _source_dict['version'] = _source_connector.version
            _source_dict['kwargs'] = kwargs
            _source_dict['query'] = query

        report['transition'].update({'description': self.pm.description,
                                     'source': _source_dict,
                                     'info': {'rows': df.shape[0], 'columns': df.shape[1],
                                              'memory': Commons.bytes2human(df.memory_usage(deep=True).sum())}})
        report['provenance'].update(self.pm.get_knowledge(catalog='transition'))
        report['dictionary'].update(self.discover.data_dictionary(df=df, inc_next_dom=True).to_dict())
        # field descriptions
        _fields = {}
        for c in df.columns.sort_values().values:
            _fields[c] = self.pm.get_knowledge(catalog='attributes', label=c, as_list=True)
        report['fields'].update(_fields)
        # analysis
        if not isinstance(analytics, dict):
            analytics = self.discover.analyse_association(df, columns_list=df.columns.to_list())
        report['analysis'].update(analytics)
        return report

    def upload_attributes(self, canonical: pd.DataFrame, label_key: str, text_key: str, constraints: list=None,
                          save=None):
        """ Allows bulk upload of notes. Assumes a dictionary of key value pairs where the key is the
        label and the value the text

        :param canonical: a DataFrame of where the key is the label and value is the text
        :param label_key: the dictionary key name for the labels
        :param text_key: the dictionary key name for the text
        :param constraints: (optional) the limited list of acceptable labels. If not in list then ignored
        :param save: if True, save to file. Default is True
        """
        super().upload_notes(canonical=canonical.to_dict(orient='list'), catalog='attributes', label_key=label_key,
                             text_key=text_key, constraints=constraints, save=save)

    def upload_notes(self, canonical: pd.DataFrame, catalog: str, label_key: str, text_key: str, constraints: list=None,
                     save=None):
        """ Allows bulk upload of notes. Assumes a dictionary of key value pairs where the key is the
        label and the value the text

        :param canonical: a DataFrame of where the key is the label and value is the text
        :param catalog: the section these notes should be put in
        :param label_key: the dictionary key name for the labels
        :param text_key: the dictionary key name for the text
        :param constraints: (optional) the limited list of acceptable labels. If not in list then ignored
        :param save: if True, save to file. Default is True
        """
        super().upload_notes(canonical=canonical.to_dict(orient='list'), catalog=catalog, label_key=label_key,
                             text_key=text_key, constraints=constraints, save=save)

    @staticmethod
    def _dtype_color(dtype: str):
        """Apply color to types"""
        if str(dtype).startswith('cat'):
            color = '#208a0f'
        elif str(dtype).startswith('int'):
            color = '#0f398a'
        elif str(dtype).startswith('float'):
            color = '#2f0f8a'
        elif str(dtype).startswith('date'):
            color = '#790f8a'
        elif str(dtype).startswith('bool'):
            color = '#08488e'
        elif str(dtype).startswith('str'):
            color = '#761d38'
        else:
            return ''
        return 'color: %s' % color
