import pandas as pd

from aistac.handlers.abstract_handlers import ConnectorContract
from aistac.components.abstract_component import AbstractComponent

from ds_discovery.intent.transition_intent import TransitionIntentModel
from ds_discovery.managers.transition_property_manager import TransitionPropertyManager
from ds_discovery.transition.commons import Commons
from ds_discovery.transition.discovery import DataDiscovery, Visualisation

__author__ = 'Darryl Oatridge'


class Transition(AbstractComponent):

    CONNECTOR_SOURCE = 'source_connector'
    CONNECTOR_PERSIST = 'persist_connector'

    def __init__(self, property_manager: TransitionPropertyManager, intent_model: TransitionIntentModel,
                 default_save=None):
        """ Encapsulation class for the discovery set of classes

        :param property_manager: The contract property manager instance for this component
        :param intent_model: the model codebase containing the parameterizable intent
        :param default_save: The default behaviour of persisting the contracts:
                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        """
        super().__init__(property_manager=property_manager, intent_model=intent_model, default_save=default_save)
        self._raw_attribute_list = []

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, pm_file_type: str=None, pm_module: str=None,
                 pm_handler: str=None, default_save=None, template_source_path: str=None,
                 template_persist_path: str=None, template_source_module: str=None,
                 template_persist_module: str=None, template_source_handler: str=None,
                 template_persist_handler: str=None, **kwargs):
        """ Class Factory Method to instantiates the component application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.

        by default the handler is local Pandas but also supports remote AWS S3 and Redis. It use these Factory
        instantiations ensure that the schema is s3:// or redis:// and the handler will be automatically redirected

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param pm_file_type: (optional) defines a specific file type for the property manager
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :param pm_module: (optional) the module or package name where the handler can be found
         :param pm_handler: (optional) the handler for retrieving the resource
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :param template_source_path: (optional) a default source root path for the source canonicals
         :param template_persist_path: (optional) a default source root path for the persisted canonicals
         :param template_source_module: (optional) a default module package path for the source handlers
         :param template_persist_module: (optional) a default module package path for the persist handlers
         :param template_source_handler: (optional) a default read only source handler
         :param template_persist_handler: (optional) a default read write persist handler
         :param kwargs: to pass to the connector contract
         :return: the initialised class instance
         """
        _pm = TransitionPropertyManager(task_name=task_name)
        _intent_model = TransitionIntentModel(property_manager=_pm)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, **kwargs)
        super()._add_templates(property_manager=_pm, source_path=template_source_path, save=default_save,
                               persist_path=template_persist_path, source_module=template_source_module,
                               persist_module=template_persist_module, source_handler=template_source_handler,
                               persist_handler=template_persist_handler)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save)

    @classmethod
    def _from_remote_s3(cls) -> (str, str):
        """ Class Factory Method that builds the connector handlers an Amazon AWS s3 remote store."""
        _module_name = 'ds_connectors.handlers.aws_s3_handlers'
        _handler = 'AwsS3PersistHandler'
        return _module_name, _handler

    @classmethod
    def _from_remote_redis(cls) -> (str, str):
        """ Class Factory Method that builds the connector handlers an Amazon AWS s3 remote store."""
        _module_name = 'ds_connectors.handlers.redis_handlers'
        _handler = 'RedisPersistHandler'
        return _module_name, _handler

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
        """The discovery instance"""
        return DataDiscovery()

    @property
    def visual(self) -> Visualisation:
        """The visualisation instance"""
        return Visualisation()

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

    def set_source(self, uri_file: str, save: bool=None):
        """sets the source contract CONNECTOR_SOURCE using the TEMPLATE_SOURCE connector contract,

        :param uri_file: the uri_file is appended to the template path
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_from_template(connector_name=self.CONNECTOR_SOURCE, uri_file=uri_file,
                                         template_name=self.TEMPLATE_SOURCE)

    def set_persist(self, uri_file: str, save: bool=None):
        """sets the persist contract CONNECTOR_PERSIST using the TEMPLATE_PERSIST connector contract

        :param uri_file: the uri_file is appended to the template path
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_from_template(connector_name=self.CONNECTOR_PERSIST, uri_file=uri_file,
                                         template_name=self.TEMPLATE_PERSIST)

    def load_source_canonical(self) -> pd.DataFrame:
        """returns the contracted source data as a DataFrame """
        return self.load_canonical(self.CONNECTOR_SOURCE)

    def load_clean_canonical(self) -> pd.DataFrame:
        """loads the clean pandas.DataFrame from the clean folder for this contract"""
        return self.load_canonical(self.CONNECTOR_PERSIST)

    def load_canonical(self, connector_name: str, **kwargs) -> pd.DataFrame:
        """returns the canonical of the referenced connector

        :param connector_name: the name or label to identify and reference the connector
        """
        canonical = super().load_canonical(connector_name=connector_name, **kwargs)
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical, orient='columns')
        return canonical

    def save_clean_canonical(self, df):
        """Saves the pandas.DataFrame to the clean files folder"""
        self.persist_canonical(self.CONNECTOR_PERSIST, df)

    def run_transition_pipeline(self):
        """Runs the transition pipeline from source to persist"""
        canonical = self.load_source_canonical()
        result = self.intent_model.run_intent_pipeline(canonical)
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
