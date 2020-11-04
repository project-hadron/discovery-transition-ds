import pandas as pd
from aistac import ConnectorContract
from aistac.components.abstract_component import AbstractComponent
from ds_discovery.components.commons import Commons
from ds_discovery.components.discovery import DataDiscovery
from ds_discovery.managers.tolerance_catalog_property_manager import ToleranceCatalogPropertyManager
from ds_discovery.intent.tolerance_catalog_intent import ToleranceCatalogIntentModel


class ConceptTolerance(AbstractComponent):

    DEFAULT_MODULE = 'ds_discovery.handlers.pandas_handlers'
    DEFAULT_SOURCE_HANDLER = 'PandasSourceHandler'
    DEFAULT_PERSIST_HANDLER = 'PandasPersistHandler'

    def __init__(self, property_manager: ToleranceCatalogPropertyManager, intent_model: ToleranceCatalogIntentModel,
                 default_save=None, reset_templates: bool = None, align_connectors: bool = None):
        """ Encapsulation class for the components set of classes

        :param property_manager: The contract property manager instance for this component
        :param intent_model: the model codebase containing the parameterizable intent
        :param default_save: The default behaviour of persisting the contracts:
                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        :param reset_templates: (optional) reset connector templates from environ variables (see `report_environ()`)
        :param align_connectors: (optional) resets aligned connectors to the template
        """
        super().__init__(property_manager=property_manager, intent_model=intent_model, default_save=default_save,
                         reset_templates=reset_templates, align_connectors=align_connectors)

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, username: str, uri_pm_repo: str=None, pm_file_type: str=None,
                 pm_module: str=None, pm_handler: str=None, pm_kwargs: dict=None, default_save=None,
                 reset_templates: bool=None, align_connectors: bool=None, default_save_intent: bool=None,
                 default_intent_level: bool=None, order_next_available: bool=None, default_replace_intent: bool=None,
                 has_contract: bool=None):
        """ Class Factory Method to instantiates the components application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.
        See class inline docs for an example method

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param username: A user name for this task activity.
         :param uri_pm_repo: (optional) A repository URI to initially load the property manager but not save to.
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
         :param has_contract: (optional) indicates the instance should have a property manager domain contract
         :return: the initialised class instance
         """
        pm_file_type = pm_file_type if isinstance(pm_file_type, str) else 'json'
        pm_module = pm_module if isinstance(pm_module, str) else 'ds_discovery.handlers.pandas_handlers'
        pm_handler = pm_handler if isinstance(pm_handler, str) else 'PandasPersistHandler'
        _pm = ToleranceCatalogPropertyManager(task_name=task_name, username=username)
        _intent_model = ToleranceCatalogIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
                                                    default_intent_level=default_intent_level,
                                                    order_next_available=order_next_available,
                                                    default_replace_intent=default_replace_intent)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, default_save=default_save,
                                 uri_pm_repo=uri_pm_repo, pm_file_type=pm_file_type, pm_module=pm_module,
                                 pm_handler=pm_handler, pm_kwargs=pm_kwargs, has_contract=has_contract)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save,
                   reset_templates=reset_templates, align_connectors=align_connectors)

    @classmethod
    def scratch_pad(cls) -> ToleranceCatalogIntentModel:
        """ A class method to use the Components intent methods as a scratch pad"""
        return super().scratch_pad()

    @property
    def intent_model(self) -> ToleranceCatalogIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> ToleranceCatalogPropertyManager:
        """The properties manager instance"""
        return self._component_pm

    @property
    def discover(self) -> DataDiscovery:
        """The components instance"""
        return DataDiscovery()

    def get_feature_contract(self, feature_name: str=None) -> ConnectorContract:
        """ gets the feature connector contract

        :param feature_name: The unique name of the feature
        :return: connector contract
        """
        return self.pm.get_connector_contract(connector_name=feature_name)

    def set_feature_contract(self, feature_name: str, connector_contract: ConnectorContract, save: bool=None):
        """ Sets the persist contract.

        :param feature_name: the unique name of the feature
        :param connector_contract: a Connector Contract for the properties persistence
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_contract(connector_name=feature_name, connector_contract=connector_contract, save=save)
        return

    def load_source_canonical(self) -> [pd.DataFrame]:
        """returns the contracted source data as a DataFrame """
        return self.load_canonical(connector_name=self.CONNECTOR_SOURCE)

    def load_catalog_feature(self, feature_name: str, reset_index: bool=None) -> pd.DataFrame:
        """returns the feature data as a DataFrame

        :param feature_name: a unique feature name
        :param reset_index: if the index should be reset bringing the key into the columns
        :return: pandas DataFrame
        """
        canonical = self.load_canonical(connector_name=feature_name)
        if isinstance(reset_index, bool) and reset_index:
            canonical.reset_index(inplace=True)
        return canonical

    def load_canonical(self, connector_name: str, **kwargs) -> pd.DataFrame:
        """returns the canonical of the referenced connector

        :param connector_name: the name or label to identify and reference the connector
        """
        canonical = super().load_canonical(connector_name=connector_name, **kwargs)
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical, orient='columns')
        return canonical

    def save_catalog_feature(self, canonical, feature_name: str):
        """Saves the pandas.DataFrame to the feature catalog

        :param canonical: the pandas DataFrame
        :param feature_name: a unique feature name
        """
        self.persist_canonical(connector_name=feature_name, canonical=canonical)

    def add_feature_description(self, feature_name: str, description: str, save: bool=None):
        """ adds a description note that is included in with the 'report_features'"""
        if isinstance(description, str) and description:
            self.pm.set_intent_description(level=feature_name, text=description)
            self.pm_persist(save)
        return

    def save_feature_schema(self, feature_name: str, canonical: pd.DataFrame=None, save: bool=None):
        """ Saves the feature schema to the Property contract. The default loads the feature canonical but optionally
        a canonical can be passed to base the schema on

        :param feature_name: the name of the schema feature to save
        :param canonical: (optional) the canonical to base the schema on
        :param save: (optional) if True, save to file. Default is True
        """
        canonical = canonical if isinstance(canonical, pd.DataFrame) else self.load_catalog_feature(feature_name)
        report = self.canonical_report(canonical=canonical, stylise=False).to_dict()
        self.pm.set_canonical_schema(name=feature_name, canonical_report=report)
        self.pm_persist(save=save)
        return

    def remove_feature(self, feature_name: str, save: bool=None):
        """completely removes a feature including connector, intent and description"""
        if self.pm.has_connector(connector_name=feature_name):
            self.remove_connector_contract(connector_name=feature_name, save=save)
        if self.pm.has_intent(level=feature_name):
            self.remove_intent(level=feature_name)
        return

    @staticmethod
    def canonical_report(canonical, stylise: bool=True, inc_next_dom: bool=False, report_header: str=None,
                         condition: str=None):
        """The Canonical Report is a data dictionary of the canonical providing a reference view of the dataset's
        attribute properties

        :param canonical: the DataFrame to view
        :param stylise: if True present the report stylised.
        :param inc_next_dom: (optional) if to include the next dominate element column
        :param report_header: (optional) filter on a header where the condition is true. Condition must exist
        :param condition: (optional) the condition to apply to the header. Header must exist. examples:
                ' > 0.95', ".str.contains('shed')"
        :return:
        """
        return DataDiscovery.data_dictionary(df=canonical, stylise=stylise, inc_next_dom=inc_next_dom,
                                             report_header=report_header, condition=condition)

    def report_connectors(self, connector_filter: [str, list] = None, inc_pm: bool = None, inc_template: bool = None,
                          stylise: bool = True):
        """ generates a report on the source contract

        :param connector_filter: (optional) filters on the connector name.
        :param inc_pm: (optional) include the property manager connector
        :param inc_template: (optional) include the template connectors
        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        report = self.pm.report_connectors(connector_filter=connector_filter, inc_pm=inc_pm,
                                           inc_template=inc_template)
        df = pd.DataFrame.from_dict(data=report, orient='columns')
        if stylise:
            return Commons.report(df, index_header='connector_name')
        df.set_index(keys='connector_name', inplace=True)
        return df

    def report_run_book(self, stylise: bool = True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_run_book(), orient='columns')
        if stylise:
            return Commons.report(df, index_header='name')
        df.set_index(keys='name', inplace=True)
        return df

    def report_intent(self, levels: [str, int, list] = None, stylise: bool = True):
        """ generates a report on all the intent

        :param levels: (optional) a filter on the levels. passing a single value will report a single parameterised view
        :param stylise: (optional) returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        if isinstance(levels, (int, str)):
            df = pd.DataFrame.from_dict(data=self.pm.report_intent_params(level=levels), orient='columns')
            if stylise:
                return Commons.report(df, index_header='order')
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(levels=levels), orient='columns')
        if stylise:
            return Commons.report(df, index_header='level')
        df.set_index(keys='level', inplace=True)
        return df

    def report_notes(self, catalog: [str, list] = None, labels: [str, list] = None, regex: [str, list] = None,
                     re_ignore_case: bool = False, stylise: bool = True, drop_dates: bool = False):
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
            return Commons.report(df, index_header='section', bold='label')
        df.set_index(keys='section', inplace=True)
        return df
