from __future__ import annotations

import pandas as pd
from aistac.components.abstract_component import AbstractComponent
from aistac.handlers.abstract_handlers import ConnectorContract

from ds_discovery.intent.feature_catalog_intent import FeatureCatalogIntentModel
from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager
from ds_discovery.components.commons import Commons
from ds_discovery.components.discovery import DataDiscovery, Visualisation


class FeatureCatalog(AbstractComponent):

    DEFAULT_MODULE = 'ds_discovery.handlers.pandas_handlers'
    DEFAULT_SOURCE_HANDLER = 'PandasSourceHandler'
    DEFAULT_PERSIST_HANDLER = 'PandasPersistHandler'

    def __init__(self, property_manager: FeatureCatalogPropertyManager, intent_model: FeatureCatalogIntentModel,
                 default_save=None, reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None, align_connectors: bool=None):
        """ Encapsulation class for the components set of classes

        :param property_manager: The contract property manager instance for this component
        :param intent_model: the model codebase containing the parameterizable intent
        :param default_save: The default behaviour of persisting the contracts:
                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        :param reset_templates: (optional) reset connector templates from environ variables (see `report_environ()`)
        :param template_path: (optional) a template path to use if the environment variable does not exist
        :param template_module: (optional) a template module to use if the environment variable does not exist
        :param template_source_handler: (optional) a template source handler to use if no environment variable
        :param template_persist_handler: (optional) a template persist handler to use if no environment variable
        :param align_connectors: (optional) resets aligned connectors to the template
        """
        super().__init__(property_manager=property_manager, intent_model=intent_model, default_save=default_save,
                         reset_templates=reset_templates, template_path=template_path, template_module=template_module,
                         template_source_handler=template_source_handler,
                         template_persist_handler=template_persist_handler, align_connectors=align_connectors)

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, username: str, uri_pm_repo: str=None, pm_file_type: str=None,
                 pm_module: str=None, pm_handler: str=None, pm_kwargs: dict=None, default_save=None,
                 reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None, align_connectors: bool=None,
                 default_save_intent: bool=None, default_intent_level: bool=None, order_next_available: bool=None,
                 default_replace_intent: bool=None, has_contract: bool=None) -> FeatureCatalog:
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
         :param template_path: (optional) a template path to use if the environment variable does not exist
         :param template_module: (optional) a template module to use if the environment variable does not exist
         :param template_source_handler: (optional) a template source handler to use if no environment variable
         :param template_persist_handler: (optional) a template persist handler to use if no environment variable
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
        username = username if isinstance(username, str) else 'Unknown'
        _pm = FeatureCatalogPropertyManager(task_name=task_name, username=username)
        _intent_model = FeatureCatalogIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
                                                  default_intent_level=default_intent_level,
                                                  order_next_available=order_next_available,
                                                  default_replace_intent=default_replace_intent)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, default_save=default_save,
                                 uri_pm_repo=uri_pm_repo, pm_file_type=pm_file_type, pm_module=pm_module,
                                 pm_handler=pm_handler, pm_kwargs=pm_kwargs, has_contract=has_contract)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save,
                   reset_templates=reset_templates, template_path=template_path, template_module=template_module,
                   template_source_handler=template_source_handler, template_persist_handler=template_persist_handler,
                   align_connectors=align_connectors)

    @classmethod
    def scratch_pad(cls) -> FeatureCatalogIntentModel:
        """ A class method to use the Components intent methods as a scratch pad"""
        return super().scratch_pad()

    @classmethod
    def discovery_pad(cls) -> DataDiscovery:
        """ A class method to use the Components discovery methods as a scratch pad"""
        return DataDiscovery()

    @classmethod
    def visual_pad(cls) -> Visualisation:
        """ A class method to use the Components visualisation methods as a scratch pad"""
        return Visualisation()

    @property
    def intent_model(self) -> FeatureCatalogIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> FeatureCatalogPropertyManager:
        """The properties manager instance"""
        return self._component_pm

    @property
    def discover(self) -> DataDiscovery:
        """The components instance"""
        return DataDiscovery()

    @property
    def visual(self) -> Visualisation:
        """The visualisation instance"""
        return Visualisation()

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

    def set_catalog_feature(self, feature_name: str, description: str=None, versioned: bool=None, stamped: bool=None,
                            file_type: str=None, save: bool=None):
        """sets the persist feature contract using the TEMPLATE_PERSIST connector contract

        :param feature_name: the unique name of the feature
        :param description: an optional description for the feature
        :param versioned: (optional) if the component version should be included as part of the pattern
        :param stamped: (optional) A string of the timestamp options ['days', 'hours', 'minutes', 'seconds', 'ns']
        :param file_type: (optional) a connector supported file extension type different from the default e.g. 'csv'
        :param save: (optional) if True, save to file. Default is True
        """
        versioned = versioned if isinstance(versioned, bool) else True
        uri_file = self.pm.file_pattern(name=feature_name, versioned=versioned, stamped=stamped,
                                        file_type=file_type)
        self.add_connector_from_template(connector_name=feature_name, uri_file=uri_file,
                                         template_name=self.TEMPLATE_PERSIST, save=save)
        if isinstance(description, str):
            self.pm.set_intent_description(level=feature_name, text=description)
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
        self.pm.set_canonical_schema(name=feature_name, schema=report)
        self.pm_persist(save=save)
        return

    def remove_feature(self, feature_name: str, save: bool=None):
        """completely removes a feature including connector, intent and description"""
        if self.pm.has_connector(connector_name=feature_name):
            self.remove_connector_contract(connector_name=feature_name, save=save)
        if self.pm.has_intent(level=feature_name):
            self.remove_intent(level=feature_name)
        return

    def run_feature_pipeline(self, canonical: pd.DataFrame=None, feature_names: [str, list]=None,
                             auto_connectors: bool=None, save: bool=None):
        """runs all features within the feature catalog or an optional set of features

        :param canonical: (optional) A canonical if the source canonical isn't to be used
        :param feature_names: (optional) a single or list of features to run
        :param auto_connectors: (optional) Adds a versioned feature connector if not yet added. Default to True
        :param save: (optional) if True, persist changes to property manager. Default is True
        """
        auto_connectors = auto_connectors if isinstance(auto_connectors, bool) else True
        if isinstance(feature_names, (str, list)):
            feature_names = Commons.list_formatter(feature_names)
        else:
            feature_names = Commons.list_formatter(self.pm.get_intent())
        if not isinstance(canonical, (pd.DataFrame, str)):
            canonical = self.load_source_canonical()
        for feature in feature_names:
            if not self.pm.has_connector(feature):
                if not auto_connectors:
                    continue
                self.set_catalog_feature(feature_name=feature, versioned=True, save=save)
            result = self.intent_model.run_intent_pipeline(canonical, feature)
            self.save_catalog_feature(feature_name=feature, canonical=result)
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

    def report_feature_schema(self, feature_name: str, stylise: bool=True):
        """ presents the current feature schema

        :param feature_name: the name of the feature schema
        :param stylise: if True present the report stylised.
        :return: pd.DataFrame
        """
        report = self.pm.get_canonical_schema(name=feature_name)
        if len(report) == 0:
            report = {'Attributes (0)': {}, 'dType': {}, '%_Null': {}, '%_Dom': {}, 'Count': {}, 'Unique': {},
                      'Observations': {}}
        return self.discover.data_schema(report=report, stylise=stylise)

    def report_feature_catalog(self, feature_names: [str, list]=None, stylise: bool=True):
        """ generates a report on the source contract

        :param feature_names: (optional) filters on specific feature names.
        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(levels=feature_names, as_description=True,
                                                               level_label='feature_name'), orient='columns')
        if stylise:
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['feature_name'], **{'font-weight': 'bold'})
            return df_style
        else:
            df.set_index(keys='feature_name', inplace=True)
        return df

    def report_connectors(self, connector_filter: [str, list]=None, inc_pm: bool=None, inc_template: bool=None,
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

    def report_intent(self, levels: [str, int, list]=None, stylise: bool = True):
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

    def report_notes(self, catalog: [str, list]=None, labels: [str, list]=None, regex: [str, list]=None,
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
