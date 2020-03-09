import pandas as pd

from aistac.handlers.abstract_handlers import ConnectorContract
from ds_discovery.intent.feature_catalog_intent import FeatureCatalogIntentModel
from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager
from aistac.components.abstract_component import AbstractComponent

from ds_discovery.transition.discovery import DataDiscovery, Visualisation


class FeatureCatalog(AbstractComponent):

    CONNECTOR_SOURCE = 'source_connector'

    def __init__(self, property_manager: FeatureCatalogPropertyManager, intent_model: FeatureCatalogIntentModel,
                 default_save=None):
        """ Encapsulation class for the transition set of classes

        :param property_manager: The contract property manager instance for this component
        :param intent_model: the model codebase containing the parameterizable intent
        :param default_save: The default behaviour of persisting the contracts:
                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        """
        super().__init__(property_manager=property_manager, intent_model=intent_model, default_save=default_save)

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
        pm_file_type = pm_file_type if isinstance(pm_file_type, str) else 'pickle'
        pm_module = pm_module if isinstance(pm_module, str) else 'ds_discovery.handlers.pandas_handlers'
        pm_handler = pm_handler if isinstance(pm_handler, str) else 'PandasPersistHandler'
        _pm = FeatureCatalogPropertyManager(task_name=task_name)
        _intent_model = FeatureCatalogIntentModel(property_manager=_pm)
        if not isinstance(template_source_module, str) or template_source_module.startswith('aistac.'):
            template_source_module = 'ds_discovery.handlers.pandas_handlers'
            template_source_handler = 'PandasSourceHandler'
        if not isinstance(template_persist_module, str) or template_persist_module.startswith('aistac.'):
            template_persist_module = 'ds_discovery.handlers.pandas_handlers'
            template_persist_handler = 'PandasPersistHandler'
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, pm_file_type=pm_file_type,
                                 pm_module=pm_module, pm_handler=pm_handler, **kwargs)
        super()._add_templates(property_manager=_pm, save=default_save,
                               source_path=template_source_path, persist_path=template_persist_path,
                               source_module=template_source_module, persist_module=template_persist_module,
                               source_handler=template_source_handler, persist_handler=template_persist_handler)
        instance = cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save)
        instance.modify_connector_from_template(connector_names=instance.pm.connector_contract_list)
        return instance

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
    def intent_model(self) -> FeatureCatalogIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> FeatureCatalogPropertyManager:
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

    def is_source_modified(self):
        """Test if the source file is modified since last load"""
        return self.pm.has_connector(connector_name=self.CONNECTOR_SOURCE)

    def get_feature_contract(self, feature_name: str=None) -> ConnectorContract:
        """ gets the feature connector contract

        :param feature_name: The unique name of the feature
        :return: connector contract
        """
        return self.pm.get_connector_contract(connector_name=feature_name)

    def set_source_contract(self, connector_contract: ConnectorContract, template_aligned: bool=None, save: bool=None):
        """ Sets the source contract

        :param connector_contract: a Connector Contract for the properties persistence
        :param template_aligned: if the source contact should align with the template changes
        :param save: (optional) if True, save to file. Default is True
        """
        save = save if isinstance(save, bool) else self._default_save
        if self.pm.has_connector(connector_name=self.CONNECTOR_SOURCE):
            self.remove_connector_contract(connector_name=self.CONNECTOR_SOURCE)
        self.pm.set_connector_contract(connector_name=self.CONNECTOR_SOURCE, connector_contract=connector_contract,
                                       aligned=template_aligned)
        self.pm_persist(save)
        return

    def set_feature_contract(self, feature_name: str, connector_contract: ConnectorContract,
                             template_aligned: bool=None, save: bool=None):
        """ Sets the persist contract.

        :param feature_name: the unique name of the feature
        :param connector_contract: a Connector Contract for the properties persistence
        :param template_aligned: if the source contact should align with the template changes
        :param save: (optional) if True, save to file. Default is True
        """
        save = save if isinstance(save, bool) else self._default_save
        if self.pm.has_connector(feature_name):
            self.remove_connector_contract(feature_name)
        self.pm.set_connector_contract(connector_name=feature_name, connector_contract=connector_contract,
                                       aligned=template_aligned)
        self.pm_persist(save)
        return

    def set_source(self, uri_file: str, save: bool=None):
        """sets the source contract CONNECTOR_SOURCE using the TEMPLATE_SOURCE connector contract,

        :param uri_file: the uri_file is appended to the template path
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_from_template(connector_name=self.CONNECTOR_SOURCE, uri_file=uri_file,
                                         template_name=self.TEMPLATE_SOURCE, save=save)

    def set_catalog_feature(self, feature_name: str, save: bool=None):
        """sets the persist feature contract using the TEMPLATE_PERSIST connector contract

        :param feature_name: the unique name of the feature
        :param save: (optional) if True, save to file. Default is True
        """
        uri_file = self.pm.file_pattern(connector_name=feature_name)
        self.add_connector_from_template(connector_name=feature_name, uri_file=uri_file,
                                         template_name=self.TEMPLATE_PERSIST, save=save)

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

    def run_feature_pipeline(self, intent_levels: [str, int, list]=None):
        """Runs the feature pipeline from source to persist"""
        canonical = self.load_source_canonical()
        result = self.intent_model.run_intent_pipeline(canonical, intent_levels=intent_levels)
        self.save_catalog_feature(result)

    @staticmethod
    def canonical_report(df, stylise: bool=True, inc_next_dom: bool=False, report_header: str=None,
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
        return DataDiscovery.data_dictionary(df=df, stylise=stylise, inc_next_dom=inc_next_dom,
                                             report_header=report_header, condition=condition)

    def report_feature_catalog(self, connector_filter: [str, list]=None, stylise: bool=True):
        """ generates a report on the source contract

        :param connector_filter: (optional) filters on the connector name.
        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        connector_filter = self.pm.list_formatter(connector_filter)
        df = pd.DataFrame.from_dict(data=self.pm.report_connectors(connector_filter=connector_filter), orient='columns')
        df.rename(columns={'connector_name': 'feature_name'}, inplace=True)
        df.set_index(keys='feature_name', inplace=True)
        df.drop(labels=[self.CONNECTOR_SOURCE, self.TEMPLATE_PERSIST,  self.TEMPLATE_SOURCE,
                        self.pm.CONNECTOR_PM_CONTRACT], inplace=True, errors='ignore')
        df.drop(columns=['module_name', 'handler', 'kwargs', 'query', 'aligned'], inplace=True, errors='ignore')

        if stylise:
            df.reset_index(inplace=True)
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['feature_name'], **{'font-weight': 'bold'})
            return df_style
        return df

    def report_connectors(self, connector_filter: [str, list]=None, stylise: bool=True):
        """ generates a report on the source contract

        :param connector_filter: (optional) filters on the connector name.
        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame.from_dict(data=self.pm.report_connectors(connector_filter=connector_filter), orient='columns')
        if stylise:
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['connector_name'], **{'font-weight': 'bold'})
            return df_style
        else:
            df.set_index(keys='connector_name', inplace=True)
        return df

    def report_run_book(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame.from_dict(data=self.pm.report_run_book(), orient='columns')
        if stylise:
            index = df[df['name'].duplicated()].index.to_list()
            df.loc[index, 'name'] = ''
            df = df.reset_index(drop=True)
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['name'],  **{'font-weight': 'bold', 'font-size': "120%"})
            return df_style
        return df

    def report_intent(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(), orient='columns')
        if stylise:
            index = df[df['level'].duplicated()].index.to_list()
            df.loc[index, 'level'] = ''
            df = df.reset_index(drop=True)
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['level'],  **{'font-weight': 'bold', 'font-size': "120%"})
            return df_style
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
        stylise = True if not isinstance(stylise, bool) else stylise
        drop_dates = False if not isinstance(drop_dates, bool) else drop_dates
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        report = self.pm.report_notes(catalog=catalog, labels=labels, regex=regex, re_ignore_case=re_ignore_case,
                                      drop_dates=drop_dates)
        df = pd.DataFrame.from_dict(data=report, orient='columns')
        if stylise:
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['section'], **{'font-weight': 'bold'})
            _ = df_style.set_properties(subset=['label', 'section'], **{'font-size': "120%"})
            return df_style
        return df
