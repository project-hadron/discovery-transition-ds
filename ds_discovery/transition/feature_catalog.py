import pandas as pd

from ds_foundation.handlers.abstract_handlers import ConnectorContract
from ds_discovery.intent.feature_catalog_intent import FeatureCatalogIntentModel
from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager
from ds_foundation.components.abstract_component import AbstractComponent

from ds_discovery.transition.discovery import DataDiscovery


class FeatureCatalog(AbstractComponent):

    CONNECTOR_SOURCE = 'source_connector'
    CONNECTOR_FRAME = 'frame_connector'

    def __init__(self, property_manager: FeatureCatalogPropertyManager, intent_model: FeatureCatalogIntentModel,
                 default_save=None):
        """ Encapsulation class for the discovery set of classes

        :param property_manager: The contract property manager instance for this component
        :param intent_model: the model codebase containing the parameterizable intent
        :param default_save: The default behaviour of persisting the contracts:

                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        """
        super().__init__(property_manager=property_manager, intent_model=intent_model, default_save=default_save)

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, default_save=None, **kwargs):
        """ Class Factory Method to instantiates the component application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.

        by default the handler is local Pandas but also supports remote AWS S3 and Redis. It use these Factory
        instantiations ensure that the schema is s3:// or redis:// and the handler will be automatically redirected

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :param kwargs: to pass to the connector contract
         :return: the initialised class instance
         """
        _pm = FeatureCatalogPropertyManager(task_name=task_name)
        _intent_model = FeatureCatalogIntentModel(property_manager=_pm)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, **kwargs)
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
    def intent_model(self) -> FeatureCatalogIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> FeatureCatalogPropertyManager:
        """The properties manager instance"""
        return self._component_pm

    def is_source_modified(self):
        """Test if the source file is modified since last load"""
        return self.pm.has_connector(connector_name=self.CONNECTOR_SOURCE)

    def get_frame_contract(self, connector_name: str=None) -> ConnectorContract:
        """ gets the frame connector contract that can be used as the next chain source

        :param connector_name: (optional) a connector name if not using the default
        :return: connector contract
        """
        connector_name = connector_name if isinstance(connector_name, str) else self.CONNECTOR_FRAME
        return self.pm.get_connector_contract(connector_name=connector_name)

    def set_source_contract(self, connector_contract: ConnectorContract, save: bool=None):
        """ Sets the source contract

        :param connector_contract: a Connector Contract for the properties persistence
        :param save: (optional) if True, save to file. Default is True
        """
        save = save if isinstance(save, bool) else self._default_save
        if self.pm.has_connector(connector_name=self.CONNECTOR_SOURCE):
            self.remove_connector_contract(connector_name=self.CONNECTOR_SOURCE)
        self.pm.set_connector_contract(connector_name=self.CONNECTOR_SOURCE, connector_contract=connector_contract)
        self.pm_persist(save)
        return

    def set_frame_contract(self, connector_contract: ConnectorContract, connector_name: str=None, save: bool=None):
        """ Sets the persist contract.

        :param connector_contract: a Connector Contract for the properties persistence
        :param connector_name: (optional) a connector name if not using the default
        :param save: (optional) if True, save to file. Default is True
        """
        save = save if isinstance(save, bool) else self._default_save
        connector_name = connector_name if isinstance(connector_name, str) else self.CONNECTOR_FRAME
        if self.pm.has_connector(connector_name):
            self.remove_connector_contract(connector_name)
        self.pm.set_connector_contract(connector_name=connector_name, connector_contract=connector_contract)
        self.pm_persist(save)
        return

    def load_source_canonical(self) -> [pd.DataFrame]:
        """returns the contracted source data as a DataFrame """
        return self.load_canonical(connector_name=self.CONNECTOR_SOURCE)

    def load_frame_canonical(self, connector_name: str=None) -> pd.DataFrame:
        """returns the contracted source data as a DataFrame

        :param connector_name: (optional) a connector name if not using the default
        :return: pandas DataFrame
        """
        connector_name = connector_name if isinstance(connector_name, str) else self.CONNECTOR_FRAME
        return self.load_canonical(connector_name=connector_name)

    def load_canonical(self, connector_name: str, **kwargs) -> pd.DataFrame:
        """returns the canonical of the referenced connector

        :param connector_name: the name or label to identify and reference the connector
        """
        canonical = super().load_canonical(connector_name=connector_name, **kwargs)
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical, orient='columns')
        return canonical

    def save_frame_canonical(self, canonical, connector_name: str=None):
        """Saves the pandas.DataFrame to the clean files folder

        :param canonical: the pandas DataFrame
        :param connector_name: (optional) a connector name if not using the default
        """
        connector_name = connector_name if isinstance(connector_name, str) else self.CONNECTOR_FRAME
        self.persist_canonical(connector_name=connector_name, canonical=canonical)

    def run_feature_pipeline(self):
        """Runs the feature pipeline from source to persist"""
        canonical = self.load_source_canonical()
        result = self.intent_model.run_intent_pipeline(canonical)
        self.save_frame_canonical(result)

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
