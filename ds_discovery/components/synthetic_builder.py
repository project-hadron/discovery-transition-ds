from __future__ import annotations

import pandas as pd
from ds_discovery.components.commons import Commons
from ds_discovery.components.discovery import DataDiscovery
from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from ds_discovery.managers.synthetic_property_manager import SyntheticPropertyManager
from aistac.components.abstract_component import AbstractComponent

__author__ = 'Darryl Oatridge'


class SyntheticBuilder(AbstractComponent):

    CONNECTOR_SOURCE = 'primary_source'
    CONNECTOR_PERSIST = 'primary_persist'
    REPORT_CATALOG = 'catalog'
    REPORT_FIELDS = 'field_description'

    DEFAULT_MODULE = 'ds_discovery.handlers.pandas_handlers'
    DEFAULT_SOURCE_HANDLER = 'PandasSourceHandler'
    DEFAULT_PERSIST_HANDLER = 'PandasPersistHandler'

    def __init__(self, property_manager: SyntheticPropertyManager, intent_model: SyntheticIntentModel,
                 default_save=None, reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None, align_connectors: bool=None):
        """ Encapsulation class for the discovery set of classes

        :param property_manager: The contract property manager instance for this components
        :param intent_model: the model codebase containing the parameterizable intent
        :param default_save: The default behaviour of persisting the contracts:
                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        :param reset_templates: (optional) reset connector templates from environ variables (see `report_environ()`)
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
                 default_replace_intent: bool=None, has_contract: bool=None) -> SyntheticBuilder:
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
        pm_module = pm_module if isinstance(pm_module, str) else cls.DEFAULT_MODULE
        pm_handler = pm_handler if isinstance(pm_handler, str) else cls.DEFAULT_PERSIST_HANDLER
        _pm = SyntheticPropertyManager(task_name=task_name, username=username)
        _intent_model = SyntheticIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
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
    def scratch_pad(cls) -> SyntheticIntentModel:
        """ A class method to use the Components intent methods as a scratch pad"""
        return super().scratch_pad()

    @classmethod
    def discovery_pad(cls) -> DataDiscovery:
        """ A class method to use the Components discovery methods as a scratch pad"""
        return DataDiscovery()

    @property
    def pm(self) -> SyntheticPropertyManager:
        return self._component_pm

    @property
    def intent_model(self) -> SyntheticIntentModel:
        return self._intent_model

    @property
    def tools(self) -> SyntheticIntentModel:
        return self._intent_model

    @property
    def discover(self) -> DataDiscovery:
        """The components instance"""
        return DataDiscovery()

    def load_synthetic_canonical(self, **kwargs) -> pd.DataFrame:
        """loads the generated synthetic pandas.DataFrame from the folder for this contract"""
        return self.load_canonical(self.CONNECTOR_PERSIST, **kwargs)

    def load_canonical(self, connector_name: str, **kwargs) -> pd.DataFrame:
        """returns the canonical of the referenced connector

        :param connector_name: the name or label to identify and reference the connector
        """
        canonical = super().load_canonical(connector_name=connector_name, **kwargs)
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical)
        return canonical

    def save_synthetic_canonical(self, canonical, auto_connectors: bool=None, **kwargs):
        """Saves the synthetic canonical , auto creating the connector from template if not set"""
        if auto_connectors if isinstance(auto_connectors, bool) else True:
            if not self.pm.has_connector(self.CONNECTOR_PERSIST):
                self.set_persist()
        self.persist_canonical(connector_name=self.CONNECTOR_PERSIST, canonical=canonical, **kwargs)

    def set_report_persist(self, connector_name: [str, list]=None, uri_file: str=None, project: str=None,
                           path: str=None, file_type: str=None, save: bool=None, **kwargs):
        """sets the report persist using the TEMPLATE_PERSIST connector contract, there are preset constants that
        should be used. These constance can be found using Transition.REPORT_<NAME> or <instance>.REPORT_<NAME>
        where <name> is the name of the report. if no report connector name is given then all the report connectors
        are set with default values.

        :param connector_name: (optional) the name(s) of the report connector to set (see class REPORT constants)
        :param uri_file: (optional) the uri_file is appended to the template path
        :param project: (optional) a project name that will replace the hadron naming, if no uri_file
        :param path: (optional) a path added to the template path default, if no uri_file
        :param file_type: (optional) a file type of the report, if no uri_file
        :param save: (optional) if True, save to file. Default is True
        """
        _default_reports = [self.REPORT_CATALOG] + self.REPORTS_BASE_LIST
        project = project if isinstance(project, str) else 'hadron'
        if not isinstance(connector_name, (str, list)):
            connector_name = _default_reports
        for _report in self.pm.list_formatter(connector_name):
            if _report not in _default_reports:
                raise ValueError(f"Report name(s) {_report} must be from the report constants {_default_reports}")
            file_pattern = uri_file
            if not isinstance(uri_file, str):
                file_type = file_type if isinstance(file_type, str) else 'csv'
                file_pattern = self.pm.file_pattern(name=_report, project=project, path=path,
                                                    file_type=file_type, versioned=True)
                if 'orient' not in kwargs.keys():
                    kwargs.update({'orient': 'records'})
            self.add_connector_from_template(connector_name=_report, uri_file=file_pattern,
                                             template_name=self.TEMPLATE_PERSIST, save=save, **kwargs)
        return

    def save_report_canonical(self, report_connector_name: str, report: [dict, pd.DataFrame],
                              auto_connectors: bool=None, **kwargs):
        """Saves the canonical to the data quality folder, auto creating the connector from template if not set"""
        if report_connector_name not in [self.REPORT_CATALOG] + self.REPORTS_BASE_LIST:
            raise ValueError("Report name must be one of the class report constants")
        if auto_connectors if isinstance(auto_connectors, bool) else True:
            if not self.pm.has_connector(report_connector_name):
                self.set_report_persist(connector_name=report_connector_name)
        self.persist_canonical(connector_name=report_connector_name, canonical=report, **kwargs)

    def save_canonical_schema(self, schema_name: str=None, canonical: pd.DataFrame=None, schema_tree: list=None,
                              save: bool=None):
        """ Saves the canonical schema to the Property contract. The default loads the clean canonical but optionally
        a canonical can be passed to base the schema on and optionally a name given other than the default

        :param schema_name: (optional) the name of the schema to save
        :param canonical: (optional) the canonical to base the schema on
        :param schema_tree: (optional) an analytics dict (see Discovery.analyse_association(...)
        :param save: (optional) if True, save to file. Default is True
        """
        schema_name = schema_name if isinstance(schema_name, str) else self.REPORT_SCHEMA
        canonical = canonical if isinstance(canonical, pd.DataFrame) else self.load_synthetic_canonical()
        schema_tree = schema_tree if isinstance(schema_tree, list) else canonical.columns.to_list()
        analytics = self.discover.analyse_association(canonical, columns_list=schema_tree)
        self.pm.set_canonical_schema(name=schema_name, schema=analytics)
        self.pm_persist(save=save)
        return

    def add_column_description(self, column_name: str, description: str, save: bool=None):
        """ adds a description note that is included in with the 'report_column_catalog'"""
        if isinstance(description, str) and description:
            self.pm.set_intent_description(level=column_name, text=description)
            self.pm_persist(save)
        return

    def run_synthetic_pipeline(self, size: int, columns: [str, list]=None, seed: int=None):
        """Runs the transition pipeline from source to persist"""
        result = self.intent_model.run_intent_pipeline(size=size, columns=columns, seed=seed)
        self.save_synthetic_canonical(canonical=result)

    @staticmethod
    def canonical_report(canonical: pd.DataFrame, stylise: bool=True, inc_next_dom: bool=False,
                         report_header: str=None, condition: str=None):
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

    def report_canonical_schema(self, schema_name: str=None, stylise: bool=True):
        """ presents the current canonical schema

        :param schema_name: (optional) the name of the schema
        :param stylise: if True present the report stylised.
        :return: pd.DataFrame
        """
        schema_name = schema_name if isinstance(schema_name, str) else self.REPORT_SCHEMA
        analytics = self.pm.get_canonical_schema(name=schema_name)
        return DataDiscovery.data_schema(analysis=analytics, stylise=stylise)

    def report_connectors(self, connector_filter: [str, list]=None, inc_pm: bool=None, inc_template: bool=None,
                          stylise: bool=True):
        """ generates a report on the source contract

        :param connector_filter: (optional) filters on the connector name.
        :param inc_pm: (optional) include the property manager connector
        :param inc_template: (optional) include the template connectors
        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        report = self.pm.report_connectors(connector_filter=connector_filter, inc_pm=inc_pm,
                                           inc_template=inc_template)
        df = pd.DataFrame.from_dict(data=report)
        if stylise:
            return Commons.report(df, index_header='connector_name')
        return df

    def report_run_book(self, stylise: bool = True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_run_book())
        if stylise:
            return Commons.report(df, index_header='name')
        return df

    def report_intent(self, levels: [str, int, list]=None, stylise: bool=True):
        """ generates a report on all the intent

        :param levels: (optional) a filter on the levels. passing a single value will report a single parameterised view
        :param stylise: (optional) returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        if isinstance(levels, (int, str)):
            df = pd.DataFrame.from_dict(data=self.pm.report_intent_params(level=levels))
            if stylise:
                Commons.report(df, index_header=['order', 'intent'], bold='intent')
                df.set_index(keys=['order', 'intent'], inplace=True)
                return df
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(levels=levels))
        if stylise:
            return Commons.report(df, index_header='level')
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
        df = pd.DataFrame.from_dict(data=report)
        if stylise:
            return Commons.report(df, index_header='section', bold='label')
        return df

    def report_column_catalog(self, column_name: [str, list]=None, stylise: bool=True):
        """ generates a report on the source contract

        :param column_name: (optional) filters on specific column names.
        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(levels=column_name, as_description=True,
                                                               level_label='column_name'))
        if stylise:
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['column_name'], **{'font-weight': 'bold'})
            return df_style
        return df

    def setup_bootstrap(self, domain: str=None, project_name: str=None, path: str=None, file_type: str=None):
        """ Creates a bootstrap simulator with a SyntheticBuilder. Note this does not set the source

        :param domain: (optional) The domain this simulator sits within for example 'Healthcare' or 'Financial Services'
        :param project_name: (optional) a project name that will replace the hadron naming on file prefix
        :param path: (optional) a path added to the template path default
        :param file_type: (optional) a file_type for the persisted file, default is 'parquet'
        """
        file_type = file_type if isinstance(file_type, str) else 'parquet'
        project_name = project_name if isinstance(project_name, str) else 'hadron'
        file_name = self.pm.file_pattern(name='dataset', project=project_name, path=path, file_type=file_type,
                                         versioned=True)
        self.set_persist(uri_file=file_name)
        self.set_report_persist(connector_name=[self.REPORT_INTENT, self.REPORT_CATALOG], project=project_name,
                                path=path)
        self.set_description(f"A domain specific {domain} simulated {project_name} dataset for {self.pm.task_name}")
