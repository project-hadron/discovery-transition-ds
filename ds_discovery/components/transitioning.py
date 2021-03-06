from __future__ import annotations

import uuid
import numpy as np
import pandas as pd
from aistac.components.abstract_component import AbstractComponent
from aistac.handlers.abstract_handlers import ConnectorContract

from ds_discovery.components.commons import Commons
from ds_discovery.intent.transition_intent import TransitionIntentModel
from ds_discovery.managers.transition_property_manager import TransitionPropertyManager
from ds_discovery.components.discovery import DataDiscovery, Visualisation

__author__ = 'Darryl Oatridge'


class Transition(AbstractComponent):

    REPORT_DICTIONARY = 'dictionary'
    REPORT_ANALYSIS = 'analysis'
    REPORT_FIELDS = 'field_description'
    REPORT_QUALITY = 'data_quality'
    REPORT_SUMMARY = 'data_quality_summary'
    REPORT_PROVENANCE = 'provenance'

    DEFAULT_MODULE = 'ds_discovery.handlers.pandas_handlers'
    DEFAULT_SOURCE_HANDLER = 'PandasSourceHandler'
    DEFAULT_PERSIST_HANDLER = 'PandasPersistHandler'

    def __init__(self, property_manager: TransitionPropertyManager, intent_model: TransitionIntentModel,
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
        self._raw_attribute_list = []

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, username: str, uri_pm_repo: str=None, pm_file_type: str=None,
                 pm_module: str=None, pm_handler: str=None, pm_kwargs: dict=None, default_save=None,
                 reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None, align_connectors: bool=None,
                 default_save_intent: bool=None, default_intent_level: bool=None, order_next_available: bool=None,
                 default_replace_intent: bool=None, has_contract: bool=None) -> Transition:
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
        _pm = TransitionPropertyManager(task_name=task_name, username=username)
        _intent_model = TransitionIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
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
    def scratch_pad(cls) -> TransitionIntentModel:
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
    def intent_model(self) -> TransitionIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def cleaners(self) -> TransitionIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> TransitionPropertyManager:
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

    def set_provenance(self, title: str=None, domain: str=None, description: str=None, license_type: str=None,
                       license_name: str=None, license_uri: str=None, cost_price: str = None, cost_code: str = None,
                       cost_type: str = None, provider_name: str=None, provider_uri: str=None, provider_note: str=None,
                       author_name: str=None, author_uri: str=None, author_contact: str=None, save: bool=None):
        """sets the provenance values. Only sets those passed

        :param title: (optional) the title of the provenance
        :param domain: (optional) the domain it sits within
        :param description: (optional) a description of the provenance
        :param license_type: (optional) The type of the license. Default 'ODC-By'
        :param license_name: (optional) The full name of the license. Default 'Open Data Commons Attribution License'
        :param license_uri: (optional) The license uri. Default https://opendatacommons.org/licenses/by/
        :param cost_price: (optional) a cost price associated with this provenance
        :param cost_code: (optional) a cost centre code or reference code
        :param cost_type: (optional) the cost type or description
        :param provider_name: (optional) the provider system or institution name or title
        :param provider_uri: (optional) a uri reference that helps identify the provider
        :param provider_note: (optional) any notes that might be useful
        :param author_name: (optional) the author of the data
        :param author_uri: (optional) the author uri
        :param author_contact: (optional)the the author contact information
        :param save: (optional) if True, save to file. Default is True
        """
        license_type = license_type if license_type else 'PDDL'
        license_name = license_name if license_name else 'Open Data Commons Attribution License'
        license_uri = license_uri if license_uri else 'https://opendatacommons.org/licenses/pddl/summary'

        self.pm.set_provenance(title=title, domain=domain, description=description, license_type=license_type,
                               license_name=license_name, license_uri=license_uri, cost_price=cost_price,
                               cost_code=cost_code, cost_type=cost_type, provider_name=provider_name,
                               provider_uri=provider_uri, provider_note=provider_note, author_name=author_name,
                               author_uri=author_uri, author_contact=author_contact)
        self.pm_persist(save=save)

    def reset_provenance(self, save: bool=None):
        """resets the provenance back to its default values"""
        self.pm.reset_provenance()
        self.pm_persist(save)

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
        _default_reports = [self.REPORT_DICTIONARY, self.REPORT_ANALYSIS, self.REPORT_QUALITY, self.REPORT_SUMMARY,
                            self.REPORT_FIELDS, self.REPORT_PROVENANCE] + self.REPORTS_BASE_LIST
        project = project if isinstance(project, str) else 'hadron'
        if not isinstance(connector_name, (str, list)):
            connector_name = _default_reports
        for _report in self.pm.list_formatter(connector_name):
            if _report not in _default_reports:
                raise ValueError(f"Report name(s) {_report} must be from the report constants {_default_reports}")
            file_pattern = uri_file
            if not isinstance(uri_file, str):
                file_type = file_type if isinstance(file_type, str) else 'csv'
                if _report in [self.REPORT_ANALYSIS, self.REPORT_QUALITY]:
                    file_type = file_type if isinstance(file_type, str) else 'json'
                file_pattern = self.pm.file_pattern(name=_report, project=project, path=path,
                                                    file_type=file_type, versioned=True)
                if 'orient' not in kwargs.keys():
                    kwargs.update({'orient': 'records'})
            self.add_connector_from_template(connector_name=_report, uri_file=file_pattern,
                                             template_name=self.TEMPLATE_PERSIST, save=save, **kwargs)
        return

    def load_source_canonical(self, **kwargs) -> pd.DataFrame:
        """returns the contracted source data as a DataFrame """
        return self.load_canonical(self.CONNECTOR_SOURCE, **kwargs)

    def load_clean_canonical(self, **kwargs) -> pd.DataFrame:
        """loads the clean pandas.DataFrame from the clean folder for this contract"""
        return self.load_canonical(self.CONNECTOR_PERSIST, **kwargs)

    def load_canonical(self, connector_name: str, **kwargs) -> pd.DataFrame:
        """returns the canonical of the referenced connector

        :param connector_name: the name or label to identify and reference the connector
        """
        canonical = super().load_canonical(connector_name=connector_name, **kwargs)
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical)
        return canonical

    def save_clean_canonical(self, canonical, auto_connectors: bool=None, **kwargs):
        """Saves the canonical to the clean files folder, auto creating the connector from template if not set"""
        if auto_connectors if isinstance(auto_connectors, bool) else True:
            if not self.pm.has_connector(self.CONNECTOR_PERSIST):
                self.set_persist()
        self.persist_canonical(connector_name=self.CONNECTOR_PERSIST, canonical=canonical, **kwargs)

    def save_quality_report(self, canonical: pd.DataFrame=None, file_type: str=None, versioned: bool=None,
                            stamped: str=None):
        """ Generates and persists the data quality

        :param canonical: the canonical to base the report on
        :param file_type: (optional) an alternative file extension to the default 'json' format
        :param versioned: (optional) if the component version should be included as part of the pattern
        :param stamped: (optional) A string of the timestamp options ['days', 'hours', 'minutes', 'seconds', 'ns']
        :return:
        """
        if isinstance(file_type, str) or isinstance(versioned, bool) or isinstance(stamped, str):
            file_pattern = self.pm.file_pattern(self.REPORT_QUALITY, file_type=file_type, versioned=versioned,
                                                stamped=stamped)
            self.set_report_persist(self.REPORT_QUALITY, uri_file=file_pattern)
        report = self.report_quality(canonical=canonical)
        self.save_report_canonical(report_connector_name=self.REPORT_QUALITY, report=report, auto_connectors=True)
        return

    def save_report_canonical(self, report_connector_name: str, report: [dict, pd.DataFrame],
                              auto_connectors: bool=None, **kwargs):
        """Saves the canonical to the data quality folder, auto creating the connector from template if not set"""
        if report_connector_name not in [self.REPORT_DICTIONARY, self.REPORT_ANALYSIS, self.REPORT_INTENT,
                                         self.REPORT_QUALITY, self.REPORT_SUMMARY, self.REPORT_PROVENANCE,
                                         self.REPORT_FIELDS] + self.REPORTS_BASE_LIST:
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
        canonical = canonical if isinstance(canonical, pd.DataFrame) else self.load_clean_canonical()
        schema_tree = schema_tree if isinstance(schema_tree, list) else canonical.columns.to_list()
        analytics = self.discover.analyse_association(canonical, columns_list=schema_tree)
        self.pm.set_canonical_schema(name=schema_name, schema=analytics)
        self.pm_persist(save=save)
        return

    def run_transition_pipeline(self, intent_levels: [str, int, list]=None):
        """Runs the components pipeline from source to persist"""
        canonical = self.load_source_canonical()
        result = self.intent_model.run_intent_pipeline(canonical, intent_levels=intent_levels, inplace=False)
        self.save_clean_canonical(result)

    def canonical_report(self, canonical, stylise: bool=True, inc_next_dom: bool=False, report_header: str=None,
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
        return self.discover.data_dictionary(df=canonical, stylise=stylise, inc_next_dom=inc_next_dom,
                                             report_header=report_header, condition=condition)

    def report_canonical_schema(self, schema_name: str=None, stylise: bool=True):
        """ presents the current canonical schema

        :param schema_name: (optional) the name of the schema
        :param stylise: if True present the report stylised.
        :return: pd.DataFrame
        """
        schema_name = schema_name if isinstance(schema_name, str) else self.REPORT_SCHEMA
        analytics = self.pm.get_canonical_schema(name=schema_name)
        return self.discover.data_schema(analysis=analytics, stylise=stylise)

    def report_attributes(self, canonical, stylise: bool=True):
        """ generates a report on the attributes and any description provided

        :param canonical: the canonical to report on
        :param stylise: if True present the report stylised.
        :return: pd.DataFrame
        """
        labels = [f'Attributes ({len(canonical.columns)})', 'dType', 'Description']
        file = []
        for c in canonical.columns.sort_values().values:
            line = [c, str(canonical[c].dtype),
                    ". ".join(self.pm.report_notes(catalog='attributes', labels=c, drop_dates=True).get('text', []))]
            file.append(line)
        df_dd = pd.DataFrame(file, columns=labels)
        if stylise:
            style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                     {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
            df_style = df_dd.style.set_table_styles(style)
            _ = df_style.applymap(self._dtype_color, subset=['dType'])
            _ = df_style.set_properties(subset=['Description'],  **{"text-align": "left"})
            _ = df_style.set_properties(subset=[f'Attributes ({len(canonical.columns)})'], **{'font-weight': 'bold',
                                                                                              'font-size': "120%"})
            return df_style
        return df_dd

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

    def report_run_book(self, stylise: bool=True):
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
                return Commons.report(df, index_header='order')
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(levels=levels))
        if stylise:
            return Commons.report(df, index_header='level')
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
        df = pd.DataFrame.from_dict(data=report)
        if stylise:
            return Commons.report(df, index_header='section', bold='label')
        return df

    def report_provenance(self, as_dict: bool=None, stylise: bool=None):
        """ a report on the provenance set as part of the domain contract

        :param as_dict: (optional) if the result should be a dictionary. Default is False
        :param stylise: (optional) if as_dict is False, if the return dataFrame should be stylised
        :return:
        """
        as_dict = as_dict if isinstance(as_dict, bool) else False
        stylise = stylise if isinstance(stylise, bool) else True
        report = self.pm.report_provenance()
        if as_dict:
            return report
        df = pd.DataFrame(report, index=['values'])
        df = df.transpose().reset_index()
        df.columns = ['provenance', 'values']
        if stylise:
            return Commons.report(df, index_header='provenance')
        return df

    def report_quality_summary(self, canonical: pd.DataFrame=None, as_dict: bool=None, stylise: bool=None):
        """ a summary quality report of the canonical

        :param canonical: (optional) the canonical to be sumarised. If not passed then loads the canonical source
        :param as_dict: (optional) if the result should be a dictionary. Default is False
        :param stylise: (optional) if as_dict is False, if the return dataFrame should be stylised
        :return: a dict or pd.DataFrame
        """
        as_dict = as_dict if isinstance(as_dict, bool) else False
        stylise = stylise if isinstance(stylise, bool) else True
        if not isinstance(canonical, pd.DataFrame):
            canonical = self._auto_transition()
        # provinance
        _provenance_headers = ['title', 'license', 'domain', 'description', 'provider',  'author', 'cost']
        _provenance_count = len(list(filter(lambda x: x in _provenance_headers, self.pm.provenance.keys())))
        _provenance_cost = self.pm.provenance.get('cost', {}).get('price', 'NA')
        # descibed
        _descibed_keys = self.pm.get_knowledge(catalog='attributes').keys()
        _descibed_count = len(list(filter(lambda x: x in canonical.columns, _descibed_keys)))
        # dictionary
        _dictionary = self.canonical_report(canonical, stylise=False)
        _total_fields = _dictionary.shape[0]
        _null_total = _dictionary['%_Null'].sum()
        _dom_fields = _dictionary['%_Dom'].sum()

        _null_columns = _dictionary['%_Null'].where(_dictionary['%_Null'] > 0.98).dropna()
        _dom_columns = _dictionary['%_Dom'].where(_dictionary['%_Dom'] > 0.98).dropna()
        _usable_fields = set(_null_columns)
        _usable_fields.update(_dom_columns)
        _numeric_fields = len(Commons.filter_headers(canonical, dtype='number'))
        _category_fields = len(Commons.filter_headers(canonical, dtype='category'))
        _date_fields = len(Commons.filter_headers(canonical, dtype='datetime'))
        _bool_fields = len(Commons.filter_headers(canonical, dtype='bool'))
        _other_fields = len(Commons.filter_headers(canonical, dtype=['category', 'datetime', 'bool',
                                                                     'number'],  exclude=True))
        _null_avg = _null_total / canonical.shape[1]
        _dom_avg = _dom_fields / canonical.shape[1]
        _quality_avg = int(round(100 - (((_null_avg + _dom_avg)/2)*100), 0))
        _correlated = self._correlated_columns(canonical)
        _usable = int(round(100 - (len(_usable_fields) / canonical.columns.size) * 100, 2))
        _field_avg = int(round(_descibed_count / canonical.shape[1] * 100, 0))
        _prov_avg = int(round(_provenance_count/len(_provenance_headers)*100, 0))
        _adjustments = self.report_intent(stylise=False).intent.size
        report = {'score': {'quality_avg': f"{_quality_avg}%", 'usability_avg': f"{_usable}%",
                            'provenance_complete': f"{_prov_avg}%", 'data_described': f"{_field_avg}%"},
                  'data_shape': {'rows': canonical.shape[0], 'columns': canonical.shape[1],
                                 'memory': Commons.bytes2human(canonical.memory_usage(deep=True).sum())},
                  'data_type': {'numeric': _numeric_fields, 'category': _category_fields,
                                'datetime': _date_fields, 'bool': _bool_fields,
                                'others': _other_fields},
                  'usability': {'mostly_null': len(_null_columns),
                                'predominance': len(_dom_columns),
                                'correlated': len(_correlated),
                                'adjustments': _adjustments},
                  'cost': {'price': _provenance_cost}}
        if as_dict:
            return report
        df = pd.DataFrame(columns=['report', 'summary', 'result'])
        counter = 0
        for index, values in report.items():
            for summary, result in values.items():
                df.loc[counter] = [index, summary, result]
                counter += 1
        if stylise:
            Commons.report(df, index_header='report', bold='summary')
        return df

    def report_quality(self, canonical: pd.DataFrame=None) -> dict:
        """A complete report of the components"""
        if not isinstance(canonical, pd.DataFrame):
            canonical = self._auto_transition()
        # meta
        report = {'meta-data': {'uid': str(uuid.uuid4()),
                                'created': str(pd.Timestamp.now()),
                                'creator': self.pm.username},
                  'description': self.pm.description,
                  'summary': self.report_quality_summary(canonical, as_dict=True)}
        # connectors
        _connectors = {}
        for connector in self.pm.connector_contract_list:
            if connector.startswith('pm_transition') or connector.startswith('template_'):
                continue
            _connector = self.pm.get_connector_contract(connector_name=connector)
            _connector_dict = {}
            if isinstance(_connector, ConnectorContract):
                kwargs = ''
                if isinstance(_connector.raw_kwargs, dict):
                    for k, v in _connector.raw_kwargs.items():
                        if len(kwargs) > 0:
                            kwargs += "  "
                        kwargs += f"{k}='{v}'"
                query = ''
                if isinstance(_connector.query, dict):
                    for k, v in _connector.query.items():
                        if len(query) > 0:
                            query += "  "
                        query += f"{k}='{v}'"
                _connector_dict['uri'] = _connector.raw_uri
                _connector_dict['version'] = _connector.version
                if len(kwargs) > 0:
                    _connector_dict['kwargs'] = kwargs
                if len(query) > 0:
                    _connector_dict['query'] = query
            _connectors[connector] = _connector_dict
        report['connectors'] = _connectors
        # provenance
        report['provenance'] = self.pm.provenance
        _provenance_headers = ['title', 'license', 'domain', 'description', 'provider',  'author', 'cost']
        _provenance_count = len(list(filter(lambda x: x in _provenance_headers, self.pm.provenance.keys())))
        # fields
        _field_count = 0
        _fields = {}
        for label, items in self.pm.get_knowledge(catalog='attributes').items():
            _fields[label] = Commons.list_formatter(items.values())
            if label in canonical.columns:
                _field_count += 1
        report['attributes'] = _fields
        # dictionary
        _data_dict = {}
        for _, row in self.canonical_report(canonical, stylise=False).iterrows():
            _data_dict[row.iloc[0]] = {}
            _att_name = None
            for index in row.index:
                if index.startswith('Attribute'):
                    _att_name = row.loc[index]
                    continue
                _data_dict[row.iloc[0]].update({index: row.loc[index]})
        report['dictionary'] = _data_dict
        # notes
        _observations = {}
        for label, items in self.pm.get_knowledge(catalog='observations').items():
            _observations[label] = Commons.list_formatter(items.values())
        _actions = {}
        for label, items in self.pm.get_knowledge(catalog='actions').items():
            _actions[label] = Commons.list_formatter(items.values())
        report['notes'] = {'observations': _observations,
                           'actions': _actions}
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

    def setup_bootstrap(self, domain: str=None, project_name: str=None, path: str=None, file_type: str=None):
        """ Creates a bootstrap Transition setup. Note this does not set the source

        :param domain: (optional) The domain this simulator sits within for example 'Healthcare' or 'Financial Services'
        :param project_name: (optional) a project name that will replace the hadron naming on file prefix
        :param path: (optional) a path added to the template path default
        :param file_type: (optional) a file_type for the persisted file, default is 'parguet'
        """
        file_type = file_type if isinstance(file_type, str) else 'parquet'
        project_name = project_name if isinstance(project_name, str) else 'hadron'
        file_name = self.pm.file_pattern(name='dataset', project=project_name.lower(), path=path, file_type=file_type,
                                         versioned=True)
        self.set_persist(uri_file=file_name)
        self.set_report_persist(connector_name=[self.REPORT_DICTIONARY, self.REPORT_SUMMARY, self.REPORT_PROVENANCE,
                                                self.REPORT_FIELDS], project=project_name, path=path)
        self.set_description(f"A domain specific {domain} transitioned {project_name} dataset for {self.pm.task_name}")
        self.set_provenance(title=f"{project_name.title()} {self.pm.task_name} Dataset ",
                            domain=domain,
                            description=f"A domain specific {domain} {project_name} dataset for {self.pm.task_name}")
        return

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

    def _correlated_columns(self, canonical: pd.DataFrame):
        """returns th percentage of useful colums"""
        threshold = 0.98
        pad = self.scratch_pad()
        canonical = pad.auto_to_category(canonical, unique_max=1000, inplace=False)
        canonical = pad.to_category_type(canonical, dtype='category', as_num=True)
        for c in canonical.columns:
            if all(Commons.valid_date(x) for x in canonical[c].dropna()):
                canonical = pad.to_date_type(canonical, dtype='datetime', as_num=True)
        canonical = Commons.filter_columns(canonical, dtype=['number'], exclude=False)
        for c in canonical.columns:
            canonical[c] = Commons.fillna(canonical[c])
        col_corr = set()
        corr_matrix = canonical.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr

    def _auto_transition(self) -> pd.DataFrame:
        """ attempts auto components on a canonical """
        pad = self.scratch_pad()
        if not self.pm.has_connector(self.CONNECTOR_SOURCE):
            raise ConnectionError("Unable to load Source canonical as the Source Connector has not been set")
        canonical = self.load_source_canonical()
        unique_max = np.log2(canonical.shape[0]) ** 2 if canonical.shape[0] > 50000 else np.sqrt(canonical.shape[0])
        return pad.auto_transition(canonical, unique_max=unique_max)
