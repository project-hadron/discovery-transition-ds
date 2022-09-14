from abc import abstractmethod
import pandas as pd
from aistac.components.abstract_component import AbstractComponent
from aistac.components.aistac_commons import DataAnalytics

from ds_discovery.components.commons import Commons
from ds_discovery.components.discovery import DataDiscovery, Visualisation

__author__ = 'Darryl Oatridge'


class AbstractCommonComponent(AbstractComponent):

    DEFAULT_MODULE = 'ds_discovery.handlers.pandas_handlers'
    DEFAULT_SOURCE_HANDLER = 'PandasSourceHandler'
    DEFAULT_PERSIST_HANDLER = 'PandasPersistHandler'

    @classmethod
    @abstractmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, creator: str, uri_pm_repo: str=None, pm_file_type: str=None,
                 pm_module: str=None, pm_handler: str=None, pm_kwargs: dict=None, default_save=None,
                 reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None, align_connectors: bool=None,
                 default_save_intent: bool=None, default_intent_level: bool=None, order_next_available: bool=None,
                 default_replace_intent: bool=None, has_contract: bool=None):
        return cls

    @classmethod
    def discovery_pad(cls) -> DataDiscovery:
        """ A class method to use the Components discovery methods as a scratch pad"""
        return DataDiscovery()

    @classmethod
    def visual_pad(cls) -> Visualisation:
        """ A class method to use the Components visualisation methods as a scratch pad"""
        return Visualisation()

    @property
    def discover(self) -> DataDiscovery:
        """The components instance"""
        return DataDiscovery()

    @property
    def visual(self) -> Visualisation:
        """The visualisation instance"""
        return Visualisation()

    def load_source_canonical(self, reset_changed: bool=None, has_changed: bool=None, return_empty: bool=None,
                              **kwargs) -> pd.DataFrame:
        """returns the contracted source data as a DataFrame

        :param reset_changed: (optional) resets the has_changed boolean to True
        :param has_changed: (optional) tests if the underline canonical has changed since last load else error returned
        :param return_empty: (optional) if has_changed is set, returns an empty canonical if set to True
        :param kwargs: arguments to be passed to the handler on load
        """
        return self.load_canonical(self.CONNECTOR_SOURCE, reset_changed=reset_changed, has_changed=has_changed,
                                   return_empty=return_empty, **kwargs)

    def load_canonical(self, connector_name: str, reset_changed: bool=None, has_changed: bool=None,
                       return_empty: bool=None, **kwargs) -> pd.DataFrame:
        """returns the canonical of the referenced connector

        :param connector_name: the name or label to identify and reference the connector
        :param reset_changed: (optional) resets the has_changed boolean to True
        :param has_changed: (optional) tests if the underline canonical has changed since last load else error returned
        :param return_empty: (optional) if has_changed is set, returns an empty canonical if set to True
        :param kwargs: arguments to be passed to the handler on load
        """
        canonical = super().load_canonical(connector_name=connector_name, reset_changed=reset_changed,
                                           has_changed=has_changed, return_empty=return_empty, **kwargs)
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical)
        return canonical

    def load_persist_canonical(self, reset_changed: bool=None, has_changed: bool=None, return_empty: bool=None, **kwargs) -> pd.DataFrame:
        """loads the clean pandas.DataFrame from the clean folder for this contract

        :param reset_changed: (optional) resets the has_changed boolean to True
        :param has_changed: (optional) tests if the underline canonical has changed since last load else error returned
        :param return_empty: (optional) if has_changed is set, returns an empty canonical if set to True
        :param kwargs: arguments to be passed to the handler on load
        """
        return self.load_canonical(self.CONNECTOR_PERSIST, reset_changed=reset_changed, has_changed=has_changed,
                                   return_empty=return_empty, **kwargs)

    def save_persist_canonical(self, canonical, auto_connectors: bool=None, **kwargs):
        """Saves the canonical to the clean files folder, auto creating the connector from template if not set"""
        if auto_connectors if isinstance(auto_connectors, bool) else True:
            if not self.pm.has_connector(self.CONNECTOR_PERSIST):
                self.set_persist()
        self.persist_canonical(connector_name=self.CONNECTOR_PERSIST, canonical=canonical, **kwargs)

    def add_column_description(self, column_name: str, description: str, save: bool=None):
        """ adds a description note that is included in with the 'report_column_catalog'"""
        if isinstance(description, str) and description:
            self.pm.set_intent_description(level=column_name, text=description)
            self.pm_persist(save)
        return

    def setup_bootstrap(self, domain: str=None, project_name: str=None, path: str=None, file_type: str=None,
                        description: str=None):
        """ Creates a bootstrap Transition setup. Note this does not set the source

        :param domain: (optional) The domain this simulators sits within e.g. 'Healthcare' or 'Financial Services'
        :param project_name: (optional) a project name that will replace the hadron naming on file prefix
        :param path: (optional) a path added to the template path default
        :param file_type: (optional) a file_type for the persisted file, default is 'parquet'
        :param description: (optional) a description of the component instance to overwrite the default
        """
        domain = domain.title() if isinstance(domain, str) else 'Unspecified'
        file_type = file_type if isinstance(file_type, str) else 'parquet'
        project_name = project_name if isinstance(project_name, str) else 'hadron'
        file_name = self.pm.file_pattern(name='dataset', project=project_name.lower(), path=path, file_type=file_type,
                                         versioned=True)
        self.set_persist(uri_file=file_name)
        component = self.pm.manager_name()
        if not isinstance(description, str):
            description = f"{domain} domain {component} component for {project_name} {self.pm.task_name} contract"
        self.set_description(description=description)

    def save_report_canonical(self, reports: [str, list], report_canonical: [dict, pd.DataFrame],
                              replace_connectors: bool=None, auto_connectors: bool=None, save: bool=None, **kwargs):
        """saves one or a list of reports using the TEMPLATE_PERSIST connector contract. Though a report can be of any
         name, for convention and consistency each component has a set of REPORT constants <Component>.REPORT_<NAME>
         where <Component> is the component Class name and <name> is the name of the report_canonical.

         The reports can be a simple string name or a list of names. The name list can be a string or a dictionary
         providing more detailed parameters on how to represent the report. These parameters keys are
            :key report: the name of the report
            :key file_type: (optional) a file type other than the default .json
            :key versioned: (optional) if the filename should be versioned
            :key stamped: (optional) A string of the timestamp options ['days', 'hours', 'minutes', 'seconds', 'ns']

        Some examples
            self.REPORT_SCHEMA
            [self.REPORT_NOTES, self.REPORT_SCHEMA]
            [self.REPORT_NOTES, {'report': self.REPORT_SCHEMA, 'uri_file': '<file_name>'}]
            [{'report': self.REPORT_NOTES, 'file_type': 'json'}]
            [{'report': self.REPORT_SCHEMA, 'file_type': 'csv', 'versioned': True, 'stamped': days}]

        :param reports: a report name or list of report names to save
        :param report_canonical: a relating canonical to base the report on
        :param auto_connectors: (optional) if a connector should be created automatically
        :param replace_connectors: (optional) replace any existing report connectors with these reports
        :param save: (optional) if True, save to file. Default is True
        :param kwargs: additional kwargs to pass to a Connector Contract
        """
        if not isinstance(reports, (str, list)):
            raise TypeError(f"The reports type must be a str or list, {type(reports)} type passed")
        auto_connectors = auto_connectors if isinstance(auto_connectors, bool) else True
        replace_connectors = replace_connectors if isinstance(replace_connectors, bool) else False
        _report_list = []
        for _report in self.pm.list_formatter(reports):
            if not isinstance(_report, (str, dict)):
                raise TypeError(f"The report type {type(_report)} is an unsupported type. Must be string or dict")
            if isinstance(_report, str):
                _report = {'report': _report}
            if not _report.get('report', None):
                raise ValueError(f"if not a string the reports list dict elements must have a 'report' key")
            _report_list.append(_report)
        if replace_connectors:
            self.set_report_persist(reports=_report_list, save=save)
        for _report in _report_list:
            connector_name = _report.get('report')
            if not self.pm.has_connector(connector_name):
                if auto_connectors:
                    self.set_report_persist(reports=[_report], save=save)
                else:
                    continue
            self.persist_canonical(connector_name=connector_name, canonical=report_canonical, **kwargs)
        return

    def save_canonical_schema(self, schema_name: str=None, canonical: pd.DataFrame=None, schema_tree: list=None,
                              exclude_associate: list=None, detail_numeric: bool=None, strict_typing: bool=None,
                              category_limit: int=None, save: bool=None):
        """ Saves the canonical schema to the Property contract. The default loads the clean canonical but optionally
        a canonical can be passed to base the schema on and optionally a name given other than the default

        :param schema_name: (optional) the name of the schema to save
        :param canonical: (optional) the canonical to base the schema on
        :param schema_tree: (optional) an analytics dict (see Discovery.analyse_association(...)
        :param exclude_associate: (optional) a list of dot notation tree of items to exclude from iteration
                (e.g. ['age.gender.salary']  will cut 'salary' branch from gender and all sub branches)
        :param detail_numeric: (optional) if numeric columns should have detail stats, slowing analysis. default False
        :param strict_typing: (optional) stops objects and string types being seen as categories. default True
        :param category_limit: (optional) a global cap on categories captured. default is 10
        :param save: (optional) if True, save to file. Default is True
        """
        schema_name = schema_name if isinstance(schema_name, str) else self.REPORT_SCHEMA
        canonical = canonical if isinstance(canonical, pd.DataFrame) else self.load_persist_canonical()
        schema_tree = schema_tree if isinstance(schema_tree, list) else canonical.columns.to_list()
        detail_numeric = detail_numeric if isinstance(detail_numeric, bool) else False
        strict_typing = strict_typing if isinstance(strict_typing, bool) else True
        category_limit = category_limit if isinstance(category_limit, int) else 10
        analytics = DataDiscovery.analyse_association(canonical, columns_list=schema_tree,
                                                      exclude_associate=exclude_associate,
                                                      detail_numeric=detail_numeric, strict_typing=strict_typing,
                                                      category_limit=category_limit)
        self.pm.set_canonical_schema(name=schema_name, schema=analytics)
        self.pm_persist(save=save)
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

    def report_canonical_schema(self, schema: [str, dict]=None, roots: [str, list]=None,
                                sections: [str, list]=None, elements: [str, list]=None, stylise: bool=True):
        """ presents the current canonical schema

        :param schema: (optional) the name of the schema
        :param roots: (optional) one or more tree roots
        :param sections: (optional) the section under the root
        :param elements: (optional) the element in the section
        :param stylise: if True present the report stylised.
        :return: pd.DataFrame
        """
        if not isinstance(schema, dict):
            schema = schema if isinstance(schema, str) else self.REPORT_SCHEMA
            if not self.pm.has_canonical_schema(name=schema):
                raise ValueError(f"There is no Schema currently stored under the name '{schema}'")
            schema = self.pm.get_canonical_schema(name=schema)
        df = pd.DataFrame(columns=['root', 'section', 'element', 'value'])
        root_list = DataAnalytics.get_tree_roots(analytics_blob=schema)
        if isinstance(roots, (str, list)):
            roots = Commons.list_formatter(roots)
            for root in roots:
                if root not in root_list:
                    raise ValueError(f"The root '{root}' can not be found in the analytics tree roots")
            root_list = roots
        for root_items in root_list:
            data_analysis = DataAnalytics.from_root(analytics_blob=schema, root=root_items)
            for section in data_analysis.section_names:
                if isinstance(sections, (str, list)):
                    if section not in Commons.list_formatter(sections):
                        continue
                for element, value in data_analysis.get(section).items():
                    if isinstance(elements, (str, list)):
                        if element not in Commons.list_formatter(elements):
                            continue
                    to_append = [root_items, section, element, value]
                    a_series = pd.Series(to_append, index=df.columns)
                    # df = df.append(a_series, ignore_index=True)
                    df = pd.concat([df, a_series.to_frame().transpose()], ignore_index=True)
        if stylise:
            return Commons.report(df, index_header=['root', 'section'], bold='element')
        return df

    def report_task(self, stylise: bool=True):
        """ generates a report on the source contract

        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        report = self.pm.report_task_meta()
        df = pd.DataFrame.from_dict(data=report, orient='index').reset_index()
        df.columns = ['name', 'value']
        # sort out any values that start with a $ as it throws formatting
        for c in df.columns:
            df[c] = [f"{x[1:]}" if str(x).startswith('$') else x for x in df[c]]
        if stylise:
            return Commons.report(df, index_header='name')
        return df

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
        # sort out any values that start with a $ as it throws formatting
        for c in df.columns:
            df[c] = [f"{x[1:]}" if str(x).startswith('$') else x for x in df[c]]
        if stylise:
            return Commons.report(df, index_header='connector_name')
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

    def report_run_book(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_run_book())
        if stylise:
            return Commons.report(df, index_header='name')
        return df

    def report_environ(self, hide_not_set: bool=True, stylise: bool=True):
        """ generates a report on all the intent

        :param hide_not_set: hide environ keys that are not set.
        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        df = pd.DataFrame.from_dict(data=super().report_environ(hide_not_set), orient='index').reset_index()
        df.columns = ["environ", "value"]
        if stylise:
            return Commons.report(df, index_header='environ')
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
