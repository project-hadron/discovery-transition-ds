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
    def from_uri(cls, task_name: str, uri_pm_path: str, username: str, uri_pm_repo: str=None, pm_file_type: str=None,
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

    def load_source_canonical(self, **kwargs) -> pd.DataFrame:
        """returns the contracted source data as a DataFrame """
        return self.load_canonical(self.CONNECTOR_SOURCE, **kwargs)

    def load_canonical(self, connector_name: str, **kwargs) -> pd.DataFrame:
        """returns the canonical of the referenced connector

        :param connector_name: the name or label to identify and reference the connector
        """
        canonical = super().load_canonical(connector_name=connector_name, **kwargs)
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical)
        return canonical

    def load_persist_canonical(self, **kwargs) -> pd.DataFrame:
        """loads the clean pandas.DataFrame from the clean folder for this contract"""
        return self.load_canonical(self.CONNECTOR_PERSIST, **kwargs)

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
        canonical = canonical if isinstance(canonical, pd.DataFrame) else self.load_persist_canonical()
        schema_tree = schema_tree if isinstance(schema_tree, list) else canonical.columns.to_list()
        analytics = DataDiscovery.analyse_association(canonical, columns_list=schema_tree)
        self.pm.set_canonical_schema(name=schema_name, schema=analytics)
        self.pm_persist(save=save)
        return

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
        return DataDiscovery.data_dictionary(df=canonical, stylise=stylise, inc_next_dom=inc_next_dom,
                                             report_header=report_header, condition=condition)

    def report_canonical_schema(self, schema: [str, dict]=None, roots: [str, list]=None,
                                sections: [str, list]=None, elements: [str, list]=None, stylise: bool=True):
        """ presents the current canonical schema

        :param schema: (optional) the name of the schema
        :param roots: (optional) one or more tree roots
        :param sections: (optional)
        :param elements: (optional)
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
                    df = df.append(a_series, ignore_index=True)
        if stylise:
            return Commons.report(df, index_header=['root', 'section'], bold='element')
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
