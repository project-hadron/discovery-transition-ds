from __future__ import annotations

import pandas as pd
from aistac.components.abstract_component import AbstractComponent
from ds_discovery.managers.controller_property_manager import ControllerPropertyManager
from ds_discovery.intent.controller_intent import ControllerIntentModel

__author__ = 'Darryl Oatridge'


class Controller(AbstractComponent):
    """Controller Class for the management and overview of task components"""

    DEFAULT_MODULE = 'ds_discovery.handlers.pandas_handlers'
    DEFAULT_SOURCE_HANDLER = 'PandasSourceHandler'
    DEFAULT_PERSIST_HANDLER = 'PandasPersistHandler'

    REPORT_USE_CASE = 'use_case'

    URI_PM_REPO = None

    def __init__(self, property_manager: ControllerPropertyManager, intent_model: ControllerIntentModel,
                 default_save=None, reset_templates: bool = None, template_path: str = None,
                 template_module: str = None,
                 template_source_handler: str = None, template_persist_handler: str = None,
                 align_connectors: bool = None):
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
                 default_replace_intent: bool=None, has_contract: bool=None) -> Controller:
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
        _pm = ControllerPropertyManager(task_name=task_name, username=username)
        _intent_model = ControllerIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
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
    def from_env(cls, task_name: str=None, default_save=None, reset_templates: bool=None, align_connectors: bool=None,
                 default_save_intent: bool=None, default_intent_level: bool=None, order_next_available: bool=None,
                 default_replace_intent: bool=None, uri_pm_repo: str=None, has_contract: bool=None,
                 **kwargs) -> Controller:
        """ Class Factory Method that builds the connector handlers taking the property contract path from
        the os.environ['HADRON_PM_PATH'] or, if not found, uses the system default,
                    for Linux and IOS '/tmp/components/contracts
                    for Windows 'os.environ['AppData']\\components\\contracts'
        The following environment variables can be set:
        'HADRON_PM_PATH': the property contract path, if not found, uses the system default
        'HADRON_PM_REPO': the property contract should be initially loaded from a read only repo site such as github
        'HADRON_PM_TYPE': a file type for the property manager. If not found sets as 'json'
        'HADRON_PM_MODULE': a default module package, if not set uses component default
        'HADRON_PM_HANDLER': a default handler. if not set uses component default

        This method calls to the Factory Method 'from_uri(...)' returning the initialised class instance

         :param task_name: (optional) The reference name that uniquely identifies the ledger. Defaults to 'primary'
         :param default_save: (optional) if the configuration should be persisted
         :param reset_templates: (optional) reset connector templates from environ variables. Default True
                                (see `report_environ()`)
         :param align_connectors: (optional) resets aligned connectors to the template. default Default True
         :param default_save_intent: (optional) The default action for saving intent in the property manager
         :param default_intent_level: (optional) the default level intent should be saved at
         :param order_next_available: (optional) if the default behaviour for the order should be next available order
         :param default_replace_intent: (optional) the default replace existing intent behaviour
         :param uri_pm_repo: The read only repo link that points to the raw data path to the contracts repo directory
         :param has_contract: (optional) indicates the instance should have a property manager domain contract
         :param kwargs: to pass to the property ConnectorContract as its kwargs
         :return: the initialised class instance
         """
        # save the controllers uri_pm_repo path
        if isinstance(uri_pm_repo, str):
            cls.URI_PM_REPO = uri_pm_repo
        task_name = task_name if isinstance(task_name, str) else 'master'
        return super().from_env(task_name=task_name, default_save=default_save, reset_templates=reset_templates,
                                align_connectors=align_connectors, default_save_intent=default_save_intent,
                                default_intent_level=default_intent_level, order_next_available=order_next_available,
                                default_replace_intent=default_replace_intent, uri_pm_repo=uri_pm_repo,
                                has_contract=has_contract, **kwargs)

    @classmethod
    def scratch_pad(cls) -> ControllerIntentModel:
        """ A class method to use the Components intent methods as a scratch pad"""
        return super().scratch_pad()

    @property
    def intent_model(self) -> ControllerIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def register(self) -> ControllerIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> ControllerPropertyManager:
        """The properties manager instance"""
        return self._component_pm

    def set_use_case(self, title: str=None, domain: str=None, overview: str=None, scope: str=None,
                     situation: str=None, opportunity: str=None, actions: str=None, project_name: str=None,
                     project_lead: str=None, project_contact: str=None, stakeholder_domain: str=None,
                     stakeholder_group: str=None, stakeholder_lead: str=None, stakeholder_contact: str=None,
                     save: bool=None):
        """ sets the use_case values. Only sets those passed

        :param title: (optional) the title of the use_case
        :param domain: (optional) the domain it sits within
        :param overview: (optional) a overview of the use case
        :param scope: (optional) the scope of responsibility
        :param situation: (optional) The inferred 'Why', 'What' or 'How' and predicted 'therefore can we'
        :param opportunity: (optional) The opportunity of the situation
        :param actions: (optional) the actions to fulfil the opportunity
        :param project_name: (optional) the name of the project this use case is for
        :param project_lead: (optional) the person who is project lead
        :param project_contact: (optional) the conact information for the project lead
        :param stakeholder_domain: (optional) the domain of the stakeholders
        :param stakeholder_group: (optional) the stakeholder group name
        :param stakeholder_lead: (optional) the stakeholder lead
        :param stakeholder_contact: (optional) contact information for the stakeholder lead
        :param save: (optional) if True, save to file. Default is True
        """
        self.pm.set_use_case(title=title, domain=domain, overview=overview, scope=scope, situation=situation,
                             opportunity=opportunity, actions=actions, project_name=project_name,
                             project_lead=project_lead, project_contact=project_contact,
                             stakeholder_domain=stakeholder_domain, stakeholder_group=stakeholder_group,
                             stakeholder_lead=stakeholder_lead, stakeholder_contact=stakeholder_contact)
        self.pm_persist(save=save)

    def reset_use_case(self, save: bool=None):
        """resets the use_case back to its default values"""
        self.pm.reset_use_case()
        self.pm_persist(save)

    def report_use_case(self, as_dict: bool=None, stylise: bool=None):
        """ a report on the use_case set as part of the domain contract

        :param as_dict: (optional) if the result should be a dictionary. Default is False
        :param stylise: (optional) if as_dict is False, if the return dataFrame should be stylised
        :return:
        """
        as_dict = as_dict if isinstance(as_dict, bool) else False
        stylise = stylise if isinstance(stylise, bool) else True
        report = self.pm.report_use_case()
        if as_dict:
            return report
        report = pd.DataFrame(report, index=['values'])
        report = report.transpose().reset_index()
        report.columns = ['use_case', 'values']
        if stylise:
            return self._report(report, index_header='use_case')
        return report

    def report_tasks(self, stylise: bool=True):
        """ generates a report for all the current component task

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        report = pd.DataFrame.from_dict(data=self.pm.report_intent())
        intent_replace = {'components': 'Transition', 'synthetic_builder': 'SyntheticBuilder', 'wrangle': 'Wrangle',
                          'feature_catalog': 'FeatureCatalog', 'data_tolerance': 'DataTolerance'}
        report['component'] = report.intent.replace(to_replace=intent_replace)
        report['task'] = [x[0][10:] for x in report['parameters']]
        report['parameters'] = [x[1:] for x in report['parameters']]
        report = report.loc[:, ['level', 'order', 'component', 'task', 'parameters', 'creator']]
        if stylise:
            return self._report(report, index_header='level')
        return report

    def report_run_book(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_run_book())
        if stylise:
            return self._report(df, index_header='name')
        return df

    def report_intent(self, levels: [str, int, list] = None, stylise: bool = True):
        """ generates a report on all the intent

        :param levels: (optional) a filter on the levels. passing a single value will report a single parameterised view
        :param stylise: (optional) returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        if isinstance(levels, (int, str)):
            df = pd.DataFrame.from_dict(data=self.pm.report_intent_params(level=levels))
            if stylise:
                return self._report(df, index_header='order')
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(levels=levels))
        if stylise:
            return self._report(df, index_header='level')
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
        df = pd.DataFrame.from_dict(data=report)
        if stylise:
            df_style = df.style.set_table_styles(style).set_properties(**{'text-align': 'left'})
            _ = df_style.set_properties(subset=['section'], **{'font-weight': 'bold'})
            _ = df_style.set_properties(subset=['label', 'section'], **{'font-size': "120%"})
            return df_style
        return df

    def run_controller(self, intent_levels: [str, int, list]=None, synthetic_sizes: dict=None):
        """ Runs the components pipeline from source to persist. The synthetic_intent_sizes allows the additional
        inclusion of the special case SyntheticBuilder dataset size to be specified and applied to the different intent
        levels. If two or more Synthetic Builds are within one intent the values will be applied to all. If no size
        is given then the default saved size is used

        :param intent_levels: (optional) list of intent labels to run in the order given
        :param synthetic_sizes: (optional) a dictionary keyed by intent level with a synthetic size parameter
        """
        if not self.pm.has_intent():
            return
        if isinstance(intent_levels, (int, str, list)):
            intent_levels = self.pm.list_formatter(intent_levels)
        else:
            intent_levels = self.pm.list_formatter(self.pm.get_intent().keys())
            if self.pm.DEFAULT_INTENT_LEVEL in intent_levels:
                intent_levels.insert(0, intent_levels.pop(intent_levels.index(self.pm.DEFAULT_INTENT_LEVEL)))
        for intent in intent_levels:
            synthetic_size = synthetic_sizes.get(intent, None) if isinstance(synthetic_sizes, dict) else None
            self.intent_model.run_intent_pipeline(intent_level=intent, controller_repo=self.URI_PM_REPO,
                                                  synthetic_size=synthetic_size)
        return

    def _report(self, canonical: pd.DataFrame, index_header: str, bold: [str, list]=None, large_font: [str, list]=None):
        """ generates a stylised report

        :param canonical
        :param index_header:
        :param bold:
        :param large_font
        :return: stylised report DataFrame
        """
        pd.set_option('max_colwidth', 200)
        pd.set_option('expand_frame_repr', True)
        bold = self.pm.list_formatter(bold)
        bold.append(index_header)
        large_font = self.pm.list_formatter(large_font)
        large_font.append(index_header)
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        index = canonical[canonical[index_header].duplicated()].index.to_list()
        canonical.loc[index, index_header] = ''
        canonical = canonical.reset_index(drop=True)
        df_style = canonical.style.set_table_styles(style)
        _ = df_style.set_properties(**{'text-align': 'left'})
        if len(bold) > 0:
            _ = df_style.set_properties(subset=bold, **{'font-weight': 'bold'})
        if len(large_font) > 0:
            _ = df_style.set_properties(subset=large_font, **{'font-size': "120%"})
        return df_style
