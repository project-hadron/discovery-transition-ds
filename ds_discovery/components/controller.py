from __future__ import annotations

import datetime
import threading
import time
import numpy as np
import pandas as pd
from aistac import ConnectorContract
from aistac.components.abstract_component import AbstractComponent

from ds_discovery import EventBookPortfolio
from ds_discovery.components.commons import Commons
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

    eb_portfolio: EventBookPortfolio

    def __init__(self, property_manager: ControllerPropertyManager, intent_model: ControllerIntentModel,
                 default_save=None, reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None,
                 align_connectors: bool=None):
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
        self.eb_portfolio = EventBookPortfolio.from_memory(has_contract=False)
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

    def remove_all_tasks(self, save: bool=None):
        """removes all tasks"""
        for level in self.pm.get_intent():
            self.pm.remove_intent(level=level)
        self.pm_persist(save)

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
        :param project_contact: (optional) the contact information for the project lead
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
        intent_replace = {'transition': 'Transition', 'synthetic_builder': 'SyntheticBuilder', 'wrangle': 'Wrangle',
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
        report = pd.DataFrame(self.pm.report_run_book())
        explode = report.explode(column='run_book', ignore_index=True)
        canonical = explode.join(pd.json_normalize(explode['run_book'])).drop(columns=['run_book']).replace(np.nan, '')
        if stylise:
            return Commons.report(canonical, index_header='name')
        return canonical

    def report_intent(self, levels: [str, int, list]=None, stylise: bool = True):
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

    def run_controller(self, run_book: [str, list, dict]=None, mod_tasks: [list, dict]=None, repeat: int=None,
                       sleep: int=None, run_time: int=None, source_check_uri: str=None, run_cycle_report: str=None):
        """ Runs the components pipeline based on the runbook instructions. The run_book can be a simple list of
        controller registered task name that will run in the given order passing the resulting outcome of one to the
        input of the next, a list of task dictionaries that contain more detailed run commands (see below) or a
        mixture of task names and task dictionaries. If no runbook is given,  all registered task names are taken from
        the intent list and run in no particular order and independent of each other using their connector source and
        persist as data input

        run book list elements can be a dictionary contain more detailed run commands for a particular task. if a
        dictionary is used it must contain the task_name as a minimum
        The dictionary keys are as follows:
            - task_name: The task name (intent level) this run detail is applied to
            - source: (optional) The task name of the source or '@<intent_name>' to reference a known event book
            - persist: (optional) if true persist to an event book named after the intent. if False do nothing
            - end_source (optional) if this task will be the last to use the source, remove it from memory on completion

        mod_tasks are a dictionary of modifications to tasks in the runbook. The run_book will still define the run
        order and modification tasks not found in the run_book will be ignored. The dictionary is indexed on the task
        name with the modifications a sub-dictionary of name value pairs.
            for example: mod_tasks = {'my_synth_gen': {source: 1000}}
            changes 'my_synth_gen' to now have a source reference of 1000 meaning it will generate 1000 synthetic rows.

        The run_cycle_report automatically generates the connector contract with the name 'run_cycle_report'. To reload
        the report for observation use the controller method 'load_canonical(...) passing the name 'run_cycle_report'.

        :param run_book: (optional) a run_book reference, a list of task names (intent levels)
        :param mod_tasks: (optional) a dict of modifications that override an existing task in the runbook
        :param repeat: (optional) the number of times this intent should be repeated. None or -1 -> never, 0 -> forever
        :param sleep: (optional) number of seconds to sleep before repeating
        :param run_time: (optional) number of seconds to run the controller using repeat and sleep cycles time is up
        :param source_check_uri: (optional) The source uri to check for change since last controller instance cycle
        :param run_cycle_report: (optional) a full name for the run cycle report
        """
        _lock = threading.Lock()
        mod_tasks = mod_tasks if isinstance(mod_tasks, (list, dict)) else []
        if isinstance(run_cycle_report, str):
            self.add_connector_persist(connector_name='run_cycle_report', uri_file=run_cycle_report)
            df_report = pd.DataFrame(columns=['time', 'text'])
        if isinstance(mod_tasks, dict):
            mod_tasks = [mod_tasks]
        if not self.pm.has_intent():
            return
        if isinstance(run_book, str):
            if not self.pm.has_run_book(run_book) and run_book not in self.pm.get_intent().keys():
                raise ValueError(f"The run book or intent level '{run_book}' can not be found in the controller")
            if self.pm.has_run_book(run_book):
                intent_levels = self.pm.get_run_book(book_name=run_book)
            else:
                intent_levels = Commons.list_formatter(run_book)
        elif isinstance(run_book, list):
            intent_levels = run_book
        elif isinstance(run_book, dict):
            intent_levels = [run_book]
        elif self.pm.has_run_book(book_name=self.pm.PRIMARY_RUN_BOOK):
            intent_levels = self.pm.get_run_book(book_name=self.pm.PRIMARY_RUN_BOOK)
        else:
            intent_levels = Commons.list_formatter(self.pm.get_intent().keys())
            # always put the DEFAULT_INTENT_LEVEL first
            if self.pm.DEFAULT_INTENT_LEVEL in intent_levels:
                intent_levels.insert(0, intent_levels.pop(intent_levels.index(self.pm.DEFAULT_INTENT_LEVEL)))
        for idx in range(len(intent_levels)):
            if isinstance(intent_levels[idx], str):
                intent_levels[idx] = {'task': intent_levels[idx]}
            if 'end_source' not in intent_levels[idx].keys():
                intent_levels[idx].update({'end_source': False})
            if 'persist' not in intent_levels[idx].keys():
                _persist = True if idx == len(intent_levels) - 1 else False
                intent_levels[idx].update({'persist': _persist})
            if 'source' not in intent_levels[idx].keys():
                _level0 = self.pm.get_intent(intent_levels[idx].get('task')).get('0', {})
                if 'synthetic_builder' in _level0.keys():
                    _source = int(_level0.get('synthetic_builder', {}).get('size', 1000))
                else:
                    _source = f'@{self.CONNECTOR_SOURCE}' if idx == 0 else intent_levels[idx - 1].get('task')
                intent_levels[idx].update({'source': _source})
            if intent_levels[idx].get('source') == '@':
                intent_levels[idx].update({'source': f'@{self.CONNECTOR_SOURCE}'})
            for mod in mod_tasks:
                if intent_levels[idx].get('task') in mod.keys():
                    intent_levels[idx].update(mod.get(intent_levels[idx].get('task'), {}))
        handler = None
        if isinstance(source_check_uri, str):
            self.add_connector_uri(connector_name='run_cycle_report', uri=source_check_uri)
            handler = self.pm.get_connector_handler(connector_name='run_cycle_report')
        repeat = repeat if isinstance(repeat, int) and repeat > 0 else 1
        run_time = run_time if isinstance(run_time, int) else 0
        if run_time > 0 and not isinstance(sleep, int):
            sleep = 1
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=run_time)
        run_count = 0
        df_report = pd.DataFrame(columns=['time', 'text'])
        while True: # run_time always runs once
            if isinstance(run_cycle_report, str):
                df_report.loc[len(df_report.index)] = [datetime.datetime.now(), f'start run-cycle {run_count}']
            for count in range(repeat):
                if isinstance(run_cycle_report, str):
                    df_report.loc[len(df_report.index)] = [datetime.datetime.now(), f'start task cycle {count}']
                if handler and handler.exists():
                    if handler.has_changed():
                        handler.reset_changed(False)
                    else:
                        if isinstance(run_cycle_report, str):
                            df_report.loc[len(df_report.index)] = [datetime.datetime.now(), 'Source has not changed']
                        if isinstance(sleep, int) and count < repeat - 1:
                            time.sleep(sleep)
                        continue
                for intent in intent_levels:
                    task = intent.get('task')
                    source = intent.get('source', '')
                    to_persist = intent.get('persist')
                    end_source = intent.get('end_source', False)
                    if isinstance(run_cycle_report, str):
                        df_report.loc[len(df_report.index)] = [datetime.datetime.now(), f'running {task}']
                    if isinstance(source, int) or (isinstance(source, str) and source.startswith('@')):
                        canonical = source
                    elif isinstance(source, str) and source.isnumeric():
                        canonical = int(source)
                    else:
                        if self.eb_portfolio.is_active_book(source):
                            canonical = self.eb_portfolio.current_state(source)
                            if end_source:
                                self.eb_portfolio.remove_event_books(book_names=task)
                        else:
                            raise ValueError(f"The task '{task}' source event book '{source}' does not exist")
                    # get the result
                    canonical = self.intent_model.run_intent_pipeline(canonical=canonical, intent_level=task,
                                                                      persist_result=to_persist,
                                                                      controller_repo=self.URI_PM_REPO)
                    if isinstance(run_cycle_report, str):
                        df_report.loc[len(df_report.index)] = [datetime.datetime.now(), f"canonical shape is "
                                                                                        f"{canonical.shape}"]
                    if to_persist:
                        continue
                    if self.eb_portfolio.is_event_book(task):
                        self.eb_portfolio.remove_event_books(task)
                    eb = self.eb_portfolio.intent_model.add_event_book(book_name=task, start_book=True)
                    self.eb_portfolio.add_book_to_portfolio(book_name=task, event_book=eb)
                    self.eb_portfolio.add_event(book_name=task, event=canonical)
                self.eb_portfolio.reset_portfolio()
                if isinstance(run_cycle_report, str):
                    df_report.loc[len(df_report.index)] = [datetime.datetime.now(), 'tasks complete']
                if isinstance(sleep, int) and count < repeat-1:
                    time.sleep(sleep)
            if isinstance(run_cycle_report, str):
                run_count += 1
            if end_time < datetime.datetime.now():
                break
            else:
                time.sleep(sleep)
        if isinstance(run_cycle_report, str):
            df_report.loc[len(df_report.index)] = [datetime.datetime.now(), 'end of report']
            self.save_canonical(connector_name='run_cycle_report', canonical=df_report)
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
        bold = Commons.list_formatter(bold)
        bold.append(index_header)
        large_font = Commons.list_formatter(large_font)
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

    @staticmethod
    def runbook2dict(task: str, source: [str, int]=None, persist: bool=None, end_source: bool=None) -> dict:
        """ a utility method to help build feature conditions by aligning method parameters with dictionary format.

        :param task: the task name (intent level) name this runbook is applied too or a number if synthetic generation
        :param source: (optional) a task name indicating where the source of this task will come from. Optionally:
                            '@' will use the source contract of this task as the source input.
                            '@<connector>' will use the connector contract that must exist in the task connectors
        :param persist: (optional) if true persist to an event book named after the intent. if False do nothing
        :param end_source: (optional) if true indicates the source canonical can be removed from in-memory
        :return: dictionary of the parameters
        """
        return Commons.param2dict(**locals())
