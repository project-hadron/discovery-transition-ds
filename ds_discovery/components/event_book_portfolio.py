from __future__ import annotations

import os
from datetime import datetime
from typing import Any
import pandas as pd
from aistac.handlers.abstract_event_book import AbstractEventBook
from aistac.handlers.abstract_handlers import ConnectorContract
from ds_discovery.components.abstract_common_component import AbstractCommonComponent
from ds_discovery.managers.event_book_property_manager import EventBookPropertyManager
from ds_discovery.intent.event_book_intent import EventBookIntentModel

__author__ = 'Darryl Oatridge'


class EventBookPortfolio(AbstractCommonComponent):

    __book_portfolio = dict()
    BOOK_TEMPLATE_CONNECTOR = 'book_template_connector'

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, creator: str, uri_pm_repo: str=None, pm_file_type: str=None,
                 pm_module: str=None, pm_handler: str=None, pm_kwargs: dict=None, default_save=None,
                 reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None, align_connectors: bool=None,
                 default_save_intent: bool=None, default_intent_level: bool=None, order_next_available: bool=None,
                 default_replace_intent: bool=None, has_contract: bool=None) -> EventBookPortfolio:
        """ Class Factory Method to instantiates the components application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.
        See class inline docs for an example method

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param creator: A user name for this task activity.
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
        pm_file_type = pm_file_type if isinstance(pm_file_type, str) else 'parquet'
        pm_module = pm_module if isinstance(pm_module, str) else cls.DEFAULT_MODULE
        pm_handler = pm_handler if isinstance(pm_handler, str) else cls.DEFAULT_PERSIST_HANDLER
        _pm = EventBookPropertyManager(task_name=task_name, creator=creator)
        _intent_model = EventBookIntentModel(property_manager=_pm, default_save_intent=default_save_intent)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, default_save=default_save,
                                 uri_pm_repo=uri_pm_repo, pm_file_type=pm_file_type, pm_module=pm_module,
                                 pm_handler=pm_handler, pm_kwargs=pm_kwargs, has_contract=has_contract)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save,
                   reset_templates=reset_templates, template_path=template_path, template_module=template_module,
                   template_source_handler=template_source_handler, template_persist_handler=template_persist_handler,
                   align_connectors=align_connectors)

    @property
    def intent_model(self) -> EventBookIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> EventBookPropertyManager:
        """The properties manager instance"""
        return self._component_pm

    def is_event_book(self, book_name: str) -> bool:
        """Tests if an event book has been added"""
        return self.pm.has_intent(level=book_name)

    def get_active_book(self, book_name: str):
        """retrieves an event book instance from the report_portfolio by name"""
        if not self.is_active_book(book_name=book_name):
            raise ValueError(f"The event book instance '{book_name}' is not active.")
        return self.__book_portfolio.get(book_name)

    def is_active_book(self, book_name: str) -> bool:
        """Tests if an event book instance exists"""
        if book_name in self.__book_portfolio.keys():
            return True
        return False

    def add_book_to_portfolio(self, book_name: str, event_book: Any):
        """

        :param book_name: the unique name of event book
        :param event_book: a concrete implementation of an AbstractEventBook
        """
        if not isinstance(event_book, AbstractEventBook):
            raise ValueError(f"The event book '{event_book}' must be a concrete implementation of an AbstractEventBook")
        if self.is_event_book(book_name):
            self.remove_event_books(book_names=book_name)
        self.__book_portfolio[book_name] = event_book
        return

    def start_portfolio(self, exclude_books: [str, list]=None,):
        """runs the intent pipeline

        :param exclude_books: (optional) a list of book_names in the portfolio not to start
        """
        portfolio = self.intent_model.run_intent_pipeline(exclude_books=exclude_books)
        self.__book_portfolio.update(portfolio)
        return

    def get_book_contract(self, book_name: str) -> ConnectorContract:
        """ retrieves a named event book connector

        :param book_name: the unique name of event book
        :return: ConnectorContract
        """
        return self.pm.get_connector_contract(connector_name=book_name)

    def set_book_contract_template(self, uri_path: str=None, module_name: str=None, handler: str=None,
                                   save: bool=None, **kwargs):
        """ sets the book template connector that is used as the base for all event book persistence. for
        parameters not given, the persist connector template is used.

        :param uri_path: a uri path
        :param module_name: a module package name
        :param handler: a handler
        :param save: override of the default save action set at initialisation.
        :param kwargs: additional kwargs
        """
        template = self.pm.get_connector_contract(self.TEMPLATE_PERSIST)
        uri_path = uri_path if isinstance(uri_path, str) else template.raw_uri
        module_name = module_name if isinstance(module_name, str) else template.raw_module_name
        handler = handler if isinstance(handler, str) else template.raw_handler
        if not isinstance(kwargs, dict):
            kwargs = {}
        kwargs.update(template.raw_kwargs)
        book_template = ConnectorContract(uri=uri_path, module_name=module_name, handler=handler, **kwargs)
        if self.pm.has_connector(self.BOOK_TEMPLATE_CONNECTOR):
            self.remove_connector_contract(connector_name=self.BOOK_TEMPLATE_CONNECTOR)
        self.pm.set_connector_contract(connector_name=self.BOOK_TEMPLATE_CONNECTOR, connector_contract=book_template)
        self.pm_persist(save=save)
        return

    def add_book_contract(self, book_name: str, with_log: bool=None, file_type: str=None, versioned: bool=None,
                          stamped: bool=None, save: bool=None, **kwargs):
        """ adds an event book connector using the book connector template and appending a book pattern to the URI path

        :param book_name: the name of the event book
        :param with_log: (optional) if an events log connector should be created
        :param file_type: (optional) a file type extension. defaults to 'pickle'
        :param versioned: (optional) if the connector uri should be versioned
        :param stamped: (optional) if the connector uri should be timestamped
        :param save: (optional) override of the default save action set at initialisation.
        :param kwargs: extra kwargs to pass to the connector
        """
        if not self.pm.has_connector(connector_name=self.BOOK_TEMPLATE_CONNECTOR):
            raise ConnectionError(f"The book template connector has not been set")
        template = self.pm.get_connector_contract(self.BOOK_TEMPLATE_CONNECTOR)
        uri_file = self.pm.file_pattern(name=book_name, file_type=file_type, versioned=versioned,
                                        stamped=stamped)
        uri = os.path.join(template.path, uri_file)
        if not isinstance(kwargs, dict):
            kwargs = {}
        kwargs.update(template.raw_kwargs)
        cc = ConnectorContract(uri=uri, module_name=template.module_name, handler=template.handler, **kwargs)
        self.add_connector_contract(connector_name=book_name, connector_contract=cc, template_aligned=True, save=save)
        # add the log persist
        if isinstance(with_log, bool) and with_log:
            log = f"{book_name}_log"
            uri_log = self.pm.file_pattern(name=log, file_type=file_type)
            lc = ConnectorContract(uri=uri, module_name=template.module_name, handler=template.handler, **kwargs)
            self.add_connector_contract(connector_name=uri_log, connector_contract=lc, template_aligned=True, save=save)
        return

    def persist_state(self, book_name: str):
        """ persists the current state of an event book"""
        if self.is_active_book(book_name=book_name):
            state = self.current_state(book_name=book_name)
            self.persist_canonical(connector_name=book_name, canonical=state)
        return

    def load_state(self, book_name: str):
        """loads the current persisted state"""
        return self.load_canonical(connector_name=book_name)

    def stop_active_books(self, book_names: [str, list]):
        """stops the event books listed in the book names"""
        book_names = self.pm.list_formatter(book_names)
        for book in book_names:
            _ = self.__book_portfolio.pop(book, None)
        return

    def reset_portfolio(self):
        """resets the event book report_portfolio removing all running event books and intent"""
        self.__book_portfolio.clear()
        self.pm.reset_intents()
        return

    def remove_event_books(self, book_names: [str, list], save: bool=None):
        """removes the event book"""
        book_names = self.pm.list_formatter(book_names)
        for book in book_names:
            state_name = book
            events_log_name = "_".join([book, '_log'])
            # remove the connectors
            if self.pm.has_connector(state_name):
                self.remove_connector_contract(state_name)
            if self.pm.has_connector(events_log_name):
                self.remove_connector_contract(events_log_name)
            # remove the intent
            self.pm.remove_intent(intent_param=book)
            # remove the report_portfolio entry
            if book in self.__book_portfolio.keys():
                self.__book_portfolio.pop(book)
            self.pm_persist(save=save)
        return

    def current_state(self, book_name: str, fillna: bool=None) -> (datetime, Any):
        event_book = self.get_active_book(book_name=book_name)
        return event_book.current_state(fillna=fillna)

    def add_event(self, book_name: str, event: Any):
        return self.get_active_book(book_name=book_name).add_event(event=event)

    def increment_event(self, book_name: str, event: Any):
        return self.get_active_book(book_name=book_name).increment_event(event=event)

    def decrement_event(self, book_name: str, event: Any):
        return self.get_active_book(book_name=book_name).decrement_event(event=event)

    def report_portfolio(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(), orient='columns')
        df['active'] = df['intent'].isin(list(self.__book_portfolio.keys()))
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
