import inspect

from aistac.intent.abstract_intent import AbstractIntentModel
from aistac.handlers.abstract_event_book import EventBookContract, EventBookFactory
from ds_discovery.engines.events.pandas_event_book import PandasEventBook
from ds_discovery.managers.event_book_property_manager import EventBookPropertyManager

__author__ = 'Darryl Oatridge'


class EventBookIntentModel(AbstractIntentModel):

    _PORTFOLIO_LEVEL = 'report_portfolio'

    def __init__(self, property_manager: EventBookPropertyManager, default_save_intent: bool=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = True
        default_intent_level = self._PORTFOLIO_LEVEL
        default_intent_order = 0
        intent_param_exclude = ['start_book', 'book_name']
        intent_type_additions = []
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, book_names: [int, str, list]=None, **kwargs):
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        :param book_names: (optional) a single or list of intent_level book names to run, if list, run in order given
        :param kwargs: additional parameters to pass beyond the contracted parameters
        """
        book_portfolio = dict()
        if self._pm.has_intent():
            # get the list of levels to run
            if isinstance(book_names, (int, str, list)):
                intent_levels = self._pm.list_formatter(book_names)
            else:
                intent_levels = sorted(self._pm.get_intent().keys())
            for level in intent_levels:
                level_key = self._pm.join(self._pm.KEY.intent_key, level)
                for order in sorted(self._pm.get(level_key, {})):
                    for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                        if method in self.__dir__():
                            # add method kwargs to the params
                            if isinstance(kwargs, dict):
                                params.update(kwargs)
                            # remove the creator param
                            _ = params.pop('intent_creator', 'Unknown')
                            # add excluded params and set to False
                            params.update({'start_book': True, 'save_intent': False})
                            book_name = level
                            eb = eval(f"self.{method}(book_name='{level}', **{params})", globals(), locals())
                            book_portfolio.update({book_name: eb})
        return book_portfolio

    def add_event_book(self, book_name: str, module_name: str=None, event_book_cls: str=None, start_book: bool=None,
                       save_intent: bool=None, intent_order: int=None, replace_intent: bool=None,
                       remove_duplicates: bool=None, **kwargs):
        """ Adds an Event Book to the intent portfolio. Note that if multiple Event Books are referenced from a single
        Event Intent, use book_name to uniquely identify each event_book within the event intent.

        :param book_name: The unique reference name for the Event Book.
        :param module_name: (optional) if passing connectors, The module name where the Event Book class can be found
        :param event_book_cls: (optional) if passing connectors. The name of the Event Book class to instantiate
        :param start_book: (optional) if the event book should be created and returned.
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param book_name: if the event has more than one book, this uniquely references the event book in the intent
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return:
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=book_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # create the event book
        if isinstance(start_book, bool) and start_book:
            if not isinstance(module_name, str) or not isinstance(event_book_cls, str):
                state_connector = kwargs.pop('state_connector', None)
                if isinstance(state_connector, str) and self._pm.has_connector(connector_name=state_connector):
                    state_connector = self._pm.get_connector_contract(connector_name=state_connector)
                events_log_connector = kwargs.pop('events_log_connector', None)
                if self._pm.has_connector(connector_name=events_log_connector):
                    events_log_connector = self._pm.get_connector_contract(connector_name=events_log_connector)
                time_distance = kwargs.pop('time_distance', 0)
                count_distance = kwargs.pop('count_distance', 0)
                events_log_distance = kwargs.pop('events_log_distance', 0)
                return PandasEventBook(book_name=book_name, time_distance=time_distance, count_distance=count_distance,
                                       events_log_distance=events_log_distance, state_connector=state_connector,
                                       events_log_connector=events_log_connector)
            else:
                event_book_contract = EventBookContract(book_name=book_name, module_name=module_name,
                                                        event_book_cls=event_book_cls, **kwargs)
                return EventBookFactory.instantiate(event_book_contract=event_book_contract)
        return

    # def _set_intend_signature(self, intent_params: dict, book_name: [int, str]=None, intent_order: int=None,
    #                           replace_intent: bool=None, remove_duplicates: bool=None, save_intent: bool=None):
    #     """ sets the intent section in the configuration file. Note: by default any identical intent, e.g.
    #     intent with the same intent (name) and the same parameter values, are removed from any level.
    #
    #     :param intent_params: a dictionary type set of configuration representing a intent section contract
    #     :param book_name: (optional) the book name that groups intent by a reference name
    #     :param intent_order: (optional) the order in which each intent should run.
    #                     If None: default's to -1
    #                     if -1: added to a level above any current instance of the intent section, level 0 if not found
    #                     if int: added to the level specified, overwriting any that already exist
    #     :param replace_intent: (optional) if the intent method exists at the level, or default level
    #                     True - replaces the current intent method with the new
    #                     False - leaves it untouched, disregarding the new intent
    #     :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
    #     :param save_intent (optional) if the intent contract should be saved to the property manager
    #     """
    #     # if not specified use the event_name as the book_name
    #     for method in intent_params.keys():
    #         if 'book_name' not in intent_params.get(method, {}).keys():
    #             intent_params.get(method).update({'book_name': book_name})
    #     super()._set_intend_signature(intent_params=intent_params, intent_level=book_name, intent_order=intent_order,
    #                                   replace_intent=replace_intent, remove_duplicates=remove_duplicates,
    #                                   save_intent=save_intent)
    #     return

