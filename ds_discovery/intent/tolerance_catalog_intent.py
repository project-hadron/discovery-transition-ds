import inspect
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
from aistac.intent.abstract_intent import AbstractIntentModel
from ds_discovery.managers.tolerance_catalog_property_manager import ToleranceCatalogPropertyManager

__author__ = 'Darryl Oatridge'


class ToleranceCatalogIntentModel(AbstractIntentModel):

    def __init__(self, property_manager: ToleranceCatalogPropertyManager, default_save_intent: bool=None,
                 default_intent_level: [str, int, float]=None, order_next_available: bool=None,
                 default_replace_intent: bool=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param default_intent_level: (optional) the default level intent should be saved at
        :param order_next_available: (optional) if the default behaviour for the order should be next available order
        :param default_replace_intent: (optional) the default replace existing intent behaviour
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = default_intent_level if isinstance(default_intent_level, (str, int, float)) else 'base'
        default_intent_order = -1 if isinstance(order_next_available, bool) and order_next_available else 0
        intent_param_exclude = []
        intent_type_additions = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64,
                                 pd.Timestamp]
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, canonical: [pd.DataFrame, str], measure: [str, list], **kwargs):
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        It is expected that all intent methods have the 'canonical' as the first parameter of the method signature
        and will contain 'save_intent' as parameters. It is also assumed that all features have a feature contract to
        save the feature outcome to

        :param canonical: this is the canonical the measure is applied too applied .
        :param measure: the measure to run
        :return distance measure
        """
        # test if there is any intent to run
        if self._pm.has_intent(level=measure):
            canonical = self._get_canonical(canonical)
            # run the feature
            level_key = self._pm.join(self._pm.KEY.intent_key, measure)
            df_measure = None
            for order in sorted(self._pm.get(level_key, {})):
                for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                    if method in self.__dir__():
                        if 'canonical' in params.keys():
                            df_measure = params.pop('canonical')
                        elif df_measure is None:
                            df_measure = canonical
                        # fail safe in case kwargs was sored as the reference
                        params.update(params.pop('kwargs', {}))
                        # add method kwargs to the params
                        if isinstance(kwargs, dict):
                            params.update(kwargs)
                        # remove the creator param
                        _ = params.pop('intent_creator', 'Unknown')
                        # add excluded params and set to False
                        params.update({'save_intent': False})
                        df_measure = eval(f"self.{method}(df_measure, **{params})", globals(), locals())
            if df_measure is None:
                raise ValueError(f"The measure '{measure}' pipeline did not run.")
            return df_measure
        raise ValueError(f"The measure '{measure}, can't be found in the tolerance catalog")

    def _get_canonical(self, data: [pd.DataFrame, str]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return deepcopy(data)
        if isinstance(data, str):
            if not self._pm.has_connector(connector_name=data):
                raise ValueError(f"The data connector name '{data}' is not in the connectors catalog")
            handler = self._pm.get_connector_handler(data)
            canonical = handler.load_canonical()
            if isinstance(canonical, dict):
                canonical = pd.DataFrame.from_dict(data=canonical, orient='columns')
            return canonical
        raise ValueError(f"The canonical format is not recognised, pd.DataFrame, "
                         f"ConnectorContract expected, {type(data)} passed")

    def tolerate_analysis(self, canonical: [pd.DataFrame, str], header: str, analytics: dict, tolerance: dict,
                          save_intent: bool=None, measure_name: [int, str]=None, intent_order: int=None,
                          replace_intent: bool=None, remove_duplicates: bool=None):
        """"""
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   measure_name=measure_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent

    def tolerate_correlation(self, canonical: [pd.DataFrame, str], header: str, analytics: dict, tolerance: dict,
                             save_intent: bool=None, measure_name: [int, str]=None, intent_order: int=None,
                             replace_intent: bool=None, remove_duplicates: bool=None):
        """"""
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   measure_name=measure_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent

    def tolerate_condition(self, canonical: [pd.DataFrame, str], header: str, analytics: dict, tolerance: dict,
                           save_intent: bool=None, measure_name: [int, str]=None, intent_order: int=None,
                           replace_intent: bool=None, remove_duplicates: bool=None):
        """"""
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   measure_name=measure_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent


    """
        PRIVATE METHODS SECTION
    """
    def _intent_builder(self, method: str, params: dict, exclude: list=None) -> dict:
        """builds the intent_params. Pass the method name and local() parameters
            Example:
                self._intent_builder(inspect.currentframe().f_code.co_name, **locals())

        :param method: the name of the method (intent). can use 'inspect.currentframe().f_code.co_name'
        :param params: the parameters passed to the method. use `locals()` in the caller method
        :param exclude: (optional) convenience parameter identifying param keys to exclude.
        :return: dict of the intent
        """
        if not isinstance(params.get('canonical', None), str):
            exclude = ['canonical']
        return super()._intent_builder(method=method, params=params, exclude=exclude)

    def _set_intend_signature(self, intent_params: dict, measure_name: [int, str]=None, intent_order: int=None,
                              replace_intent: bool=None, remove_duplicates: bool=None, save_intent: bool=None):
        """ sets the intent section in the configuration file. Note: by default any identical intent, e.g.
        intent with the same intent (name) and the same parameter values, are removed from any level.

        :param intent_params: a dictionary type set of configuration representing a intent section contract
        :param measure_name: (optional) the measure name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :param save_intent (optional) if the intent contract should be saved to the property manager
        """
        if save_intent or (not isinstance(save_intent, bool) and self._default_save_intent):
            if not isinstance(measure_name, (str, int)) or not measure_name:
                raise ValueError(f"if the intent is to be saved then a measure name must be provided")
        super()._set_intend_signature(intent_params=intent_params, intent_level=measure_name, intent_order=intent_order,
                                      replace_intent=replace_intent, remove_duplicates=remove_duplicates,
                                      save_intent=save_intent)
        return
