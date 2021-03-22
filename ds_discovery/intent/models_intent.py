import inspect
from copy import deepcopy

import numpy as np
import pandas as pd
from aistac.handlers.abstract_handlers import HandlerFactory
from aistac.intent.abstract_intent import AbstractIntentModel

from ds_discovery.components.commons import Commons
from ds_discovery.managers.models_property_manager import ModelsPropertyManager

__author__ = 'Darryl Oatridge'


class ModelsIntentModel(AbstractIntentModel):

    TRAIN_INTENT_LEVEL = 'train_level'
    PREDICT_INTENT_LEVEL = 'predict_level'

    def __init__(self, property_manager: ModelsPropertyManager, default_save_intent: bool=None,
                 default_intent_level: bool=None, order_next_available: bool=None, default_replace_intent: bool=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param default_intent_level: (optional) the default level intent should be saved at
        :param order_next_available: (optional) if the default behaviour for the order should be next available order
        :param default_replace_intent: (optional) the default replace existing intent behaviour
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = default_intent_level if isinstance(default_intent_level, (str, int, float)) else 'A'
        default_intent_order = -1 if isinstance(order_next_available, bool) and order_next_available else 0
        intent_param_exclude = ['canonical']
        intent_type_additions = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, canonical: pd.DataFrame, intent_levels: [int, str, list]=None, **kwargs):
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        It is expected that all intent methods have the 'canonical' as the first parameter of the method signature
        and will contain 'inplace' and 'save_intent' as parameters.

        :param canonical: this is the iterative value all intent are applied to and returned.
        :param intent_levels: (optional) an single or list of levels to run, if list, run in order given
        :param kwargs: additional kwargs to add to the parameterised intent, these will replace any that already exist
        :return Canonical with parameterised intent applied or None if inplace is True
        """
        # test if there is any intent to run
        return

    def register_estimator(self, canonical: pd.DataFrame, target: str, headers: list, class_name: str,
                           module_name: str, hyper_param: dict=None, test_size: float=None, random_state: int=None,
                           save_intent: bool=None, model_name: str=None, intent_order: int=None,
                           replace_intent: bool=None, remove_duplicates: bool=None):
        """ registers and fits an estimator model returning the model fit

        :param canonical: the model canonical
        :param class_name: the name of the model class
        :param target: the model target
        :param headers: the model features header names
        :param hyper_param: (optional) hyper parameters for the model instance
        :param test_size:  (optional) the size of the test sample (default tp 0.33)
        :param random_state:  (optional) a random state value for the test sample
        :param module_name: (optional) the name of the module
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param model_name: (optional) the name of the model
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: CatBoostClassifier.
        """
        # resolve intent persist options
        _method = inspect.currentframe().f_code.co_name
        self._set_intend_signature(self._intent_builder(method=_method, params=locals()),
                                   model_name=model_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        local_intent = {}
        if model_name and self._pm.has_intent(model_name):
            local_intent = self._pm.get_intent(level=model_name, intent=_method)
        module_name = module_name if isinstance(module_name, str) else local_intent.get('module_name', None)
        X = Commons.filter_columns(canonical, headers=headers)
        y = Commons.filter_columns(canonical, headers=target)
        module = HandlerFactory.get_module(module_name='ds_behavioral')


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
        if not isinstance(params.get('canonical', None), (str, dict)):
            exclude = ['canonical']
        return super()._intent_builder(method=method, params=params, exclude=exclude)

    def _set_intend_signature(self, intent_params: dict, model_name: [int, str]=None, intent_order: int=None,
                              replace_intent: bool=None, remove_duplicates: bool=None, save_intent: bool=None):
        """ sets the intent section in the configuration file. Note: by default any identical intent, e.g.
        intent with the same intent (name) and the same parameter values, are removed from any level.

        :param intent_params: a dictionary type set of configuration representing a intent section contract
        :param model_name: (optional) the model name that groups intent by a reference name
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
            if not isinstance(model_name, (str, int)) or not model_name:
                raise ValueError(f"if the intent is to be saved then a feature name must be provided")
        super()._set_intend_signature(intent_params=intent_params, intent_level=model_name, intent_order=intent_order,
                                      replace_intent=replace_intent, remove_duplicates=remove_duplicates,
                                      save_intent=save_intent)
        return

    def _get_canonical(self, data: [pd.DataFrame, pd.Series, list, str, dict], header: str=None) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return deepcopy(data)
        if isinstance(data, dict):
            method = data.pop('method', None)
            if method is None:
                raise ValueError(f"The data dictionary has no 'method' key.")
            if str(method).startswith('@generate'):
                task_name = data.pop('task_name', None)
                if task_name is None:
                    raise ValueError(f"The data method '@generate' requires a 'task_name' key.")
                repo_uri = data.pop('repo_uri', None)
                module = HandlerFactory.get_module(module_name='ds_behavioral')
                inst = module.SyntheticBuilder.from_env(task_name=task_name, uri_pm_repo=repo_uri, default_save=False)
                size = data.pop('size', None)
                seed = data.get('seed', None)
                run_book = data.pop('run_book', None)
                result = inst.tools.run_intent_pipeline(size=size, columns=run_book, seed=seed)
                return inst.tools.frame_selection(canonical=result, save_intent=False, **data)
            else:
                raise ValueError(f"The data 'method' key {method} is not a recognised intent method")
        elif isinstance(data, (list, pd.Series)):
            header = header if isinstance(header, str) else 'default'
            return pd.DataFrame(data=deepcopy(data), columns=[header])
        elif isinstance(data, str):
            if data == '@empty':
                return pd.DataFrame()
            if not self._pm.has_connector(connector_name=data):
                raise ValueError(f"The data connector name '{data}' is not in the connectors catalog")
            handler = self._pm.get_connector_handler(data)
            canonical = handler.load_canonical()
            if isinstance(canonical, dict):
                canonical = pd.DataFrame.from_dict(data=canonical, orient='columns')
            return canonical
        raise ValueError(f"The canonical format is not recognised, pd.DataFrame, pd.Series"
                         f"str, list or dict expected, {type(data)} passed")

