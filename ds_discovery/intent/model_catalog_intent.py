import inspect
import pandas as pd
import numpy as np

from aistac.intent.abstract_intent import AbstractIntentModel
from aistac.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'

from ds_discovery.transition.commons import Commons


class ModelCatalogIntentModel(AbstractIntentModel):
    """A set of methods to help build features as pandas.Dataframe"""

    def __init__(self, property_manager: AbstractPropertyManager, default_save_intent: bool=True,
                 intent_next_available: bool=None, default_replace_intent: bool=None, intent_type_additions: list=None):
        """initialisation of the Intent class. The 'intent_param_exclude' is used to exclude commonly used method
         parameters from being included in the intent contract, this is particularly useful if passing a canonical, or
         non relevant parameters to an intent method pattern. Any named parameter in the intent_param_exclude list
         will not be included in the recorded intent contract for that method

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param intent_next_available: (optional) if the default level should be set to next available level or zero
        :param default_replace_intent: (optional) the default replace strategy for the same intent found at that level
        :param intent_type_additions: (optional) if additional data types need to be supported as an intent param
        """
        # set all the defaults
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = -1 if isinstance(intent_next_available, bool) and intent_next_available else 0
        intent_param_exclude = ['df', 'canonical']
        intent_type_additions = intent_type_additions if isinstance(intent_type_additions, list) else list()
        intent_type_additions += [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]
        super().__init__(property_manager=property_manager, intent_param_exclude=intent_param_exclude,
                         default_save_intent=default_save_intent, default_intent_level=default_intent_level,
                         default_replace_intent=default_replace_intent, intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, canonical, intent_level: [int, str, list]=None, **kwargs):
        # test if there is any intent to run
        if self._pm.has_intent():
            # get the list of levels to run
            if isinstance(intent_level, (int, str, list)):
                intent_level = Commons.list_formatter(intent_level)
            else:
                intent_level = sorted(self._pm.get_intent().keys())
            for level in intent_level:
                for method, params in self._pm.get_intent(level=level).items():
                    if method in self.__dir__():
                        if isinstance(kwargs, dict):
                            params.update(kwargs)
                        canonical = eval(f"self.{method}(canonical, save_intent=False, **{params})")
        return canonical

