import numpy as np
import pandas as pd
from typing import Any
from ds_discovery.components.commons import Commons
from aistac.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.intent.abstract_builder_get_intent import AbstractBuilderGetIntent
from ds_discovery.intent.abstract_builder_correlate_intent import AbstractBuilderCorrelateIntent
from ds_discovery.intent.abstract_builder_model_intent import AbstractBuilderModelIntent
from ds_discovery.intent.abstract_builder_frame_intent import AbstractBuilderFrameIntent

__author__ = 'Darryl Oatridge'


class AbstractBuilderIntentModel(AbstractBuilderGetIntent, AbstractBuilderCorrelateIntent, AbstractBuilderModelIntent,
                                 AbstractBuilderFrameIntent):

    _INTENT_PARAMS = ['self', 'save_intent', 'column_name', 'intent_order',
                      'replace_intent', 'remove_duplicates', 'seed']

    def __init__(self, property_manager: AbstractPropertyManager, default_save_intent: bool=None,
                 default_intent_level: [str, int, float]=None, default_intent_order: int=None,
                 default_replace_intent: bool=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param default_intent_level: (optional) the default level intent should be saved at
        :param default_intent_order: (optional) if the default behaviour for the order should be next available order
        :param default_replace_intent: (optional) the default replace existing intent behaviour
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = default_intent_level if isinstance(default_intent_level, (str, int, float)) else 'A'
        default_intent_order = default_intent_order if isinstance(default_intent_order, int) else 0
        intent_param_exclude = ['size']
        intent_type_additions = [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, pd.Timestamp]
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, canonical: Any=None, intent_levels: [str, int, list]=None, run_book: str=None,
                            seed: int=None, simulate: bool=None, **kwargs) -> pd.DataFrame:
        """Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract. The whole run can be seeded though any parameterised seeding in the intent
        contracts will take precedence

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param intent_levels: (optional) a single or list of intent_level to run in order given
        :param run_book: (optional) a preset runbook of intent_level to run in order
        :param seed: (optional) a seed value that will be applied across the run: default to None
        :param simulate: (optional) returns a report of the order of run and return the indexed column order of run
        :return: a pandas dataframe
        """
        simulate = simulate if isinstance(simulate, bool) else False
        col_sim = {"column": [], "order": [], "method": []}
        # legacy
        if 'size' in kwargs.keys():
            canonical = kwargs.pop('size')
        canonical = self._get_canonical(canonical)
        size = canonical.shape[0] if canonical.shape[0] > 0 else 1000
        # test if there is any intent to run
        if self._pm.has_intent():
            # get the list of levels to run
            if isinstance(intent_levels, (str, list)):
                column_names = Commons.list_formatter(intent_levels)
            elif isinstance(run_book, str) and self._pm.has_run_book(book_name=run_book):
                column_names = self._pm.get_run_book(book_name=run_book)
            else:
                # put all the intent in order of model, get, correlate, associate
                _model = []
                _get = []
                _correlate = []
                _frame_start = []
                _frame_end = []
                for column in self._pm.get_intent().keys():
                    for order in self._pm.get(self._pm.join(self._pm.KEY.intent_key, column), {}):
                        for method in self._pm.get(self._pm.join(self._pm.KEY.intent_key, column, order), {}).keys():
                            if str(method).startswith('get_'):
                                if column in _correlate + _frame_start + _frame_end:
                                    continue
                                _get.append(column)
                            elif str(method).startswith('model_'):
                                _model.append(column)
                            elif str(method).startswith('correlate_'):
                                if column in _get:
                                    _get.remove(column)
                                _correlate.append(column)
                            elif str(method).startswith('frame_'):
                                if column in _get:
                                    _get.remove(column)
                                if str(method).startswith('frame_starter'):
                                    _frame_start.append(column)
                                else:
                                    _frame_end.append(column)
                column_names = Commons.list_unique(_frame_start + _get + _model + _correlate + _frame_end)
            for column in column_names:
                level_key = self._pm.join(self._pm.KEY.intent_key, column)
                for order in sorted(self._pm.get(level_key, {})):
                    for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                        try:
                            if method in self.__dir__():
                                if simulate:
                                    col_sim['column'].append(column)
                                    col_sim['order'].append(order)
                                    col_sim['method'].append(method)
                                    continue
                                result = []
                                params.update(params.pop('kwargs', {}))
                                if isinstance(seed, int):
                                    params.update({'seed': seed})
                                _ = params.pop('intent_creator', 'Unknown')
                                if str(method).startswith('get_'):
                                    result = eval(f"self.{method}(size=size, save_intent=False, **params)",
                                                  globals(), locals())
                                elif str(method).startswith('correlate_'):
                                    result = eval(f"self.{method}(canonical=canonical, save_intent=False, **params)",
                                                  globals(), locals())
                                elif str(method).startswith('model_'):
                                    canonical = eval(f"self.{method}(canonical=canonical, save_intent=False, **params)",
                                                     globals(), locals())
                                    continue
                                elif str(method).startswith('frame_starter'):
                                    canonical = self._get_canonical(params.pop('canonical', canonical), deep_copy=False)
                                    size = canonical.shape[0]
                                    canonical = eval(f"self.{method}(canonical=canonical, save_intent=False, **params)",
                                                     globals(), locals())
                                    continue
                                elif str(method).startswith('frame_'):
                                    canonical = eval(f"self.{method}(canonical=canonical, save_intent=False, **params)",
                                                     globals(), locals())
                                    continue
                                if 0 < size != len(result):
                                    raise IndexError(f"The index size of '{column}' is '{len(result)}', "
                                                     f"should be {size}")
                                canonical[column] = result
                        except ValueError as ve:
                            raise ValueError(f"intent '{column}', order '{order}', method '{method}' failed with: {ve}")
                        except TypeError as te:
                            raise TypeError(f"intent '{column}', order '{order}', method '{method}' failed with: {te}")
        if simulate:
            return pd.DataFrame.from_dict(col_sim)
        return canonical

