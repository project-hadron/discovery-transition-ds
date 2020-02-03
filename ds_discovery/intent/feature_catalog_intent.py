import inspect
import threading
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

from ds_foundation.intent.abstract_intent import AbstractIntentModel
from ds_foundation.properties.abstract_properties import AbstractPropertyManager

from ds_discovery.intent.transition_intent import TransitionIntentModel
from ds_discovery.transition.discovery import DataDiscovery, DataAnalytics

__author__ = 'Darryl Oatridge'


class FeatureCatalogIntentModel(AbstractIntentModel):
    """A set of methods to help build features as pandas.Dataframe"""

    def __init__(self, property_manager: AbstractPropertyManager, default_save_intent: bool = True,
                 intent_next_available: bool = False):
        # set all the defaults
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_intent_level = -1 if isinstance(intent_next_available, bool) and intent_next_available else 0
        intent_param_exclude = ['inplace', 'canonical']
        super().__init__(property_manager=property_manager, intent_param_exclude=intent_param_exclude,
                         default_save_intent=default_save_intent, default_intent_level=default_intent_level)

    def run_intent_pipeline(self, canonical, levels: [int, str, list]=None, inplace: bool=False, **kwargs):
        inplace = inplace if isinstance(inplace, bool) else False
        # test if there is any intent to run
        if self._pm.has_intent() and not inplace:
            # create the copy and use this for all the operations
            if not inplace:
                with threading.Lock():
                    canonical = deepcopy(canonical)
            # get the list of levels to run
            if isinstance(levels, (int, str, list)):
                levels = self._pm.list_formatter(levels)
            else:
                levels = sorted(self._pm.get_intent().keys())
            for level in levels:
                for method, params in self._pm.get_intent(level=level).items():
                    if method in self.__dir__():
                        if isinstance(kwargs, dict):
                            params.update(kwargs)
                        canonical = eval(f"self.{method}(canonical, inplace=False, save_intent=False, **{params})")
        if not inplace:
            return canonical
        return

    def remove_outliers(self, canonical, headers: list, lower_quantile: float=None, upper_quantile: float=None,
                        inplace: bool=False, save_intent: bool=True, intent_level: [int, str]=None):
        """ removes outliers by removing the boundary quantiles

        :param canonical: the DataFrame to apply
        :param headers: the header name of the columns to be included
        :param lower_quantile: (optional) the lower quantile in the range 0 < lower_quantile < 1, deafault to 0.001
        :param upper_quantile: (optional) the upper quantile in the range 0 < upper_quantile < 1, deafault to 0.999
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: the revised values
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        lower_quantile = lower_quantile if isinstance(lower_quantile, float) and 0 < lower_quantile < 1 else 0.001
        upper_quantile = upper_quantile if isinstance(upper_quantile, float) and 0 < upper_quantile < 1 else 0.999

        remove_idx = set()
        for column_name in headers:
            values = canonical[column_name]
            result = DataDiscovery.analyse_number(values, granularity=[lower_quantile, upper_quantile])
            analysis = DataAnalytics(result)
            canonical = canonical[canonical[column_name] > analysis.selection[0][1]]
            canonical = canonical[canonical[column_name] < analysis.selection[2][0]]
        return canonical

    def date_matrix(self, canonical, key, column, index_key=True, save_intent: bool=True,
                    intent_level: [int, str]=None) -> pd.DataFrame:
        """ returns a pandas.Dataframe of the datetime broken down

        :param canonical: the pandas.Dataframe to take the columns from
        :param key: the key column
        :param column: the date column
        :param index_key: if to index the key. Default to True
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: a pandas.DataFrame of the datetime breakdown
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        if key not in canonical:
            raise NameError("The key {} can't be found in the DataFrame".format(key))
        if column not in canonical:
            raise NameError("The column {} can't be found in the DataFrame".format(column))
        if not canonical[column].dtype.name.startswith('datetime'):
            raise TypeError("the column {} is not of dtype datetime".format(column))
        df_time = canonical.filter([key, column], axis=1)
        df_time['{}_yr'.format(column)] = canonical[column].dt.year
        df_time['{}_dec'.format(column)] = (canonical[column].dt.year - canonical[column].dt.year % 10).astype('category')
        df_time['{}_mon'.format(column)] = canonical[column].dt.month
        df_time['{}_day'.format(column)] = canonical[column].dt.day
        df_time['{}_dow'.format(column)] = canonical[column].dt.dayofweek
        df_time['{}_hr'.format(column)] = canonical[column].dt.hour
        df_time['{}_min'.format(column)] = canonical[column].dt.minute
        df_time['{}_woy'.format(column)] = canonical[column].dt.weekofyear
        df_time['{}_doy'.format(column)] = canonical[column].dt.dayofyear
        df_time['{}_ordinal'.format(column)] = mdates.date2num(canonical[column])

        if index_key:
            df_time = df_time.set_index(key)
        return df_time

