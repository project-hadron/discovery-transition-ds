import inspect
from copy import deepcopy
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_datetime64_any_dtype
import numpy as np
import matplotlib.dates as mdates

from aistac.intent.abstract_intent import AbstractIntentModel

from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager
from ds_discovery.transition.commons import Commons
from ds_discovery.transition.discovery import DataDiscovery, DataAnalytics
# scratch_pads
from ds_discovery.transition.transitioning import Transition
from ds_behavioral.components.synthetic_builder import SyntheticBuilder

__author__ = 'Darryl Oatridge'


class FeatureCatalogIntentModel(AbstractIntentModel):
    """A set of methods to help build features as pandas.Dataframe"""

    def __init__(self, property_manager: FeatureCatalogPropertyManager, default_save_intent: bool=None,
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
        intent_param_exclude = ['df', 'inplace', 'canonical', 'feature']
        intent_type_additions = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64,
                                 pd.Timestamp]
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, canonical, feature_name: [int, str], **kwargs):
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        It is expected that all intent methods have the 'canonical' as the first parameter of the method signature
        and will contain 'save_intent' as parameters. It is also assumed that all features have a feature contract to
        save the feature outcome to

        :param canonical: this is the iterative value all intent are applied to and returned.
        :param feature_name: features to run
        :param kwargs: additional kwargs to add to the parameterised intent, these will replace any that already exist
        :return
        """
        # test if there is any intent to run
        if self._pm.has_intent():
            # run the feature
            df_feature = canonical.copy()
            level_key = self._pm.join(self._pm.KEY.intent_key, feature_name)
            for order in sorted(self._pm.get(level_key, {})):
                for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                    if method in self.__dir__():
                        # fail safe in case kwargs was sored as the reference
                        params.update(params.pop('kwargs', {}))
                        # add method kwargs to the params
                        if isinstance(kwargs, dict):
                            params.update(kwargs)
                        # add excluded params and set to False
                        params.update({'save_intent': False})
                        df_feature = eval(f"self.{method}(df_feature, **{params})", globals(), locals())
            return df_feature
        raise ValueError(f"The feature '{feature_name}, can't be found in the feature catalog")

    def flatten_date_diff(self, canonical: pd.DataFrame, key: str, first_date: str, second_date: str,
                          aggregator: str=None, units: str=None, label: str=None, precision: int=None,
                          unindex: bool=None, save_intent: bool=None, feature_name: [int, str]=None,
                          intent_order: int=None, replace_intent: bool=None,
                          remove_duplicates: bool=None) -> pd.DataFrame:
        """ adds a column for the difference between a primary and secondary date where the primary is an early date
        than the secondary.

        :param canonical: the DataFrame containing the column headers
        :param key: the key label to group by and index on
        :param first_date: the primary or older date field
        :param second_date: the secondary or newer date field
        :param aggregator: (optional) the aggregator as a function of Pandas DataFrame 'groupby'
        :param units: (optional) The Timedelta units e.g. 'D', 'W', 'M', 'Y'. default is 'D'
        :param label: (optional) a label for the new column.
        :param precision: the precision of the result
        :param unindex: if the passed canonical should be un-index before processing
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: the DataFrame with the extra column
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        precision = precision if isinstance(precision, int) else 0
        units = units if isinstance(units, str) else 'D'
        label = label if isinstance(label, str) else f'{second_date}-{first_date}'
        df = Commons.filter_columns(canonical, headers=[key, first_date, second_date]).dropna(axis='index', how='any')
        df_diff = pd.DataFrame()
        df_diff[key] = df[key]
        result = df[second_date].sub(df[first_date], axis=0) / np.timedelta64(1, units)
        df_diff[label] = [np.round(v, precision) for v in result]
        return self.group_features(df_diff, headers=label, group_by=key, aggregator=aggregator)

    def select_feature(self, canonical: pd.DataFrame, key: [str, list], headers: [str, list]=None,
                       drop: bool=False, dtype: [str, list]=None, exclude: bool=False, regex: [str, list]=None,
                       re_ignore_case: bool=False, drop_dup_index: str=None, rename: dict=None, unindex: bool=None,
                       save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                       replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ used for feature attribution allowing columns to be selected directly from the canonical attributes

        :param canonical: the Pandas.DataFrame to get the selection from
        :param key: the key column to index on
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt' columns
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param drop_dup_index: if any duplicate index should be removed passing either 'First' or 'last'
        :param rename: a dictionary of headers to rename
        :param unindex: if the passed canonical should be un-index before processing
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: selected list of headers indexed on key
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        filter_headers = Commons.filter_headers(df=canonical, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                regex=regex, re_ignore_case=re_ignore_case)

        filter_headers += self._pm.list_formatter(key)
        df_rtn = Commons.filter_columns(canonical, headers=filter_headers, inplace=False)
        if isinstance(drop_dup_index, str) and drop_dup_index.lower() in ['first', 'last']:
            df_rtn = df_rtn.loc[~df_rtn.index.duplicated(keep=drop_dup_index)]
        if isinstance(rename, dict):
            df_rtn.rename(columns=rename, inplace=True)
        return df_rtn.set_index(key)

    def apply_condition(self, canonical: pd.DataFrame, key: [str, list], column: str, conditions: [tuple, list],
                        default: [int, float, str]=None, label: str=None, unindex: bool=None, save_intent: bool=None,
                        feature_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                        remove_duplicates: bool=None) -> pd.DataFrame:
        """ applies a selections choice based on a set of conditions to a condition to a named column
        Example: conditions = tuple('< 5',  'red')
             or: conditions = [('< 5',  'green'), ('> 5 & < 10',  'red')]

        :param canonical: the Pandas.DataFrame to get the column headers from
        :param key: the key column to index on
        :param column: a list of headers to apply the condition on,
        :param unindex: if the passed canonical should be un-index before processing
        :param conditions: a tuple or list of tuple conditions
        :param default: (optional) a value if no condition is met. 0 if not set
        :param label: (optional) a label for the new column
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
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
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = self._pm.list_formatter(key)
        label = label if isinstance(label, str) else column
        df_rtn = Commons.filter_columns(canonical, headers=key)
        str_code = ''
        if isinstance(conditions, tuple):
            conditions = [conditions]
        choices = []
        str_code = []
        for item, choice in conditions:
            choices.append(choice)
            or_list = []
            for _or in item.split('|'):
                and_list = []
                for _and in _or.split('&'):
                    and_list.append(f"(canonical[column]{_and})")
                    and_list.append('&')
                _ = and_list.pop(-1)
                _or = "".join(and_list)
                or_list.append(f"({_or})")
                or_list.append('|')
            _ = or_list.pop(-1)
            str_code.append("".join(or_list))
        selection = []
        for item in str_code:
            selection.append(eval(item, globals(), locals()))
        if isinstance(default, (str, int, float)):
            df_rtn[label] = np.select(selection, choices, default=default)
        else:
            df_rtn[label] = np.select(selection, choices)
        return df_rtn.set_index(key)

    def apply_where(self, canonical: pd.DataFrame, key: [str, list], column: [str, list], condition: str,
                    outcome: str=None, otherwise: str=None, unindex: bool=None, save_intent: bool=None,
                    feature_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                    remove_duplicates: bool=None) -> pd.DataFrame:
        """applies a condition to the given column label returning the result with the key as index. For the
        'condition', 'outcome' or 'otherwise' parameters that reference the original DataFrame, use 'canonical' as
        the DataFrame name e.g "canonical['ref_column'] > 23"

        :param canonical: the Pandas.DataFrame to get the column headers from
        :param key: the key column to index on
        :param column: a list of headers to apply the condition on,
        :param condition: (optional) the condition to apply to the header. Header must exist. examples:
                 example:  "condition= > 0.98"
                 or:       ".str.contains('shed')"
        :param outcome: an optional outcome if the condition is true
                 example: "'red'"
                 or       "canonical['alt']"
        :param otherwise an alternative to the outcome condition and has the same examples as outcome
        :param unindex: if the passed canonical should be un-index before processing
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
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
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        column = self._pm.list_formatter(column)
        key = self._pm.list_formatter(key)
        result_df = Commons.filter_columns(canonical, headers=set(column + key))
        if isinstance(condition, str):
            for label in column:
                str_code = f"result_df['{label}']{condition}"
                if isinstance(outcome, str):
                    str_code = f"{str_code}, {outcome}"
                    if isinstance(otherwise, str):
                        str_code = f"{str_code}, {otherwise}"
                result_df[label] = result_df.where(eval(str_code, globals(), locals()))
        return result_df.dropna(axis='index', how='all', inplace=False)

    def remove_outliers(self, canonical: pd.DataFrame, key: [str, list], column: str, lower_quantile: float=None,
                        upper_quantile: float=None, unindex: bool=None, save_intent: bool=None,
                        feature_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                        remove_duplicates: bool=None) -> [None, pd.DataFrame]:
        """ removes outliers by removing the boundary quantiles

        :param canonical: the DataFrame to apply
        :param key: the key column to index on
        :param column: the column name to remove outliers
        :param lower_quantile: (optional) the lower quantile in the range 0 < lower_quantile < 1, deafault to 0.001
        :param upper_quantile: (optional) the upper quantile in the range 0 < upper_quantile < 1, deafault to 0.999
        :param unindex: if the passed canonical should be un-index before processing
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: the revised values
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # intend code block on the canonical
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = self._pm.list_formatter(key)
        df_rtn = Commons.filter_columns(canonical, headers=key + [column])
        lower_quantile = lower_quantile if isinstance(lower_quantile, float) and 0 < lower_quantile < 1 else 0.00001
        upper_quantile = upper_quantile if isinstance(upper_quantile, float) and 0 < upper_quantile < 1 else 0.99999

        result = DataDiscovery.analyse_number(df_rtn[column], granularity=[lower_quantile, upper_quantile])
        analysis = DataAnalytics(result)
        df_rtn = df_rtn[(df_rtn[column] > analysis.selection[0][1]) & (df_rtn[column] < analysis.selection[2][0])]
        return df_rtn.set_index(key)

    def group_features(self, canonical: pd.DataFrame, headers: [str, list], group_by: [str, list], aggregator: str=None,
                       drop_group_by: bool=False, include_weighting: bool=False, weighting_precision: int=None,
                       remove_weighting_zeros: bool=False, remove_aggregated: bool=False, drop_dup_index: str=None,
                       unindex: bool=None, save_intent: bool=None,
                       feature_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                       remove_duplicates: bool=None) -> [None, pd.DataFrame]:
        """ groups features according to the aggregator passed. The list of aggregators are mean, sum, size, count,
        nunique, first, last, min, max, std var, describe.

        :param canonical: the pd.DataFrame to group
        :param headers: the column headers to apply the aggregation too
        :param group_by: the column headers to group by
        :param aggregator: (optional) the aggregator as a function of Pandas DataFrame 'groupby'
        :param drop_group_by: drops the group by headers
        :param include_weighting: include a percentage weighting column for each
        :param weighting_precision: a precision for the weighting values
        :param remove_aggregated: if used in conjunction with the weighting then drops the aggrigator column
        :param remove_weighting_zeros: removes zero values
        :param drop_dup_index: if any duplicate index should be removed passing either 'First' or 'last'
        :param unindex: if the passed canonical should be un-index before processing
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pd.DataFrame
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # intend code block on the canonical
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        canonical = deepcopy(canonical)
        weighting_precision = weighting_precision if isinstance(weighting_precision, int) else 3
        aggregator = aggregator if isinstance(aggregator, str) else 'sum'
        headers = self._pm.list_formatter(headers)
        group_by = self._pm.list_formatter(group_by)
        df_sub = Commons.filter_columns(canonical, headers=headers + group_by).dropna()
        df_sub = df_sub.groupby(group_by).agg(aggregator)
        if include_weighting:
            df_sub['sum'] = df_sub.sum(axis=1, numeric_only=True)
            total = df_sub['sum'].sum()
            df_sub['weighting'] = df_sub['sum'].\
                apply(lambda x: round((x / total), weighting_precision) if isinstance(x, (int, float)) else 0)
            df_sub = df_sub.drop(columns='sum')
            if remove_weighting_zeros:
                df_sub = df_sub[df_sub['weighting'] > 0]
            df_sub = df_sub.sort_values(by='weighting', ascending=False)
        if isinstance(drop_dup_index, str) and drop_dup_index.lower() in ['first', 'last']:
            df_sub = df_sub.loc[~df_sub.index.duplicated(keep=drop_dup_index)]
        if remove_aggregated:
            df_sub = df_sub.drop(headers, axis=1)
        if drop_group_by:
            df_sub = df_sub.drop(columns=group_by, errors='ignore')
        return df_sub

    def interval_categorical(self, canonical: pd.DataFrame, key: str, column: str, granularity: [int, float, list]=None,
                             lower: [int, float]=None, upper: [int, float]=None, label: str=None, categories: list=None,
                             precision: int=None, unindex: bool=None, save_intent: bool = None,
                             feature_name: [int, str] = None, intent_order: int=None, replace_intent: bool=None,
                             remove_duplicates: bool=None) -> [None, pd.DataFrame]:
        """ converts continuous representation into discrete representation through interval categorisation

        :param canonical: the dataset where the column and target can be found
        :param key: the key column to index one
        :param column: the column name to be converted
        :param granularity: (optional) the granularity of the analysis across the range. Default is 3
                int passed - represents the number of periods
                float passed - the length of each interval
                list[tuple] - specific interval periods e.g []
                list[float] - the percentile or quantities, All should fall between 0 and 1
        :param lower: (optional) the lower limit of the number value. Default min()
        :param upper: (optional) the upper limit of the number value. Default max()
        :param precision: (optional) The precision of the range and boundary values. by default set to 5.
        :param label: (optional) a label to give the new column
        :param categories:(optional)  a set of labels the same length as the intervals to name the categories
        :param unindex: if the passed canonical should be un-index before processing
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: the converted fields
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # intend code block on the canonical
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        if key not in canonical.columns:
            raise ValueError(f"The key value '{key}' is not a column name in the canonica passed")
        if column not in canonical.columns:
            raise ValueError(f"The column value '{column}' is not a column name in the canonical passed")
        granularity = 3 if not isinstance(granularity, (int, float, list)) or granularity == 0 else granularity
        precision = precision if isinstance(precision, int) else 5
        label = label if isinstance(label, str) else f"{column}_cat"
        # firstly get the granularity
        lower = canonical[column].min() if not isinstance(lower, (int, float)) else lower
        upper = canonical[column].max() if not isinstance(upper, (int, float)) else upper
        if lower >= upper:
            upper = lower
            granularity = [(lower, upper, 'both')]
        if isinstance(granularity, (int, float)):
            # if granularity float then convert frequency to intervals
            if isinstance(granularity, float):
                # make sure frequency goes beyond the upper
                _end = upper + granularity - (upper % granularity)
                periods = pd.interval_range(start=lower, end=_end, freq=granularity).drop_duplicates()
                periods = periods.to_tuples().to_list()
                granularity = []
                while len(periods) > 0:
                    period = periods.pop(0)
                    if len(periods) == 0:
                        granularity += [(period[0], period[1], 'both')]
                    else:
                        granularity += [(period[0], period[1], 'left')]
            # if granularity int then convert periods to intervals
            else:
                periods = pd.interval_range(start=lower, end=upper, periods=granularity).drop_duplicates()
                granularity = periods.to_tuples().to_list()
        if isinstance(granularity, list):
            if all(isinstance(value, tuple) for value in granularity):
                if len(granularity[0]) == 2:
                    granularity[0] = (granularity[0][0], granularity[0][1], 'both')
                granularity = [(t[0], t[1], 'right') if len(t) == 2 else t for t in granularity]
            elif all(isinstance(value, float) and 0 < value < 1 for value in granularity):
                quantiles = list(set(granularity + [0, 1.0]))
                boundaries = canonical[column].quantile(quantiles).values
                boundaries.sort()
                granularity = [(boundaries[0], boundaries[1], 'both')]
                granularity += [(boundaries[i - 1], boundaries[i], 'right') for i in range(2, boundaries.size)]
            else:
                granularity = (lower, upper, 'both')

        granularity = [(np.round(p[0], precision), np.round(p[1], precision), p[2]) for p in granularity]
        df_rtn = Commons.filter_columns(canonical, headers=[key, column])
        # now create the categories
        conditions = []
        for interval in granularity:
            lower, upper, closed = interval
            if str.lower(closed) == 'neither':
                conditions.append((df_rtn[column] > lower) & (df_rtn[column] < upper))
            elif str.lower(closed) == 'right':
                conditions.append((df_rtn[column] > lower) & (df_rtn[column] <= upper))
            elif str.lower(closed) == 'both':
                conditions.append((df_rtn[column] >= lower) & (df_rtn[column] <= upper))
            else:
                conditions.append((df_rtn[column] >= lower) & (df_rtn[column] < upper))
        if isinstance(categories, list) and len(categories) == len(conditions):
            choices = categories
        else:
            if df_rtn[column].dtype.name.startswith('int'):
                choices = [f"{int(i[0])}->{int(i[1])}" for i in granularity]
            else:
                choices = [f"{i[0]}->{i[1]}" for i in granularity]
        # noinspection PyTypeChecker
        df_rtn[label] = np.select(conditions, choices, default="<NA>")
        df_rtn[label] = df_rtn[label].astype('category', copy=False)
        df_rtn = df_rtn.drop(column, axis=1).set_index(key)
        return df_rtn

    def flatten_categorical(self, canonical: pd.DataFrame, key, column, prefix=None, index_key=True, dups=True,
                            unindex: bool=None, save_intent: bool=None,
                            feature_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                            remove_duplicates: bool=None) -> [None, pd.DataFrame]:
        """ flattens a categorical as a sum of one-hot

        :param canonical: the Dataframe to reference
        :param key: the key column to sum on
        :param column: the category type column break into the category columns
        :param prefix: a prefix for the category columns
        :param index_key: set the key as the index. Default to True
        :param dups: id duplicates should be removed from the original canonical
        :param unindex: if the passed canonical should be un-index before processing
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.Dataframe of the flattened categorical
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # intend code block on the canonical
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        canonical = deepcopy(canonical)
        if key not in canonical:
            raise NameError("The key {} can't be found in the DataFrame".format(key))
        if column not in canonical:
            raise NameError("The column {} can't be found in the DataFrame".format(column))
        if canonical[column].dtype.name != 'category':
            canonical[column] = canonical[column].astype('category')
        if prefix is None:
            prefix = column
        if not dups:
            canonical.drop_duplicates(inplace=True)
        dummy_df = pd.get_dummies(canonical[[key, column]], columns=[column], prefix=prefix)
        dummy_cols = dummy_df.columns[dummy_df.columns.to_series().str.contains('{}_'.format(prefix))]
        dummy_df = dummy_df.groupby([pd.Grouper(key=key)])[dummy_cols].sum()
        if index_key:
            dummy_df = dummy_df.set_index(key)
        return dummy_df

    def date_categorical(self, canonical: pd.DataFrame, key: [str, list], column: str, matrix: [str, list]=None,
                         label: str=None, save_intent: bool=None, feature_name: [int, str]=None,
                         intent_order: int=None, replace_intent: bool=None,
                         remove_duplicates: bool=None) -> pd.DataFrame:
        """ breaks a date down into value representations of the various parts that date.

        :param canonical: the DataFrame to take the columns from
        :param key: the key column
        :param column: the date column
        :param matrix: the matrix options (see below)
        :param label: (optional) a label alternative to the column name
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pandas.DataFrame of the datetime breakdown

        Matrix options are:
        - yr: year
        - dec: decade
        - mon: month
        - day: day
        - dow: day of week
        - hr: hour
        - min: minute
        - woy: week of year
        = doy: day of year
        - ordinal: numeric float value of date
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # intend code block on the canonical
        if key not in canonical:
            raise NameError(f"The key {key} can't be found in the DataFrame")
        if column not in canonical:
            raise NameError(f"The column {column} can't be found in the DataFrame")
        values_type = canonical[column].dtype.name.lower()
        if not values_type.startswith('date'):
            raise TypeError(f"the column {column} is not a date type")
        label = label if isinstance(label, str) else column
        matrix = self._pm.list_formatter(matrix)
        df_time = Commons.filter_columns(canonical, headers=key)
        df_time[f"{label}_yr"] = canonical[column].dt.year
        df_time[f"{label}_dec"] = (canonical[column].dt.year-canonical[column].dt.year % 10).astype('category')
        df_time[f"{label}_mon"] = canonical[column].dt.month
        df_time[f"{label}_day"] = canonical[column].dt.day
        df_time[f"{label}_dow"] = canonical[column].dt.dayofweek
        df_time[f"{label}_hr"] = canonical[column].dt.hour
        df_time[f"{label}_min"] = canonical[column].dt.minute
        df_time[f"{label}_woy"] = canonical[column].dt.weekofyear
        df_time[f"{label}_doy"] = canonical[column].dt.dayofyear
        df_time[f"{label}_ordinal"] = mdates.date2num(canonical[column])
        if matrix:
            headers = [key]
            for item in matrix:
                headers.append(f"{label}_{item}")
            df_time = Commons.filter_columns(df_time, headers=headers)
        return df_time.set_index(key)

    def replace_missing(self, canonical: pd.DataFrame, key: [str, list], headers: [str, list],
                        granularity: [int, float]=None, lower: [int, float]=None, upper: [int, float]=None,
                        nulls_list: list=None, replace_zero: [int, float]=None, precision: int=None, unindex: bool=None,
                        day_first: bool=False, year_first: bool=False, save_intent: bool=None,
                        feature_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                        remove_duplicates: bool=None) -> [None, pd.DataFrame]:
        """ imputes missing data with a weighted distribution based on the analysis of the other elements in the
            column

        :param canonical: the pd.DataFrame to replace missing values in
        :param key: the key column
        :param headers: the headers in the pd.DataFrame to apply the substitution too
        :param granularity: (optional) the granularity of the analysis across the range.
                int passed - the number of sections to break the value range into
                pd.Timedelta passed - a frequency time delta
        :param lower: (optional) the lower limit of the number or date value. Takes min() if not set
        :param upper: (optional) the upper limit of the number or date value. Takes max() if not set
        :param nulls_list: (optional) a list of nulls other than np.nan that should be considered null
        :param replace_zero: (optional) if zero what to replace the weighting value with to avoid zero probability
        :param precision: (optional) by default set to 3.
        :param unindex:
        :param day_first: if the date provided has day first
        :param year_first: if the date provided has year first
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
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
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # intend code block on the canonical
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        sim = SyntheticBuilder.scratch_pad()
        tr = Transition.scratch_pad()
        headers = self._pm.list_formatter(headers)
        if not isinstance(canonical, pd.DataFrame):
            raise TypeError("The canonical given is not a pandas DataFrame")
        nulls_list = nulls_list if isinstance(nulls_list, list) else ['nan', '']
        df_rtn = pd.DataFrame()
        df_rtn[key] = canonical[key].copy()
        for c in headers:
            col = deepcopy(canonical[c])
            # replace alternative nulls with pd.nan
            if nulls_list is not None:
                for null in self._pm.list_formatter(nulls_list):
                    col.replace(null, np.nan, inplace=True)
            size = len(col[col.isna()])
            if size > 0:
                if is_numeric_dtype(col):
                    result = DataDiscovery.analyse_number(col, granularity=granularity, lower=lower, upper=upper,
                                                          precision=precision)
                    result = DataAnalytics(result)
                    col[col.isna()] = sim.get_number(from_value=result.lower, to_value=result.upper,
                                                     weight_pattern=result.weight_pattern, precision=0, size=size,
                                                     save_intent=False)
                elif is_datetime64_any_dtype(col):
                    result = DataDiscovery.analyse_date(col, granularity=granularity, lower=lower, upper=upper,
                                                        day_first=day_first, year_first=year_first)
                    synthetic = sim.associate_analysis(result, size=size, save_intent=False)
                    col = col.apply(lambda x:  synthetic.pop(0) if x == pd.to_datetime(pd.NaT) else x)
                else:
                    result = DataDiscovery.analyse_category(col, replace_zero=replace_zero)
                    result = DataAnalytics(result)
                    col[col.isna()] = sim.get_category(selection=result.selection,
                                                       weight_pattern=result.weight_pattern, size=size,
                                                       save_intent=False)
                    col = col.astype('category')
            df_rtn[c] = col
        return df_rtn.set_index(key)

    def apply_substitution(self, canonical: pd.DataFrame, key: str, headers: [str, list],
                           save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                           replace_intent: bool=None, remove_duplicates: bool=None, **kwargs) -> pd.DataFrame:
        """ regular expression substitution of key value pairs to the value string

        :param canonical: the value to apply the substitution to
        :param key:
        :param headers:
        :param kwargs: a set of keys to replace with the values
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: the amended value
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # intend code block on the canonical
        headers = self._pm.list_formatter(headers, headers)
        df_rtn = Commons.filter_columns(canonical, headers=headers + [key])
        for c in headers:
            for k, v in kwargs.items():
                df_rtn[c].replace(k, v)
        return df_rtn

    def custom_builder(self, canonical: pd.DataFrame, code_str: str, use_exec: bool=False,
                       save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                       replace_intent: bool=None, remove_duplicates: bool=None, **kwargs) -> [None, pd.DataFrame]:
        """ enacts a code_str on a dataFrame, returning the output of the code_str or the DataFrame if using exec or
        the evaluation returns None. Note that if using the input dataframe in your code_str, it is internally
        referenced as it's parameter name 'canonical'.

        :param canonical: a pd.DataFrame used in the action
        :param code_str: an action on those column values
        :param use_exec: (optional) By default the code runs as eval if set to true exec would be used
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :param kwargs: a set of kwargs to include in any executable function
        :return: a list or pandas.DataFrame
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # intend code block on the canonical
        canonical = deepcopy(canonical)
        local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
        if 'canonical' not in local_kwargs:
            local_kwargs['canonical'] = canonical

        result = exec(code_str, globals(), local_kwargs) if use_exec else eval(code_str, globals(), local_kwargs)
        if result is None:
            return canonical
        return result

    """
        PRIVATE METHODS SECTION
    """
    def _set_intend_signature(self, intent_params: dict, feature_name: [int, str]=None, intent_order: int=None,
                              replace_intent: bool=None, remove_duplicates: bool=None, save_intent: bool=None):
        """ sets the intent section in the configuration file. Note: by default any identical intent, e.g.
        intent with the same intent (name) and the same parameter values, are removed from any level.

        :param intent_params: a dictionary type set of configuration representing a intent section contract
        :param feature_name: (optional) the feature name that groups intent by a reference name
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
            if not isinstance(feature_name, (str, int)) or not feature_name:
                raise ValueError(f"if the intent is to be saved then a feature name must be provided")
        super()._set_intend_signature(intent_params=intent_params, intent_level=feature_name, intent_order=intent_order,
                                      replace_intent=replace_intent, remove_duplicates=remove_duplicates,
                                      save_intent=save_intent)
        return
