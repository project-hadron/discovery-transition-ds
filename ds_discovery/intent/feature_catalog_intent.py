import inspect
import threading
from copy import deepcopy
import pandas as pd
from ds_engines.engines.event_books.pandas_event_book import PandasEventBook
from pandas.core.dtypes.common import is_numeric_dtype, is_datetime64_any_dtype
import numpy as np
import matplotlib.dates as mdates

from aistac.intent.abstract_intent import AbstractIntentModel
from aistac.properties.abstract_properties import AbstractPropertyManager

from ds_discovery.intent.transition_intent import TransitionIntentModel
from ds_discovery.transition.commons import Commons
from ds_discovery.transition.discovery import DataDiscovery, DataAnalytics

__author__ = 'Darryl Oatridge'


class FeatureCatalogIntentModel(AbstractIntentModel):
    """A set of methods to help build features as pandas.Dataframe"""

    def __init__(self, property_manager: AbstractPropertyManager, default_save_intent: bool=True,
                 default_intent_level: [int, float, str]=None, default_replace_intent: bool=None,
                 intent_type_additions: list=None):
        """initialisation of the Intent class. The 'intent_param_exclude' is used to exclude commonly used method
         parameters from being included in the intent contract, this is particularly useful if passing a canonical, or
         non relevant parameters to an intent method pattern. Any named parameter in the intent_param_exclude list
         will not be included in the recorded intent contract for that method

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param default_intent_level: (optional) The default intent level
        :param default_replace_intent: (optional) the default replace strategy for the same intent found at that level
        :param intent_type_additions: (optional) if additional data types need to be supported as an intent param
        """
        # set all the defaults
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = default_intent_level if isinstance(default_intent_level, (int, float, str)) else -1
        intent_param_exclude = ['df', 'canonical', 'feature', 'inplace']
        intent_type_additions = intent_type_additions if isinstance(intent_type_additions, list) else list()
        intent_type_additions += [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]
        super().__init__(property_manager=property_manager, intent_param_exclude=intent_param_exclude,
                         default_save_intent=default_save_intent, default_intent_level=default_intent_level,
                         default_replace_intent=default_replace_intent, intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, canonical: pd.DataFrame, event_book: PandasEventBook, event_type: str=None,
                            fillna: bool=None, intent_levels: [int, str, list]=None, **kwargs):
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        :param canonical: this is the iterative value all intent are applied to and returned.
        :param event_book: the event book to pass the resulting events into
        :param event_type: the method of how to add the events
        :param fillna: if the resulting DataFrame should have any NaN values replaced with default values. Default True
        :param intent_levels: an single or list of levels to run, if list, run in order given
        :param kwargs: additional kwargs to add to the parameterised intent, these will replace any that already exist
        :return Canonical with parameterised intent applied or None if inplace is True
        """
        event_type = event_type if isinstance(event_type, str) else 'add'
        fillna = fillna if isinstance(fillna, bool) else True
        # test if there is any intent to run
        if self._pm.has_intent():
            if event_type not in ['add', 'increment', 'decrement']:
                raise ValueError(f"The event type '{event_type}' is not one of 'add', 'increment' or 'decrement' ")
            # get the list of levels to run
            if isinstance(intent_levels, (int, str, list)):
                intent_levels = self._pm.list_formatter(intent_levels)
            else:
                intent_levels = sorted(self._pm.get_intent().keys())
            for level in intent_levels:
                for method, params in self._pm.get_intent(level=level).items():
                    if method in self.__dir__():
                        params.update(params.pop('kwargs', {}))
                        if isinstance(kwargs, dict):
                            params.update(kwargs)
                        params.update({'inplace': False, 'save_intent': False})
                        event = eval(f"self.{method}(canonical, **params)", globals(), locals())
                        _ = eval(f"event_book.{event_type}_event(event)", globals(), locals())
        return event_book.current_state(fillna=fillna)

    def flatten_date_diff(self, canonical: pd.DataFrame, key: [str, list], first_date: str, second_date: str,
                          aggregator: str=None, units: str=None, label: str=None, inplace: bool=None,
                          save_intent: bool=None, intent_level: [int, str] = None) -> [None, pd.DataFrame]:
        """ adds a column for the difference between a primary and secondary date where the primary is an early date
        than the secondary.

        :param canonical: the DataFrame containing the column headers
        :param key: the key label to group by and index on
        :param first_date: the primary or older date field
        :param second_date: the secondary or newer date field
        :param aggregator: (optional) the aggregator as a function of Pandas DataFrame 'groupby'
        :param units: (optional) The Timedelta units e.g. 'D', 'W', 'M', 'Y'. default is 'D'
        :param label: (optional) a label for the new column.
        :param inplace: adds the calculated column to the canonical before flattening
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
        :return: the DataFrame with the extra column
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        units = units if isinstance(units, str) else 'D'
        label = label if isinstance(label, str) else f'{second_date}-{first_date}'
        inplace = inplace if isinstance(inplace, bool) else False
        result = canonical[second_date].sub(canonical[first_date], axis=0) / np.timedelta64(1, units)
        if inplace:
            canonical[label] = result.round(1)
        df_diff = pd.DataFrame()
        for col in self._pm.list_formatter(key):
            df_diff[col] = canonical[col].copy()
        df_diff[label] = canonical[label].copy()
        return self.group_features(df_diff, headers=label, group_by=key, aggregator=aggregator)

    def apply_selection(self, canonical: pd.DataFrame, key: [str, list], column: str, conditions: [tuple, list],
                        default: [int, float, str]=None, label: str=None, save_intent: bool=None,
                        intent_level: [int, str]=None) -> pd.DataFrame:
        """ applies a selections choice based on a set of conditions to a condition to a named column
        Example: conditions = tuple('< 5',  'red')
             or: conditions = [('< 5',  'green'), ('> 5 & < 10',  'red')]

        :param canonical: the Pandas.DataFrame to get the column headers from
        :param key: the key column to index on
        :param column: a list of headers to apply the condition on,
        :param conditions: a tuple or list of tuple conditions
        :param default: (optional) a value if no condition is met. 0 if not set
        :param label: (optional) a label for the new column
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
        :return:
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
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

    #     df = pd.DataFrame({'Type': list('ABBC'), 'Set': list('ZZXY')})
    #     # conditions = [
    #     #     (df['Set'] == 'Z') & (df['Type'] == 'A'),
    #     #     (df['Set'] == 'Z') & (df['Type'] == 'B'),
    #     #     (df['Type'] == 'B')]
    #     # choices = ['yellow', 'blue', 'purple']
    #     # df['color'] = np.select(conditions, choices, default='black')

    def apply_condition(self, canonical: pd.DataFrame, key: [str, list], column: [str, list], condition: str,
                        outcome: str=None, otherwise: str=None, save_intent: bool=None,
                        intent_level: [int, str]=None) -> pd.DataFrame:
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
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
        :return:
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
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
        return result_df

    def remove_outliers(self, canonical: pd.DataFrame, key: [str, list], column: str, lower_quantile: float=None,
                        upper_quantile: float=None, inplace: bool=False, save_intent: bool=None,
                        intent_level: [int, str]=None) -> [None, pd.DataFrame]:
        """ removes outliers by removing the boundary quantiles

        :param canonical: the DataFrame to apply
        :param key: the key column to index on
        :param column: the column name to remove outliers
        :param lower_quantile: (optional) the lower quantile in the range 0 < lower_quantile < 1, deafault to 0.001
        :param upper_quantile: (optional) the upper quantile in the range 0 < upper_quantile < 1, deafault to 0.999
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: the revised values
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        key = self._pm.list_formatter(key)
        df_rtn = Commons.filter_columns(canonical, headers=key + [column])
        lower_quantile = lower_quantile if isinstance(lower_quantile, float) and 0 < lower_quantile < 1 else 0.001
        upper_quantile = upper_quantile if isinstance(upper_quantile, float) and 0 < upper_quantile < 1 else 0.999

        result = DataDiscovery.analyse_number(df_rtn[column], granularity=[lower_quantile, upper_quantile])
        analysis = DataAnalytics(result)
        df_rtn = df_rtn[(df_rtn[column] > analysis.selection[0][1]) & (df_rtn[column] < analysis.selection[2][0])]
        return df_rtn.set_index(key)

    def group_features(self, canonical: pd.DataFrame, headers: [str, list], group_by: [str, list], aggregator: str=None,
                       drop_group_by: bool=False, include_weighting: bool=False, weighting_precision: int=None,
                       remove_weighting_zeros: bool=False, remove_aggregated: bool=False, drop_dup_index: str=None,
                       inplace: bool=False, save_intent: bool=None,
                       intent_level: [int, str]=None) -> [None, pd.DataFrame]:
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
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: pd.DataFrame
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        if not inplace:
            with threading.Lock():
                canonical = deepcopy(canonical)
        weighting_precision = weighting_precision if isinstance(weighting_precision, int) else 3
        aggregator = aggregator if isinstance(aggregator, str) else 'sum'
        if drop_group_by and str(aggregator).startswith('nunique'):
            raise ValueError(f"drop_group_by must be False when aggregator is 'nunique'")
        headers = self._pm.list_formatter(headers)
        group_by = self._pm.list_formatter(group_by)
        df_sub = TransitionIntentModel.filter_columns(canonical, headers=headers + group_by).dropna()
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
            df_sub = df_sub.reset_index()
            df_sub = df_sub.drop(group_by, axis=1)
        if inplace:
            return
        return df_sub

    def interval_categorical(self, canonical: pd.DataFrame, key: str, column: str, granularity: [int, float, list]=None,
                             lower: [int, float]=None, upper: [int, float]=None, label: str=None, categories: list=None,
                             precision: int=None, inplace: bool=False, save_intent: bool = None,
                             intent_level: [int, str] = None) -> [None, pd.DataFrame]:
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
        :param inplace: (optional) if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: the converted fields
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
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
        if inplace:
            canonical[label] = df_rtn[label].copy()
        return df_rtn

    def flatten_categorical(self, canonical: pd.DataFrame, key, column, prefix=None, index_key=True, dups=True,
                            inplace: bool=False, save_intent: bool=None,
                            intent_level: [int, str]=None) -> [None, pd.DataFrame]:
        """ flattens a categorical as a sum of one-hot

        :param canonical: the Dataframe to reference
        :param key: the key column to sum on
        :param column: the category type column break into the category columns
        :param prefix: a prefix for the category columns
        :param index_key: set the key as the index. Default to True
        :param dups: id duplicates should be removed from the original canonical
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: a pd.Dataframe of the flattened categorical
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        if not inplace:
            with threading.Lock():
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
        if inplace:
            return
        return dummy_df

    def date_categorical(self, canonical: pd.DataFrame, key: [str, list], column: str, matrix: [str, list]=None,
                         label: str=None, save_intent: bool=None, intent_level: [int, str]=None) -> pd.DataFrame:
        """ breaks a date down into value representations of the various parts that date.

        :param canonical: the DataFrame to take the columns from
        :param key: the key column
        :param column: the date column
        :param matrix: the matrix options (see below)
        :param label: (optional) a label alternative to the column name
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
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
                                   intent_level=intent_level, save_intent=save_intent)
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

    def replace_missing(self, canonical: pd.DataFrame, headers: [str, list], granularity: [int, float]=None,
                        lower: [int, float]=None, upper: [int, float]=None, nulls_list: [bool, list]=None,
                        replace_zero: [int, float]=None, precision: int=None, day_first: bool=False,
                        year_first: bool=False, date_format: str = None, inplace: bool=False, save_intent: bool=None,
                        intent_level: [int, str]=None) -> [None, pd.DataFrame]:
        """ imputes missing data with a weighted distribution based on the analysis of the other elements in the
            column

        :param canonical: the pd.DataFrame to replace missing values in
        :param headers: the headers in the pd.DataFrame to apply the substitution too
        :param granularity: (optional) the granularity of the analysis across the range.
                int passed - the number of sections to break the value range into
                pd.Timedelta passed - a frequency time delta
        :param lower: (optional) the lower limit of the number or date value. Takes min() if not set
        :param upper: (optional) the upper limit of the number or date value. Takes max() if not set
        :param nulls_list: (optional) a list of nulls other than np.nan
        :param replace_zero: (optional) if zero what to replace the weighting value with to avoid zero probability
        :param precision: (optional) by default set to 3.
        :param day_first: if the date provided has day first
        :param year_first: if the date provided has year first
        :param date_format: the format of the output dates, if None then pd.Timestamp
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return:
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        if not inplace:
            with threading.Lock():
                canonical = deepcopy(canonical)
        headers = self._pm.list_formatter(canonical)
        if not isinstance(canonical, pd.DataFrame):
            raise TypeError("The canonical given is not a pandas DataFrame")
        if isinstance(nulls_list, bool) and nulls_list:
            nulls_list = ['NaN', 'nan', 'null', '', 'None', np.inf, -np.inf]
        elif not isinstance(nulls_list, list):
            nulls_list = None
        for c in headers:
            col = deepcopy(canonical[c])
            # replace alternative nulls with pd.nan
            if nulls_list is not None:
                col.replace(nulls_list, np.nan, inplace=True)
            size = len(col[col.isna()])
            if size > 0:
                if is_numeric_dtype(col):
                    result = DataDiscovery.analyse_number(col, granularity=granularity, lower=lower, upper=upper,
                                                          precision=precision)
                    col[col.isna()] = self._get_number(from_value=result.get('lower'), to_value=result.get('upper'),
                                                       weight_pattern=result.get('weighting'), precision=0, size=size)
                elif is_datetime64_any_dtype(col):
                    result = DataDiscovery.analyse_date(col, granularity=granularity, lower=lower, upper=upper,
                                                        day_first=day_first, year_first=year_first,
                                                        date_format=date_format)
                    synthetic = self._get_datetime(start=result.get('lower'), until=result.get('upper'),
                                                   weight_pattern=result.get('weighting'), date_format=date_format,
                                                   day_first=day_first, year_first=year_first, size=size)
                    col = col.apply(lambda x: synthetic.pop() if x is pd.NaT else x)
                else:
                    result = DataDiscovery.analyse_category(col, replace_zero=replace_zero)
                    col[col.isna()] = self._get_category(selection=result.get('selection'),
                                                         weight_pattern=result.get('weighting'), size=size)
            canonical[c] = col
        if inplace:
            return
        return canonical

    def apply_substitution(self, canonical: pd.DataFrame, headers: [str, list], inplace: bool=False,
                           save_intent: bool=None, intent_level: [int, str]=None, **kwargs) -> [None, pd.DataFrame]:
        """ regular expression substitution of key value pairs to the value string

        :param canonical: the value to apply the substitution to
        :param headers:
        :param kwargs: a set of keys to replace with the values
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: the amended value
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        if not inplace:
            with threading.Lock():
                canonical = deepcopy(canonical)
        headers = self._pm.list_formatter(headers)
        for c in headers:
            for k, v in kwargs.items():
                canonical[c].replace(k, v)
        if inplace:
            return
        return canonical

    def custom_builder(self, canonical: pd.DataFrame, code_str: str, use_exec: bool=False, inplace: bool=False,
                       save_intent: bool=None, intent_level: [int, str]=None, **kwargs) -> [None, pd.DataFrame]:
        """ enacts a code_str on a dataFrame, returning the output of the code_str or the DataFrame if using exec or
        the evaluation returns None. Note that if using the input dataframe in your code_str, it is internally
        referenced as it's parameter name 'canonical'.

        :param canonical: a pd.DataFrame used in the action
        :param code_str: an action on those column values
        :param use_exec: (optional) By default the code runs as eval if set to true exec would be used
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :param kwargs: a set of kwargs to include in any executable function
        :return: a list or pandas.DataFrame
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        if not inplace:
            with threading.Lock():
                canonical = deepcopy(canonical)
        local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
        if 'canonical' not in local_kwargs:
            local_kwargs['canonical'] = canonical

        result = exec(code_str, globals(), local_kwargs) if use_exec else eval(code_str, globals(), local_kwargs)
        if inplace:
            return
        if result is None:
            return canonical
        return result

    """
        PRIVATE METHODS SECTION
    """
    def _get_number(self, from_value: [int, float], to_value: [int, float]=None, weight_pattern: list=None,
                    offset: int=None, precision: int=None, bounded_weighting: bool=True, at_most: int=None,
                    dominant_values: [float, list]=None, dominant_percent: float=None, dominance_weighting: list=None,
                    size: int = None) -> list:
        """ returns a number in the range from_value to to_value. if only to_value given from_value is zero

        :param from_value: range from_value to_value if to_value is used else from 0 to from_value if to_value is None
        :param to_value: optional, (signed) integer to end from.
        :param weight_pattern: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param offset: an offset multiplier, if None then assume 1
        :param bounded_weighting: if the weighting pattern should have a soft or hard boundary constraint
        :param at_most: the most times a selection should be chosen
        :param dominant_values: a value or list of values with dominant_percent. if used MUST provide a dominant_percent
        :param dominant_percent: a value between 0 and 1 representing the dominant_percent of the dominant value(s)
        :param dominance_weighting: a weighting of the dominant values
        :param size: the size of the sample
        :return: a random number
        """
        (from_value, to_value) = (0, from_value) if not isinstance(to_value, (float, int)) else (from_value, to_value)
        at_most = 0 if not isinstance(at_most, int) else at_most
        if at_most > 0 and (at_most * (to_value-from_value)) < size:
            raise ValueError("When using 'at_most', the selectable values must be greater than the size. selectable "
                             "value count is '{}',size requested is '{}'".format(at_most * (to_value-from_value), size))
        size = 1 if size is None else size
        offset = 1 if offset is None else offset
        dominant_percent = 0 if not isinstance(dominant_percent, (int, float)) else dominant_percent
        dominant_percent = dominant_percent / 100 if 1 < dominant_percent <= 100 else dominant_percent
        _limit = 10000
        precision = 3 if not isinstance(precision, int) else precision
        if precision == 0:
            from_value = int(round(from_value, 0))
            to_value = int(round(to_value, 0))
        is_int = True if isinstance(to_value, int) and isinstance(from_value, int) else False
        if is_int:
            precision = 0
        dominant_list = []
        if isinstance(dominant_values, (int, float, list)):
            sample_count = int(round(size * dominant_percent, 1)) if size > 1 else 0
            dominance_weighting = [1] if not isinstance(dominance_weighting, list) else dominance_weighting
            if sample_count > 0:
                if isinstance(dominant_values, list):
                    dominant_list = self._get_category(selection=dominant_values, weight_pattern=dominance_weighting,
                                                       size=sample_count, bounded_weighting=True)
                else:
                    dominant_list = [dominant_values] * sample_count
            size -= sample_count
        if weight_pattern is not None:
            counter = [0] * len(weight_pattern)
            if bounded_weighting:
                unit = size/sum(weight_pattern)
                for i in range(len(weight_pattern)):
                    counter[i] = int(round(weight_pattern[i] * unit, 0))
                    if 0 < at_most < counter[i]:
                        counter[i] = at_most
                    if counter[i] == 0 and weight_pattern[i] > 0:
                        if counter[self._weighted_choice(weight_pattern)] == i:
                            counter[i] = 1
            else:
                for _ in range(size):
                    counter[self._weighted_choice(weight_pattern)] += 1
                for i in range(len(counter)):
                    if 0 < at_most < counter[i]:
                        counter[i] = at_most
            while sum(counter) != size:
                if at_most > 0:
                    for index in range(len(counter)):
                        if counter[index] >= at_most:
                            counter[index] = at_most
                            weight_pattern[index] = 0
                if sum(counter) < size:
                    counter[self._weighted_choice(weight_pattern)] += 1
                else:
                    weight_idx = self._weighted_choice(weight_pattern)
                    if counter[weight_idx] > 0:
                        counter[weight_idx] -= 1

        else:
            counter = [size]
        rtn_list = []
        if is_int:
            value_bins = []
            ref = from_value
            counter_len = len(counter)
            select_len = to_value - from_value
            for index in range(1, counter_len):
                position = int(round(select_len / counter_len * index, 1)) + from_value
                value_bins.append((ref, position))
                ref = position
            value_bins.append((ref, to_value))
            for index in range(counter_len):
                low, high = value_bins[index]
                if low >= high:
                    rtn_list += [low] * counter[index]
                elif at_most > 0:
                    choice = []
                    for _ in range(at_most):
                        choice += list(range(low, high))
                    np.random.shuffle(choice)
                    rtn_list += [int(np.round(value, precision)) for value in choice[:counter[index]]]
                else:
                    _remaining = counter[index]
                    while _remaining > 0:
                        _size = _limit if _remaining > _limit else _remaining
                        rtn_list += np.random.randint(low=low, high=high, size=_size).tolist()
                        _remaining -= _limit
        else:
            value_bins = pd.interval_range(start=from_value, end=to_value, periods=len(counter), closed='both')
            for index in range(len(counter)):
                low = value_bins[index].left
                high = value_bins[index].right
                if low >= high:
                    rtn_list += [low] * counter[index]
                elif at_most > 0:
                    choice = []
                    for _ in range(at_most):
                        choice += list(range(low, high))
                    np.random.shuffle(choice)
                    rtn_list += [np.round(value, precision) for value in choice[:counter[index]]]
                else:
                    _remaining = counter[index]
                    while _remaining > 0:
                        _size = _limit if _remaining > _limit else _remaining
                        rtn_list += np.round((np.random.random(size=_size)*(high-low)+low), precision).tolist()
                        _remaining -= _limit
        if offset != 1:
            rtn_list = [value*offset for value in rtn_list]
        # add in the dominant values
        rtn_list = rtn_list + dominant_list
        np.random.shuffle(rtn_list)
        return rtn_list

    def _get_category(self, selection: list, weight_pattern: list=None, size: int=None, at_most: int=None,
                      bounded_weighting: bool=None) -> list:
        """ returns a category from a list. Of particular not is the at_least parameter that allows you to
        control the number of times a selection can be chosen.

        :param selection: a list of items to select from
        :param weight_pattern: a weighting pattern that does not have to add to 1
        :param size: an optional size of the return. default to 1
        :param at_most: the most times a selection should be chosen
        :param bounded_weighting: if the weighting pattern should have a soft or hard boundary (default False)
        :return: an item or list of items chosen from the list
        """
        if not isinstance(selection, list) or len(selection) == 0:
            return [None]*size
        bounded_weighting = False if not isinstance(bounded_weighting, bool) else bounded_weighting
        select_index = self._get_number(len(selection), weight_pattern=weight_pattern, at_most=at_most,
                                        size=size, bounded_weighting=bounded_weighting)
        rtn_list = [selection[i] for i in select_index]
        return rtn_list

    def _get_datetime(self, start, until, weight_pattern: list=None, at_most: int=None, date_format: str=None,
                      as_num: bool=False, ignore_time: bool=False, size: int=None,
                      day_first: bool=False, year_first: bool=False) -> list:
        """ returns a random date between two date and times. weighted patterns can be applied to the overall date
        range, the year, month, day-of-week, hours and minutes to create a fully customised random set of dates.
        Note: If no patterns are set this will return a linearly random number between the range boundaries.
              Also if no patterns are set and a default date is given, that default date will be returnd each time

        :param start: the start boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
        :param until: then up until boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
        :param weight_pattern: (optional) A pattern across the whole date range.
        :param at_most: the most times a selection should be chosen
        :param ignore_time: ignore time elements and only select from Year, Month, Day elements. Default is False
        :param date_format: the string format of the date to be returned. if not set then pd.Timestamp returned
        :param as_num: returns a list of Matplotlib date values as a float. Default is False
        :param size: the size of the sample to return. Default to 1
        :param year_first: specifies if to parse with the year first
                If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
                If both dayfirst and yearfirst are True, yearfirst is preceded (same as dateutil).
        :param day_first: specifies if to parse with the day first
                If True, parses dates with the day first, eg %d-%m-%Y.
                If False default to the a prefered preference, normally %m-%d-%Y (but not strict)
        :return: a date or size of dates in the format given.
         """
        as_num = False if not isinstance(as_num, bool) else as_num
        ignore_time = False if not isinstance(ignore_time, bool) else ignore_time
        if start is None or until is None:
            raise ValueError("The start or until parameters cannot be of NoneType")
        size = 1 if size is None else size
        _dt_start = self._convert_date2value(start, day_first=day_first, year_first=year_first)[0]
        _dt_until = self._convert_date2value(until, day_first=day_first, year_first=year_first)[0]
        precision = 15
        if ignore_time:
            _dt_start = int(_dt_start)
            _dt_until = int(_dt_until)
            precision = 0
        rtn_list = self._get_number(from_value=_dt_start, to_value=_dt_until, weight_pattern=weight_pattern,
                                    at_most=at_most, precision=precision, size=size)
        if not as_num:
            rtn_list = mdates.num2date(rtn_list)
            if isinstance(date_format, str):
                rtn_list = pd.Series(data=rtn_list).dt.strftime(date_format).tolist()
        return rtn_list

    @staticmethod
    def _weighted_choice(weights: list):
        """ a probability weighting based on the values in the integer list

        :param weights: a list of integers representing a pattern of weighting
        :return: an index of which weight was randomly chosen
        """
        if not isinstance(weights, list) or not all(isinstance(x, (int, float, list)) for x in weights):
            raise ValueError("The weighted pattern must be an list of integers")
        rnd = np.random.random() * sum(weights)
        for i, w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                return i

    @staticmethod
    def _convert_date2value(dates, day_first: bool = True, year_first: bool = False):
        values = pd.to_datetime(dates, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                                yearfirst=year_first)
        return mdates.date2num(pd.Series(values)).tolist()
