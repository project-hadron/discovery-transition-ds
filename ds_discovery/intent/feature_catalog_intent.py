import inspect
from copy import deepcopy
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_datetime64_any_dtype
import numpy as np
import matplotlib.dates as mdates

from ds_foundation.intent.abstract_intent import AbstractIntentModel
from ds_foundation.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.transition.discovery import DataDiscovery, DataAnalytics

__author__ = 'Darryl Oatridge'


class FeatureCatalogIntentModel(AbstractIntentModel):
    """A set of methods to help build features as pandas.Dataframe"""

    def __init__(self, property_manager: AbstractPropertyManager, default_save_intent: bool=True,
                 intent_next_available: bool=None):
        """initialisation of the Intent class. The 'intent_param_exclude' is used to exclude commonly used method
         parameters from being included in the intent contract, this is particularly useful if passing a canonical, or
         non relevant parameters to an intent method pattern. Any named parameter in the intent_param_exclude list
         will not be included in the recorded intent contract for that method

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param intent_next_available: (optional) if the default level should be set to next available level or zero
        """
        # set all the defaults
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_intent_level = -1 if isinstance(intent_next_available, bool) and intent_next_available else 0
        intent_param_exclude = ['df']
        super().__init__(property_manager=property_manager, intent_param_exclude=intent_param_exclude,
                         default_save_intent=default_save_intent, default_intent_level=default_intent_level)

    def run_intent_pipeline(self, canonical, intent_level: [int, str, list]=None, **kwargs):
        # test if there is any intent to run
        if self._pm.has_intent():
            # get the list of levels to run
            if isinstance(intent_level, (int, str, list)):
                intent_level = self._pm.list_formatter(intent_level)
            else:
                intent_level = sorted(self._pm.get_intent().keys())
            for level in intent_level:
                for method, params in self._pm.get_intent(level=level).items():
                    if method in self.__dir__():
                        if isinstance(kwargs, dict):
                            params.update(kwargs)
                        canonical = eval(f"self.{method}(canonical, save_intent=False, **{params})")
        return canonical

    def remove_outliers(self, canonical: pd.DataFrame, headers: list, lower_quantile: float=None, upper_quantile: float=None,
                        save_intent: bool=None, intent_level: [int, str]=None):
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

    def group_features(self, canonical: pd.DataFrame, headers:[str, list], group_by: [str, list], aggregator: str=None,
                       drop_group_by: bool=False, include_weighting=False, remove_zeros: bool=False,
                       remove_aggregated: bool=True, save_intent: bool=None, intent_level: [int, str]=None):
        """ groups features according to the stategy passed. strategy options are None, 'sum', 'nunique', 'max', 'min'

        :param canonical: the pd.DataFrame to group
        :param headers: the headers to apply the strategy to
        :param group_by: the headers to group by
        :param aggregator: the aggregator function
        :param drop_group_by: drops the group by headers
        :param remove_aggregated: if used in conjunction with the weighting then drops the aggrigator column
        :param remove_zeros: removes zero values
        :param include_weighting: include a percentage weighting column
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: pd.DataFrame
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        aggregator = aggregator if isinstance(aggregator, str) else 'sum'
        df_sub = canonical[headers].groupby(group_by).agg(aggregator)
        df_sub = df_sub.sort_values(ascending=False).reset_index()
        if include_weighting:
            total = df_sub[headers].sum()
            df_sub['weighting'] = df_sub[headers].apply(lambda x: round((x / total) * 100, 2) if isinstance(x, (int, float)) else 0)
            if remove_zeros:
                df_sub = df_sub[df_sub['weighting'] > 0]
        if remove_aggregated:
            df_sub = df_sub.drop(headers, axis=1)
        else:
            if not include_weighting and remove_zeros:
                df_sub = df_sub[df_sub[headers] > 0]
        return df_sub

    def flatten_categorical(self, canonical: pd.DataFrame, key, column, prefix=None, index_key=True, dups=True,
                            save_intent: bool=None, intent_level: [int, str]=None) -> pd.DataFrame:
        """ flattens a categorical as a sum of one-hot

        :param canonical: the Dataframe to reference
        :param key: the key column to sum on
        :param column: the category type column break into the category columns
        :param prefix: a prefix for the category columns
        :param index_key: set the key as the index. Default to True
        :param dups: id duplicates should be removed from the origional canonical
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: a pd.Dataframe of the flattened categorical
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
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

    def date_matrix(self, canonical: pd.DataFrame, key, column, index_key=True, save_intent: bool=None,
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

    def replace_missing(self, canonical: pd.DataFrame, headers: [str, list], granularity: [int, float]=None,
                        lower: [int, float]=None, upper: [int, float]=None, nulls_list: [bool, list]=None,
                        replace_zero: [int, float]=None, precision: int=None, day_first: bool=False,
                        year_first: bool=False, date_format: str = None, save_intent: bool=None,
                        intent_level: [int, str]=None):
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
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return:
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        df = canonical
        headers = DataDiscovery.list_formatter(df)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The canonical given is not a pandas DataFrame")
        if isinstance(nulls_list, bool) and nulls_list:
            nulls_list = ['NaN', 'nan', 'null', '', 'None', np.inf, -np.inf]
        elif not isinstance(nulls_list, list):
            nulls_list = None
        for c in df.columns:
            col = deepcopy(df[c])
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
                                                    day_first=day_first, year_first=year_first, date_format=date_format)
                    synthetic = self._get_datetime(start=result.get('lower'), until=result.get('upper'),
                                                   weight_pattern=result.get('weighting'), date_format=date_format,
                                                   day_first=day_first, year_first=year_first, size=size)
                    col = col.apply(lambda x: synthetic.pop() if x is pd.NaT else x)
                else:
                    result = DataDiscovery.analyse_category(col, replace_zero=replace_zero)
                    col[col.isna()] = self._get_category(selection=result.get('selection'),
                                                             weight_pattern=result.get('weighting'), size=size)
            df[c] = col
        return df

    def apply_substitution(self, canonical: pd.DataFrame, headers: [str, list], save_intent: bool=None, intent_level: [int, str]=None, **kwargs):
        """ regular expression substitution of key value pairs to the value string

        :param value: the value to apply the substitution to
        :param kwargs: a set of keys to replace with the values
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: the amended value
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        headers = DataDiscovery.list_formatter(headers)
        for c in headers:
            for k, v in kwargs.items():
                canonical[c].replace(k, v)
        return canonical

    def custom_builder(self, canonical: pd.DataFrame, code_str: str, use_exec: bool=False, save_intent: bool=None,
                       intent_level: [int, str]=None, **kwargs):
        """ enacts a code_str on a dataFrame, returning the output of the code_str or the DataFrame if using exec or
        the evaluation returns None. Note that if using the input dataframe in your code_str, it is internally
        referenced as it's parameter name 'canonical'.

        :param canonical: a pd.DataFrame used in the action
        :param code_str: an action on those column values
        :param use_exec: (optional) By default the code runs as eval if set to true exec would be used
        :param kwargs: a set of kwargs to include in any executable function
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: a list or pandas.DataFrame
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
        if 'canonical' not in local_kwargs:
            local_kwargs['canonical'] = canonical

        result = exec(code_str, globals(), local_kwargs) if use_exec else eval(code_str, globals(), local_kwargs)
        if result is None:
            return canonical
        return result

    def build_frame(self, canonical_left, canonical_right, on=None, left_on=None, right_on=None, how='left',
                    left_index=False, right_index=False, sort=True, suffixes=('_x', '_y'), indicator=False,
                    validate=None, save_intent: bool=None, intent_level: [int, str]=None):
        """ converts columns to object type

        :param canonical_left: A DataFrame object.
        :param canonical_right: Another DataFrame object.
        :param on: Column or index level names to join on. Must be found in both the left and right DataFrame objects.
                If not passed and left_index and right_index are False, the intersection of the columns in the
                DataFrames will be inferred to be the join keys.
        :param left_on: Columns or index levels from the left DataFrame to use as keys. Can either be column names,
                index level names, or arrays with length equal to the length of the DataFrame.
        :param right_on: Columns or index levels from the right DataFrame to use as keys. Can either be column names,
                index level names, or arrays with length equal to the length of the DataFrame.
        :param left_index: If True, use the index (row labels) from the left DataFrame as its join key(s). In the case
                of a DataFrame with a MultiIndex (hierarchical), the number of levels must match the number of join
                keys from the right DataFrame.
        :param right_index: Same usage as left_index for the right DataFrame
        :param how: One of 'left', 'right', 'outer', 'inner'. Defaults to inner.
        :param sort: Sort the result DataFrame by the join keys in lexicographical order. Defaults to True, setting
                to False will improve performance substantially in many cases.
        :param suffixes: A tuple of string suffixes to apply to overlapping columns. Defaults to ('_x', '_y').
        :param in place: Always copy data (default True) from the passed DataFrame objects, even when reindexing is
                not necessary. Cannot be avoided in many cases but may improve performance / memory usage. The cases
                where copying can be avoided are somewhat pathological but this option is provided nonetheless.
        :param validate : string, default None. If specified, checks if merge is of specified type.
                    “one_to_one” or “1:1”: checks if merge keys are unique in both left and right datasets.
                    “one_to_many” or “1:m”: checks if merge keys are unique in left dataset.
                    “many_to_one” or “m:1”: checks if merge keys are unique in right dataset.
                    “many_to_many” or “m:m”: allowed, but does not result in checks.
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) a level to place the intent
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # intend code block on the canonical
        df = pd.merge(left=canonical_left, right=canonical_right, how=how, on=on, left_on=left_on, right_on=right_on,
                      left_index=left_index, right_index=right_index, copy=True, sort=sort, suffixes=suffixes,
                      indicator=indicator, validate=validate)
        return df

    """
        PRIVATE METHODS SECTION
    """
    def _get_number(self, from_value: [int, float], to_value: [int, float]=None, weight_pattern: list=None, offset: int=None,
                    precision: int=None, bounded_weighting: bool=True, at_most: int=None,
                    dominant_values: [float, list]=None, dominant_percent: float=None, dominance_weighting: list=None,
                    size: int = None) -> list:
        """ returns a number in the range from_value to to_value. if only to_value given from_value is zero

        :param from_value: range from_value to_value if to_value is used else from 0 to from_value if to_value is None
        :param to_value: optional, (signed) integer to end from.
        :param weight_pattern: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param offset: an offset multiplier, if None then assume 1
        :param currency: a currency symbol to prefix the value with. returns string with commas
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
                    dominant_list = self._get_category(selection=dominant_values,
                                                                  weight_pattern=dominance_weighting,
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
                rtn_list = pd.Series(rtn_list).dt.strftime(date_format).tolist()
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
