import re
from copy import deepcopy
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.dates as mdates

from pandas.api.types import is_datetime64_any_dtype, is_categorical_dtype, is_numeric_dtype, is_string_dtype

from ds_foundation.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.transition.discovery import DataDiscovery as Discovery
from ds_behavioral.generator.data_builder import DataBuilderTools as Tools

__author__ = 'Darryl Oatridge'


class FeatureBuilderTools(object):
    """A set of methods to help build features as pandas.Dataframe"""

    def __dir__(self):
        rtn_list = []
        for m in dir(FeatureBuilderTools):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def date_matrix(df, key, column, index_key=True) -> pd.DataFrame:
        """ returns a pandas.Dataframe of the datetime broken down

        :param df: the pandas.Dataframe to take the columns from
        :param key: the key column
        :param column: the date column
        :param index_key: if to index the key. Default to True
        :return: a pandas.DataFrame of the datetime breakdown
        """
        if key not in df:
            raise NameError("The key {} can't be found in the DataFrame".format(key))
        if column not in df:
            raise NameError("The column {} can't be found in the DataFrame".format(column))
        if not df[column].dtype.name.startswith('datetime'):
            raise TypeError("the column {} is not of dtype datetime".format(column))
        df_time = df.filter([key, column], axis=1)
        df_time['{}_yr'.format(column)] = df[column].dt.year
        df_time['{}_dec'.format(column)] = (df[column].dt.year - df[column].dt.year % 10).astype('category')
        df_time['{}_mon'.format(column)] = df[column].dt.month
        df_time['{}_day'.format(column)] = df[column].dt.day
        df_time['{}_dow'.format(column)] = df[column].dt.dayofweek
        df_time['{}_hr'.format(column)] = df[column].dt.hour
        df_time['{}_min'.format(column)] = df[column].dt.minute
        df_time['{}_woy'.format(column)] = df[column].dt.weekofyear
        df_time['{}_doy'.format(column)] = df[column].dt.dayofyear
        df_time['{}_ordinal'.format(column)] = mdates.date2num(df[column])

        if index_key:
            df_time = df_time.set_index(key)
        return df_time

    @staticmethod
    def replace_missing(dataset: Any, granularity: [int, float]=None, chunk_size: int=None, lower: [int, float]=None,
                        upper: [int, float]=None, nulls_list: [bool, list]=None, replace_zero: [int, float]=None,
                        precision: int=None, day_first: bool=False, year_first: bool=False, date_format: str = None):
        """ imputes missing data with a weighted distribution based on the analysis of the other elements in the
            column

        :param dataset: the series of dataframe to replace missing values in
        :param granularity: (optional) the granularity of the analysis across the range.
                int passed - the number of sections to break the value range into
                pd.Timedelta passed - a frequency time delta
        :param chunk_size: (optional) number of chuncks if you want weighting over the length of the dataset
        :param lower: (optional) the lower limit of the number or date value. Takes min() if not set
        :param upper: (optional) the upper limit of the number or date value. Takes max() if not set
        :param nulls_list: (optional) a list of nulls other than np.nan
        :param replace_zero: (optional) if zero what to replace the weighting value with to avoid zero probability
        :param precision: (optional) by default set to 3.
        :param day_first: if the date provided has day first
        :param year_first: if the date provided has year first
        :param date_format: the format of the output dates, if None then pd.Timestamp
        :return:
        """
        if not isinstance(dataset, (str, int, float, list, pd.Series, pd.DataFrame)):
            raise TypeError("The parameter values is not an accepted type")
        df = dataset
        if isinstance(df, (str, int, float)):
            df = AbstractPropertyManager.list_formatter(df)
        if isinstance(df, (list, pd.Series)):
            tmp = pd.DataFrame()
            tmp['_default'] = df
            df = tmp
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The dataset given is not or could not be convereted to a pandas DataFrame")
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
                    result = Discovery.analyse_number(col, granularity=granularity, lower=lower, upper=upper,
                                                        chunk_size=chunk_size, replace_zero=replace_zero,
                                                        precision=precision)
                    col[col.isna()] = Tools.get_number(from_value=result.get('lower'), to_value=result.get('upper'),
                                                           weight_pattern=result.get('weighting'), precision=0, size=size)
                elif is_datetime64_any_dtype(col):
                    result = Discovery.analyse_date(col, granularity=granularity, lower=lower, upper=upper,
                                                      chunk_size=chunk_size, replace_zero=replace_zero,
                                                      day_first=day_first, year_first=year_first, date_format=date_format)
                    synthetic = Tools.get_datetime(start=result.get('lower'), until=result.get('upper'),
                                                             date_pattern=result.get('weighting'), date_format=date_format,
                                                             day_first=day_first, year_first=year_first, size=size)
                    col = col.apply(lambda x: synthetic.pop() if x is pd.NaT else x)

                else:
                    result = Discovery.analyse_category(col, chunk_size=chunk_size, replace_zero=replace_zero)
                    col[col.isna()] = Tools.get_category(selection=result.get('selection'),
                                                             weight_pattern=result.get('weighting'), size=size)
            df[c] = col
        return df

    @staticmethod
    def custom_builder(df: pd.DataFrame, code_str: str, use_exec: bool=False, **kwargs):
        """ enacts a code_str on a dataFrame, returning the output of the code_str or the DataFrame if using exec or
        the evaluation returns None. Note that if using the input dataframe in your code_str, it is internally
        referenced as it's parameter name 'df'.

        :param df: a pd.DataFrame used in the action
        :param code_str: an action on those column values
        :param use_exec: (optional) By default the code runs as eval if set to true exec would be used
        :param kwargs: a set of kwargs to include in any executable function
        :return: a list or pandas.DataFrame
        """
        local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
        if 'df' not in local_kwargs:
            local_kwargs['df'] = df

        result = exec(code_str, globals(), local_kwargs) if use_exec else eval(code_str, globals(), local_kwargs)
        if result is None:
            return df
        return result

    @staticmethod
    def association_builder(dataset: Any, associations: list, actions: dict, header_name: str=None,
                          default_value: Any=None, default_header: str=None,
                          day_first: bool=False, year_first: bool=False):
        """ Associates a a set of criteria of an input values to a set of actions
            The association dictionary takes the form of a set of dictionaries in a list with each item in the list
            representing an index key for the action dictionary. Each dictionary are to associated relationship.
            In this example for the first index the associated values should be header1 is within a date range
            and header2 has a value of 'M'
                association = [{'header1': {'expect': 'date',
                                            'value': ['12/01/1984', '14/01/2014']},
                                'header2': {'expect': 'category',
                                            'value': ['M']}},
                                {...}]

            if the dataset is not a DataFrame then the header should be omitted. in this example the association is
            a range comparison between 2 and 7 inclusive.
                association= [{'expect': 'number', 'value': [2, 7]},
                              {...}]

            The actions dictionary takes the form of an index referenced dictionary of actions, where the key value
            of the dictionary corresponds to the index of the association list. In other words, if a match is found
            in the association, that list index is used as reference to the action to execute.
                {0: {'action': '', 'kwargs' : {}},
                 1: {...}}
            you can also use the action to specify a specific value:
                {0: {'action': ''},
                 1: {'action': ''}}

        :param dataset: the dataset to map against, this can be a str, int, float, list, Series or DataFrame
        :param associations: a list of categories (can also contain lists for multiple references.
        :param actions: the action set that should map to the index
        :param default_header: (optional) if no association, the default column header to take the value from.
                    if None then the default_value is taken.
                    Note for non-DataFrame datasets the default header is '_default'
        :param default_value: (optional) if no default header then this value is taken if no association
        :param header_name: if passing a pandas dataframe, the name of the new column created
        :param day_first: (optional) if expected type is date, indicates if the day is first. Default to true
        :param year_first: (optional) if expected type is date, indicates if the year is first. Default to true
        :return: a list of equal length to the one passed
        """
        if not isinstance(dataset, (str, int, float, list, pd.Series, pd.DataFrame)):
            raise TypeError("The parameter values is not an accepted type")
        if not isinstance(associations, (list, dict)):
            raise TypeError("The parameter reference must be a list or dict")
        _dataset = dataset
        _associations = associations
        if isinstance(_dataset, (str, int, float)):
            _dataset = AbstractPropertyManager.list_formatter(_dataset)
        if isinstance(_dataset, (list, pd.Series)):
            tmp = pd.DataFrame()
            tmp['_default'] = _dataset
            _dataset = tmp
            tmp = []
            for item in _associations:
                tmp.append({'_default': item})
            _associations = tmp
        if not isinstance(_dataset, pd.DataFrame):
            raise TypeError("The dataset given is not or could not be convereted to a pandas DataFrame")
        class_methods = FeatureBuilderTools().__dir__()

        rtn_list = []
        for index in range(_dataset.shape[0]):
            action_idx = None
            for idx in range(len(_associations)):
                associate_dict = _associations[idx]
                is_match = [0] * len(associate_dict.keys())
                match_idx = 0
                for header, lookup in associate_dict.items():
                    df_value = _dataset[header].iloc[index]
                    expect = lookup.get('expect')
                    chk_value = AbstractPropertyManager.list_formatter(lookup.get('value'))
                    if expect.lower() in ['number', 'n']:
                        if len(chk_value) == 1:
                            [s] = [e] = chk_value
                        else:
                            [s, e] = chk_value
                        if s <= df_value <= e:
                            is_match[match_idx] = True
                    elif expect.lower() in ['date', 'datetime', 'd']:
                        [s, e] = chk_value
                        value_date = pd.to_datetime(df_value, errors='coerce', infer_datetime_format=True,
                                                    dayfirst=day_first, yearfirst=year_first)
                        s_date = pd.to_datetime(s, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                                                yearfirst=year_first)
                        e_date = pd.to_datetime(e, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                                                yearfirst=year_first)
                        if value_date is pd.NaT or s_date is pd.NaT or e_date is pd.NaT:
                            break
                        if s_date <= value_date <= e_date:
                            is_match[match_idx] = True
                    elif expect.lower() in ['category', 'c']:
                        if df_value in chk_value:
                            is_match[match_idx] = True
                    else:
                        break
                    match_idx += 1
                if all(x for x in is_match):
                    action_idx = idx
                    break
            if action_idx is None or actions.get(action_idx) is None:
                if default_header is not None and default_header in _dataset.columns:
                    rtn_list.append(_dataset[default_header].iloc[index])
                else:
                    rtn_list.append(default_value)
                continue
            method = actions.get(action_idx).get('action')
            if method is None:
                raise ValueError("There is no 'action' key at index [{}]".format(action_idx))
            if method in class_methods:
                kwargs = actions.get(action_idx).get('kwargs').copy()
                for k, v in kwargs.items():
                    if isinstance(v, dict) and '_header' in v.keys():
                        if v.get('_header') not in _dataset.columns:
                            raise ValueError("Dataset header '{}' does not exist: see action: {} -> key: {}".format(
                                v.get('_header'), action_idx, k))
                        kwargs[k] = _dataset[v.get('_header')].iloc[index]
                result = eval("FeatureBuilderTools.{}(**{})".format(method, kwargs).replace('nan', 'None'))
                if isinstance(result, list):
                    if len(result) > 0:
                        rtn_list.append(result.pop())
                    else:
                        rtn_list.append(None)
                else:
                    rtn_list.append(result)
            elif method == 'drop':
                _dataset.drop(index=index)
            elif isinstance(method, dict):
                if method.get('_header') not in _dataset.columns:
                    raise ValueError("Dataset header '{}' does not exist: see action: {} -> key: action".format(
                        method.get('_header'), action_idx))
                rtn_list.append(_dataset[method.get('_header')].iloc[index])
            else:
                rtn_list.append(method)
        if header_name is not None:
            _dataset[header_name] = rtn_list
        return _dataset

    @staticmethod
    def apply_substitution(value: str, **kwargs):
        """ reular expression subsitution of key value pairs to the value string

        :param value: the value to apply the subsitutions to
        :param kwargs: a set of keys to replace with the values
        :return: the amended value
        """
        for k, v in kwargs.items():
            value = re.sub(str(k), str(v), value)
        return value

    @staticmethod
    def get_custom(code_str: str, use_exec: bool=False, **kwargs):
        """returns a number based on the random func. The code should generate a value per line
        example:
            code_str = 'round(np.random.normal(loc=loc, scale=scale), 3)'
            fbt.get_custom(code_str, loc=0.4, scale=0.1)

        :param code_str: an evaluable code as a string
        :param quantity: (optional) a number between 0 and 1 representing data that isn't null
        :param size: (optional) the size of the sample
        :param seed: (optional) a seed value for the random function: default to None
        :return: a random value based on function called
        """
        local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
        result = exec(code_str, globals(), local_kwargs) if use_exec else eval(code_str, globals(), local_kwargs)
        return result

    @staticmethod
    def get_groups_sum(def_df: pd.DataFrame, group_headers: list, sum_header: str, include_weighting=True,
                       remove_zeros: bool = True, remove_sum=True):
        df_sub = def_df.groupby(group_headers)[sum_header].agg('sum')
        df_sub = df_sub.sort_values(ascending=False).reset_index()
        if include_weighting:
            total = df_sub[sum_header].sum()
            df_sub['weighting'] = df_sub[sum_header].apply(lambda x: round((x / total) * 100, 2))
            if remove_zeros:
                df_sub = df_sub[df_sub['weighting'] > 0]
        if remove_sum:
            df_sub = df_sub.drop(sum_header, axis=1)
        else:
            if not include_weighting and remove_zeros:
                df_sub = df_sub[df_sub[sum_header] > 0]
        return df_sub

    @staticmethod
    def normalise_values(values: Any, precision: int=None):
        """normalises numberic data to between -1 and 1 (or 0 and 1 if all positive)

        :param values: a list or Series of values
        :param precision: the return precision of the values
        :return: an normalised set of data
        """

        values = AbstractPropertyManager.list_formatter(values)
        norm = values / np.linalg.norm(values)
        if isinstance(precision, int):
            norm = np.round(norm, precision)
        return norm

    @staticmethod
    def flatten_categorical(df, key, column, prefix=None, index_key=True, dups=True) -> pd.DataFrame:
        """ flattens a categorical as a sum of one-hot

        :param df: the Dataframe to reference
        :param key: the key column to sum on
        :param column: the category type column break into the category columns
        :param prefix: a prefix for the category columns
        :param index_key: set the key as the index. Default to True
        :param dups: id duplicates should be removed from the origional df
        :return: a pd.Dataframe of the flattened categorical
        """
        if key not in df:
            raise NameError("The key {} can't be found in the DataFrame".format(key))
        if column not in df:
            raise NameError("The column {} can't be found in the DataFrame".format(column))
        if df[column].dtype.name != 'category':
            raise TypeError("the column {} is not of dtype category".format(column))
        if prefix is None:
            prefix = 'HOT'
        if not dups:
            df.drop_duplicates(inplace=True)
        dummy_df = pd.get_dummies(df[[key, column]], columns=[column], prefix=prefix)
        dummy_cols = dummy_df.columns[dummy_df.columns.to_series().str.contains('{}_'.format(prefix))]
        dummy_df = dummy_df.groupby([pd.Grouper(key=key)])[dummy_cols].sum()
        if index_key:
            dummy_df = dummy_df.set_index(key)
        return dummy_df

    @staticmethod
    def date_offset_limits(df: pd.DataFrame, date_header: str, offset_header: str):
        """returns the min, mean and max timedelta between the two Pandas Timestamp columns

        :param df: the dataframe to get the columns from
        :param date_header: the base date column name
        :param offset_header: the offset date colun name
        :return: min, mean, max as pd.Timedelta
        """
        diff_list = []
        for index in range(df.shape[0]):
            diff_list.append(df[date_header] - df[offset_header])
        max_diff = max(diff_list)
        min_diff = min(diff_list)
        mean_diff = np.mean(diff_list)
        return min_diff, mean_diff, max_diff

    @staticmethod
    def remove_outliers(df: pd.DataFrame, lower_quantile: float=None, upper_quantile: float=None) -> pd.DataFrame:
        """ removes outliers by removing the boundary quantiles

        :param df: the DataFrame to apply
        :param lower_quantile: (optional) the lower quantile, default is 0.25
        :param upper_quantile: (optional) the upper quantile
        :return: the revised values
        """
        lower_quantile = 0.25 if not isinstance(lower_quantile, float) else lower_quantile
        upper_quantile = 0.75 if not isinstance(upper_quantile, float) else upper_quantile

        df_out = pd.DataFrame()
        for column_name in df.columns:
            if df[column_name].min() == df[column_name].max():
                df_out[column_name] = df[column_name]
                continue
            q1 = df[column_name].quantile(lower_quantile)
            q3 = df[column_name].quantile(upper_quantile)
            iqr = q3-q1
            fence_low = q1-1.5*iqr
            fence_high = q3+1.5*iqr
            df_out[column_name] = df[column_name].loc[(df[column_name] > fence_low) & (df[column_name] < fence_high)]
        return df_out

    @staticmethod
    def merge(df_left, df_right, how='inner', on=None, left_on=None, right_on=None, left_index=False,
              right_index=False, sort=True, suffixes=('_x', '_y'), indicator=False, validate=None):
        """ converts columns to object type

        :param left: A DataFrame object.
        :param right: Another DataFrame object.
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
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        df = pd.merge(left=df_left, right=df_right, how=how, on=on, left_on=left_on, right_on=right_on,
                      left_index=left_index, right_index=right_index, copy=True, sort=sort, suffixes=suffixes,
                      indicator=indicator, validate=validate)
        return df

