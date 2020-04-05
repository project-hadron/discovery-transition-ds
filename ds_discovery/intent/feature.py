import inspect
import threading
from copy import deepcopy
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from aistac.intent.abstract_intent import AbstractIntentModel
from aistac.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'

from ds_discovery.transition.commons import Commons, DataAnalytics
from ds_discovery.transition.discovery import DataDiscovery


class Feature(object):
    """A set of methods to help build features as pandas.Dataframe"""


    @staticmethod
    def association_builder(dataset: Any, associations: list, actions: dict, header_name: str=None,
                          default_value: Any=None, default_header: str=None,
                          day_first: bool=False, year_first: bool=False):
        """ Associates a set of criteria of an input values to a set of actions
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
        # TODO: Need to add key to this to make it a Feature Build, probably only accept DataFrame and Series
        if not isinstance(dataset, (str, int, float, list, pd.Series, pd.DataFrame)):
            raise TypeError("The parameter values is not an accepted type")
        if not isinstance(associations, (list, dict)):
            raise TypeError("The parameter reference must be a list or dict")
        _dataset = dataset
        _associations = associations
        if isinstance(_dataset, (str, int, float)):
            _dataset = Commons.list_formatter(_dataset)
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
        class_methods = Feature().__dir__()

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
                    chk_value = Commons.list_formatter(lookup.get('value'))
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
    def get_groups_sum(df: pd.DataFrame, group_headers: list, sum_header: str, include_weighting=True,
                       remove_zeros: bool = True, remove_sum=True):
        # TODO: This need to be considerably improved to be able to add, sum, count
        # Need to add a key to make it a feature
        df_sub = df.groupby(group_headers)[sum_header].agg('sum')
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
        # TODO: This needs to be a frame builder
        df = pd.merge(left=df_left, right=df_right, how=how, on=on, left_on=left_on, right_on=right_on,
                      left_index=left_index, right_index=right_index, copy=True, sort=sort, suffixes=suffixes,
                      indicator=indicator, validate=validate)
        return df
