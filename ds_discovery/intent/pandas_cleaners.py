import threading
import re
from builtins import staticmethod
from copy import deepcopy
import datetime
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from typing import Union, List

from ds_foundation.intent.abstract_cleaners import AbstractCleaners

__author__ = 'Darryl Oatridge'


class PandasCleaners(AbstractCleaners):
    """A set of methods to help clean columns with a Pandas.DataFrame"""

    def __dir__(self):
        rtn_list = []
        for m in dir(PandasCleaners):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def run_contract_pipeline(df, cleaner_contract, inplace=False) -> pd.DataFrame:
        """ run the contract pipeline
        Example:
            The root dictionary should have the data cleaning optional key values::
                ``remove``, ``to_bool``, ``to_float``, ``to_int``, ``to_category``, ``to_date``, ``excel_to_date``
            Each key should have the filter optional sub-key values::
                ``columns: <list>``, ``drop: <bool>``, ``dtype: <list>``, ``exclude: <bool>``
            for to_bool there should be an additional mandatory map dictionary mapping True and False
                ``map: <dict>``

        :param inplace: change pandas.DataFrame in place or to return a deep copy. default True
        :param df: the pandas.DataFrame to be cleaned
        :param cleaner_contract: the configuration dictionary
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return the cleaned pandas.DataFrame
        .. see filter
        """

        # test there is a managers to run
        if not isinstance(cleaner_contract, dict) or cleaner_contract is None:
            if not inplace:
                return df

        # create the copy and use this for all the operations
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)

        # auto clean header
        if cleaner_contract.get('auto_clean_header') is not None:
            settings = cleaner_contract.get('auto_clean_header')
            PandasCleaners.auto_clean_header(df, case=settings.get('case'), rename_map=settings.get('rename_map'),
                                             replace_spaces=settings.get('replace_spaces'), inplace=True)
        # auto remove
        if cleaner_contract.get('auto_remove_columns') is not None:
            settings = cleaner_contract.get('auto_remove_columns')
            PandasCleaners.auto_remove_columns(df, null_min=settings.get('null_min'),
                                               nulls_list=settings.get('nulls_list'),
                                               predominant_max=settings.get('predominant_max'), inplace=True)
        # auto category
        if cleaner_contract.get('auto_to_category') is not None:
            settings = cleaner_contract.get('auto_to_category')
            PandasCleaners.auto_to_category(df, unique_max=settings.get('unique_max'),
                                            null_max=settings.get('null_max'), inplace=True)
        # auto drop duplicates
        if cleaner_contract.get('auto_drop_duplicates') is not None:
            settings = cleaner_contract.get('auto_drop_duplicates')
            PandasCleaners.auto_drop_duplicates(df, headers=settings.get('headers'), drop=settings.get('drop'),
                                                dtype=settings.get('dtype'), exclude=settings.get('exclude'),
                                                regex=settings.get('regex'),
                                                re_ignore_case=settings.get('re_ignore_case'), inplace=True)
        # auto drop correlated
        if cleaner_contract.get('auto_drop_correlated') is not None:
            settings = cleaner_contract.get('auto_drop_correlated')
            PandasCleaners.auto_drop_correlated(df, threshold=settings.get('threshold'),
                                                inc_category=settings.get('inc_category'), inplace=True)
        # 'to remove'
        if cleaner_contract.get('to_remove') is not None:
            settings = cleaner_contract.get('to_remove')
            PandasCleaners.to_remove(df, headers=settings.get('headers'), drop=settings.get('drop'),
                                     dtype=settings.get('dtype'), exclude=settings.get('exclude'),
                                     regex=settings.get('regex'), re_ignore_case=settings.get('re_ignore_case'),
                                     inplace=True)
        # 'to select'
        if cleaner_contract.get('to_select') is not None:
            settings = cleaner_contract.get('to_select')
            PandasCleaners.to_select(df, headers=settings.get('headers'), drop=settings.get('drop'),
                                     dtype=settings.get('dtype'), exclude=settings.get('exclude'),
                                     regex=settings.get('regex'), re_ignore_case=settings.get('re_ignore_case'),
                                     inplace=True)
        # 'to bool'
        if cleaner_contract.get('to_bool_type') is not None:
            settings = cleaner_contract.get('to_bool_type')
            PandasCleaners.to_bool_type(df, bool_map=settings.get('bool_map'), headers=settings.get('headers'),
                                        drop=settings.get('drop'), dtype=settings.get('dtype'),
                                        exclude=settings.get('exclude'), regex=settings.get('regex'),
                                        re_ignore_case=settings.get('re_ignore_case'), inplace=True)
        # 'to category'
        if cleaner_contract.get('to_category_type') is not None:
            settings = cleaner_contract.get('to_category_type')
            PandasCleaners.to_category_type(df, headers=settings.get('headers'), drop=settings.get('drop'),
                                            dtype=settings.get('dtype'), exclude=settings.get('exclude'),
                                            regex=settings.get('regex'), re_ignore_case=settings.get('re_ignore_case'),
                                            inplace=True)
        # 'to date'
        if cleaner_contract.get('to_date_type') is not None:
            settings = cleaner_contract.get('to_date_type')
            PandasCleaners.to_date_type(df, headers=settings.get('headers'), drop=settings.get('drop'),
                                        dtype=settings.get('dtype'), exclude=settings.get('exclude'),
                                        regex=settings.get('regex'), re_ignore_case=settings.get('re_ignore_case'),
                                        as_num=settings.get('as_num'), day_first=settings.get('day_first'),
                                        year_first=settings.get('year_first'), inplace=True)
        # 'to numeric'
        if cleaner_contract.get('to_numeric_type') is not None:
            settings = cleaner_contract.get('to_numeric_type')
            PandasCleaners.to_numeric_type(df, headers=settings.get('headers'), drop=settings.get('drop'),
                                           dtype=settings.get('dtype'), exclude=settings.get('exclude'),
                                           regex=settings.get('regex'), re_ignore_case=settings.get('re_ignore_case'),
                                           precision=settings.get('precision'), fillna=settings.get('fillna'),
                                           errors=settings.get('errors'), inplace=True)
        # 'to int'
        if cleaner_contract.get('to_int_type') is not None:
            settings = cleaner_contract.get('to_int_type')
            PandasCleaners.to_int_type(df, headers=settings.get('headers'), drop=settings.get('drop'),
                                       dtype=settings.get('dtype'), exclude=settings.get('exclude'),
                                       regex=settings.get('regex'), re_ignore_case=settings.get('re_ignore_case'),
                                       fillna=settings.get('fillna'), errors=settings.get('errors'), inplace=True)
        # 'to float'
        if cleaner_contract.get('to_float_type') is not None:
            settings = cleaner_contract.get('to_float_type')
            PandasCleaners.to_float_type(df, headers=settings.get('headers'), drop=settings.get('drop'),
                                         dtype=settings.get('dtype'), exclude=settings.get('exclude'),
                                         regex=settings.get('regex'), re_ignore_case=settings.get('re_ignore_case'),
                                         precision=settings.get('precision'), fillna=settings.get('fillna'),
                                         errors=settings.get('errors'), inplace=True)
        # 'to str'
        if cleaner_contract.get('to_str_type') is not None:
            settings = cleaner_contract.get('to_str_type')
            PandasCleaners.to_str_type(df, headers=settings.get('headers'), drop=settings.get('drop'),
                                       dtype=settings.get('dtype'), exclude=settings.get('exclude'),
                                       regex=settings.get('regex'), re_ignore_case=settings.get('re_ignore_case'),
                                       nulls_list=settings.get('nulls_list'), inplace=True)
        # 'to date from excel'
        if cleaner_contract.get('to_date_from_excel_type') is not None:
            settings = cleaner_contract.get('to_date_from_excel_type')
            PandasCleaners.to_date_from_excel_type(df, headers=settings.get('headers'), drop=settings.get('drop'),
                                                   dtype=settings.get('dtype'), exclude=settings.get('exclude'),
                                                   regex=settings.get('regex'),
                                                   re_ignore_case=settings.get('re_ignore_case'),
                                                   inplace=True)
        if not inplace:
            return df

    @staticmethod
    def filter_headers(df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                       exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None) -> list:
        """ returns a list of headers based on the filter criteria

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes. Default is False
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt' headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :return: a filtered list of headers

        :raise: TypeError if any of the types are not as expected
        """
        if drop is None or not isinstance(drop, bool):
            drop = False
        if exclude is None or not isinstance(exclude, bool):
            exclude = False
        if re_ignore_case is None or not isinstance(re_ignore_case, bool):
            re_ignore_case = False

        if not isinstance(df, pd.DataFrame):
            raise TypeError("The first function attribute must be a pandas 'DataFrame'")
        _headers = PandasCleaners.list_formatter(headers)
        dtype = PandasCleaners.list_formatter(dtype)
        regex = PandasCleaners.list_formatter(regex)
        _obj_cols = df.columns
        _rtn_cols = set()
        unmodified = True

        if _headers is not None:
            _rtn_cols = set(_obj_cols).difference(_headers) if drop else set(_obj_cols).intersection(_headers)
            unmodified = False

        if regex is not None and regex:
            re_ignore_case = re.I if re_ignore_case else 0
            _regex_cols = list()
            for exp in regex:
                _regex_cols += [s for s in _obj_cols if re.search(exp, s, re_ignore_case)]
            _rtn_cols = _rtn_cols.union(set(_regex_cols))
            unmodified = False

        if unmodified:
            _rtn_cols = set(_obj_cols)

        if dtype is not None and len(dtype) > 0:
            _df_selected = df.loc[:, _rtn_cols]
            _rtn_cols = (_df_selected.select_dtypes(exclude=dtype) if exclude
                         else _df_selected.select_dtypes(include=dtype)).columns

        return [c for c in _rtn_cols]

    @staticmethod
    def filter_columns(df, headers=None, drop=False, dtype=None, exclude=False, regex=None, re_ignore_case=None,
                       inplace=False) -> Union[dict, pd.DataFrame]:
        """ Returns a subset of columns based on the filter criteria

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt' columns
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return:
        """
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = PandasCleaners.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)
        return df.loc[:, obj_cols]

    @staticmethod
    def auto_clean_header(df, case=None, rename_map: dict=None, replace_spaces: str=None, inplace=False):
        """ clean the headers of a pandas DataFrame replacing space with underscore

        :param df: the pandas.DataFrame to drop duplicates from
        :param rename_map: a from: to dictionary of headers to rename
        :param case: changes the headers to lower, upper, title. if none of these then no change
        :param replace_spaces: character to replace spaces with. Default is '_' (underscore)
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        if isinstance(rename_map, dict):
            df.rename(mapper=rename_map, axis='columns', inplace=True)
        # removes any hidden characters
        for c in df.columns:
            df[c].column = str(c)
        # convert case
        if case is not None and isinstance(case, str):
            if case.lower() == 'lower':
                df.rename(mapper=str.lower, axis='columns', inplace=True)
            elif case.lower() == 'upper':
                df.rename(mapper=str.upper, axis='columns', inplace=True)
            elif case.lower() == 'title':
                df.rename(mapper=str.title, axis='columns', inplace=True)
        # replaces spaces at the end just in case title is used
        replace_spaces = '_' if not isinstance(replace_spaces, str) else replace_spaces
        df.columns = df.columns.str.replace(' ', replace_spaces)

        if inplace:
            return PandasCleaners._build_section('auto_clean_header', case=case, rename_map=rename_map,
                                                 replace_spaces=replace_spaces)
        return df

    @staticmethod
    def auto_to_category(df, unique_max: int=None, null_max: float=None, auto_contract: bool=True,
                         inplace=False) -> Union[dict, pd.DataFrame]:
        """ auto categorises columns that have a max number of uniqueness with a min number of nulls
        and are object dtype

        :param df: the pandas.DataFrame to auto categorise
        :param unique_max: the max number of unique values in the column. default to 20
        :param null_max: maximum number of null in the column between 0 and 1. default to 0.7 (70% nulls allowed)
        :param auto_contract: if the auto_category or to_category should be returned
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        unique_max = 20 if not isinstance(unique_max, int) else unique_max
        null_max = 0.7 if not isinstance(null_max, (int, float)) else null_max
        df_len = len(df)
        obj_cols = PandasCleaners.filter_headers(df, dtype='object')
        col_cat = []
        for c in obj_cols:
            if df[c].nunique() < unique_max and round(df[c].isnull().sum() / df_len, 2) < null_max:
                col_cat.append(c)
        result = PandasCleaners.to_category_type(df, headers=col_cat, inplace=inplace)
        if inplace and auto_contract:
            return PandasCleaners._build_section('auto_to_category', unique_max=unique_max, null_max=null_max)
        return result

    # drop column that only have 1 value in them
    @staticmethod
    def auto_remove_columns(df, null_min: float=None, predominant_max: float=None, nulls_list: [bool, list]=None,
                            auto_contract: bool=True, inplace=False) -> Union[dict, pd.DataFrame]:
        """ auto removes columns that are np.NaN, a single value or have a predominant value greater than.

        :param df: the pandas.DataFrame to auto remove
        :param null_min: the minimum number of null values default to 0.998 (99.8%) nulls
        :param predominant_max: the percentage max a single field predominates default is 0.998
        :param nulls_list: can be boolean or a list:
                    if boolean and True then null_list equals ['NaN', 'nan', 'null', '', 'None', ' ']
                    if list then this is considered potential null values.
        :param auto_contract: if the auto_category or to_category should be returned
        :param inplace: if to change the passed pandas.DataFrame or return a copy (see return)
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        null_min = 0.998 if not isinstance(null_min, (int, float)) else null_min
        predominant_max = 0.998 if not isinstance(predominant_max, (int, float)) else predominant_max
        if isinstance(nulls_list, bool) and nulls_list:
            nulls_list = ['NaN', 'nan', 'null', '', 'None', ' ']
        elif not isinstance(nulls_list, list):
            nulls_list = None
        df_len = len(df)
        col_drop = []
        for c in df.columns:
            col = deepcopy(df[c])
            if nulls_list is not None:
                col.replace(nulls_list, np.nan, inplace=True)
            if round(col.isnull().sum() / df_len, 5) > null_min:
                col_drop.append(c)
            elif col.nunique() == 1:
                col_drop.append(c)
            elif round((col.value_counts() / np.float(len(col.dropna()))).sort_values(
                    ascending=False).values[0], 5) >= predominant_max:
                col_drop.append(c)

        result = PandasCleaners.to_remove(df, headers=col_drop, inplace=inplace)
        if inplace and auto_contract:
            return PandasCleaners._build_section('auto_remove_columns', null_min=null_min,
                                                 predominant_max=predominant_max, nulls_list=nulls_list)
        return result

    # drop duplicate columns
    @staticmethod
    def auto_drop_duplicates(df, headers=None, drop=False, dtype=None, exclude=False, regex=None, re_ignore_case=None,
                             inplace=False) -> Union[dict, pd.DataFrame]:
        """ drops duplicate columns from the pd.DataFrame.

        :param df: the pandas.DataFrame to drop duplicates from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        df_filter = PandasCleaners.filter_columns(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)
        duplicated_col = []
        for i in range(0, len(df_filter)):
            col_1 = df_filter.columns[i]

            for col_2 in df_filter.columns[i + 1:]:
                if df_filter[col_1].equals(df_filter[col_2]):
                    duplicated_col.append(col_2)
        df.drop(labels=duplicated_col, axis=1, inplace=True)
        if inplace:
            return PandasCleaners._build_section('auto_drop_duplicates', headers=headers, drop=drop, dtype=dtype,
                                                 exclude=exclude, regex=regex, re_ignore_case=re_ignore_case)
        return df

    # drops highly correlated columns
    @staticmethod
    def auto_drop_correlated(df, threshold: float=None, inc_category: bool=False, inplace=False) -> [dict, pd.DataFrame]:
        """ uses 'brute force' techniques to removes highly correlated columns based on the threshold,
        set by default to 0.998.

        :param df: data: the Canonical data to drop duplicates from
        :param threshold: (optional) threshold correlation between columns. default 0.998
        :param inc_category: (optional) if category type columns should be converted to numeric representations
        :param inplace: if the passed Canonical, should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy Canonical,.
        """
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        threshold = threshold if isinstance(threshold, float) and 0 < threshold < 1 else 0.998
        df_filter = PandasCleaners.filter_columns(df, dtype=['number'], exclude=False)
        if inc_category:
            for col in PandasCleaners.filter_columns(df, dtype=['category'], exclude=False):
                df_filter[col] = df[col].cat.codes
        col_corr = set()
        corr_matrix = df_filter.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        df.drop(labels=col_corr, axis=1, inplace=True)
        if inplace:
            return PandasCleaners._build_section('auto_drop_correlated', threshold=threshold, inc_category=inc_category)
        return df

    # drop unwanted
    @staticmethod
    def to_remove(df, headers=None, drop=False, dtype=None, exclude=False, regex=None, re_ignore_case=None,
                  inplace=False) -> Union[dict, pd.DataFrame]:
        """ remove columns from the pandas.DataFrame

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = PandasCleaners.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)
        df.drop(obj_cols, axis=1, inplace=True)
        if inplace:
            return PandasCleaners._build_section('to_remove', headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)
        return df

    # drop unwanted
    @staticmethod
    def to_select(df, headers=None, drop=False, dtype=None, exclude=False, regex=None, re_ignore_case=None,
                  inplace=False) -> Union[dict, pd.DataFrame]:
        """ remove columns from the pandas.DataFrame

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = PandasCleaners.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)

        PandasCleaners.to_remove(df, headers=obj_cols, drop=True, inplace=True)

        if inplace:
            return PandasCleaners._build_section('to_select', headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)
        return df

    # convert boolean
    @staticmethod
    def to_bool_type(df, bool_map, headers=None, drop=False, dtype=None, exclude=False, regex=None, re_ignore_case=None,
                     inplace=False) -> Union[dict, pd.DataFrame]:
        """ converts column to bool based on the map

        :param df: the Pandas.DataFrame to get the column headers from
        :param bool_map: a mapping of what to make True and False
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        if not isinstance(bool_map, dict):
            raise TypeError("The map attribute must be of type 'dict'")
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        if not bool_map:  # map is empty so nothing to map
            return df
        obj_cols = PandasCleaners.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)
        for c in obj_cols:
            if df[c].dtype.name != 'bool':
                df[c] = df[c].map(bool_map)
                df[c] = df[c].fillna(False)
                df[c] = df[c].astype('bool')
        if inplace:
            return PandasCleaners._build_section('to_bool_type', bool_map=bool_map, headers=headers, drop=drop,
                                                 dtype=dtype, exclude=exclude, regex=regex,
                                                 re_ignore_case=re_ignore_case)
        else:
            return df

    # convert objects to categories
    @staticmethod
    def to_category_type(df, headers=None, drop=False, dtype=None, exclude=False, regex=None, re_ignore_case=None,
                         inplace=False) -> Union[dict, pd.DataFrame]:
        """ converts columns to categories

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = PandasCleaners.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = df[c].astype('category')
        if inplace:
            return PandasCleaners._build_section('to_category_type', headers=headers, drop=drop, dtype=dtype,
                                                 exclude=exclude, regex=regex, re_ignore_case=re_ignore_case)
        else:
            return df

    @staticmethod
    def _to_numeric(df, numeric_type, fillna, errors=None, headers=None, drop=False, dtype=None, exclude=False,
                    regex=None, re_ignore_case=None, precision=None, inplace=False) -> Union[dict, pd.DataFrame]:
        """ Code reuse method for all the numeric types. see calling methods for inline docs"""
        if errors is None or str(errors) not in ['ignore', 'raise', 'coerce']:
            errors = 'coerce'
        if not inplace:
            df = deepcopy(df)
        obj_cols = PandasCleaners.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace('[$£€, ]', ''), errors=errors)
            if fillna is None:
                df[c] = df[c].fillna(np.nan)
            elif str(fillna).lower() == 'mean':
                df[c] = df[c].fillna(df[c].mean())
            elif str(fillna).lower() == 'mode':
                _value = []
                for _ in df[df[c].isna()].index:
                    _value.append(df[c].mode().sample().iloc[0])
                df.loc[df[c].isna(), c] = _value
            elif str(fillna).lower() == 'median':
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(fillna)

            if not isinstance(precision, int):
                try:
                    precision = df[c].dropna().apply(str).str.extract('\.(.*)')[0].map(len).max()
                except:
                    precision = 15
            if str(numeric_type).lower().startswith('int'):
                df[c] = df[c].round(0).astype(int)
            elif str(numeric_type).lower().startswith('float'):
                df[c] = df[c].round(precision).astype(float)

        if inplace:
            return PandasCleaners._build_section(numeric_type, fillna=fillna, headers=headers, drop=drop, dtype=dtype,
                                                 exclude=exclude, regex=regex, re_ignore_case=re_ignore_case,
                                                 errors=errors, precision=precision)
        else:
            return df

    @staticmethod
    def to_numeric_type(df, headers=None, drop=False, dtype=None, exclude=False,  regex=None, re_ignore_case=None,
                        precision=None, fillna=None, errors=None, inplace=False) -> Union[dict, pd.DataFrame]:
        """ converts columns to int type

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param precision: how many decimal places to set the return values. if None then the number is unchanged
        :param fillna: { num_value, 'mean', 'mode', 'median' }. Default to np.nan
                    - If num_value, then replaces NaN with this number value. Must be a value not a string
                    - If 'mean', then replaces NaN with the mean of the column
                    - If 'mode', then replaces NaN with a mode of the column. random sample if more than 1
                    - If 'median', then replaces NaN with the median of the column
        :param errors : {'ignore', 'raise', 'coerce'}, default 'coerce'
                    - If 'raise', then invalid parsing will raise an exception
                    - If 'coerce', then invalid parsing will be set as NaN
                    - If 'ignore', then invalid parsing will return the input
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        if fillna is None or not fillna:
            fillna = np.nan
        return PandasCleaners._to_numeric(df, 'to_numeric_type', fillna=fillna, errors=errors, headers=headers,
                                          drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case, precision=precision, inplace=inplace)

    @staticmethod
    def to_int_type(df, headers=None, drop=False, dtype=None, exclude=False,  regex=None, re_ignore_case=None,
                    fillna=None, errors=None, inplace=False) -> Union[dict, pd.DataFrame]:
        """ converts columns to int type

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param fillna: { num_value, 'mean', 'mode', 'median' }. Default to 0
                    - If num_value, then replaces NaN with this number value
                    - If 'mean', then replaces NaN with the mean of the column
                    - If 'mode', then replaces NaN with a mode of the column. random sample if more than 1
                    - If 'median', then replaces NaN with the median of the column
        :param errors : {'ignore', 'raise', 'coerce'}, default 'coerce'
                    - If 'raise', then invalid parsing will raise an exception
                    - If 'coerce', then invalid parsing will be set as NaN
                    - If 'ignore', then invalid parsing will return the input
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        if fillna is None or not fillna:
            fillna = 0
        return PandasCleaners._to_numeric(df, 'to_int_type', fillna=fillna, errors=errors, headers=headers, drop=drop,
                                          dtype=dtype, exclude=exclude, regex=regex, re_ignore_case=re_ignore_case,
                                          inplace=inplace)

    @staticmethod
    def to_float_type(df, headers=None, drop=False, dtype=None, exclude=False, regex=None, re_ignore_case=None,
                      precision=None, fillna=None, errors=None, inplace=False) -> Union[dict, pd.DataFrame]:
        """ converts columns to float type

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param precision: how many decimal places to set the return values. if None then the number is unchanged
        :param fillna: { num_value, 'mean', 'mode', 'median' }. Default to np.nan
                    - If num_value, then replaces NaN with this number value
                    - If 'mean', then replaces NaN with the mean of the column
                    - If 'mode', then replaces NaN with a mode of the column. random sample if more than 1
                    - If 'median', then replaces NaN with the median of the column
        :param errors : {'ignore', 'raise', 'coerce'}, default 'coerce' }. Default to 'coerce'
                    - If 'raise', then invalid parsing will raise an exception
                    - If 'coerce', then invalid parsing will be set as NaN
                    - If 'ignore', then invalid parsing will return the input
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        if fillna is None or not fillna:
            fillna = np.nan
        return PandasCleaners._to_numeric(df, 'to_float_type', fillna=fillna, errors=errors, headers=headers, drop=drop,
                                          dtype=dtype, exclude=exclude, regex=regex, re_ignore_case=re_ignore_case,
                                          precision=precision, inplace=inplace)

    @staticmethod
    def to_str_type(df, headers=None, drop=False, dtype=None, exclude=False, regex=None, re_ignore_case=None,
                    inplace=False, nulls_list: [bool, list]=None) -> Union[dict, pd.DataFrame]:
        """ converts columns to object type

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param nulls_list: can be boolean or a list:
                    if boolean and True then null_list equals ['NaN', 'nan', 'null', '', 'None']
                    if list then this is considered potential null values.
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
       """
        if isinstance(nulls_list, bool) and nulls_list:
            nulls_list = ['NaN', 'nan', 'null', 'NULL', ' ', '', 'None']
        elif not isinstance(nulls_list, list):
            nulls_list = None

        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = PandasCleaners.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = df[c].astype(str)
            if nulls_list is not None:
                df[c] = df[c].replace(nulls_list, np.nan)

        if inplace:
            return PandasCleaners._build_section('to_str_type', headers=headers, drop=drop, dtype=dtype,
                                                 exclude=exclude, regex=regex, re_ignore_case=re_ignore_case,
                                                 nulls_list=nulls_list)
        else:
            return df

    @staticmethod
    def to_date_type(df, headers=None, drop=False, dtype=None, exclude=False, regex=None, re_ignore_case=None,
                     as_num=False, day_first=False, year_first=False, date_format=None,
                     inplace=False) -> [dict, pd.DataFrame]:
        """ converts columns to date types

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param as_num: if true returns number of days since 0001-01-01 00:00:00 with fraction being hours/mins/secs
        :param year_first: specifies if to parse with the year first
                If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
                If both dayfirst and yearfirst are True, yearfirst is preceded (same as dateutil).
        :param day_first: specifies if to parse with the day first
                If True, parses dates with the day first, eg %d-%m-%Y.
                If False default to the a prefered preference, normally %m-%d-%Y (but not strict)
        :param date_format: if the date can't be inferred uses date format eg format='%Y%m%d'
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        infer_datetime_format = date_format is None
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = PandasCleaners.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = df[c].fillna(np.nan)
            df[c] = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=infer_datetime_format,
                                   dayfirst=day_first, yearfirst=year_first, format=date_format)
            if as_num:
                df[c] = mdates.date2num(df[c])
        if inplace:
            return PandasCleaners._build_section('to_date_type', headers=headers, drop=drop, dtype=dtype,
                                                 exclude=exclude, regex=regex, re_ignore_case=re_ignore_case,
                                                 as_num=as_num, day_first=day_first, year_first=year_first)
        else:
            return df

    # converts excel object to dates
    @staticmethod
    def _excel_date_converter(date_float):
        if pd.isnull(date_float):
            return np.nan
        if isinstance(date_float, float):
            temp = datetime.datetime(1900, 1, 1)
            delta = datetime.timedelta(days=date_float)
            return temp + delta
        return date_float

    @staticmethod
    def to_date_from_excel_type(df, headers=None, drop=False, dtype=None, exclude=False, regex=None,
                                re_ignore_case=None, inplace=False) -> Union[dict, pd.DataFrame]:
        """converts excel date formats into datetime

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regiar expression to seach the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
       """
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        if dtype is None:
            dtype = ['float64']
        obj_cols = PandasCleaners.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                 regex=regex, re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = [PandasCleaners._excel_date_converter(d) for d in df[c]]
        if inplace:
            return PandasCleaners._build_section('excel_to_date', headers=headers, drop=drop, dtype=dtype,
                                                 exclude=exclude, regex=regex, re_ignore_case=re_ignore_case)
        else:
            return df

    @staticmethod
    def list_formatter(value) -> [List[str], list, None]:
        """ Useful utility method to convert any type of str, list, tuple or pd.Series into a list"""
        if isinstance(value, (int, float, str, pd.Timestamp)):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, dict):
            return list(value.items())
        return None

    @staticmethod
    def _build_section(key, headers=None, drop=None, dtype=None, exclude=None, fillna=None, bool_map=None,
                       null_min=None, null_max=None, single_value=None, nulls_list=None, unique_max=None,
                       regex=None, re_ignore_case=None, case=None, rename_map=None, replace_spaces=None,
                       predominant_max=None, errors=None, precision=None, as_num=None, day_first=None,
                       year_first=None, threshold: float=None, inc_category: bool=None) -> dict:
        section = {}
        if headers is not None:
            section['headers'] = headers
            section['drop'] = drop if drop is not None else False
        if dtype is not None:
            section['dtype'] = dtype
            section['exclude'] = exclude if exclude is not None else False
        if regex is not None:
            section['regex'] = regex
            section['re_ignore_case'] = re_ignore_case if re_ignore_case is not None else False
        if as_num is not None:
            section['as_num'] = as_num
        if day_first is not None:
            section['day_first'] = day_first
        if year_first is not None:
            section['year_first'] = year_first
        if fillna is not None:
            section['fillna'] = fillna
        if errors is not None:
            section['errors'] = errors
        if precision is not None:
            section['precision'] = precision
        if bool_map is not None:
            section['bool_map'] = bool_map
        if null_min is not None:
            section['null_min'] = null_min
        if null_max is not None:
            section['null_max'] = null_max
        if predominant_max is not None:
            section['predominant_max'] = predominant_max
        if nulls_list is not None:
            section['nulls_list'] = nulls_list
        if single_value is not None:
            section['single_value'] = single_value
        if unique_max is not None:
            section['unique_max'] = unique_max
        if case is not None:
            section['case'] = case
        if rename_map is not None:
            section['rename_map'] = rename_map
        if replace_spaces is not None:
            section['replace_spaces'] = replace_spaces
        if threshold is not None:
            section['threshold'] = threshold
        if inc_category is not None:
            section['inc_category'] = inc_category
        return {key: section}
