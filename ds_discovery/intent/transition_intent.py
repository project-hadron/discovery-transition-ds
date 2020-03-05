import inspect
import threading
import re
from builtins import staticmethod
from copy import deepcopy
import datetime
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from aistac.properties.abstract_properties import AbstractPropertyManager
from aistac.intent.abstract_intent import AbstractIntentModel

__author__ = 'Darryl Oatridge'

from ds_discovery.transition.commons import Commons


class TransitionIntentModel(AbstractIntentModel):
    """A set of methods to help clean columns with a Pandas.DataFrame"""

    def __init__(self, property_manager: AbstractPropertyManager, default_save_intent: bool=None,
                 intent_next_available: bool=None, default_replace_intent: bool=None, intent_type_additions: list=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param intent_next_available: (optional) if the default level should be set to next available level or zero
        :param default_replace_intent: (optional) the default replace strategy for the same intent found at that level
        :param intent_type_additions: (optional) if additional data types need to be supported as an intent param
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = -1 if isinstance(intent_next_available, bool) and intent_next_available else 0
        intent_param_exclude = ['df', 'inplace', 'canonical']
        intent_type_additions = intent_type_additions if isinstance(intent_type_additions, list) else list()
        intent_type_additions += [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]
        super().__init__(property_manager=property_manager, intent_param_exclude=intent_param_exclude,
                         default_save_intent=default_save_intent, default_intent_level=default_intent_level,
                         default_replace_intent=default_replace_intent, intent_type_additions=intent_type_additions)

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
        if self._pm.has_intent():
            # get the list of levels to run
            if isinstance(intent_levels, (int, str, list)):
                intent_levels = Commons.list_formatter(intent_levels)
            else:
                intent_levels = sorted(self._pm.get_intent().keys())
            for level in intent_levels:
                for method, params in self._pm.get_intent(level=level).items():
                    if method in self.__dir__():
                        if isinstance(kwargs, dict):
                            params.update(kwargs)
                        method_params = {'self': self, 'canonical': canonical, 'params': params}
                        canonical = eval(f"self.{method}(canonical, inplace=False, save_intent=False, **params)",
                                         globals(), method_params)
        return canonical

    @staticmethod
    def filter_headers(df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                       exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None) -> list:
        """ returns a list of headers based on the filter criteria

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None.
                    example: int, float, bool, 'category', 'object', 'number'. 'datetime', 'datetimetz', 'timedelta'
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
        _headers = Commons.list_formatter(headers)
        dtype = Commons.list_formatter(dtype)
        regex = Commons.list_formatter(regex)
        _obj_cols = df.columns
        _rtn_cols = set()
        unmodified = True

        if _headers:
            _rtn_cols = set(_obj_cols).difference(_headers) if drop else set(_obj_cols).intersection(_headers)
            unmodified = False

        if regex and regex:
            re_ignore_case = re.I if re_ignore_case else 0
            _regex_cols = list()
            for exp in regex:
                _regex_cols += [s for s in _obj_cols if re.search(exp, s, re_ignore_case)]
            _rtn_cols = _rtn_cols.union(set(_regex_cols))
            unmodified = False

        if unmodified:
            _rtn_cols = set(_obj_cols)

        if dtype and len(dtype) > 0:
            _df_selected = df.loc[:, _rtn_cols]
            _rtn_cols = (_df_selected.select_dtypes(exclude=dtype) if exclude
                         else _df_selected.select_dtypes(include=dtype)).columns

        return [c for c in _rtn_cols]

    @staticmethod
    def filter_columns(df, headers: [str, list]=None, drop: bool=False, dtype: [str, list]=None, exclude: bool=False,
                       regex: [str, list]=None, re_ignore_case: bool=False, inplace=False) -> [dict, pd.DataFrame]:
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
        obj_cols = TransitionIntentModel.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                        regex=regex, re_ignore_case=re_ignore_case)
        return df.loc[:, obj_cols]

    def auto_clean_header(self, df, case=None, rename_map: dict=None, replace_spaces: str=None, inplace: bool=False,
                          save_intent: bool=None, intent_level: [int, str]=None):
        """ clean the headers of a pandas DataFrame replacing space with underscore

        :param df: the pandas.DataFrame to drop duplicates from
        :param rename_map: a from: to dictionary of headers to rename
        :param case: changes the headers to lower, upper, title, snake. if none of these then no change
        :param replace_spaces: character to replace spaces with. Default is '_' (underscore)
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
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
        if not inplace:
            return df
        return

    def auto_to_category(self, df, unique_max: int=None, null_max: float=None, fill_nulls: str=None,
                         nulls_list: [bool, list]=None,inplace: bool=False, save_intent: bool=None,
                         intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """ auto categorises columns that have a max number of uniqueness with a min number of nulls
        and are object dtype

        :param df: the pandas.DataFrame to auto categorise
        :param unique_max: the max number of unique values in the column. default to 20
        :param null_max: maximum number of null in the column between 0 and 1. default to 0.7 (70% nulls allowed)
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        unique_max = 20 if not isinstance(unique_max, int) else unique_max
        null_max = 0.7 if not isinstance(null_max, (int, float)) else null_max
        df_len = len(df)
        col_cat = []
        for c in df.columns:
            if df[c].nunique() < unique_max and round(df[c].isnull().sum() / df_len, 2) < null_max:
                col_cat.append(c)
        result = self.to_category_type(df, headers=col_cat, fill_nulls=fill_nulls, nulls_list=nulls_list,
                                       inplace=inplace, save_intent=False)
        if not inplace:
            return result
        return

    # drop column that only have 1 value in them
    def auto_remove_columns(self, df, null_min: float=None, predominant_max: float=None, nulls_list: [bool, list]=None,
                            auto_contract: bool=True, test_size: float=None, random_state: int=None,
                            inplace: bool=False, save_intent: bool=None,
                            intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """ auto removes columns that are np.NaN, a single value or have a predominant value greater than.

        :param df: the pandas.DataFrame to auto remove
        :param null_min: the minimum number of null values default to 0.998 (99.8%) nulls
        :param predominant_max: the percentage max a single field predominates default is 0.998
        :param nulls_list: can be boolean or a list:
                    if boolean and True then null_list equals ['NaN', 'nan', 'null', '', 'None', ' ']
                    if list then this is considered potential null values.
        :param auto_contract: if the auto_category or to_category should be returned
        :param test_size: a test percentage split from the df to avoid over-fitting. Default is 0 for no split
        :param random_state: a random state should be applied to the test train split. Default is None
        :param inplace: if to change the passed pandas.DataFrame or return a copy (see return)
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        null_min = 0.998 if not isinstance(null_min, (int, float)) else null_min
        predominant_max = 0.998 if not isinstance(predominant_max, (int, float)) else predominant_max
        if isinstance(nulls_list, bool) and nulls_list:
            nulls_list = ['NaN', 'nan', 'null', '', 'None', ' ']
        elif not isinstance(nulls_list, list):
            nulls_list = None
        if isinstance(test_size, float) and 0 < test_size < 1:
            df_filter, _ = train_test_split(deepcopy(df), test_size=test_size, random_state=random_state)
        else:
            df_filter = deepcopy(df)
        df_len = len(df_filter)
        col_drop = []
        for c in df_filter.columns:
            if nulls_list is not None:
                df_filter[c].replace(nulls_list, np.nan, inplace=True)
            if round(df_filter[c].isnull().sum() / df_len, 5) > null_min:
                col_drop.append(c)
            elif df_filter[c].nunique() == 1:
                col_drop.append(c)
            elif round((df_filter[c].value_counts() / np.float(len(df_filter[c].dropna()))).sort_values(
                    ascending=False).values[0], 5) >= predominant_max:
                col_drop.append(c)
        result = self.to_remove(df, headers=col_drop, inplace=inplace, save_intent=False)
        if not inplace:
            return result
        return

    # drop duplicate columns
    def auto_drop_duplicates(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=False,
                             dtype: [str, list]=None, exclude: bool=False, regex: [str, list]=None,
                             re_ignore_case: bool=False, test_size: float=None, random_state: int=None,
                             inplace: bool=False, save_intent: bool=None,
                             intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """ drops duplicate columns from the pd.DataFrame.

        :param df: the pandas.DataFrame to drop duplicates from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param test_size: a test percentage split from the df to avoid over-fitting. Default is 0 for no split
        :param random_state: a random state should be applied to the test train split. Default is None
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Intent task
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        df_filter = self.filter_columns(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                        re_ignore_case=re_ignore_case)
        if isinstance(test_size, float) and 0 < test_size < 1:
            df_filter, _ = train_test_split(df_filter, test_size=test_size, random_state=random_state)
        duplicated_col = []
        for i in range(0, len(df_filter)):
            col_1 = df_filter.columns[i]

            for col_2 in df_filter.columns[i + 1:]:
                if df_filter[col_1].equals(df_filter[col_2]):
                    duplicated_col.append(col_2)
        df.drop(labels=duplicated_col, axis=1, inplace=True)
        if not inplace:
            return df
        return

    # drops highly correlated columns
    def auto_drop_correlated(self, df: pd.DataFrame, threshold: float=None, inc_category: bool=False,
                             inc_str: bool=False, test_size: float=None, random_state: int=None, inplace: bool=False,
                             save_intent: bool=None, intent_level: [int, str]=None) -> [dict, pd.DataFrame]:
        """ uses 'brute force' techniques to removes highly correlated columns based on the threshold,
        set by default to 0.998.

        :param df: data: the Canonical data to drop duplicates from
        :param threshold: (optional) threshold correlation between columns. default 0.998
        :param inc_category: (optional) if category type columns should be converted to numeric representations
        :param inc_str: (optional) if str type columns should be converted to numeric representations
        :param test_size: a test percentage split from the df to avoid over-fitting. Default is 0 for no split
        :param random_state: a random state should be applied to the test train split. Default is None
        :param inplace: if the passed Canonical, should be used or a deep copy
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy Canonical,.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        threshold = threshold if isinstance(threshold, float) and 0 < threshold < 1 else 0.998
        df_filter = self.filter_columns(df, dtype=['number'], exclude=False)
        if isinstance(test_size, float) and 0 < test_size < 1:
            df_filter, _ = train_test_split(df_filter, test_size=test_size, random_state=random_state)
        if inc_category:
            for col in self.filter_columns(df, dtype=['category'], exclude=False):
                df_filter[col] = df[col].cat.codes
        if inc_str:
            for col in self.filter_columns(df, dtype=['object', 'string'], exclude=False):

                label_encoder = LabelEncoder().fit(df[col])
                df_filter[col] = label_encoder.transform(df[col])
        col_corr = set()
        corr_matrix = df_filter.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        df.drop(labels=col_corr, axis=1, inplace=True)
        if not inplace:
            return df
        return

    # drop unwanted
    def to_remove(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=False, dtype: [str, list]=None,
                  exclude: bool=False, regex: [str, list]=None, re_ignore_case: bool=False, inplace: bool=False,
                  save_intent: bool=None, intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """ remove columns from the pandas.DataFrame

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = self.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                       re_ignore_case=re_ignore_case)
        df.drop(obj_cols, axis=1, inplace=True)
        if not inplace:
            return df
        return

    # drop unwanted
    def to_select(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=False, dtype: [str, list]=None,
                  exclude: bool=False, regex: [str, list]=None, re_ignore_case: bool=False, inplace: bool=False,
                  save_intent: bool=None, intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """ remove columns from the pandas.DataFrame

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = self.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                       re_ignore_case=re_ignore_case)

        self.to_remove(df, headers=obj_cols, drop=True, inplace=True, save_intent=False)
        if not inplace:
            return df
        return

    # convert boolean
    def to_bool_type(self, df: pd.DataFrame, bool_map, headers: [str, list]=None, drop: bool=False,
                     dtype: [str, list]=None, exclude: bool=False, regex: [str, list]=None,
                     re_ignore_case: bool=False, inplace: bool=False, save_intent: bool=None,
                     intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """ converts column to bool based on the map

        :param df: the Pandas.DataFrame to get the column headers from
        :param bool_map: a mapping of what to make True and False
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        if not isinstance(bool_map, dict):
            raise TypeError("The map attribute must be of type 'dict'")
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        if not bool_map:  # map is empty so nothing to map
            return df
        obj_cols = self.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                       re_ignore_case=re_ignore_case)
        for c in obj_cols:
            if df[c].dtype.name != 'bool':
                df[c] = df[c].map(bool_map)
                df[c] = df[c].fillna(False)
                df[c] = df[c].astype('bool')
        if not inplace:
            return df
        return

    # convert objects to categories
    def to_category_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=False, dtype: [str, list]=None,
                         exclude: bool=False, regex: [str, list]=None, re_ignore_case: bool=False, inplace: bool=False,
                         as_num: bool=False, fill_nulls: str=None, nulls_list: [bool, list]=None,
                         save_intent: bool=None, intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """ converts columns to categories

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param as_num: if true returns the category as a category code
        :param fill_nulls: a string value to fill nullsthat then can be idenfied as a category type
        :param nulls_list: can be boolean or a list:
                    if boolean and True then null_list equals ['NaN', 'nan', 'null', '', 'None'. np.nan, None]
                    if list then this is considered potential null values.
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        if isinstance(nulls_list, bool) or not isinstance(nulls_list, list):
            nulls_list = ['NaN', 'nan', 'null', 'NULL', ' ', '', 'None', np.nan, None]
        if fill_nulls is None:
            fill_nulls = np.nan

        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = self.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                       re_ignore_case=re_ignore_case)
        for c in obj_cols:
            if isinstance(fill_nulls, str):
                df[c] = df[c].replace(nulls_list, fill_nulls)
            df[c] = df[c].astype('category')
            if as_num:
                df[c] = df[c].cat.codes
        if not inplace:
            return df
        return

    def _to_numeric(self, df: pd.DataFrame, numeric_type, fillna, errors=None, headers: [str, list]=None,
                    drop: bool=False, dtype: [str, list]=None, exclude: bool=False, regex: [str, list]=None,
                    re_ignore_case: bool=False, precision=None, inplace=False) -> [dict, pd.DataFrame]:
        """ Code reuse method for all the numeric types. see calling methods for inline docs"""
        if errors is None or str(errors) not in ['ignore', 'raise', 'coerce']:
            errors = 'coerce'
        if not inplace:
            df = deepcopy(df)
        obj_cols = self.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                       re_ignore_case=re_ignore_case)
        for c in obj_cols:
            if not isinstance(precision, int):
                try:
                    precision = df[c].dropna().apply(str).str.extract('\.(.*)')[0].map(len).max()
                except (ValueError, TypeError, IndexError, KeyError, ReferenceError, NameError, RecursionError):
                    precision = 15
            df[c] = pd.to_numeric(df[c].apply(str).str.replace('[$£€, ]', ''), errors=errors)
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

            if str(numeric_type).lower().startswith('int'):
                df[c] = df[c].round(0).astype(int)
            elif str(numeric_type).lower().startswith('float'):
                df[c] = df[c].round(precision).astype(float)
        return df

    def to_numeric_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=False, dtype: [str, list]=None,
                        exclude: bool=False, regex: [str, list]=None, re_ignore_case: bool=False, precision=None,
                        fillna=None, errors=None, inplace: bool=False,
                        save_intent: bool=None, intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """ converts columns to int type

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
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
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        if fillna is None or not fillna:
            fillna = np.nan
        df = self._to_numeric(df, numeric_type='numeric', fillna=fillna, errors=errors, headers=headers, drop=drop,
                              dtype=dtype, exclude=exclude, regex=regex, re_ignore_case=re_ignore_case,
                              precision=precision, inplace=inplace)
        if not inplace:
            return df
        return

    def to_int_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=False, dtype: [str, list]=None,
                    exclude: bool=False,  regex: [str, list]=None, re_ignore_case: bool=False, fillna=None,
                    errors=None, inplace: bool=False, save_intent: bool=None,
                    intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """ converts columns to int type

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
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
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        if fillna is None or not fillna:
            fillna = 0
        df = self._to_numeric(df, numeric_type='int', fillna=fillna, errors=errors, headers=headers, drop=drop,
                              dtype=dtype, exclude=exclude, regex=regex, re_ignore_case=re_ignore_case, inplace=inplace)
        if not inplace:
            return df
        return

    def to_float_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=False, dtype: [str, list]=None,
                      exclude: bool=False, regex: [str, list]=None, re_ignore_case: bool=False, precision=None,
                      fillna=None, errors=None, inplace: bool=False, save_intent: bool=None,
                      intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """ converts columns to float type

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
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
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        if fillna is None or not fillna:
            fillna = np.nan
        df = self._to_numeric(df, numeric_type='float', fillna=fillna, errors=errors, headers=headers, drop=drop,
                              dtype=dtype, exclude=exclude, regex=regex, re_ignore_case=re_ignore_case,
                              precision=precision, inplace=inplace)
        if not inplace:
            return df
        return

    def to_normalised(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=False, dtype: [str, list]=None,
                      exclude: bool=False, regex: [str, list]=None, re_ignore_case: bool=False, precision=None,
                      inplace: bool=False, save_intent: bool=None, intent_level: [int, str]=None):
        """ converts columns to float type

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param precision: how many decimal places to set the return values. if None then the number is unchanged
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        self.to_numeric_type(df, dtype='number', fillna=0, inplace=True, save_intent=False)
        obj_cols = self.filter_headers(df, dtype=['number'], exclude=False)
        for c in obj_cols:
            df[c] = df[c] / np.linalg.norm(df[c])
            if isinstance(precision, int):
                df[c] = np.round(df[c], precision)
        if not inplace:
            return df
        return

    def to_str_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=False, dtype: [str, list]=None,
                    exclude: bool=False, regex: [str, list]=None, re_ignore_case: bool=False, inplace: bool=False,
                    nulls_list: [bool, list]=None, as_num: bool=False,
                    save_intent: bool=None, intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """ converts columns to object type

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param nulls_list: can be boolean or a list:
                    if boolean and True then null_list equals ['NaN', 'nan', 'null', '', 'None'. np.nan, None]
                    if list then this is considered potential null values.
        :param as_num: if true returns the string as a number using Scikit-learn LabelEncoder
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
       """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        if isinstance(nulls_list, bool) and nulls_list:
            nulls_list = ['NaN', 'nan', 'null', 'NULL', ' ', '', 'None']
        elif not isinstance(nulls_list, list):
            nulls_list = ['nan']

        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = self.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                       re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = df[c].apply(str).astype('string', errors='ignore')
            if nulls_list is not None:
                df[c] = df[c].replace(nulls_list, np.nan)
            if as_num:
                label_encoder = LabelEncoder().fit(df[c])
                df[c] = label_encoder.transform(df[c])
        if not inplace:
            return df
        return

    def to_date_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=False, dtype: [str, list]=None,
                     exclude: bool=False, regex: [str, list]=None, re_ignore_case: bool=False, as_num=False,
                     day_first=False, year_first=False, date_format=None, inplace: bool=False, save_intent: bool=None,
                     intent_level: [int, str]=None) -> [dict, pd.DataFrame]:
        """ converts columns to date types

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
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
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        infer_datetime_format = date_format is None
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        obj_cols = self.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                       re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = df[c].fillna(np.nan)
            df[c] = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=infer_datetime_format,
                                   dayfirst=day_first, yearfirst=year_first, format=date_format)
            if as_num:
                df[c] = mdates.date2num(df[c])
        if not inplace:
            return df
        return

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

    def to_date_from_excel_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=False,
                                dtype: [str, list]=None, exclude: bool=False, regex: [str, list]=None,
                                re_ignore_case: bool=False, inplace: bool=False, save_intent: bool=None,
                                intent_level: [int, str]=None) -> [dict, pd.DataFrame, None]:
        """converts excel date formats into datetime

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param inplace: if the passed pandas.DataFrame should be used or a deep copy
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level of the intent,
                        If None: default's 0 unless the global intent_next_available is true then -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy pandas.DataFrame.
       """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, save_intent=save_intent)
        # Code block for intent
        if not inplace:
            with threading.Lock():
                df = deepcopy(df)
        if dtype is None:
            dtype = ['float64']
        obj_cols = self.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                       re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = [self._excel_date_converter(d) for d in df[c]]
        if not inplace:
            return df
        return
