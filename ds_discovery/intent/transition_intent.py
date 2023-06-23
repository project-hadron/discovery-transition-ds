import ast
import datetime
import inspect
from builtins import staticmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from aistac.intent.abstract_intent import AbstractIntentModel
from ds_discovery.managers.transition_property_manager import TransitionPropertyManager
from ds_discovery.components.commons import Commons

__author__ = 'Darryl Oatridge'


class TransitionIntentModel(AbstractIntentModel):
    """This component provides a set of actions that focuses on tidying raw data by removing data
    columns that are not useful to the final feature set, also known as data selection. These may include
    null columns, single value columns, duplicate columns and noise etc. We can also ensure the data is
    properly canonicalized through enforcing data typing."""

    def __init__(self, property_manager: TransitionPropertyManager, default_save_intent: bool=None,
                 default_intent_level: [str, int, float]=None, order_next_available: bool=None,
                 default_replace_intent: bool=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param default_intent_level: (optional) the default level intent should be saved at
        :param order_next_available: (optional) if the default behaviour for the order should be next available order
        :param default_replace_intent: (optional) the default replace existing intent behaviour
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = default_intent_level if isinstance(default_intent_level, (str, int, float)) else 'base'
        default_intent_order = -1 if isinstance(order_next_available, bool) and order_next_available else 0
        intent_param_exclude = ['df', 'inplace', 'canonical']
        intent_type_additions = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, canonical: pd.DataFrame, intent_levels: [int, str, list]=None, run_book: str=None,
                            **kwargs):
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        It is expected that all intent methods have the 'canonical' as the first parameter of the method signature
        and will contain'save_intent' as parameters.

        :param canonical: this is the iterative value all intent are applied to and returned.
        :param intent_levels: (optional) an single or list of levels to run, if list, run in order given
        :param run_book: (optional) a preset runbook of intent_level to run in order
        :param kwargs: additional kwargs to add to the parameterised intent, these will replace any that already exist
        :return: Canonical with parameterised intent applied 
        """
        # test if there is any intent to run
        if self._pm.has_intent():
            # get the list of levels to run
            if isinstance(intent_levels, (int, str, list)):
                intent_levels = Commons.list_formatter(intent_levels)
            elif isinstance(run_book, str) and self._pm.has_run_book(book_name=run_book):
                intent_levels = self._pm.get_run_book(book_name=run_book)
            else:
                intent_levels = sorted(self._pm.get_intent().keys())
            for level in intent_levels:
                level_key = self._pm.join(self._pm.KEY.intent_key, level)
                for order in sorted(self._pm.get(level_key, {})):
                    for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                        if method in self.__dir__():
                            # fail safe in case kwargs was stored as the reference
                            params.update(params.pop('kwargs', {}))
                            # add method kwargs to the params
                            if isinstance(kwargs, dict):
                                params.update(kwargs)
                            # remove the creator param
                            _ = params.pop('intent_creator', 'Unknown')
                            # add excluded params and set to False
                            params.update({'save_intent': False})
                            canonical = eval(f"self.{method}(canonical, **{params})", globals(), locals())
        return canonical

    def auto_projection(self, df, headers: list = None, drop: bool = None, n_components: [int, float] = None,
                        seed: int = None, save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None, **kwargs) -> pd.DataFrame:
        """Principal component analysis (PCA) is a linear dimensionality reduction using Singular Value Decomposition
        of the data to project it to a lower dimensional space.

        :param df: a pd.DataFrame as the reference dataframe
        :param headers: (optional) a list of headers to select (default) or drop from the dataset
        :param drop: (optional) if True then srop the headers. False by default
        :param n_components: (optional) Number of components to keep.
        :param seed: (optional) placeholder
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :param kwargs: additional parameters to pass the PCA model
        :return: a pd.DataFrame
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        sample = Commons.filter_columns(df, headers=headers, drop=drop, dtype='number')
        n_components = n_components if isinstance(n_components, (int, float)) \
                                       and 0 < n_components < sample.shape[1]  else sample.shape[1]
        # data_prep
        train_size = int(sample.shape[0] * 0.7)
        train, test = sample.iloc[:train_size], sample.iloc[train_size:]
        # standardise
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        # pca
        pca = PCA(n_components=n_components, **kwargs)
        train = pca.fit_transform(train)
        gen = Commons.label_gen(prefix='pca_')
        names = []
        for n in range(train.shape[1]):
            names.append(next(gen))
        train = pd.DataFrame(data=train, columns=names)
        canonical = df.drop(sample.columns, axis=1)
        return pd.concat([canonical, train], axis=1)

    def auto_remove_null_rows(self, df, nulls_list: list=None, save_intent: bool=None,
                              intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                              remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
        """ automatically removes rows where the full row is null

        :param df: the pandas DataFrame to remove null rows from
        :param nulls_list: (optional) potential null values to consider other than just np.nan
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        nulls_list = nulls_list if isinstance(nulls_list, list) else ['', ' ', 'nan']
        
        for c in df.columns:
            for item in nulls_list:
                df[c] = df[c].replace(item, np.nan)
        return df.dropna(axis='index', how='all', inplace=False)

    def auto_reinstate_nulls(self, df, nulls_list=None, headers: [str, list]=None, drop: bool=None,
                             dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                             re_ignore_case: bool=None, save_intent: bool=None,
                             intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                             remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
        """ automatically reinstates nulls that have been masked with alternate values such as space or question-mark.
        By default, the nulls list is ['',' ','NaN','nan','None','null','Null','NULL']

        :param df: the pandas DataFrame to remove null rows from
        :param nulls_list: (optional) potential null values to replace with a null.
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        nulls_list = nulls_list if isinstance(nulls_list, list) else ['',' ','NaN','nan','None','null','Null','NULL']
        
        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = df[c].where(~df[c].isin(nulls_list))
        return df

    def auto_clean_header(self, df, case=None, rename_map: [dict, str, list]=None, replace_spaces: str=None,
                          save_intent: bool=None, intent_level: [int, str]=None,
                          intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ clean the headers of a pandas DataFrame replacing space with underscore. If the rename_map is passed as a
        name of a connector contract

        :param df: the pandas.DataFrame to drop duplicates from
        :param rename_map: a dict of name value pairs or a correct length list of column names
        :param case: changes the headers to lower, upper, title, snake. if none of these then no change
        :param replace_spaces: character to replace spaces with. Default is '_' (underscore)
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        # removes any hidden characters
        df.columns = df.columns.astype(str)
        # auto mapping
        if isinstance(rename_map, str):
            if self._pm.has_connector(rename_map):
                handler = self._pm.get_connector_handler(rename_map)
                mapper = handler.load_canonical()
                if mapper.shape[1] == 1:
                    rename_map = mapper.iloc[:, 0].values.tolist()
                else:
                    rename_map = dict(zip(mapper.iloc[:, 0].values, mapper.iloc[:, 1].values))
            else:
                mapper = None
        if isinstance(rename_map, dict):
            df.rename(mapper=rename_map, axis='columns', inplace=True)
        elif isinstance(rename_map, list) and len(rename_map) == df.shape[1]:
            df.columns = rename_map
        # convert case
        if isinstance(case, str):
            if case.lower() == 'lower':
                df.rename(mapper=str.lower, axis='columns', inplace=True)
            elif case.lower() == 'upper':
                df.rename(mapper=str.upper, axis='columns', inplace=True)
            elif case.lower() == 'title':
                df.rename(mapper=str.title, axis='columns', inplace=True)
        # replaces spaces at the end just in case title is used
        replace_spaces = '_' if not isinstance(replace_spaces, str) else replace_spaces
        df.columns = df.columns.str.replace(' ', replace_spaces, regex=False)
        return df

    def auto_transition(self, df, unique_max: int=None, null_max: float=None, inc_category: bool=None,
                        save_intent: bool=None, intent_level: [int, str]=None,
                        intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ automatically tries to convert a passes DataFrame to appropriate types

        :param df: the pandas DataFrame to remove null rows from
        :param unique_max: the max number of unique values in the column. default to 20
        :param null_max: maximum number of null in the column between 0 and 1. default to 0.7 (70% nulls allowed)
        :param inc_category: create categorical data types
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        if not isinstance(unique_max, int):
            unique_max = np.log2(df.shape[0]) ** 2 if df.shape[0] > 50000 else np.sqrt(df.shape[0])
        null_max = 0.9 if not isinstance(null_max, (int, float)) else null_max
        inc_category = inc_category if isinstance(inc_category, bool) else False
        
        _null_headers = []
        _date_headers = []
        _bool_headers = []
        _cat_headers = []
        _num_headers = []
        _str_headers = []
        for c in Commons.filter_headers(df, dtype=object):
            try:
                if df[c].isnull().all():
                    _null_headers.append(c)
                elif any(Commons.valid_date(x) for x in df[c].dropna()) or df[c].dropna().dtype.kind in 'nM':
                    _date_headers.append(c)
                elif df[c].dropna().isin([1, 0,'1','0']).all() or df[c].dropna().dtype.kind in 'b':
                    _bool_headers.append(c)
                elif df[c].nunique() < unique_max and round(df[c].isnull().sum() / df.shape[0], 3) < null_max:
                    _cat_headers.append(c)
                elif df[c].dropna().dtype.kind in 'iufc':
                    _num_headers.append(c)
                elif all(isinstance(v, str) for v in df[c].dropna()):
                    _str_headers.append(c)
            except TypeError:
                pass
        if len(_bool_headers) > 0:
            bool_map = {1: True}
            df = self.to_bool_type(df, headers=_bool_headers, bool_map=bool_map, save_intent=False)
        if len(_date_headers) > 0:
            df = self.to_date_type(df, headers=_date_headers, save_intent=False)
        if len(_cat_headers) > 0 and inc_category:
            df = self.to_category_type(df, headers=_cat_headers, save_intent=False)
        if len(_num_headers) > 0:
            df = self.to_numeric_type(df, headers=_num_headers, save_intent=False)
        if len(_str_headers) > 0:
            df = self.to_str_type(df, headers=_str_headers, use_string_type=True, save_intent=False)
        return df

    def auto_to_date(self, df: pd.DataFrame, iso_format: bool=None, timezone: str=None, day_first: bool=None,
                     year_first: bool=None, date_format: str=None, save_intent: bool=None,
                     intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                     remove_duplicates: bool=None):
        """ looks through the dataset for valid date formats and converts them to a common datetime.

        :param df: the Pandas.DataFrame to get the column headers from
        :param iso_format: rather than return a Timestamp, return an ISO formatted string
        :param timezone: set the timezone else data set to native
        :param year_first: specifies if to parse with the year first
                If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
                If both dayfirst and yearfirst are True, yearfirst is preceded (same as dateutil).
        :param day_first: specifies if to parse with the day first
                If True, parses dates with the day first, eg %d-%m-%Y.
                If False default to the a preferred preference, normally %m-%d-%Y (but not strict)
        :param date_format: if the date can't be inferred uses date format eg format='%Y%m%d'
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        _date_headers = []
        for c in Commons.filter_headers(df, dtype=object):
            try:
                if df[c].isnull().all():
                    continue
                if all(Commons.valid_date(x) for x in df[c].dropna()):
                    _date_headers.append(c)
            except TypeError:
                pass
        if len(_date_headers) > 0:
            df = self.to_date_type(df, headers=_date_headers, iso_format=iso_format, timezone=timezone,
                                   day_first=day_first, year_first=year_first, date_format=date_format,
                                   save_intent=False)
        return df

    def auto_to_category(self, df: pd.DataFrame, unique_max: int=None, null_max: float=None, fill_nulls: str=None,
                         nulls_list: list=None, save_intent: bool=None,
                         intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                         remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
        """ auto categorises columns that have a max number of uniqueness with a min number of nulls
        and are object dtype

        :param df: the pandas.DataFrame to auto categorise
        :param unique_max: the max number of unique values in the column. default to 20
        :param null_max: maximum number of null in the column between 0 and 1. default to 0.7 (70% nulls allowed)
        :param fill_nulls: a value to fill nulls that then can be identified as a category type
        :param nulls_list:  potential null values to replace.
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        unique_max = 100 if not isinstance(unique_max, int) else unique_max
        null_max = 0.8 if not isinstance(null_max, (int, float)) else null_max
        df_len = len(df)
        col_cat = []
        for c in Commons.filter_headers(df, dtype=['object', 'number']):
            if df[c].nunique() < unique_max and round(df[c].isnull().sum() / df_len, 2) < null_max:
                col_cat.append(c)
        result = self.to_category_type(df, headers=col_cat, fill_nulls=fill_nulls, nulls_list=nulls_list,
                                        save_intent=False)
        return result

    def auto_drop_columns(self, df, null_min: float=None, predominant_max: float=None, nulls_list: [bool, list]=None,
                          drop_predominant: bool=None, drop_empty_row: bool=None, drop_unknown: bool=None,
                          save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                          replace_intent: bool=None, remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
        """ auto removes columns that are at least 0.998 percent np.NaN, a single value, std equal zero or have a
        predominant value greater than the default 0.998 percent.

        :param df: the pandas.DataFrame to auto remove
        :param null_min: the minimum number of null values default to 0.998 (99.8%) nulls
        :param predominant_max: the percentage max a single field predominates default is 0.998 (99.8%) unique value
        :param nulls_list: can be boolean or a list:
                    if boolean and True then null_list equals ['NaN', 'nan', 'null', '', 'None', ' ']
                    if list then this is considered potential null values.
        :param drop_predominant: drop columns that have a predominant value of the given predominant max
        :param drop_empty_row: also drop any rows where all the values are empty
        :param drop_unknown:  (optional) drop objects that are not string types such as binary
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        null_min = 0.998 if not isinstance(null_min, (int, float)) else null_min
        predominant_max = 0.998 if not isinstance(predominant_max, (int, float)) else predominant_max
        drop_predominant = drop_predominant if isinstance(drop_predominant, bool) else True
        drop_unknown = drop_unknown if isinstance(drop_unknown, bool) else False
        if isinstance(nulls_list, bool) and nulls_list:
            nulls_list = ['NaN', 'nan', 'null', '', 'None', ' ']
        elif not isinstance(nulls_list, list):
            nulls_list = None
        df_filter = df.copy()
        df_len = len(df_filter)
        col_drop = []
        for c in df_filter.columns:
            if nulls_list is not None:
                df_filter[c].replace(nulls_list, np.nan, inplace=True)
            if round(df_filter[c].isnull().sum() / df_len, 5) > null_min:
                col_drop.append(c)
            elif drop_predominant and df_filter[c].nunique() == 1:
                col_drop.append(c)
            elif drop_predominant and round((df_filter[c].value_counts() / np.float64(len(df_filter[c].dropna()))).
                                            sort_values(ascending=False).values[0], 5) >= predominant_max:
                col_drop.append(c)
            elif drop_predominant and isinstance(df_filter[c], (int, float)) and df_filter[c].dropna().std() == 0:
                col_drop.append(c)
            elif drop_unknown and not df[c].dropna().dtype.kind in 'iufcbnM':
                if not all(isinstance(v, str) for v in df_filter[c].dropna()):
                    col_drop.append(c)
        result = self.to_remove(df, headers=col_drop, save_intent=False)
        if isinstance(drop_empty_row, bool) and drop_empty_row:
            result = result.dropna(how='all')
        return result

    # drops highly correlated columns
    def auto_drop_correlated(self, df: pd.DataFrame, threshold: float=None, save_intent: bool=None,
                             intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                             remove_duplicates: bool=None) -> [dict, pd.DataFrame]:
        """ uses 'brute force' techniques to removes highly correlated numeric columns based on the threshold,
        set by default to 0.998.

        :param df: data: the Canonical data to drop duplicates from
        :param threshold: (optional) threshold correlation between columns. default 0.998
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: Canonical,.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        threshold = threshold if isinstance(threshold, float) and 0 < threshold < 1 else 0.998
        df_filter = Commons.filter_columns(df, dtype=[float, int], exclude=False)
        col_corr = set()
        corr_matrix = df_filter.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        df.drop(labels=col_corr, axis=1, inplace=True)
        return df

    # drops duplicate columns
    def auto_drop_duplicates(self, df: pd.DataFrame, save_intent: bool=None,
                             intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                             remove_duplicates: bool=None) -> [dict, pd.DataFrame]:
        """ Removes columns that are duplicates of each other

        :param df: data: the Canonical data to drop duplicates from
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: Canonical,.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        df_filter = Commons.filter_columns(df, dtype=['number'], exclude=False)
        duplicated_feat = []
        for i in range(0, len(df_filter.columns)):
            col_1 = df_filter.columns[i]
            for col_2 in df_filter.columns[i + 1:]:
                if df_filter[col_1].equals(df_filter[col_2]):
                    duplicated_feat.append(col_2)
        df.drop(labels=duplicated_feat, axis=1, inplace=True)
        return df

    # drop unwanted
    def to_remove(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                  exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None,
                  save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                  replace_intent: bool=None, remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
        """ remove columns from the pandas.DataFrame

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case)
        df.drop(obj_cols, axis=1, inplace=True)
        return df

    # select wanted
    def to_select(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                  exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None,
                  save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                  replace_intent: bool=None, remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
        """ selects columns from the pandas.DataFrame

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case)
        return self.to_remove(df, headers=obj_cols, drop=True, save_intent=False)


    def to_sample(self, df, sample_size: [int, float], shuffle: bool=None, seed: int=None,
                  save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                  replace_intent: bool=None, remove_duplicates: bool=None):
        """ allows a certain sample size to be selected from the dataframe.

        :param df: the pandas.DataFrame to drop duplicates from
        :param sample_size: If float, should be between 0.0 and 1.0 and represent the proportion of the
                            data set to return as a sample. If int, represents the absolute number of samples.
        :param shuffle: (optional) if the canonical should be shuffled
        :param seed: (optional) if shuffle is not None a seed value for the sample_size
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        shuffle = shuffle if isinstance(shuffle, bool) else False
        if isinstance(sample_size, float):
            if not 0 < sample_size < 1:
                raise ValueError(f"if passing a test_size as a float the number must be tween 0 and 1")
            if shuffle:
                df = df.sample(frac=sample_size, random_state=seed).reset_index(drop=True)
            else:
                df = df.iloc[:int(df.shape[0] * sample_size)]
        elif isinstance(sample_size, int):
            if sample_size > df.shape[0]:
                raise ValueError(f"The sample size '{sample_size}' can't be greater than the canonical "
                                 f"number the rows '{df.shape[0]}'")
            if shuffle:
                df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
            else:
                df = df.iloc[:sample_size]
        else:
            raise ValueError(f"sample_size must be an int less than the number of rows or a float between 0 and 1")
        return df

    # convert boolean
    def to_bool_type(self, df: pd.DataFrame, bool_map: dict=None, headers: [str, list]=None, drop: bool=None,
                     dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                     re_ignore_case: bool=None, save_intent: bool=None,
                     intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                     remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
        """ converts column to bool based on the map

        :param df: the Pandas.DataFrame to get the column headers from
        :param bool_map: a mapping of what to make True and False
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        bool_map = bool_map if isinstance(bool_map, dict) else {1: True, '1': True, 'True': True, 'T': True}
        if not bool_map:  # map is empty so nothing to map
            return df
        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case)
        for c in obj_cols:
            if df[c].dtype.name != 'bool':
                df[c] = df[c].map(bool_map)
                df[c] = df[c].fillna(False)
                df[c] = df[c].astype('bool')
        return df

    # convert objects to list
    def to_list_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                     exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None,
                     fill_nulls: str=None, nulls_list: list=None, save_intent: bool=None, intent_level: [int, str]=None,
                     intent_order: int=None, replace_intent: bool=None,
                     remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
        """ converts a string representation of a list into a list type

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param fill_nulls: a value to fill nulls, default to an empty list
        :param nulls_list:  potential null values to replace.
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        nulls_list = nulls_list if isinstance(nulls_list, list) else ['', ' ', 'nan']
        fill_nulls = fill_nulls if isinstance(fill_nulls, str) else '[]'

        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case)
        for c in obj_cols:
            for item in nulls_list:
                df[c] = df[c].replace(item, fill_nulls)
            df[c].loc[df[c].isna()] = '[]'

            df[c] = [ast.literal_eval(x) if isinstance(x, str) and
                                            x.startswith('[') and x.endswith(']') else x for x in df[c]]
            # replace all other items with list
            df[c] = [x if isinstance(x, list) else [x] for x in df[c]]
            df[c] = df[c].astype('object')
        return df

    # convert objects to categories
    def to_category_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                         exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None,
                         as_num: bool=None, fill_nulls: str=None, nulls_list: list=None,
                         save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                         replace_intent: bool=None, remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
        """ converts columns to categories

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param as_num: if true returns the category as a category code
        :param fill_nulls: a value to fill nulls that then can be identified as a category type
        :param nulls_list:  potential null values to replace.
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        nulls_list = nulls_list if isinstance(nulls_list, list) else ['', ' ', 'nan']
        fill_nulls = fill_nulls if isinstance(fill_nulls, str) else np.nan

        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case)
        for c in obj_cols:
            for item in nulls_list:
                df[c] = df[c].replace(item, fill_nulls)
            if not all(df[c].astype(str).str.isnumeric()):
                df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].astype('category')
            if as_num:
                df[c] = df[c].cat.codes
        return df

    @staticmethod
    def _to_numeric(df: pd.DataFrame, numeric_type, fillna, errors=None, headers: [str, list]=None,
                    drop: bool=None, dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                    re_ignore_case: bool=None, precision=None) -> [dict, pd.DataFrame]:
        """ Code reuse method for all the numeric types. see calling methods for inline docs"""
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        if errors is None or str(errors) not in ['ignore', 'raise', 'coerce']:
            errors = 'coerce'
        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case)
        for c in obj_cols:
            if not isinstance(precision, int):
                try:
                    precision = df[c].dropna().apply(str).str.extract('\.(.*)')[0].map(len).max()
                except (ValueError, TypeError, IndexError, KeyError, ReferenceError, NameError, RecursionError):
                    precision = 15
            df[c] = pd.to_numeric(df[c].apply(str).str.replace('[$£€, ]', '', regex=True), errors=errors)
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

            if str(numeric_type).lower().startswith('int') or precision == 0:
                df[c] = pd.to_numeric(df[c], errors='raise').as_type('int')
            elif str(numeric_type).lower().startswith('float'):
                df[c] = pd.to_numeric(df[c], downcast='float', errors='coerce').round(precision)
        return df

    def to_numeric_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                        exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None, precision=None,
                        fillna=None, errors=None,
                        save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
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

        :param errors: {'ignore', 'raise', 'coerce'}, default 'coerce'
                    - If 'raise', then invalid parsing will raise an exception
                    - If 'coerce', then invalid parsing will be set as NaN
                    - If 'ignore', then invalid parsing will return the input

        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        if not isinstance(fillna, (int, float, str)):
            fillna = np.nan
        df = self._to_numeric(df, numeric_type='numeric', fillna=fillna, errors=errors, headers=headers, drop=drop,
                              dtype=dtype, exclude=exclude, regex=regex, re_ignore_case=re_ignore_case,
                              precision=precision)
        return df

    def to_int_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                    exclude: bool=None,  regex: [str, list]=None, re_ignore_case: bool=None, fillna=None,
                    errors=None, save_intent: bool=None,
                    intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                    remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
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

        :param errors: {'ignore', 'raise', 'coerce'}, default 'coerce'
                    - If 'raise', then invalid parsing will raise an exception
                    - If 'coerce', then invalid parsing will be set as NaN
                    - If 'ignore', then invalid parsing will return the input

        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        if fillna is None or not fillna:
            fillna = 0
        df = self._to_numeric(df, numeric_type='int', fillna=fillna, errors=errors, headers=headers, drop=drop,
                              dtype=dtype, exclude=exclude, regex=regex, re_ignore_case=re_ignore_case)
        return df

    def to_float_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                      exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None, precision=None,
                      fillna=None, errors=None, save_intent: bool=None,
                      intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                      remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
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

        :param errors: {'ignore', 'raise', 'coerce'}, default 'coerce' }. Default to 'coerce'
                    - If 'raise', then invalid parsing will raise an exception
                    - If 'coerce', then invalid parsing will be set as NaN
                    - If 'ignore', then invalid parsing will return the input

        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        precision = precision if isinstance(precision, int) else 5
        if fillna is None or not fillna:
            fillna = np.nan

        columns = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                         re_ignore_case=re_ignore_case)
        for c in columns:
            if df[c].astype(str).str.isnumeric().all():
                df[c] = pd.to_numeric(df[c], errors='coerce')
            if str(df[c].dtype.name).startswith('int'):
                df[c] = df[c].astype(float)
            if str(df[c].dtype.name).startswith('float'):
                df[c] = df[c].round(precision)
                df[c] = df[c].replace(np.nan, fillna)
        return df

    def to_str_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                    exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None, fixed_len_pad: str=None,
                    use_string_type: bool=None, fill_nulls: str=None, nulls_list: list=None,
                    save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                    replace_intent: bool=None, remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
        """ converts columns to str type

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param fixed_len_pad: a padding character that when passed pads all values to the length of the longest
        :param use_string_type: if the dtype 'string' should be used or keep as object type
        :param fill_nulls: a value to fill nulls that then can be identified as a category type
        :param nulls_list:  potential null values to replace.
        :param nulls_list: can be boolean or a list:
                    if boolean and True then null_list equals ['NaN', 'nan', 'null', '', 'None'. np.nan, None]
                    if list then this is considered potential null values.
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
       """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        nulls_list = nulls_list if isinstance(nulls_list, list) else ['', ' ', 'nan']
        fill_nulls = fill_nulls if isinstance(fill_nulls, str) else np.nan

        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = df[c].apply(str)
            df[c] = df[c].str.strip()
            for item in nulls_list:
                df[c] = df[c].replace(item, fill_nulls)
            if isinstance(use_string_type, bool) and use_string_type:
                df[c] = df[c].astype('string', errors='ignore')
            if isinstance(fixed_len_pad, str) and len(fixed_len_pad) > 0:
                idx = df[c].dropna().index
                max_len = len(max(df[c].iloc[idx], key=len))
                df[c].iloc[idx] = [x.rjust(max_len, fixed_len_pad[0]) for x in df[c].iloc[idx]]
        return df

    def to_date_element(self, df: pd.DataFrame, matrix: [str, list], headers: [str, list]=None, drop: bool=None,
                        dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None,
                        day_first: bool=None, year_first: bool=None, date_format: str=None,
                        save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None) -> [dict, pd.DataFrame]:
        """ breaks a date down into value representations of the various parts that date.

        :param df: the Pandas.DataFrame to get the column headers from
        :param matrix: the matrix options (see below)
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param year_first: specifies if to parse with the year first
                If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
                If both dayfirst and yearfirst are True, yearfirst is preceded (same as dateutil).
        :param day_first: specifies if to parse with the day first
                If True, parses dates with the day first, eg %d-%m-%Y.
                If False default to the a prefered preference, normally %m-%d-%Y (but not strict)
        :param date_format: if the date can't be inferred uses date format eg format='%Y%m%d'
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.

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
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        day_first = day_first if isinstance(day_first, bool) else False
        year_first = year_first if isinstance(year_first, bool) else False
        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = df[c].fillna(np.nan)
            df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=day_first, yearfirst=year_first, format=date_format)
            matrix = Commons.list_formatter(matrix)
            if 'yr' in matrix:
                df[f"{c}_yr"] = df[c].dt.year
            if 'dec' in matrix:
                df[f"{c}_dec"] = (df[c].dt.year - df[c].dt.year % 10).astype('category')
            if 'mon' in matrix:
                df[f"{c}_mon"] = df[c].dt.month
            if 'day' in matrix:
                df[f"{c}_day"] = df[c].dt.day
            if 'dow' in matrix:
                df[f"{c}_dow"] = df[c].dt.dayofweek
            if 'hr' in matrix:
                df[f"{c}_hr"] = df[c].dt.hour
            if 'min' in matrix:
                df[f"{c}_min"] = df[c].dt.minute
            if 'woy' in matrix:
                df[f"{c}_woy"] = df[c].dt.isocalendar().week
            if 'doy' in matrix:
                df[f"{c}_doy"] = df[c].dt.dayofyear
        return df

    def to_date_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                     exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None, iso_format: bool=None,
                     timezone: str=None, day_first: bool=None, year_first: bool=None, date_format: str=None,
                     save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                     replace_intent: bool=None, remove_duplicates: bool=None) -> [dict, pd.DataFrame]:
        """ converts columns to date types

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param iso_format: rather than return a Timestamp, return an ISO formatted string
        :param timezone: set the timezone else data set to native
        :param year_first: specifies if to parse with the year first
                If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
                If both dayfirst and yearfirst are True, yearfirst is preceded (same as dateutil).
        :param day_first: specifies if to parse with the day first
                If True, parses dates with the day first, eg %d-%m-%Y.
                If False default to the a prefered preference, normally %m-%d-%Y (but not strict)
        :param date_format: if the date can't be inferred uses date format eg format='%Y%m%d'
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        iso_format = iso_format if isinstance(iso_format, bool) else False
        
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        timezone = timezone if isinstance(timezone, str) else None
        day_first = day_first if isinstance(day_first, bool) else False
        year_first = year_first if isinstance(year_first, bool) else False
        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = df[c].fillna(np.nan)
            df[c] = df[c].astype('object')
            df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=day_first, yearfirst=year_first, format=date_format)
            if df[c].dt.tz:
                df[c] = df[c].dt.tz_convert(timezone)
            else:
                df[c] =df[c].dt.tz_localize(timezone)
            if iso_format:
                df[c] = [x.isoformat() for x in df[c]]
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

    def to_date_from_excel_type(self, df: pd.DataFrame, headers: [str, list]=None, drop: bool=None,
                                dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                                re_ignore_case: bool=None, save_intent: bool=None,
                                intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                                remove_duplicates: bool=None) -> [dict, pd.DataFrame, None]:
        """converts excel date formats into datetime

        :param df: the Pandas.DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pandas.DataFrame.
       """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        if dtype is None:
            dtype = ['float64']
        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude, regex=regex,
                                          re_ignore_case=re_ignore_case)
        for c in obj_cols:
            df[c] = [self._excel_date_converter(d) for d in df[c]]
        return df
