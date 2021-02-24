import inspect
from typing import Any

import numpy as np
import pandas as pd
from aistac.handlers.abstract_handlers import HandlerFactory

from ds_discovery.intent.abstract_common_intent import AbstractCommonsIntentModel
from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager
from ds_discovery.components.commons import Commons
from aistac.components.aistac_commons import DataAnalytics
from ds_discovery.components.discovery import DataDiscovery

__author__ = 'Darryl Oatridge'


class FeatureCatalogIntentModel(AbstractCommonsIntentModel):
    """A set of methods to help build features as pandas.Dataframe"""

    def __init__(self, property_manager: FeatureCatalogPropertyManager, default_save_intent: bool=None,
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
        intent_param_exclude = []
        intent_type_additions = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64,
                                 pd.Timestamp]
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, canonical: Any, feature_name: [int, str], train_size: [float, int]=None,
                            seed: int=None, shuffle: bool=None, **kwargs) -> [pd.DataFrame, pd.Series]:
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        It is expected that all intent methods have the 'canonical' as the first parameter of the method signature
        and will contain 'save_intent' as parameters. It is also assumed that all features have a feature contract to
        save the feature outcome to

        :param canonical: this is the iterative value all intent are applied to and returned.
        :param feature_name: feature to run
        :param train_size: (optional) If float, should be between 0.0 and 1.0 and represent the proportion of the
                            dataset to include in the train split. If int, represents the absolute number of train
                            samples. If None, then not used
        :param seed: (optional) if shuffle is True a seed value for the choice
        :param shuffle: (optional) Whether or not to shuffle the data before splitting or just split on train size.
        :param kwargs: additional kwargs to add to the parameterised intent, these will replace any that already exist
        :return
        """
        # test if there is any intent to run
        if self._pm.has_intent(level=feature_name):
            canonical = self._get_canonical(canonical)
            if isinstance(train_size, (float, int)):
                canonical = self.canonical_sampler(canonical, sample_size=train_size, shuffle=shuffle, seed=seed)
            # run the feature
            level_key = self._pm.join(self._pm.KEY.intent_key, feature_name)
            df_feature = None
            for order in sorted(self._pm.get(level_key, {})):
                for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                    if method in self.__dir__():
                        if 'canonical' in params.keys():
                            df_feature = params.pop('canonical')
                        elif df_feature is None:
                            df_feature = canonical
                        # fail safe in case kwargs was sored as the reference
                        params.update(params.pop('kwargs', {}))
                        # add method kwargs to the params
                        if isinstance(kwargs, dict):
                            params.update(kwargs)
                        # remove the creator param
                        _ = params.pop('intent_creator', 'Unknown')
                        # add excluded params and set to False
                        params.update({'save_intent': False})
                        df_feature = eval(f"self.{method}(df_feature, **{params})", globals(), locals())
            if df_feature is None:
                raise ValueError(f"The feature '{feature_name}' pipeline did not run. ")
            return df_feature
        raise ValueError(f"The feature '{feature_name}, can't be found in the feature catalog")

    def apply_date_diff(self, canonical: Any, key: [str, list], first_date: str, second_date: str,
                        aggregator: str=None, units: str=None, precision: int=None, rtn_columns: list=None,
                        regex: bool=None, rename: str=None, unindex: bool=None, save_intent: bool=None,
                        feature_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                        remove_duplicates: bool=None) -> pd.DataFrame:
        """ adds a column for the difference between a primary and secondary date where the primary is an early date
        than the secondary.

        :param canonical: the DataFrame containing the column headers
        :param key: the key label to group by and index on
        :param first_date: the primary or older date field
        :param second_date: the secondary or newer date field
        :param aggregator: (optional) the aggregator as a function of Pandas DataFrame 'groupby'
        :param units: (optional) The Timedelta units e.g. 'D', 'W', 'M', 'Y'. default is 'D'
        :param precision: the precision of the result
        :param rtn_columns: (optional) return columns, the header must be listed to be included.
                    If None then header
                    if 'all' then all original headers
        :param regex: if True then treat the rtn_columns as a regular expression
        :param rename: a new name for the column, else primary and secondary name used
        :param unindex: (optional) if the passed canonical should be un-index before processing
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
        if second_date not in canonical.columns:
            raise ValueError(f"The column header '{second_date}' is not in the canonical DataFrame")
        if first_date not in canonical.columns:
            raise ValueError(f"The column header '{first_date}' is not in the canonical DataFrame")
        canonical = self._get_canonical(canonical)
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        rename = rename if isinstance(rename, str) else f'{second_date}-{first_date}'
        if rtn_columns == 'all':
            rtn_columns = Commons.filter_headers(canonical, headers=key + [rename], drop=True)
        if isinstance(regex, bool) and regex:
            rtn_columns = Commons.filter_headers(canonical, regex=rtn_columns)
        rtn_columns = Commons.list_formatter(rtn_columns) if isinstance(rtn_columns, list) else [rename]
        precision = precision if isinstance(precision, int) else 0
        units = units if isinstance(units, str) else 'D'
        selected = canonical[[first_date, second_date]].dropna(axis='index', how='any')
        canonical[rename] = (selected[second_date].sub(selected[first_date], axis=0) / np.timedelta64(1, units))
        canonical[rename] = [np.round(v, precision) for v in canonical[rename]]
        return Commons.filter_columns(canonical, headers=list(set(key + [rename] + rtn_columns))).set_index(key)

    def select_feature(self, canonical: Any, key: [str, list], headers: [str, list]=None,
                       drop: bool=None, dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                       re_ignore_case: bool=None, drop_dup_index: str=None, rename: dict=None, unindex: bool=None,
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
        canonical = self._get_canonical(canonical)
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        filter_headers = Commons.filter_headers(df=canonical, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                                regex=regex, re_ignore_case=re_ignore_case)
        filter_headers += self._pm.list_formatter(key)
        df_rtn = Commons.filter_columns(canonical, headers=filter_headers)
        if isinstance(drop_dup_index, str) and drop_dup_index.lower() in ['first', 'last']:
            df_rtn = df_rtn.loc[~df_rtn.index.duplicated(keep=drop_dup_index)]
        if isinstance(rename, dict):
            df_rtn.rename(columns=rename, inplace=True)
        return df_rtn.set_index(key)

    def apply_merge(self, canonical: Any, merge_connector: str, key: [str, list], how: str=None,
                    on: str=None, left_on: str=None, right_on: str=None, left_index: bool=None, right_index: bool=None,
                    sort: bool=None, suffixes: tuple=None, indicator: bool=None, validate: str=None,
                    rtn_columns: list=None, regex: bool=None, unindex: bool=None, save_intent: bool=None,
                    feature_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                    remove_duplicates: bool=None):
        """ merges the canonical with another canonical obtained from a connector contract

        :param canonical: the canonical to merge on the left
        :param merge_connector: the name of the Connector Contract to load to merge on the right
        :param key: the key column to index on
        :param how: (optional) One of 'left', 'right', 'outer', 'inner'. Defaults to inner. See below for more detailed
                    description of each method.
        :param on: (optional) Column or index level names to join on. Must be found in both the left and right
                    DataFrame and/or Series objects. If not passed and left_index and right_index are False,  the
                    intersection of the columns in the DataFrames and/or Series will be inferred to be the join keys
        :param left_on: (optional) Columns or index levels from the left DataFrame or Series to use as keys. Can either
                    be column names, index level names, or arrays with length equal to the length of the DataFrame
                    or Series.
        :param right_on: (optional) Columns or index levels from the right DataFrame or Series to use as keys. Can
                    either be column names, index level names, or arrays with length equal to the length of the
                    DataFrame or Series.
        :param left_index: (optional) If True, use the index (row labels) from the left DataFrame or Series as its join
                    key(s). In the case of a DataFrame or Series with a MultiIndex (hierarchical), the number of levels
                    must match the number of join keys from the right DataFrame or Series.
        :param right_index: (optional) Same usage as left_index for the right DataFrame or Series
        :param sort: (optional) Sort the result DataFrame by the join keys in lexicographical order. Defaults to True,
                    setting to False will improve performance substantially in many cases.
        :param suffixes: (optional) A tuple of string suffixes to apply to overlapping columns. Defaults ('', '_dup').
        :param indicator: (optional) Add a column to the output DataFrame called _merge with information on the source
                    of each row. _merge is Categorical-type and takes on a value of left_only for observations whose
                    merge key only appears in 'left' DataFrame or Series, right_only for observations whose merge key
                    only appears in 'right' DataFrame or Series, and both if the observation’s merge key is found
                    in both.
        :param validate: (optional) validate : string, default None. If specified, checks if merge is of specified type.
                            “one_to_one” or “1:1”: checks if merge keys are unique in both left and right datasets.
                            “one_to_many” or “1:m”: checks if merge keys are unique in left dataset.
                            “many_to_one” or “m:1”: checks if merge keys are unique in right dataset.
                            “many_to_many” or “m:m”: allowed, but does not result in checks.
        :param rtn_columns: (optional) return columns, the header must be listed to be included. If None then header
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt' columns
        :param unindex: (optional) if the passed canonical should be un-index before processing
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
        canonical = self._get_canonical(canonical)
        how = how if isinstance(how, str) and how in ['left', 'right', 'outer', 'inner'] else 'inner'
        left_index = left_index if isinstance(left_index, bool) else False
        right_index = right_index if isinstance(right_index, bool) else False
        sort = sort if isinstance(sort, bool) else True
        indicator = indicator if isinstance(indicator, bool) else False
        suffixes = suffixes if isinstance(suffixes, tuple) and len(suffixes) == 2 else ('', '_dup')
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        if not self._pm.has_connector(connector_name=merge_connector):
            raise ValueError(f"The connector name '{merge_connector}' is not in the connectors catalog")
        handler = self._pm.get_connector_handler(merge_connector)
        other = handler.load_canonical()
        if isinstance(other, dict):
            other = pd.DataFrame.from_dict(data=other)
        df = pd.merge(left=canonical, right=other, how=how, on=on, left_on=left_on, right_on=right_on,
                      left_index=left_index, right_index=right_index, sort=sort, suffixes=suffixes, indicator=indicator,
                      validate=validate)
        if isinstance(regex, bool) and regex:
            rtn_columns = Commons.filter_headers(df, regex=rtn_columns)
        rtn_columns = rtn_columns if isinstance(rtn_columns, list) else df.columns.to_list()
        return Commons.filter_columns(df, headers=list(set(key + rtn_columns))).set_index(key)

    def apply_map(self, canonical: Any, key: [str, list], header: str, value_map: dict,
                  default_to: Any=None, replace_na: bool=None, rtn_columns: list=None, regex: bool=None,
                  rename: str=None, unindex: bool=None, save_intent: bool=None, feature_name: [int, str]=None,
                  intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ Apply mapping and filtering based on a key value pair of find and replace values

        :param canonical: the value to apply the substitution to
        :param key: the key column to index on
        :param header: the column header name to apply the value map too
        :param value_map: a dictionary of keys and their replace value
        :param default_to: (optional) a default value if no map if found. If None then NaN
        :param replace_na: (optional) if existing NaN values should be replaced with default_value. if None then True
        :param rtn_columns: (optional) return columns, the header must be listed to be included.
                    If None then header
                    if 'all' then all original headers
        :param regex: if True then treat the rtn_columns as a regular expression
        :param rename: a new name for the column, else current column header
        :param unindex: (optional) if the passed canonical should be un-index before processing
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
        if header not in canonical.columns:
            raise ValueError(f"The column header '{header}' is not in the canonical DataFrame")
        canonical = self._get_canonical(canonical)
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        rename = rename if isinstance(rename, str) else header
        if rtn_columns == 'all':
            rtn_columns = Commons.filter_headers(canonical, headers=key, drop=True)
        if isinstance(regex, bool) and regex:
            rtn_columns = Commons.filter_headers(canonical, regex=rtn_columns)
        rtn_columns = Commons.list_formatter(rtn_columns) if isinstance(rtn_columns, list) else [rename]
        replace_na = replace_na if isinstance(replace_na, bool) else True
        if default_to is not None:
            value_map = Commons.dict_with_missing(value_map, default=default_to)
        na_action = 'ignore' if replace_na else None
        canonical[rename] = canonical[header].map(value_map, na_action=na_action)
        canonical.dropna(subset=[rename], inplace=True)
        return Commons.filter_columns(canonical, headers=list(set(key + rtn_columns))).set_index(key)

    def apply_numeric_typing(self, canonical: Any, key: [str, list], header: str, normalise: bool=None,
                             precision: int=None, fillna: [int, float]=None, errors: str=None, rtn_columns: list=None,
                             rtn_regex: bool=None, unindex: bool=None, rename: str=None, save_intent: bool=None,
                             feature_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                             remove_duplicates: bool=None) -> pd.DataFrame:
        """ converts columns to categories

        :param canonical: the Pandas.DataFrame to get the column headers from
        :param key: the key column to index on
        :param header: the header to apply typing to
        :param normalise: if the resulting column should be normalised
        :param precision: how many decimal places to set the return values.
                        if None then precision is based on the most decimal places of all data points
                        if 0 (zero) the int is assumed
        :param fillna: { num_value, 'mean', 'mode', 'median' }. Default to np.nan
                    - If num_value, then replaces NaN with this number value. Must be a value not a string
                    - If 'mean', then replaces NaN with the mean of the column
                    - If 'mode', then replaces NaN with a mode of the column. random sample if more than 1
                    - If 'median', then replaces NaN with the median of the column
        :param errors : {'ignore', 'raise', 'coerce'}, default 'coerce'
                    - If 'raise', then invalid parsing will raise an exception
                    - If 'coerce', then invalid parsing will be set as NaN
                    - If 'ignore', then invalid parsing will return the input
        :param rtn_columns: (optional) return columns, the header must be listed to be included.
                    If None then header
                    if 'all' then all original headers
        :param rtn_regex: if True then treat the rtn_columns as a regular expression
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
        canonical = self._get_canonical(canonical)
        if header not in canonical.columns:
            raise ValueError(f"The column header '{header}' is not in the canonical DataFrame")
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        rename = rename if isinstance(rename, str) else header
        if rtn_columns == 'all':
            rtn_columns = Commons.filter_headers(canonical, headers=key, drop=True)
        if isinstance(rtn_regex, bool) and rtn_regex:
            rtn_columns = Commons.filter_headers(canonical, regex=rtn_columns)
        rtn_columns = Commons.list_formatter(rtn_columns) if isinstance(rtn_columns, list) else [rename]
        if canonical[header].dtype.name.startswith('int') and not isinstance(precision, int):
            precision = 0
        if not isinstance(fillna, (int, float)) and isinstance(precision, int) and precision == 0:
            fillna = 0
        module = HandlerFactory.get_module(module_name='ds_discovery')
        canonical = module.Transition.scratch_pad().to_numeric_type(df=canonical, headers=header, precision=precision,
                                                                    fillna=fillna, errors=errors, inplace=False)
        if isinstance(normalise, bool) and normalise:
            s_column = canonical[rename]
            s_column /= np.linalg.norm(s_column)
            if isinstance(precision, int):
                s_column = np.round(s_column, precision)
            canonical[rename] = s_column
        return Commons.filter_columns(canonical, headers=list(set(key + rtn_columns))).set_index(key)

    def apply_category_typing(self, canonical: Any, key: [str, list], header: str, as_num: bool=None,
                              rtn_columns: list=None, rtn_regex: bool=None, unindex: bool=None, rename: str=None,
                              save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                              replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ converts columns to categories

        :param canonical: the Pandas.DataFrame to get the column headers from
        :param key: the key column to index on
        :param header: the header to apply typing to
        :param as_num: if true returns the category as a category code
        :param rtn_columns: (optional) return columns, the header must be listed to be included.
                    If None then header
                    if 'all' then all original headers
        :param rtn_regex: if True then treat the rtn_columns as a regular expression
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
        canonical = self._get_canonical(canonical)
        if header not in canonical.columns:
            raise ValueError(f"The column header '{header}' is not in the canonical DataFrame")
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        rename = rename if isinstance(rename, str) else header
        if rtn_columns == 'all':
            rtn_columns = Commons.filter_headers(canonical, headers=key, drop=True)
        if isinstance(rtn_regex, bool) and rtn_regex:
            rtn_columns = Commons.filter_headers(canonical, regex=rtn_columns)
        rtn_columns = Commons.list_formatter(rtn_columns) if isinstance(rtn_columns, list) else [rename]
        module = HandlerFactory.get_module(module_name='ds_discovery')
        canonical = module.Transition.scratch_pad().to_category_type(df=canonical, headers=header, as_num=as_num,
                                                                     inplace=False)
        return Commons.filter_columns(canonical, headers=list(set(key + rtn_columns))).set_index(key)

    def apply_replace(self, canonical: Any, key: [str, list], header: str, to_replace: dict,
                      regex: bool=None, rtn_columns: list=None, rtn_regex: bool=None, unindex: bool=None,
                      rename: str=None, save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                      replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ Apply replacement based on a key value pair of find and replace values. if you wish to replace null values
        or put in null values use the tag '$null' to represent None or np.nan

        :param canonical: the value to apply the substitution to
        :param key: the key column to index on
        :param header: the column header name to apply the value map too
        :param to_replace: a dictionary of keys and their replace value
        :param regex: if the to_replace is regular expression
        :param rtn_columns: (optional) return columns, the header must be listed to be included.
                    If None then header
                    if 'all' then all original headers
        :param rtn_regex: if True then treat the rtn_columns as a regular expression
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
        :return: the amended value
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # intend code block on the canonical
        canonical = self._get_canonical(canonical)
        if header not in canonical.columns:
            raise ValueError(f"The column header '{header}' is not in the canonical DataFrame")
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        rename = rename if isinstance(rename, str) else header
        if rtn_columns == 'all':
            rtn_columns = Commons.filter_headers(canonical, headers=key, drop=True)
        if isinstance(rtn_regex, bool) and rtn_regex:
            rtn_columns = Commons.filter_headers(canonical, regex=rtn_columns)
        rtn_columns = Commons.list_formatter(rtn_columns) if isinstance(rtn_columns, list) else [rename]
        # replace null tag with np.nan
        for _ref, _value in to_replace.copy().items():
            if _ref == '$null':
                to_replace.pop(_ref)
                to_replace[np.nan] = _value
            if _value == '$null':
                to_replace[_ref] = np.nan
        regex = regex if isinstance(regex, bool) else False
        canonical[rename] = canonical[header].replace(to_replace=to_replace, inplace=False, regex=regex)
        return Commons.filter_columns(canonical, headers=list(set(key + rtn_columns))).set_index(key)

    def apply_condition(self, canonical: Any, key: [str, list], header: str, conditions: [tuple, list],
                        default: [int, float, str]=None, inc_columns: list=None, rename: str=None, unindex: bool=None,
                        save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ applies a selections choice based on a set of conditions to a condition to a named column
        Example: conditions = tuple('< 5',  'red')
             or: conditions = [('< 5',  'green'), ('> 5 & < 10',  'red')]

        :param canonical: the Pandas.DataFrame to get the column headers from
        :param key: the key column to index on
        :param header: a list of headers to apply the condition on,
        :param unindex: if the passed canonical should be un-index before processing
        :param conditions: a tuple or list of tuple conditions
        :param default: (optional) a value if no condition is met. 0 if not set
        :param inc_columns: additional columns to include in the returning DataFrame
        :param rename: (optional) if the column should have an alternative name
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
        canonical = self._get_canonical(canonical)
        if header not in canonical.columns:
            raise ValueError(f"The column header '{header}' is not in the canonical DataFrame")
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        rename = rename if isinstance(rename, str) else header
        inc_columns = self._pm.list_formatter(inc_columns)
        if not inc_columns:
            inc_columns = Commons.filter_headers(canonical, headers=key, drop=True)
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
                    and_list.append(f"(canonical[header]{_and})")
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
            canonical[rename] = np.select(selection, choices, default=default)
        else:
            canonical[rename] = np.select(selection, choices)
        return Commons.filter_columns(canonical, headers=list(set(key + inc_columns))).set_index(key)

    def select_where(self, canonical: Any, key: [str, list], selection: list, inc_columns: list=None,
                     unindex: bool=None, save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                     replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ returns a selected result based upon a set of conditions.

        :param canonical: the Pandas.DataFrame to get the column headers from
        :param key: the key column to index on
        :param selection: a list of dictionaries of selection where conditions to filter on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
        :param inc_columns: additional columns to include in the returning DataFrame
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
        :return: apandas DataFrame of the resulting select

        Conditions are a list of dictionaries of conditions and optional additional parameters to filter.
        To help build conditions there is a static helper method called 'conditions2dict(...)' that has parameter
        options available to build a condition.
        An example of a condition with the minimum requirements is
                [{'column': 'genre', 'condition': "=='Comedy'"}]

        an example of using the helper method
                selection = [self.select2dict(column='gender', condition="=='M'"),
                             self.select2dict(column='age', condition=">65", logic='XOR')]

        Using the 'select2dict' method ensure the correct keys are used and the dictionary is properly formed
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        if not isinstance(selection, list) or not all(isinstance(x, dict) for x in selection):
            raise ValueError("The 'selection' parameter must be a 'list' of 'dict' types")
        for _where in selection:
            if 'column' not in _where or 'condition' not in _where:
                raise ValueError("all 'dict' in the 'selection' list must have a 'column' and 'condition' key "
                                 "as a minimum")
        canonical = self._get_canonical(canonical)
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        inc_columns = self._pm.list_formatter(inc_columns)
        if not inc_columns:
            inc_columns = Commons.filter_headers(canonical, headers=key, drop=True)
        select_idx = None
        for _where in selection:
            select_idx = self._condition_index(canonical=canonical, condition=_where, select_idx=select_idx)
        canonical = canonical.iloc[select_idx]
        return Commons.filter_columns(canonical, headers=list(set(key + inc_columns))).set_index(key)

    def remove_outliers(self, canonical: Any, key: [str, list], column: str, lower_quantile: float=None,
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
        canonical = self._get_canonical(canonical)
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        df_rtn = Commons.filter_columns(canonical, headers=key + [column])
        lower_quantile = lower_quantile if isinstance(lower_quantile, float) and 0 < lower_quantile < 1 else 0.00001
        upper_quantile = upper_quantile if isinstance(upper_quantile, float) and 0 < upper_quantile < 1 else 0.99999

        result = DataDiscovery.analyse_number(df_rtn[column], granularity=[lower_quantile, upper_quantile],
                                              detail_stats=False)
        analysis = DataAnalytics(result)
        df_rtn = df_rtn[(df_rtn[column] > analysis.intent.intervals[0][1]) & (df_rtn[column] <
                                                                              analysis.intent.intervals[2][0])]
        return df_rtn.set_index(key)

    def group_features(self, canonical: Any, headers: [str, list], group_by: [str, list],
                       aggregator: str=None, drop_group_by: bool=False, include_weighting: bool=False,
                       freq_precision: int=None, remove_weighting_zeros: bool=False, remove_aggregated: bool=False,
                       drop_dup_index: str=None, unindex: bool=None, save_intent: bool=None,
                       feature_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                       remove_duplicates: bool=None) -> pd.DataFrame:
        """ groups features according to the aggregator passed. The list of aggregators are mean, sum, size, count,
        nunique, first, last, min, max, std var, describe.

        :param canonical: the pd.DataFrame to group
        :param headers: the column headers to apply the aggregation too
        :param group_by: the column headers to group by
        :param aggregator: (optional) the aggregator as a function of Pandas DataFrame 'groupby'
        :param drop_group_by: drops the group by headers
        :param include_weighting: include a percentage weighting column for each
        :param freq_precision: a precision for the weighting values
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
        canonical = self._get_canonical(canonical)
        freq_precision = freq_precision if isinstance(freq_precision, int) else 3
        aggregator = aggregator if isinstance(aggregator, str) else 'sum'
        headers = self._pm.list_formatter(headers)
        group_by = self._pm.list_formatter(group_by)
        df_sub = Commons.filter_columns(canonical, headers=headers + group_by).dropna()
        df_sub = df_sub.groupby(group_by).agg(aggregator)
        if include_weighting:
            df_sub['sum'] = df_sub.sum(axis=1, numeric_only=True)
            total = df_sub['sum'].sum()
            df_sub['weighting'] = df_sub['sum'].\
                apply(lambda x: round((x / total), freq_precision) if isinstance(x, (int, float)) else 0)
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

    def interval_categorical(self, canonical: Any, key: [str, list], column: str,
                             inc_columns: list=None, granularity: [int, float, list]=None, lower: [int, float]=None,
                             upper: [int, float]=None, rename: str=None, categories: list=None, precision: int=None,
                             unindex: bool=None, save_intent: bool=None, feature_name: [int, str]=None,
                             intent_order: int=None, replace_intent: bool=None,
                             remove_duplicates: bool=None) -> [None, pd.DataFrame]:
        """ converts continuous representation into discrete representation through interval categorisation

        :param canonical: the dataset where the column and target can be found
        :param key: the key column to index one
        :param column: the column name to be converted
        :param inc_columns: additional columns to include in the returning DataFrame
        :param granularity: (optional) the granularity of the analysis across the range. Default is 3
                int passed - represents the number of periods
                float passed - the length of each interval
                list[tuple] - specific interval periods e.g []
                list[float] - the percentile or quantities, All should fall between 0 and 1
        :param lower: (optional) the lower limit of the number value. Default min()
        :param upper: (optional) the upper limit of the number value. Default max()
        :param precision: (optional) The precision of the range and boundary values. by default set to 5.
        :param rename: (optional) if the column should have an alternative name
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
        # exceptions check
        canonical = self._get_canonical(canonical)
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        if column not in canonical.columns:
            raise ValueError(f"The column value '{column}' is not a column name in the canonical passed")
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # intend code block on the canonical
        inc_columns = self._pm.list_formatter(inc_columns)
        if not inc_columns:
            inc_columns = Commons.filter_headers(canonical, headers=key, drop=True)
        granularity = 3 if not isinstance(granularity, (int, float, list)) or granularity == 0 else granularity
        precision = precision if isinstance(precision, int) else 5
        rename = rename if isinstance(rename, str) else f"{column}_cat"
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
        df_rtn = Commons.filter_columns(canonical, headers=key + [column])
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
        df_rtn[rename] = np.select(conditions, choices, default="<NA>")
        df_rtn[rename] = df_rtn[rename].astype('category', copy=False)
        df_rtn = df_rtn.drop(column, axis=1).set_index(key)
        return df_rtn

    def group_flatten_multihot(self, canonical: Any, key: [str, list], header: str, prefix=None,
                               prefix_sep: str=None, dummy_na: bool=False, drop_first: bool=False,  dtype: Any=None,
                               aggregator: str=None, dups=True, title_rename_map: dict=None, title_case=None,
                               title_replace_spaces: str=None, inc_columns: list=None, unindex: bool=None,
                               save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                               replace_intent: bool=None, remove_duplicates: bool=None) -> [None, pd.DataFrame]:
        """ groups flattens a one-hot or multi-hot encoding of a categorical

        :param canonical: the Dataframe to reference
        :param key: the key column to sum on
        :param header: the category type column break into the category columns
        :param aggregator: (optional) the aggregator as a function of Pandas DataFrame 'groupby'
        :param title_rename_map: dictionary map of title header mapping
        :param title_case: changes the column header title to lower, upper, title, snake.
        :param title_replace_spaces: character to replace spaces in title headers. Default is '_' (underscore)
        :param prefix : str, list of str, or dict of str, default None
                String to append DataFrame column names.
                Pass a list with length equal to the number of columns
                when calling get_dummies on a DataFrame. Alternatively, `prefix`
                can be a dictionary mapping column names to prefixes.
        :param prefix_sep : str, default '_'
                If appending prefix, separator/delimiter to use. Or pass a
                list or dictionary as with `prefix`.
        :param dummy_na : bool, default False
                Add a column to indicate NaNs, if False NaNs are ignored.
        :param drop_first : bool, default False
                Whether to get k-1 dummies out of k categorical levels by removing the
                first level.
        :param dtype : dtype, default np.uint8
                Data type for new columns. Only a single dtype is allowed.
        :param inc_columns: (optional) additional columns to include in the returning canonical.
                If extra columsn are included the group aggriation key will be on all these columns not just the key.
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
        if header not in canonical:
            raise NameError(f"The column {header} can't be found in the DataFrame")
        canonical = self._get_canonical(canonical)
        aggregator = aggregator if isinstance(aggregator, str) else 'sum'
        prefix = prefix if isinstance(prefix, str) else header
        prefix_sep = prefix_sep if isinstance(prefix_sep, str) else "_"
        dummy_na = dummy_na if isinstance(dummy_na, bool) else False
        drop_first = drop_first if isinstance(drop_first, bool) else False
        dtype = dtype if dtype else np.uint8
        if isinstance(unindex, bool) and unindex:
            canonical.reset_index(inplace=True)
        key = Commons.list_formatter(key)
        if canonical[header].dtype.name != 'category':
            canonical[header] = canonical[header].astype('category')
        inc_columns = self._pm.list_formatter(inc_columns)
        df = Commons.filter_columns(canonical, headers=list(set(key + [header] + inc_columns)))
        if not dups:
            df.drop_duplicates(inplace=True)
        dummy_df = pd.get_dummies(canonical, columns=[header], prefix=prefix, prefix_sep=prefix_sep, dummy_na=dummy_na,
                                  drop_first=drop_first, dtype=dtype)
        dummy_cols = Commons.filter_headers(dummy_df, regex=f'{prefix}{prefix_sep}')
        group_cols = Commons.filter_headers(dummy_df, headers=dummy_cols, drop=True)
        dummy_df = self.group_features(dummy_df, headers=dummy_cols, group_by=group_cols, aggregator=aggregator,
                                       save_intent=False).reset_index()
        module = HandlerFactory.get_module(module_name='ds_discovery')
        module.Transition.scratch_pad().auto_clean_header(dummy_df, case=title_case, rename_map=title_rename_map,
                                                          replace_spaces=title_replace_spaces, inplace=True)
        return dummy_df.set_index(key)

    def custom_builder(self, canonical: Any, code_str: str, use_exec: bool=False,
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
        canonical = self._get_canonical(canonical)
        local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
        if 'canonical' not in local_kwargs:
            local_kwargs['canonical'] = canonical

        result = exec(code_str, globals(), local_kwargs) if use_exec else eval(code_str, globals(), local_kwargs)
        if result is None:
            return canonical
        return result

    @staticmethod
    def select2dict(column: str, condition: str, operator: str=None, logic: str=None, date_format: str=None,
                    offset: int=None):
        """ a utility method to help build feature conditions by aligning method parameters with dictionary format.

        :param column: the column name to apply the condition to
        :param condition: the condition string (special conditions are 'date.now' for current date
        :param operator: (optional) an operator to place before the condition if not included in the condition
        :param logic: (optional) the logic to provide, options are 'and', 'or', 'not', 'xor'
        :param date_format: (optional) a format of the date if only a specific part of the date and time is required
        :param offset: (optional) a time delta in days (+/-) from the current date and time (minutes not supported)
        :return: dictionary of the parameters

        logic:
            and: the intersect of the left and the right (common to both)
            or: the union of the left and the right (everything in both)
            diff: the left minus the intersect of the right (only things in the left excluding common to both)


        """
        return Commons.param2dict(**locals())

    @staticmethod
    def canonical_sampler(canonical: [pd.DataFrame, pd.Series], sample_size: [int, float], shuffle: bool=None,
                          train_only: bool=True, seed: int=None) -> [tuple, pd.DataFrame, pd.Series]:
        """ returns a tuple of the canonical split of sample size and the remaining

        :param canonical: a canonical to take the sampler from
        :param sample_size: If float, should be between 0.0 and 1.0 and represent the proportion of the
                            data set to return as a sample. If int, represents the absolute number of samples.
        :param shuffle: (optional) if the canonical should be shuffled
        :param train_only: (optional) if only the train data-set should be returned rather than the train, test tuple
        :param seed: (optional) if shuffle is not None a seed value for the sample_size
        :return: a (sample, remaining) tuple
        """
        if not isinstance(canonical, (pd.DataFrame, pd.Series)):
            raise ValueError(f"The canonical must be a pandas DataFrame or Series")
        shuffle = shuffle if isinstance(shuffle, bool) else False
        if isinstance(sample_size, float):
            if not 0 < sample_size < 1:
                raise ValueError(f"if passing a test_size as a float the number must be tween 0 and 1")
            if shuffle:
                train = canonical.sample(frac=sample_size, random_state=seed)
            else:
                train = canonical.iloc[:int(canonical.shape[0] * sample_size)]
        elif isinstance(sample_size, int):
            if sample_size > canonical.shape[0]:
                raise ValueError(f"The sample size '{sample_size}' can't be greater than the canonical "
                                 f"number the rows '{canonical.shape[0]}'")
            if shuffle:
                train = canonical.sample(n=sample_size, random_state=seed)
            else:
                train = canonical.iloc[:sample_size]
        else:
            raise ValueError(f"sample_size must be an int less than the number of rows or a float between 0 and 1")
        test = canonical.loc[~canonical.index.isin(train.index), :]
        if isinstance(train_only, bool) and train_only:
            return train
        return train, test

    """
        PRIVATE METHODS SECTION
    """
    def _intent_builder(self, method: str, params: dict, exclude: list=None) -> dict:
        """builds the intent_params. Pass the method name and local() parameters
            Example:
                self._intent_builder(inspect.currentframe().f_code.co_name, **locals())

        :param method: the name of the method (intent). can use 'inspect.currentframe().f_code.co_name'
        :param params: the parameters passed to the method. use `locals()` in the caller method
        :param exclude: (optional) convenience parameter identifying param keys to exclude.
        :return: dict of the intent
        """
        if not isinstance(params.get('canonical', None), (str, dict)):
            exclude = ['canonical']
        return super()._intent_builder(method=method, params=params, exclude=exclude)

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

    @staticmethod
    def _condition_index(canonical: pd.DataFrame, condition: dict, select_idx: pd.Int64Index):
        """ private method to select index from the selection conditions

        :param canonical: a pandas DataFrame to select from
        :param condition: the dict conditions
        :param select_idx: the current selection index of the canonical
        :return: returns the current select_idx of the condition
        """
        _column = condition.get('column')
        _condition = condition.get('condition')
        _operator = condition.get('operator', '')
        _logic = condition.get('logic', 'and')
        if _condition == 'date.now':
            _date_format = condition.get('date_format', "%Y-%m-%dT%H:%M:%S")
            _offset = condition.get('offset', 0)
            _condition = f"'{(pd.Timestamp.now() + pd.Timedelta(days=_offset)).strftime(_date_format)}'"
        s_values = canonical[_column]
        idx = eval(f"s_values.where(s_values{_operator}{_condition}).dropna().index", globals(), locals())
        if select_idx is None:
            select_idx = idx
        else:
            if str(_logic).lower() == 'and':
                select_idx = select_idx.intersection(idx)
            elif str(_logic).lower() == 'or':
                select_idx = select_idx.union(idx)
            elif str(_logic).lower() == 'not':
                select_idx = select_idx.difference(idx)
            elif str(_logic).lower() == 'xor':
                select_idx = select_idx.union(idx).difference(select_idx.intersection(idx))
            else:
                raise ValueError(f"The logic '{_logic}' for column '{_column}' is not recognised logic. "
                                 f"Use 'AND', 'OR', 'NOT', 'XOR'")
        return select_idx
