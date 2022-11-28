import inspect
import pandas as pd
from typing import Any
from ds_discovery.intent.abstract_builder_intent import AbstractBuilderIntentModel
from ds_discovery.managers.synthetic_property_manager import SyntheticPropertyManager
from ds_discovery.managers.wrangle_property_manager import WranglePropertyManager

__author__ = 'Darryl Oatridge'


class WrangleIntentModel(AbstractBuilderIntentModel):
    
    def __init__(self, property_manager: [WranglePropertyManager, SyntheticPropertyManager],
                 default_save_intent: bool=None, default_intent_level: [str, int, float]=None,
                 order_next_available: bool=None, default_replace_intent: bool=None):
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
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         default_intent_level=default_intent_level, default_intent_order=default_intent_order,
                         default_replace_intent=default_replace_intent)

    def frame_starter(self, canonical: Any, selection: list=None, choice: int=None, headers: [str, list]=None,
                      drop: bool=None, dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                      re_ignore_case: bool=None, rename_map: dict=None, seed: int=None, save_intent: bool=None,
                      column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                      remove_duplicates: bool=None) -> pd.DataFrame:
        """ Selects rows and/or columns changing the shape of the DatFrame. This is always run first in a pipeline
        Rows are filtered before columns are filtered so columns can be referenced even though they might not be
        included in the final column list.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param choice: a number of rows to select, randomly selected from the index
        :param selection: a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt' columns
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param seed: this is a place holder, here for compatibility across methods
        :param rename_map: a from: to dictionary of headers to rename
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pd.DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - int -> generates an empty pd.Dataframe with an index size of the int passed.
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        Selections are a list of dictionaries of conditions and optional additional parameters to filter.
        To help build conditions there is a static helper method called 'select2dict(...)' that has parameter
        options available to build a condition.
        An example of a condition with the minimum requirements is
                [{'column': 'genre', 'condition': "=='Comedy'"}]

        an example of using the helper method
                selection = [inst.select2dict(column='gender', condition="=='M'"),
                             inst.select2dict(column='age', condition=">65", logic='XOR')]

        Using the 'select2dict' method ensure the correct keys are used and the dictionary is properly formed. It also
        helps with building the logic that is executed in order
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._frame_starter(seed=seed, **params)

    def frame_selection(self, canonical: Any, selection: list=None, choice: int=None, headers: [str, list]=None,
                        drop: bool=None, dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                        re_ignore_case: bool=None, seed: int=None, save_intent: bool=None,
                        column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                        remove_duplicates: bool=None) -> pd.DataFrame:
        """ Selects rows and/or columns changing the shape of the DatFrame. This is always run last in a pipeline
        Rows are filtered before the column filter so columns can be referenced even though they might not be included
        the final column list.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param selection: a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
        :param choice: a number of rows to select, randomly selected from the index
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or excluse. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt' columns
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param seed: this is a place holder, here for compatibility across methods
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: pd.DataFrame

        Selections are a list of dictionaries of conditions and optional additional parameters to filter.
        To help build conditions there is a static helper method called 'select2dict(...)' that has parameter
        options available to build a condition.
        An example of a condition with the minimum requirements is
                [{'column': 'genre', 'condition': "=='Comedy'"}]

        an example of using the helper method
                selection = [inst.select2dict(column='gender', condition="=='M'"),
                             inst.select2dict(column='age', condition=">65", logic='XOR')]

        Using the 'select2dict' method ensure the correct keys are used and the dictionary is properly formed. It also
        helps with building the logic that is executed in order

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._frame_selection(seed=seed, **params)

    def model_custom(self, canonical: Any, code_str: str, seed: int=None, save_intent: bool=None,
                     column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                     remove_duplicates: bool=None, **kwargs):
        """ Commonly used for custom methods, takes code string that when executed changes the the canonical returning
        the modified canonical. If the method passes returns a pd.Dataframe this will be returned else the assumption is
        the canonical has been changed inplace and thus the modified canonical will be returned
        When referencing the canonical in the code_str it should be referenced either by use parameter label 'canonical'
        or the short cut '@' symbol. kwargs can also be passed into the code string but must be preceded by a '$' symbol
        for example:
            code_str =

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param code_str: an action on those column values
        :param kwargs: a set of kwargs to include in any executable function
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list or pandas.DataFrame
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        params.update(params.pop('kwargs', {}))
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_custom(seed=seed, **params)

    def model_group(self, canonical: Any, group_by: [str, list], headers: [str, list]=None, regex: bool=None,
                    aggregator: str=None, list_choice: int=None, list_max: int=None, drop_group_by: bool=False,
                    seed: int=None, include_weighting: bool=False, freq_precision: int=None,
                    remove_weighting_zeros: bool=False, remove_aggregated: bool=False, save_intent: bool=None,
                    column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                    remove_duplicates: bool=None) -> pd.DataFrame:
        """ returns the full column values directly from another connector data source. in addition the the
        standard groupby aggregators there is also 'list' and 'set' that returns an aggregated list or set.
        These can be using in conjunction with 'list_choice' and 'list_size' allows control of the return values.
        if list_max is set to 1 then a single value is returned rather than a list of size 1.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param headers: the column headers to apply the aggregation too
        :param group_by: the column headers to group by
        :param regex: if the column headers is q regex
        :param aggregator: (optional) the aggregator as a function of Pandas DataFrame 'groupby' or 'list' or 'set'
        :param list_choice: (optional) used in conjunction with list or set aggregator to return a random n choice
        :param list_max: (optional) used in conjunction with list or set aggregator restricts the list to a n size
        :param drop_group_by: (optional) drops the group by headers
        :param include_weighting: (optional) include a percentage weighting column for each
        :param freq_precision: (optional) a precision for the relative_freq values
        :param remove_aggregated: (optional) if used in conjunction with the weighting then drops the aggrigator column
        :param remove_weighting_zeros: (optional) removes zero values
        :param seed: (optional) this is a place holder, here for compatibility across methods
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.DataFrame
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_group(seed=seed, **params)

    def model_modifier(self, canonical: Any, other: Any, targets_header: str=None, values_header: str=None,
                       modifier: str=None, aggregator: str=None, agg_header: str=None, seed: int=None,
                       precision: int=None, save_intent: bool=None, column_name: [int, str]=None,
                       intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """Modifies a given set of target header names, within the canonical with the target value for that name. The
        aggregator indicates the type of modification to be performed. It is assumed the other DataFrame has the
        target headers as the first column and the target values as the second column, if this is not the case the
        targets_header and values_handler parameters can be used to specify the other header names.

        Additionally, the given headers, from other, can be aggregated to a single value. The new aggregated column
        can be given a header name with agg_header.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param other: a direct or generated pd.DataFrame. see context notes below
        :param targets_header: (optional) the name of the target header where the header names are listed
        :param values_header: (optional) The name of the value header where the target values are listed
        :param modifier: (optional) how the value is to be modified. Options are 'add', 'sub', 'mul', 'div'
        :param aggregator: (optional) the aggregation function name to enact. The available functions are:
                           'sum', 'prod', 'count', 'min', 'max', 'mean' and 'list' which combines the columns as a list
        :param agg_header: (optional) the name to give the aggregated column. By default 'latent_aggregator'
        :param precision: (optional) the value precision of the return values
        :param seed: (optional) this is a placeholder, here for compatibility across methods
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.DataFrame

        The other is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrame of one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_modifier(seed=seed, **params)

    def model_merge(self, canonical: Any, other: Any, left_on: str=None, right_on: str=None, on: str=None,
                    how: str=None, headers: list=None, suffixes: tuple=None, indicator: bool=None, validate: str=None,
                    replace_nulls: bool=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                    intent_order: int=None, replace_intent: bool=None,  remove_duplicates: bool=None) -> pd.DataFrame:
        """ returns the full column values directly from another connector data source.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param other: a direct or generated pd.DataFrame. see context notes below
        :param left_on: the canonical key column(s) to join on
        :param right_on: the merging dataset key column(s) to join on
        :param on: if th left and right join have the same header name this can replace left_on and right_on
        :param how: (optional) One of 'left', 'right', 'outer', 'inner'. Defaults to inner. See below for more detailed
                    description of each method.
        :param headers: (optional) a filter on the headers included from the right side
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
        :param replace_nulls: (optional) replaces nulls with an appropriate value dependent upon the field type
        :param seed: this is a place holder, here for compatibility across methods
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.DataFrame

        The other is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrame of one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_merge(seed=seed, **params)

    def model_concat(self, canonical: Any, other: Any, as_rows: bool=None, headers: [str, list]=None,
                     drop: bool=None, dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                     re_ignore_case: bool=None, shuffle: bool=None, seed: int=None, save_intent: bool=None,
                     column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                     remove_duplicates: bool=None) -> pd.DataFrame:
        """ returns the full column values directly from another connector data source.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param other: a direct or generated pd.DataFrame. see context notes below
        :param as_rows: (optional) how to concatenate, True adds the connector dataset as rows, False as columns
        :param headers: (optional) a filter of headers from the 'other' dataset
        :param drop: (optional) to drop or not drop the headers if specified
        :param dtype: (optional) a filter on data type for the 'other' dataset. int, float, bool, object
        :param exclude: (optional) to exclude or include the data types if specified
        :param regex: (optional) a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt'
        :param re_ignore_case: (optional) true if the regex should ignore case. Default is False
        :param shuffle: (optional) if the rows in the loaded canonical should be shuffled
        :param seed: this is a place holder, here for compatibility across methods
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.DataFrame

        The other is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_concat(seed=seed, **params)

    def model_dict_column(self, canonical: Any, header: str, convert_str: bool=None, seed: int=None,
                          save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                          replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ takes a column that contains dict and expands them into columns. Note, the column must be a flat dictionary.
        Complex structures will not work.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header of the column to be convert
        :param convert_str: (optional) if the header has the dict as a string convert to dict using ast.literal_eval()
        :param seed: (optional) this is a place holder, here for compatibility across methods
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.DataFrame
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_dict_column(seed=seed, **params)

    def model_missing_cca(self, canonical: Any, threshold: float=None, seed: int=None, save_intent: bool=None,
                          column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                          remove_duplicates: bool=None) -> pd.DataFrame:
        """ Applies Complete Case Analysis to the canonical. Complete-case analysis (CCA), also called "list-wise
        deletion" of cases, consists of discarding observations with any missing values. In other words, we only keep
        observations with data on all the variables. CCA works well when the data is missing completely at random.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param threshold: (optional) a null threshold between 0 and 1 where 1 is all nulls. Default to 0.5
        :param seed: (optional) a placeholder
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.DataFrame
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_missing_cca(seed=seed, **params)

    def model_drop_outliers(self, canonical: Any, header: str, measure: [int, float]=None, method: str=None,
                            seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                            replace_intent: bool=None, remove_duplicates: bool=None):
        """ Drops rows in the canonical where the values are deemed outliers based on the method and measure.
        There are three selectable methods of choice, interquartile or empirical, of which interquartile
        is the default.

        The 'empirical' rule states that for a normal distribution, nearly all of the data will fall within three
        standard deviations of the mean. Given mu and sigma, a simple way to identify outliers is to compute a z-score
        for every value, which is defined as the number of standard deviations away a value is from the mean. therefor
        measure given should be the z-score or the number of standard deviations away a value is from the mean.
        The 68–95–99.7 rule, guide the percentage of values that lie within a band around the mean in a normal
        distribution with a width of two, four and six standard deviations, respectively and thus the choice of z-score

        For the 'interquartile' range (IQR), also called the midspread, middle 50%, or H‑spread, is a measure of
        statistical dispersion, being equal to the difference between 75th and 25th percentiles, or between upper
        and lower quartiles of a sample set. The IQR can be used to identify outliers by defining limits on the sample
        values that are a factor k of the IQR below the 25th percentile or above the 75th percentile. The common value
        for the factor k is 1.5. A factor k of 3 or more can be used to identify values that are extreme outliers.

         :param canonical: a pd.DataFrame as the reference dataframe
         :param header: the header in the DataFrame to correlate
         :param method: (optional) A method to run to identify outliers. interquartile (default) or empirical
         :param measure: (optional) A measure against each method, respectively factor k, z-score, quartile (see above)
         :param seed: (optional) the random seed
         :param save_intent: (optional) if the intent contract should be saved to the property manager
         :param column_name: (optional) the column name that groups intent to create a column
         :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
         :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
         :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
         :return: an equal length list of correlated values
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_drop_outliers(seed=seed, **params)

    # convert objects to categories
    def model_to_category(self, canonical: Any, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                          exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None, seed: int=None,
                          save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                          replace_intent: bool=None, remove_duplicates: bool=None):
        """ converts columns to categories

        :param canonical: a pd.DataFrame as the reference dataframe
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param seed: (optional) a placeholder
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.DataFrame
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_to_category(seed=seed, **params)

    def model_encode_one_hot(self, canonical: Any, headers: [str, list], prefix=None, dtype: Any=None,
                             prefix_sep: str=None, dummy_na: bool=False, drop_first: bool=False, seed: int=None,
                             save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                             replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ encodes categorical data types, One hot encoding, consists in encoding each categorical variable with
        different boolean variables (also called dummy variables) which take values 0 or 1, indicating if a category
        is present in an observation.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param headers: the header(s) to apply multi-hot
        :param prefix : str, list of str, or dict of str, String to append DataFrame column names,
                list with length equal to the number of columns. Alternatively, dictionary mapping column names to prefixes.
        :param prefix_sep : str separator, default '_'
        :param dummy_na : Add a column to indicate null values, if False nullss are ignored.
        :param drop_first :  Whether to get k-1 dummies out of k categorical levels by removing the first level.
        :param dtype : Data type for new columns. Only a single dtype is allowed.
        :param seed: seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.DataFrame
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_encode_one_hot(seed=seed, **params)

    def model_encode_ordinal(self, canonical: Any, headers: [str, list], prefix=None, seed: int=None,
                             save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                             replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ encodes categorical data types, Integer encoding consist in replacing the categories by digits from 1 to n
        (or 0 to n-1, depending the implementation), where n is the number of distinct categories of the variable.
        The numbers are assigned arbitrarily. This encoding method allows for quick benchmarking of machine learning
        models.

        Advantages
        - Straightforward to implement
        - Does not expand the feature space
        Limitations
        - Does not capture any information about the categories labels
        - Not suitable for linear models.

        Integer encoding is better suited for non-linear methods which are able to navigate through the arbitrarily
        assigned digits to try and find patters that relate them to the target.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param headers: the header(s) to apply the encoding
        :param prefix: a str to prefix the column
        :param seed: seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.DataFrame
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_encode_ordinal(seed=seed, **params)

    def model_encode_count(self, canonical: Any, headers: [str, list], prefix=None, seed: int=None,
                           save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                           replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ encodes categorical data types, In count encoding we replace the categories by the count of the
        observations that show that category in the dataset. This techniques capture's the representation of each label
        in a dataset, but the encoding may not necessarily be predictive of the outcome.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param headers: the header(s) to apply the encoding
        :param prefix: a str to prefix the column
        :param seed: seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.DataFrame
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_encode_count(seed=seed, **params)

    # def model_encode_woe(self, canonical: Any, headers: [str, list], target: str=None, prefix=None, seed: int=None,
    #                        save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
    #                        replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
    #     """ encodes categorical data types, Weight of Evidence (WoE) was developed primarily for the credit and
    #     financial industries to help build more predictive models to evaluate the risk of loan default. That is, to
    #     predict how likely the money lent to a person or institution is to be lost. Thus, Weight of Evidence is a
    #     measure of the "strength” of a grouping technique to separate good and bad risk (default).
    #
    #     - WoE will be 0 if the P(Goods) / P(Bads) = 1, that is, if the outcome is random for that group.
    #     - If P(Bads) > P(Goods) the odds ratio will be < 1 and,
    #     - WoE will be < 0 if, P(Goods) > P(Bads).
    #     WoE is well suited for Logistic Regression, because the Logit transformation is simply the log of the odds,
    #     i.e., ln(P(Goods)/P(Bads)). Therefore, by using WoE-coded predictors in logistic regression, the predictors are
    #     all prepared and coded to the same scale, and the parameters in the linear logistic regression equation can be
    #     directly compared.
    #
    #     The WoE transformation has three advantages:
    #     - It creates a monotonic relationship between the target and the independent variables.
    #     - It orders the categories on a "logistic" scale which is natural for logistic regression
    #     - The transformed variables can then be compared because they are on the same scale. Therefore, it is possible
    #       to determine which one is more predictive.
    #
    #     The WoE also has a limitation:
    #     - Prone to cause over-fitting
    #
    #     :param canonical: a pd.DataFrame as the reference dataframe
    #     :param headers: the header(s) to apply multi-hot
    #     :param target: The woe target
    #     :param prefix: a str value to put before the column name
    #     :param seed: seed: (optional) a seed value for the random function: default to None
    #     :param save_intent (optional) if the intent contract should be saved to the property manager
    #     :param column_name: (optional) the column name that groups intent to create a column
    #     :param intent_order: (optional) the order in which each intent should run.
    #                     If None: default's to -1
    #                     if -1: added to a level above any current instance of the intent section, level 0 if not found
    #                     if int: added to the level specified, overwriting any that already exist
    #     :param replace_intent: (optional) if the intent method exists at the level, or default level
    #                     True - replaces the current intent method with the new
    #                     False - leaves it untouched, disregarding the new intent
    #     :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
    #     :return: a pd.DataFrame
    #     """
    #     self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
    #                                column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
    #                                remove_duplicates=remove_duplicates, save_intent=save_intent)
    #     # remove intent params
    #     params = locals()
    #     [params.pop(k) for k in self._INTENT_PARAMS]
    #     # set the seed and call the method
    #     seed = self._seed(seed=seed)
    #     return self._model_encode_woe(seed=seed, **params)

    def model_explode(self, canonical: Any, header: str, seed: int=None, save_intent: bool=None,
                      column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                      remove_duplicates: bool=None) -> pd.DataFrame:
        """ takes a single column of list values and explodes the DataFrame so row is represented by each elements
        in the row list

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header of the column to be exploded
        :param seed: (optional) this is a place holder, here for compatibility across methods
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a pd.DataFrame
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_explode(seed=seed, **params)

    def model_sample(self, canonical: Any, other: Any, headers: list, replace: bool=None, relative_freq: list=None,
                     seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                     replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """Takes a target dataset and samples from that target to the size of the canonical

        :param canonical: a pd.DataFrame as the reference dataframe
        :param other: a direct or generated pd.DataFrame. see context notes below
        :param headers: the headers to be selected from the other DataFrame
        :param replace: assuming other is bigger than canonical, selects without replacement when True
        :param relative_freq: (optional) a weighting pattern that does not have to add to 1
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a DataFrame
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_sample(seed=seed, **params)

    def model_analysis(self, canonical: Any, other: Any, columns_list: list=None, exclude_associate: list=None,
                       detail_numeric: bool=None, strict_typing: bool=None, category_limit: int=None, seed: int=None,
                       save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                       replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ builds a set of columns based on an analysis dictionary of weighting (see analyse_association)
        if a reference DataFrame is passed then as the analysis is run if the column already exists the row
        value will be taken as the reference to the sub category and not the random value. This allows already
        constructed association to be used as reference for a sub category.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param other: a direct or generated pd.DataFrame. see context notes below
        :param columns_list: a list structure of columns to select for association
        :param exclude_associate: (optional) a list of dot separated tree of items to exclude from iteration
                (e.g. ['age.gender.salary']
        :param detail_numeric: (optional) as a default, if numeric columns should have detail stats, slowing analysis
        :param strict_typing: (optional) stops objects and string types being seen as categories
        :param category_limit: (optional) a global cap on categories captured. zero value returns no limits
        :param seed: seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a DataFrame

        The other is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and parameters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._model_analysis(seed=seed, **params)

    def get_noise(self, size: int, ones: bool=None, seed: int=None, save_intent: bool=None,
                  column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                  remove_duplicates: bool=None) -> list:
        """ A noise bias column of ones unless ones is False, then returns zeros

        :param size: size of the list to return
        :param ones: (optional) by default set to True returning a list of ones, else returning a list of zeros
        :param seed: (optional) placeholder for continuity
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a DataFrame
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._get_noise(seed=seed, **params)

    def correlate_selection(self, canonical: Any, selection: list, action: [str, int, float, dict],
                            default_action: [str, int, float, dict]=None, seed: int=None, rtn_type: str=None,
                            save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                            replace_intent: bool=None, remove_duplicates: bool=None):
        """ returns a value set based on the selection list and the action enacted on that selection. If
        the selection criteria is not fulfilled then the default_action is taken if specified, else null value.

        If a DataFrame is not passed, the values column is referenced by the header '_default'

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param selection: a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
        :param action: a value or dict to act upon if the select is successful. see below for more examples
                An example of an action as a dict: (see 'action2dict(...)')
                {'method': 'get_category', 'selection': ['M', 'F', 'U']}
        :param default_action: (optional) a default action to take if the selection is not fulfilled
        :param seed: (optional) a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: value set based on the selection list and the action

        Selections are a list of dictionaries of conditions and optional additional parameters to filter.
        To help build conditions there is a static helper method called 'select2dict(...)' that has parameter
        options available to build a condition.
        An example of a condition with the minimum requirements is
                [{'column': 'genre', 'condition': "=='Comedy'"}]

        an example of using the helper method
                selection = [inst.select2dict(column='gender', condition="=='M'"),
                             inst.select2dict(column='age', condition=">65", logic='XOR')]

        Using the 'select2dict' method ensure the correct keys are used and the dictionary is properly formed. It also
        helps with building the logic that is executed in order

        Actions are the resulting outcome of the selection (or the default). An action can be just a value or a dict
        that executes a intent method such as get_number(). To help build actions there is a helper function called
        action2dict(...) that takes a method as a mandatory attribute.

        With actions there are special keyword 'method' values:
            @header: use a column as the value reference, expects the 'header' key
            @constant: use a value constant, expects the key 'value'
            @sample: use to get sample values, expected 'name' of the Sample method, optional 'shuffle' boolean
            @eval: evaluate a code string, expects the key 'code_str' and any locals() required

        An example of a simple action to return a selection from a list:
                {'method': 'get_category', selection: ['M', 'F', 'U']}

        This same action using the helper method would look like:
                inst.action2dict(method='get_category', selection=['M', 'F', 'U'])

        an example of using the helper method, in this example we use the keyword @header to get a value from another
        column at the same index position:
                inst.action2dict(method="@header", header='value')

        We can even execute some sort of evaluation at run time:
                inst.action2dict(method="@eval", code_str='sum(values)', values=[1,4,2,1])
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_selection(seed=seed, **params)

    def correlate_date_diff(self, canonical: Any, first_date: str, second_date: str, units: str=None,
                            precision: int=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                            intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None, **kwargs):
        """ returns a column for the difference between a primary and secondary date where the primary is an early date
        than the secondary.

        :param canonical: the DataFrame containing the column headers
        :param first_date: the primary or older date field
        :param second_date: the secondary or newer date field
        :param units: (optional) The Timedelta units e.g. 'D', 'W', 'M', 'Y'. default is 'D'
        :param precision: the precision of the result
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :param kwargs: a set of kwargs to include in any executable function
        :return: value set based on the selection list and the action
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        params.update(params.pop('kwargs', {}))
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_date_diff(seed=seed, **params)

    def correlate_custom(self, canonical: Any, code_str: str, seed: int=None, save_intent: bool=None,
                         column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                         remove_duplicates: bool=None, **kwargs):
        """ Commonly used for custom list comprehension, takes code string that when evaluated returns a list of values
        Before using this method, consider the method correlate_selection(...)

        When referencing the canonical in the code_str it should be referenced either by use parameter label 'canonical'
        or the short cut '@' symbol.
        for example:
            code_str = "[x + 2 for x in @['A']]" # where 'A' is a header in the canonical

        kwargs can also be passed into the code string but must be preceded by a '$' symbol
        for example:
            code_str = "[True if x == $v1 else False for x in @['A']]" # where 'v1' is a kwargs

        :param canonical: a pd.DataFrame as the reference dataframe
        :param code_str: an action on those column values. to reference the canonical use '@'
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :param kwargs: a set of kwargs to include in any executable function
        :return: value set based on the selection list and the action
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        params.update(params.pop('kwargs', {}))
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_custom(seed=seed, **params)

    def correlate_aggregate(self, canonical: Any, headers: list, agg: str, seed: int=None, rtn_type: str=None,
                            save_intent: bool=None, precision: int=None, column_name: [int, str]=None,
                            intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ correlate two or more columns with each other through a finite set of aggregation functions. The
        aggregation function names are limited to 'sum', 'prod', 'count', 'min', 'max' and 'mean' for numeric columns
        and a special 'list' function name to combine the columns as a list

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param headers: a list of headers to correlate
        :param agg: the aggregation function name enact. The available functions are:
                        'sum', 'prod', 'count', 'min', 'max', 'mean' and 'list' which combines the columns as a list
        :param precision: the value precision of the return values
        :param seed: (optional) a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal length to the one passed
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_aggregate(seed=seed, **params)

    def correlate_choice(self, canonical: Any, header: str, list_size: int=None, random_choice: bool=None,
                         replace: bool=None, shuffle: bool=None, convert_str: bool=None, rtn_type: str=None,
                         seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                         replace_intent: bool=None, remove_duplicates: bool=None):
        """ correlate a column where the elements of the columns contains a list, and a choice is taken from that list.
        if the list_size == 1 then a single value is correlated otherwise a list is correlated

        Null values are passed through but all other elements must be a list with at least 1 value in.

        if 'random' is true then all returned values will be a random selection from the list and of equal length.
        if 'random' is false then each list will not exceed the 'list_size'

        Also if 'random' is true and 'replace' is False then all lists must have more elements than the list_size.
        By default 'replace' is True and 'shuffle' is False.

        In addition 'convert_str' allows lists that have been formatted as a string can be converted from a string
        to a list using 'ast.literal_eval(x)'

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: The header containing a list to chose from.
        :param list_size: (optional) the number of elements to return, if more than 1 then list
        :param random_choice: (optional) if the choice should be a random choice.
        :param replace: (optional) if the choice selection should be replaced or selected only once
        :param shuffle: (optional) if the final list should be shuffled
        :param convert_str: if the header has the list as a string convert to list using ast.literal_eval()
        :param seed: (optional) a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal length to the one passed
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_element_choice(seed=seed, **params)

    def correlate_join(self, canonical: Any, header: str, action: [str, dict], sep: str=None, rtn_type: str=None,
                       seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                       replace_intent: bool=None, remove_duplicates: bool=None):
        """ correlate a column and join it with the result of the action, This allows for composite values to be
        build from. an example might be to take a forename and add the surname with a space separator to create a
        composite name field, of to join two primary keys to create a single composite key.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: an ordered list of columns to join
        :param action: (optional) a string or a single action whose outcome will be joined to the header value
        :param sep: (optional) a separator between the values
        :param seed: (optional) a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal length to the one passed

        Actions are the resulting outcome of the selection (or the default). An action can be just a value or a dict
        that executes a intent method such as get_number(). To help build actions there is a helper function called
        action2dict(...) that takes a method as a mandatory attribute.

        With actions there are special keyword 'method' values:
            @header: use a column as the value reference, expects the 'header' key
            @constant: use a value constant, expects the key 'value'
            @sample: use to get sample values, expected 'name' of the Sample method, optional 'shuffle' boolean
            @eval: evaluate a code string, expects the key 'code_str' and any locals() required

        An example of a simple action to return a selection from a list:
                {'method': 'get_category', selection=['M', 'F', 'U']

        an example of using the helper method, in this example we use the keyword @header to get a value from another
        column at the same index position:
                inst.action2dict(method="@header", header='value')

        We can even execute some sort of evaluation at run time:
                inst.action2dict(method="@eval", code_str='sum(values)', values=[1,4,2,1])
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_join(seed=seed, **params)

    def correlate_sigmoid(self, canonical: Any, header: str, precision: int=None, seed: int=None, rtn_type: str=None,
                          save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                          replace_intent: bool=None, remove_duplicates: bool=None):
        """ logistic sigmoid a.k.a logit, takes an array of real numbers and transforms them to a value
        between (0,1) and is defined as
                                        f(x) = 1/(1+exp(-x)

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param precision: (optional) how many decimal places. default to 3
        :param seed: (optional) the random seed. defaults to current datetime
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: an equal length list of correlated values
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_sigmoid(seed=seed, **params)

    def correlate_polynomial(self, canonical: Any, header: str, coefficient: list, rtn_type: str=None, seed: int=None,
                             keep_zero: bool=None, save_intent: bool=None, column_name: [int, str]=None,
                             intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ creates a polynomial using the reference header values and apply the coefficients where the
        index of the list represents the degree of the term in reverse order.

                  e.g  [6, -2, 0, 4] => f(x) = 4x**3 - 2x + 6

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param coefficient: the reverse list of term coefficients
        :param seed: (optional) the random seed. defaults to current datetime
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :param keep_zero: (optional) if True then zeros passed remain zero, Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: an equal length list of correlated values
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_polynomial(seed=seed, **params)

    def correlate_mark_outliers(self, canonical: Any, header: str, measure: [int, float]=None, method: str=None,
                                seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                                replace_intent: bool=None, remove_duplicates: bool=None):
        """ Drops rows in the canonical where the values are deemed outliers based on the method and measure.
        There are three selectable methods of choice, interquartile or empirical, of which interquartile
        is the default.

        The 'empirical' rule states that for a normal distribution, nearly all of the data will fall within three
        standard deviations of the mean. Given mu and sigma, a simple way to identify outliers is to compute a z-score
        for every value, which is defined as the number of standard deviations away a value is from the mean. therefor
        measure given should be the z-score or the number of standard deviations away a value is from the mean.
        The 68–95–99.7 rule, guide the percentage of values that lie within a band around the mean in a normal
        distribution with a width of two, four and six standard deviations, respectively and thus the choice of z-score

        For the 'interquartile' range (IQR), also called the midspread, middle 50%, or H‑spread, is a measure of
        statistical dispersion, being equal to the difference between 75th and 25th percentiles, or between upper
        and lower quartiles of a sample set. The IQR can be used to identify outliers by defining limits on the sample
        values that are a factor k of the IQR below the 25th percentile or above the 75th percentile. The common value
        for the factor k is 1.5. A factor k of 3 or more can be used to identify values that are extreme outliers.

         :param canonical: a pd.DataFrame as the reference dataframe
         :param header: the header in the DataFrame to correlate
         :param method: (optional) A method to run to identify outliers. interquartile (default) or empirical
         :param measure: (optional) A measure against each method, respectively factor k, z-score, quartile (see above)
         :param seed: (optional) the random seed
         :param save_intent: (optional) if the intent contract should be saved to the property manager
         :param column_name: (optional) the column name that groups intent to create a column
         :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
         :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
         :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
         :return: an equal length list of correlated values
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_flag_outliers(seed=seed, **params)

    def correlate_missing(self, canonical: Any, header: str, method: str=None, weights: str=None, constant: Any=None, precision: int=None,
                          seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                          replace_intent: bool=None, remove_duplicates: bool=None):
        """ imputes missing data with statistical estimates of the missing values. The methods are 'mean', 'median',
        'mode' and 'random' with the addition of 'constant' and 'indicator'

        Mean/median imputation consists of replacing all occurrences of missing values (NA) within a variable by the
        mean (if the variable has a Gaussian distribution) or median (if the variable has a skewed distribution). Can
        only be applied to numeric values.

        Mode imputation consists of replacing all occurrences of missing values (NA) within a variable by the mode,
        which is the most frequent value or most frequent category. Can be applied to both numerical and categorical
        variables.

        Random sampling imputation is in principle similar to mean, median, and mode imputation in that it considers
        that missing values, should look like those already existing in the distribution. Random sampling consists of
        taking random observations from the pool of available data and using them to replace the NA. In random sample
        imputation, we take as many random observations as missing values exist in the variable. Can be applied to both
        numerical and categorical variables.

        Neighbour imputation is for filling in missing values using the k-Nearest Neighbors approach. Each missing
        feature is imputed using values from five nearest neighbors that have a value for the feature. The
        feature of the neighbors are averaged uniformly or weighted by distance to each neighbor. If a sample has
        more than one feature missing, then the neighbors for that sample can be different depending on the particular
        feature being imputed. When the number of available neighbors is less than five the average for that feature
        is used during imputation. If there is at least one neighbor with a defined distance, the weighted or
        unweighted average of the remaining neighbors will be used during imputation.

        Constant or Arbitrary value imputation consists of replacing all occurrences of missing values (NA) with an
        arbitrary constant value. Can be applied to both numerical and categorical variables. A value must be passed
        in the constant parameter relevant to the column type.

        Indicator is not an imputation method but imputation techniques, such as mean, median and random will affect
        the variable distribution quite dramatically and is a good idea to flag them with a missing indicator. This
        must be done before imputation of the column.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param method: (optional) the replacement method, 'mean', 'median', 'mode', 'constant', 'random', 'indicator'
        :param weights: (optional) Weight function used in prediction of nearest neighbour if used as method. Options
                    ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                    ‘distance’ : weight points by the inverse of their distance.
        :param constant: (optional) a value to us when the method is constant
        :param precision: (optional) if numeric, the precision of the outcome, by default set to 3.
        :param seed: (optional) the random seed. defaults to current datetime
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: an equal length list of correlated values
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_missing(seed=seed, **params)

    def correlate_missing_weighted(self, canonical: Any, header: str, granularity: [int, float, list]=None,
                                   as_type: str=None, lower: [int, float]=None, upper: [int, float]=None,
                                   exclude_dominant: bool=None, replace_zero: [int, float]=None,
                                   precision: int=None, day_first: bool=None, year_first: bool=None, seed: int=None,
                                   rtn_type: str=None, save_intent: bool=None, column_name: [int, str]=None,
                                   intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ imputes missing data with a weighted distribution based on the analysis of the other elements in the
            column

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param granularity: (optional) the granularity of the analysis across the range. Default is 5
                int passed - represents the number of periods
                float passed - the length of each interval
                list[tuple] - specific interval periods e.g []
                list[float] - the percentile or quantities, All should fall between 0 and 1
        :param as_type: (optional) specify the type to analyse
        :param lower: (optional) the lower limit of the number value. Default min()
        :param upper: (optional) the upper limit of the number value. Default max()
        :param exclude_dominant: (optional) if overly dominant are to be excluded from analysis to avoid bias (numbers)
        :param replace_zero: (optional) with categories, a non-zero minimal chance relative frequency to replace zero
                This is useful when the relative frequency of a category is so small the analysis returns zero
        :param precision: (optional) by default set to 3.
        :param day_first: (optional) if the date provided has day first
        :param year_first: (optional) if the date provided has year first
        :param seed: (optional) the random seed. defaults to current datetime
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: an equal length list of correlated values
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_missing_weighted(seed=seed, **params)

    def correlate_numbers(self, canonical: Any, header: str, to_numeric: bool=None, standardize: bool=None,
                          normalize: tuple=None, scalarize: bool=None, transform: str=None, jitter: float=None,
                          jitter_freq: list=None, offset: [int, float, str]=None, precision: int=None, seed: int=None,
                          keep_zero: bool=None, replace_nulls: [int, float]=None, rtn_type: str=None,
                          min_value: [int, float]=None, max_value: [int, float]=None, save_intent: bool=None,
                          column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                          remove_duplicates: bool=None):
        """ Returns a number that correlates to the value given. The numbers can be standardized, normalize between
        given limits, scalarized or transformed and offers offset and jitters. The jitter is based on a normal
        distribution with the correlated value being the mean and the jitter its  standard deviation from that mean

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param to_numeric: (optional) ensures numeric type. None convertable strings are set to null
        :param standardize: (optional) if the column should be standardised
        :param normalize: (optional) normalise the column between two values. the tuple is the lower and upper bounds
        :param scalarize: (optional) assuming standard normally distributed, removes the mean and scaling
        :param transform: (optional) attempts normal distribution of values.
                            options are log, sqrt, cbrt, reciprocal, boxcox, yeojohnson
        :param transform: (optional) transform the columns options are log, sqrt, cbrt, reciprocal
        :param offset: (optional) a fixed value to offset or if str an operation to perform using @ as the header value.
        :param jitter: (optional) a perturbation of the value where the jitter is a std. defaults to 0
        :param jitter_freq: (optional)  a relative freq with the pattern mid point the mid point of the jitter
        :param precision: (optional) how many decimal places. default to 3
        :param replace_nulls: (optional) a numeric value to replace nulls
        :param seed: (optional) the random seed. defaults to current datetime
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :param keep_zero: (optional) if True then zeros passed remain zero, Default is False
        :param min_value: a minimum value not to go below
        :param max_value: a max value not to go above
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: an equal length list of correlated values

        The offset can be a numeric offset that is added to the value, e.g. passing 2 will add 2 to all values.
        If a string is passed if format should be a calculation with the '@' character used to represent the column
        value. e.g.
            '1-@' would subtract the column value from 1,
            '@*0.5' would multiply the column value by 0.5
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_numbers(seed=seed, **params)

    def correlate_categories(self, canonical: Any, header: str, correlations: list, actions: dict,
                             default_action: [str, int, float, dict]=None, rtn_type: str=None,
                             seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                             intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ correlation of a set of values to an action, the correlations must map to the dictionary index values.
        Note. to use the current value in the passed values as a parameter value pass an empty dict {} as the keys
        value. If you want the action value to be the current value of the passed value then again pass an empty dict
        action to be the current value
            simple correlation list:
                ['A', 'B', 'C'] # if values is 'A' then action is 0 and so on
            multiple choice correlation
                [['A','B'], 'C'] # if values is 'A' OR 'B' then action is 0 and so on
            actions dictionary where the method is a class method followed by its parameters
                {0: {'method': 'get_numbers', 'from_value': 0, to_value: 27}}
            you can also use the action to specify a specific value:
                {0: 'F', 1: {'method': 'get_numbers', 'from_value': 0, to_value: 27}}

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param correlations: a list of categories (can also contain lists for multiple correlations.
        :param actions: the correlated set of categories that should map to the index
        :param default_action: (optional) a default action to take if the selection is not fulfilled
        :param seed: a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal length to the one passed

        Actions are the resulting outcome of the selection (or the default). An action can be just a value or a dict
        that executes a intent method such as get_number(). To help build actions there is a helper function called
        action2dict(...) that takes a method as a mandatory attribute.

        With actions there are special keyword 'method' values:
            @header: use a column as the value reference, expects the 'header' key
            @constant: use a value constant, expects the key 'value'
            @sample: use to get sample values, expected 'name' of the Sample method, optional 'shuffle' boolean
            @eval: evaluate a code string, expects the key 'code_str' and any locals() required

        An example of a simple action to return a selection from a list:
                {'method': 'get_category', selection=['M', 'F', 'U']

        an example of using the helper method, in this example we use the keyword @header to get a value from another
        column at the same index position:
                inst.action2dict(method="@header", header='value')

        We can even execute some sort of evaluation at run time:
                inst.action2dict(method="@eval", code_str='sum(values)', values=[1,4,2,1])

        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_categories(seed=seed, **params)

    def correlate_discrete_intervals(self, canonical: Any, header: str, granularity: [int, float, list]=None,
                                     lower: [int, float]=None, upper: [int, float]=None, categories: list=None,
                                     precision: int=None, seed: int=None, save_intent: bool=None,
                                     column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                                     remove_duplicates: bool=None):
        """ converts continuous representation into discrete representation through interval categorisation

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param granularity: (optional) the granularity of the analysis across the range. Default is 3
                int passed - represents the number of periods
                float passed - the length of each interval
                list[tuple] - specific interval periods e.g []
                list[float] - the percentile or quantities, All should fall between 0 and 1
        :param lower: (optional) the lower limit of the number value. Default min()
        :param upper: (optional) the upper limit of the number value. Default max()
        :param precision: (optional) The precision of the range and boundary values. by default set to 5.
        :param categories:(optional)  a set of labels the same length as the intervals to name the categories
        :param seed: seed: (optional) a seed value for the random function: default to None
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal length to the one passed
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_discrete_intervals(seed=seed, **params)

    def correlate_dates(self, canonical: Any, header: str, offset: [int, dict]=None, jitter: int=None,
                        jitter_units: str=None, jitter_freq: list=None, now_delta: str=None, date_format: str=None,
                        min_date: str=None, max_date: str=None, fill_nulls: bool=None, day_first: bool=None,
                        year_first: bool=None, seed: int=None, rtn_type: str=None, save_intent: bool=None,
                        column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                        remove_duplicates: bool=None):
        """ correlates dates to an existing date or list of dates. The return is a list

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param offset: (optional) and offset to the date. if int then assumed a 'days' offset
                int or dictionary associated with pd. eg {'days': 1}
        :param jitter: (optional) the random jitter or deviation in days
        :param jitter_units: (optional) the units of the jitter, Options: 'W', 'D', 'h', 'm', 's'. default 'D'
        :param jitter_freq: (optional) a relative freq with the pattern mid point the mid point of the jitter
        :param now_delta: (optional) returns a delta from now as an int list, Options: 'Y', 'M', 'W', 'D', 'h', 'm', 's'
        :param min_date: (optional)a minimum date not to go below
        :param max_date: (optional)a max date not to go above
        :param fill_nulls: (optional) if no date values should remain untouched or filled based on the list mode date
        :param day_first: (optional) if the dates given are day first firmat. Default to True
        :param year_first: (optional) if the dates given are year first. Default to False
        :param date_format: (optional) the format of the output
        :param seed: (optional) a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of equal size to that given
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        return self._correlate_dates(seed=seed, **params)

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
        exclude = []
        if 'canonical' in params.keys() and not isinstance(params.get('canonical'), (str, dict, list)):
            exclude.append('canonical')
        return super()._intent_builder(method=method, params=params, exclude=exclude)

    def _set_intend_signature(self, intent_params: dict, column_name: [int, str]=None, intent_order: int=None,
                              replace_intent: bool=None, remove_duplicates: bool=None, save_intent: bool=None):
        """ sets the intent section in the configuration file. Note: by default any identical intent, e.g.
        intent with the same intent (name) and the same parameter values, are removed from any level.

        :param intent_params: a dictionary type set of configuration representing a intent section contract
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        if save_intent or (not isinstance(save_intent, bool) and self._default_save_intent):
            if not isinstance(column_name, (str, int)) or not column_name:
                raise ValueError(f"if the intent is to be saved then a column name must be provided")
        super()._set_intend_signature(intent_params=intent_params, intent_level=column_name, intent_order=intent_order,
                                      replace_intent=replace_intent, remove_duplicates=remove_duplicates,
                                      save_intent=save_intent)
        return
