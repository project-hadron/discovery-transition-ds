import inspect
import random
import numpy as np
import pandas as pd
from typing import Any
from ds_discovery.components.commons import Commons
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

    def run_intent_pipeline(self, columns: [str, list]=None, seed: int=None, **kwargs) -> pd.DataFrame:
        """Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract. The whole run can be seeded though any parameterised seeding in the intent
        contracts will take precedence

        :param columns: (optional) a single or list of intent_level to run, if list, run in order given
        :param seed: a seed value that will be applied across the run: default to None
        :return: a pandas dataframe
        """
        df = pd.DataFrame()
        # test if there is any intent to run
        if self._pm.has_intent():
            # get the list of levels to run
            if isinstance(columns, (str, list)):
                column_names = self._pm.list_formatter(columns)
            else:
                # put all the intent in order of model, get, correlate, associate
                _model = []
                _get = []
                _correlate = []
                _frame = []
                for column in self._pm.get_intent().keys():
                    for order in self._pm.get(self._pm.join(self._pm.KEY.intent_key, column), {}):
                        for method in self._pm.get(self._pm.join(self._pm.KEY.intent_key, column, order), {}).keys():
                            if str(method).startswith('get_'):
                                _get.append(column)
                            elif str(method).startswith('model_'):
                                _model.append(column)
                            elif str(method).startswith('correlate_'):
                                if column in _get:
                                    _get.remove(column)
                                _correlate.append(column)
                            elif str(method).startswith('frame_'):
                                if column in _get:
                                    _get.remove(column)
                                _frame.append(column)
                column_names = Commons.list_unique(_get + _model + _correlate + _frame)
            for column in column_names:
                level_key = self._pm.join(self._pm.KEY.intent_key, column)
                for order in sorted(self._pm.get(level_key, {})):
                    for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                        if method in self.__dir__():
                            result = []
                            params.update(params.pop('kwargs', {}))
                            if isinstance(seed, int):
                                params.update({'seed': seed})
                            _ = params.pop('intent_creator', 'Unknown')
                            if str(method).startswith('get_'):
                                result = eval(f"self.{method}(size=size, save_intent=False, **params)",
                                              globals(), locals())
                            elif str(method).startswith('correlate_'):
                                result = eval(f"self.{method}(canonical=df, save_intent=False, **params)",
                                              globals(), locals())
                            elif str(method).startswith('model_'):
                                df = eval(f"self.{method}(canonical=df, save_intent=False, **params)",
                                          globals(), locals())
                                continue
                            elif str(method).startswith('frame_'):
                                df = eval(f"self.{method}(canonical=df, save_intent=False, **params)",
                                          globals(), locals())
                                continue
                            df[column] = result
        return df

    def frame_selection(self, canonical: Any, selection: list=None, headers: [str, list]=None, drop: bool=None,
                        dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None,
                        seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ Selects rows and/or columns changing the shape of the DatFrame. This is always run last in a pipeline
        Rows are filtered before the column filter so columns can be referenced even though they might not be included
        the final column list.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
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
        return self._frame_selection(seed=seed, **params)

    def model_iterator(self, canonical: Any, marker_col: str=None, starting_frame: str=None, selection: list=None,
                       default_action: dict=None, iteration_actions: dict=None, iter_start: int=None,
                       iter_stop: int=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                       intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ This method allows one to model repeating data subset that has some form of action applied per iteration.
        The optional marker column must be included in order to apply actions or apply an iteration marker
        An example of use might be a recommender generator where a cohort of unique users need to be selected, for
        different recommendation strategies but users can be repeated across recommendation strategy

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param marker_col: (optional) the marker column name for the action outcome. default is to not include
        :param starting_frame: (optional) a str referencing an existing connector contract name as the base DataFrame
        :param selection: (optional) a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
        :param default_action: (optional) a default action to take on all iterations. defaults to iteration value
        :param iteration_actions: (optional) a dictionary of actions where the key is a specific iteration
        :param iter_start: (optional) the start value of the range iteration default is 0
        :param iter_stop: (optional) the stop value of the range iteration default is start iteration + 1
        :param seed: (optional) this is a place holder, here for compatibility across methods
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
        return self._model_iterator(seed=seed, **params)

    def model_group(self, canonical: Any, headers: [str, list], group_by: [str, list], aggregator: str=None,
                    list_choice: int=None, list_max: int=None, drop_group_by: bool=False, seed: int=None,
                    include_weighting: bool=False, freq_precision: int=None, remove_weighting_zeros: bool=False,
                    remove_aggregated: bool=False, save_intent: bool=None, column_name: [int, str]=None,
                    intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ returns the full column values directly from another connector data source. in addition the the
        standard groupby aggregators there is also 'list' and 'set' that returns an aggregated list or set.
        These can be using in conjunction with 'list_choice' and 'list_size' allows control of the return values.
        if list_max is set to 1 then a single value is returned rather than a list of size 1.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param headers: the column headers to apply the aggregation too
        :param group_by: the column headers to group by
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        return self._model_group(seed=seed, **params)

    def model_merge(self, canonical: Any, other: [str, dict], left_on: str, right_on: str, how: str=None,
                    suffixes: tuple=None, indicator: bool=None, validate: str=None, seed: int=None,
                    save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                    replace_intent: bool=None,  remove_duplicates: bool=None) -> pd.DataFrame:
        """ returns the full column values directly from another connector data source.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param other: a direct or generated pd.DataFrame. see context notes below
        :param left_on: the canonical key column(s) to join on
        :param right_on: the merging dataset key column(s) to join on
        :param how: (optional) One of 'left', 'right', 'outer', 'inner'. Defaults to inner. See below for more detailed
                    description of each method.
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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

    def model_analysis(self, canonical: Any, analytics_model: dict, apply_bias: bool=None, seed: int=None,
                       save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                       replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ builds a set of columns based on an analysis dictionary of weighting (see analyse_association)
        if a reference DataFrame is passed then as the analysis is run if the column already exists the row
        value will be taken as the reference to the sub category and not the random value. This allows already
        constructed association to be used as reference for a sub category.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param analytics_model: the analytics model from discovery-transition-ds discovery model train
        :param apply_bias: (optional) if dominant values have been excluded, re-include to maintain bias
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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

    def correlate_selection(self, canonical: Any, selection: list, action: [str, int, float, dict],
                            default_action: [str, int, float, dict]=None, quantity: float=None, seed: int=None,
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
        :param quantity: (optional) a number between 0 and 1 presenting the percentage quantity of the data
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
        :return: value set based on the selection list and the action

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._correlate_selection(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def correlate_custom(self, canonical: Any, code_str: str, use_exec: bool=None, quantity: float=None, seed: int=None,
                         save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                         replace_intent: bool=None, remove_duplicates: bool=None, **kwargs):
        """ enacts an action on a dataFrame, returning the output of the action or the DataFrame if using exec or
        the evaluation returns None. Note that if using the input dataframe in your action, it is internally referenced
        as it's parameter name 'canonical'.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param code_str: an action on those column values
        :param use_exec: (optional) By default the code runs as eval if set to true exec would be used
        :param kwargs: a set of kwargs to include in any executable function
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._correlate_custom(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def correlate_aggregate(self, canonical: Any, headers: list, agg: str, quantity: float=None, seed: int=None,
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
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) a seed value for the random function: default to None
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._correlate_aggregate(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def correlate_choice(self, canonical: Any, header: str, list_size: int=None, random_choice: bool=None,
                         replace: bool=None, shuffle: bool=None, convert_str: bool=None, quantity: float=None,
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
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) a seed value for the random function: default to None
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._correlate_choice(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def correlate_join(self, canonical: Any, header: str, action: [str, dict], sep: str=None, quantity: float=None,
                       seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                       replace_intent: bool=None, remove_duplicates: bool=None):
        """ correlate a column and join it with the result of the action

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: an ordered list of columns to join
        :param action: (optional) a string or a single action whose outcome will be joined to the header value
        :param sep: (optional) a separator between the values
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) a seed value for the random function: default to None
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._correlate_join(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def correlate_sigmoid(self, canonical: Any, header: str, precision: int=None, quantity: float=None, seed: int=None,
                          save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                          replace_intent: bool=None, remove_duplicates: bool=None):
        """ logistic sigmoid a.k.a logit, takes an array of real numbers and transforms them to a value
        between (0,1) and is defined as
                                        f(x) = 1/(1+exp(-x)

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param precision: (optional) how many decimal places. default to 3
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) the random seed used with quantity. defaults to current datetime
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._correlate_sigmoid(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def correlate_polynomial(self, canonical: Any, header: str, coefficient: list, quantity: float=None,
                             seed: int=None, keep_zero: bool=None, save_intent: bool=None, column_name: [int, str]=None,
                             intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ creates a polynomial using the reference header values and apply the coefficients where the
        index of the list represents the degree of the term in reverse order.

                  e.g  [6, -2, 0, 4] => f(x) = 4x**3 - 2x + 6

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param coefficient: the reverse list of term coefficients
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) the random seed used with quantity. defaults to current datetime
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._correlate_polynomial(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def correlate_numbers(self, canonical: Any, header: str, offset: float=None, jitter: float=None,
                          jitter_freq: list=None, multiply_offset: bool=None, precision: int=None,
                          fill_nulls: bool=None, quantity: float=None, seed: int=None, keep_zero: bool=None,
                          min_value: [int, float]=None, max_value: [int, float]=None, save_intent: bool=None,
                          column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                          remove_duplicates: bool=None):
        """ returns a number that correlates to the value given. The jitter is based on a normal distribution
        with the correlated value being the mean and the jitter its standard deviation from that mean

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param header: the header in the DataFrame to correlate
        :param offset: (optional) how far from the value to offset. defaults to zero
        :param jitter: (optional) a perturbation of the value where the jitter is a std. defaults to 0
        :param jitter_freq: (optional)  a relative freq with the pattern mid point the mid point of the jitter
        :param multiply_offset: (optional) if true then the offset is multiplied else added
        :param precision: (optional) how many decimal places. default to 3
        :param fill_nulls: (optional) if True then fills nulls with the most common values
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) the random seed. defaults to current datetime
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._correlate_numbers(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def correlate_categories(self, canonical: Any, header: str, correlations: list, actions: dict,
                             default_action: [str, int, float, dict]=None, quantity: float=None,
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
        :param quantity: (optional) a number between 0 and 1 presenting the percentage quantity of the data
        :param seed: a seed value for the random function: default to None
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


        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._correlate_categories(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def correlate_dates(self, canonical: Any, header: str, offset: [int, dict]=None, jitter: int=None,
                        jitter_units: str=None, jitter_freq: list=None, now_delta: str=None, date_format: str=None,
                        min_date: str=None, max_date: str=None, fill_nulls: bool=None, day_first: bool=None,
                        year_first: bool=None, quantity: float=None, seed: int=None, save_intent: bool=None,
                        column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                        remove_duplicates: bool=None):
        """ correlates dates to an existing date or list of dates. The return is a list of pd

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
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param seed: (optional) a seed value for the random function: default to None
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

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._correlate_dates(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    """
        PRIVATE METHODS SECTION
    """

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

    def _set_quantity(self, selection, quantity, seed=None):
        """Returns the quantity percent of good values in selection with the rest fill"""
        if quantity == 1:
            return selection
        seed = self._seed(seed=seed)
        generator = np.random.default_rng(seed=seed)

        def replace_fill():
            """Used to run through all the possible fill options for the list type"""
            if isinstance(selection[i], float):
                selection[i] = np.nan
            elif isinstance(selection[i], str):
                selection[i] = ''
            else:
                selection[i] = None
            return

        if len(selection) < 100:
            for i in range(len(selection)):
                if generator.random() > quantity:
                    replace_fill()
        else:
            sample_count = int(round(len(selection) * (1 - quantity), 0))
            indices = random.sample(list(range(len(selection))), sample_count)
            for i in indices:
                replace_fill()
        return selection

    @staticmethod
    def _quantity(quantity: [float, int]) -> float:
        """normalises quantity to a percentate float between 0 and 1.0"""
        if not isinstance(quantity, (int, float)) or not 0 <= quantity <= 100:
            return 1.0
        if quantity > 1:
            return round(quantity / 100, 2)
        return float(quantity)