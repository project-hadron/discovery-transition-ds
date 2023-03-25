import ast
from difflib import ndiff

import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Any
import pandas.api.types as ptypes
from aistac.components.aistac_commons import DataAnalytics
from ds_discovery.components.commons import Commons
from ds_discovery.components.discovery import DataDiscovery
from ds_discovery.intent.abstract_common_intent import AbstractCommonsIntentModel

__author__ = 'Darryl Oatridge'


class AbstractBuilderModelIntent(AbstractCommonsIntentModel):

    @abstractmethod
    def run_intent_pipeline(self, *args, **kwargs) -> [None, tuple]:
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.
        """

    def _model_custom(self, canonical: Any, code_str: str, seed: int = None, **kwargs):
        """ Commonly used for custom methods, takes code string that when executed changes the the canonical returning
        the modified canonical. If the method passes returns a pd.Dataframe this will be returned else the assumption is
        the canonical has been changed inplace and thus the modified canonical will be returned
        When referencing the canonical in the code_str it should be referenced either by use parameter label 'canonical'
        or the short cut '@' symbol. kwargs can also be passed into the code string but must be preceded by a '$' symbol
        for example:
            assume canonical['gender'] = ['M', 'F', 'U']
            code_str ='''
                \n@['new_gender'] = [True if x in $value else False for x in @[$header]]
                \n@['value'] = [4, 5, 6]
            '''
            where kwargs are header="'gender'" and value=['M', 'F']

        :param canonical: a pd.DataFrame as the reference dataframe
        :param code_str: an action on those column values. to reference the canonical use '@'
        :param seed: (optional) a seed value for the random function: default to None
        :param kwargs: a set of kwargs to include in any executable function
        :return: a list (optionally a pd.DataFrame
        """
        canonical = self._get_canonical(canonical)
        _seed = seed if isinstance(seed, int) else self._seed()
        local_kwargs = locals()
        for k, v in local_kwargs.pop('kwargs', {}).items():
            local_kwargs.update({k: v})
            code_str = code_str.replace(f'${k}', str(v))
        code_str = code_str.replace('@', 'canonical')
        df = exec(code_str, globals(), local_kwargs)
        if df is None:
            return canonical
        return df

    def _model_group(self, canonical: Any, group_by: [str, list], headers: [str, list] = None, regex: bool = None,
                     aggregator: str = None, list_choice: int = None, list_max: int = None,
                     drop_group_by: bool = False,
                     seed: int = None, include_weighting: bool = False, freq_precision: int = None,
                     remove_weighting_zeros: bool = False, remove_aggregated: bool = False) -> pd.DataFrame:
        """ groups a given set of headers, or all headers, using the aggregator to calculate the given headers.
        in addition the standard groupby aggregators there is also 'list' and 'set' that returns an aggregated list or
        set. These can be using in conjunction with 'list_choice' and 'list_size' allows control of the return values.
        if list_max is set to 1 then a single value is returned rather than a list of size 1.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param group_by: the column headers to group by
        :param headers: the column headers to apply the aggregation too and return
        :param regex: if the column headers is q regex
        :param aggregator: (optional) the aggregator as a function of Pandas DataFrame 'groupby' or 'list' or 'set'
        :param list_choice: (optional) used in conjunction with list or set aggregator to return a random n choice
        :param list_max: (optional) used in conjunction with list or set aggregator restricts the list to a n size
        :param drop_group_by: (optional) drops the group by headers
        :param include_weighting: (optional) include a percentage weighting column for each
        :param freq_precision: (optional) a precision for the relative_freq values
        :param remove_aggregated: (optional) if used in conjunction with the weighting then drops the aggregator column
        :param remove_weighting_zeros: (optional) removes zero values
        :param seed: (optional) this is a place holder, here for compatibility across methods
        :return: a pd.DataFrame
        """
        canonical = self._get_canonical(canonical)
        _seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=_seed)
        freq_precision = freq_precision if isinstance(freq_precision, int) else 3
        aggregator = aggregator if isinstance(aggregator, str) else 'sum'
        headers = Commons.filter_headers(canonical, regex=headers) if isinstance(regex, bool) and regex else None
        headers = Commons.list_formatter(headers) if isinstance(headers, (list,str)) else canonical.columns.to_list()
        group_by = Commons.list_formatter(group_by)
        df_sub = Commons.filter_columns(canonical, headers=headers + group_by).dropna()
        if aggregator.startswith('set') or aggregator.startswith('list'):
            df_tmp = df_sub.groupby(group_by)[headers[0]].apply(eval(aggregator)).apply(lambda x: list(x))
            df_tmp = df_tmp.reset_index()
            for idx in range(1, len(headers)):
                result = df_sub.groupby(group_by)[headers[idx]].apply(eval(aggregator)).apply(lambda x: list(x))
                df_tmp = df_tmp.merge(result, how='left', left_on=group_by, right_index=True)
            for idx in range(len(headers)):
                header = headers[idx]
                if isinstance(list_choice, int):
                    df_tmp[header] = df_tmp[header].apply(lambda x: generator.choice(x, size=list_choice))
                if isinstance(list_max, int):
                    df_tmp[header] = df_tmp[header].apply(lambda x: x[0] if list_max == 1 else x[:list_max])
            df_sub = df_tmp
        else:
            df_sub = df_sub.groupby(group_by, as_index=False).agg(aggregator)
        if include_weighting:
            df_sub['sum'] = df_sub.sum(axis=1, numeric_only=True)
            total = df_sub['sum'].sum()
            df_sub['weighting'] = df_sub['sum'].\
                apply(lambda x: round((x / total), freq_precision) if isinstance(x, (int, float)) else 0)
            df_sub = df_sub.drop(columns='sum')
            if remove_weighting_zeros:
                df_sub = df_sub[df_sub['weighting'] > 0]
            df_sub = df_sub.sort_values(by='weighting', ascending=False)
        if remove_aggregated:
            df_sub = df_sub.drop(headers, axis=1)
        if drop_group_by:
            df_sub = df_sub.drop(columns=group_by, errors='ignore')
        return df_sub

    def _model_modifier(self, canonical: Any, other: Any, targets_header: str=None, values_header: str=None,
                        modifier: str=None, precision: int=None, seed: int=None) -> pd.DataFrame:
        """Modifies a given set of target header names, within the canonical with the target value for that name. The
        aggregator indicates the type of modification to be performed. It is assumed the other DataFrame has the
        target headers as the first column and the target values as the second column, if this is not the case the
        targets_header and values_handler parameters can be used to specify the other header names.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param other: a direct or generated pd.DataFrame. see context notes below
        :param targets_header: (optional) the name of the target header where the header names are listed
        :param values_header: (optional) The name of the value header where the target values are listed
        :param modifier: (optional) how the value is to be modified. Options are 'add', 'sub', 'mul', 'div'
        :param precision: (optional) the value precision of the return values
        :param seed: (optional) this is a placeholder, here for compatibility across methods
        :return: pd.DataFrame
        """
        canonical = self._get_canonical(canonical)
        other = self._get_canonical(other, size=canonical.shape[0])
        _seed = self._seed() if seed is None else seed
        modifier = modifier if isinstance(modifier, str) and modifier in ['add', 'sub', 'mul', 'div'] else 'add'
        precision = precision if isinstance(precision, int) else 3
        targets_header = targets_header if isinstance(targets_header, str) else other.columns[0]
        values_header = values_header if isinstance(values_header, str) else other.columns[1]
        target_headers = []
        for index, row in other.iterrows():
            if row.loc[targets_header] in canonical.columns:
                if not ptypes.is_numeric_dtype(canonical[row.loc[targets_header]]):
                    raise TypeError(f"The column {row.loc[targets_header]} is not of type numeric")
                canonical[row.loc[targets_header]] = eval(f"canonical[row.loc[targets_header]].{modifier}(row.loc["
                                                          f"values_header])", globals(), locals()).round(precision)
                target_headers.append(row.loc[targets_header])
        return canonical

    def _model_merge(self, canonical: Any, other: Any, left_on: str=None, right_on: str=None, on: str=None,
                     how: str=None, headers: list=None, suffixes: tuple=None, indicator: bool=None, validate: str=None,
                     replace_nulls: bool=None, seed: int=None) -> pd.DataFrame:
        """ returns the full column values directly from another connector data source. The indicator parameter can be
        used to mark the merged items.

        :param canonical: a pd.DataFrame as the reference dataframe
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
        :return: a pd.DataFrame

        The other is a pd.DataFrame, a pd.Series, int or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrame of one column with the 'header' name or 'default' if not given
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

        """
        canonical = self._get_canonical(canonical)
        other = self._get_canonical(other, size=canonical.shape[0])
        _seed = self._seed() if seed is None else seed
        how = how if isinstance(how, str) and how in ['left', 'right', 'outer', 'inner'] else 'inner'
        replace_nulls = replace_nulls if isinstance(replace_nulls, bool) else False
        indicator = indicator if isinstance(indicator, bool) else False
        suffixes = suffixes if isinstance(suffixes, tuple) and len(suffixes) == 2 else ('', '_dup')
        # Filter on the columns
        if isinstance(headers, list):
            headers.append(right_on if isinstance(right_on, str) else on)
            other = Commons.filter_columns(other, headers=headers)
        df_rtn = canonical.merge(right=other, how=how, left_on=left_on, right_on=right_on, on=on,
                          suffixes=suffixes, indicator=indicator, validate=validate)
        if replace_nulls:
            for column in df_rtn.columns.to_list():
                Commons.fillna(df_rtn[column])
        return df_rtn

    def _model_profiling(self, canonical: Any, profiling: str, headers: [str, list]=None, drop: bool=None,
                         dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                         re_ignore_case: bool=None, seed: int=None):
        """ Data profiling provides, analyzing, and creating useful summaries of data. The process yields a high-level
        overview which aids in the discovery of data quality issues, risks, and overall trends. It can be used to
        identify any errors, anomalies, or patterns that may exist within the data. There are three types of data
        profiling available 'canonical', 'schema' or 'quality'

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param profiling: The profiling name. Options are 'canonical', 'schema' or 'quality'
        :param headers: (optional) a filter of headers from the 'other' dataset
        :param drop: (optional) to drop or not drop the headers if specified
        :param dtype: (optional) a filter on data type for the 'other' dataset. int, float, bool, object
        :param exclude: (optional) to exclude or include the data types if specified
        :param regex: (optional) a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt'
        :param re_ignore_case: (optional) true if the regex should ignore case. Default is False
        :param seed:(optional) this is a placeholder, here for compatibility across methods
        :return: pd.DataFrame
        """
        canonical = self._get_canonical(canonical)
        columns = Commons.filter_headers(canonical, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                        regex=regex, re_ignore_case=re_ignore_case)
        _seed = self._seed() if seed is None else seed
        if profiling == 'canonical':
            return DataDiscovery.data_dictionary(df=canonical, stylise=False, inc_next_dom=True)
        if profiling == 'schema':
            blob = DataDiscovery.analyse_association(df=canonical, columns_list=columns)
            df = pd.DataFrame(columns=['root', 'section', 'element', 'value'])
            root_list = DataAnalytics.get_tree_roots(analytics_blob=blob)
            for root_items in root_list:
                data_analysis = DataAnalytics.from_root(analytics_blob=blob, root=root_items)
                for section in data_analysis.section_names:
                    for element, value in data_analysis.get(section).items():
                        to_append = [root_items, section, element, value]
                        a_series = pd.Series(to_append, index=df.columns)
                        df = pd.concat([df, a_series.to_frame().transpose()], ignore_index=True)
            return df
        if profiling == 'quality':
            return DataDiscovery.data_quality(df=canonical)
        raise ValueError(f"The report name '{profiling}' is not recognised. Use 'canonical', 'schema' or 'quality'")

    def _model_difference(self, canonical: Any, other: Any, on_key: str, drop_no_diff: bool=None,
                          index_on_key: bool=None, seed: int=None):
        """returns the difference, by Levenshtein distance, between two canonicals, joined on a common and unique key.
        The ``on_key`` parameter can be a direct reference to the canonical column header or to an environment variable.
        If the environment variable is used ``on_key`` should be set to ``"${<<YOUR_ENVIRON>>}"`` where
        <<YOUR_ENVIRON>> is the environment variable name

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param other: a direct or generated pd.DataFrame. to concatenate
        :param on_key: The name of the key that uniquely joins the canonical to others
        :param drop_no_diff: (optional) drops columns with no difference
        :param index_on_key: (optional) set the index to be the key
        :param seed: (optional) this is a placeholder, here for compatibility across methods

        The other is a pd.DataFrame, a pd.Series, int or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrame of one column with the 'header' name or 'default' if not given
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

        """
        canonical = self._get_canonical(canonical)
        other = self._get_canonical(other, size=canonical.shape[0])
        _ = seed if isinstance(seed, int) else self._seed()
        drop_no_diff = drop_no_diff if isinstance(drop_no_diff, bool) else False
        index_on_key = index_on_key if isinstance(index_on_key, bool) else False
        on_key = self._extract_value(on_key)
        canonical.sort_values(on_key, inplace=True)
        other.sort_values(on_key, inplace=True)
        # concat
        df = pd.concat([canonical, other])
        df = df.reset_index(drop=True)
        # group by
        grouped = df.groupby(list(df.columns))
        # get index of unique records
        idx = [x[0] for x in grouped.groups.values() if len(x) == 1]
        df = df.reindex(idx)
        # ensure union of the ket

        def levenshtein_distance(str1, str2, ):
            counter = {"+": 0, "-": 0}
            distance = 0
            for edit_code, *_ in ndiff(str1, str2):
                if edit_code == " ":
                    distance += max(counter.values())
                    counter = {"+": 0, "-": 0}
                else:
                    counter[edit_code] += 1
            distance += max(counter.values())
            return distance

        # get the distance between differences
        diff = pd.DataFrame()
        for idx in range(0, df.shape[0], 2):
            line = pd.Series(data=0, index=df.iloc[idx].index, dtype='int')
            try:
                for index, value in df.iloc[idx].items():
                    if index == on_key:
                        line.at[index] = value
                    else:
                        line.at[index] = levenshtein_distance(str(value), str(df.iloc[idx + 1].loc[index]))
            except IndexError:
                continue
            diff = pd.concat([diff, line], axis=1)
        diff = diff.T.reset_index(drop=True)
        # set the index to the key
        if index_on_key:
            diff = diff.set_index(on_key)
        # drop zeros
        if drop_no_diff:
            return diff.loc[:, (diff != 0).any(axis=0)]
        return diff

    def _model_concat(self, canonical: Any, other: Any, as_rows: bool=None, headers: [str, list]=None,
                      drop: bool=None, dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                      re_ignore_case: bool=None, shuffle: bool=None, seed: int=None) -> pd.DataFrame:
        """ returns the full column values directly from another connector data source.

        :param canonical: a pd.DataFrame as the reference dataframe
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
        :return: a pd.DataFrame

        The other is a pd.DataFrame, a pd.Series, int or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrame of one column with the 'header' name or 'default' if not given
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

        """
        canonical = self._get_canonical(canonical)
        other = self._get_canonical(other, size=canonical.shape[0])
        _seed = self._seed() if seed is None else seed
        shuffle = shuffle if isinstance(shuffle, bool) else False
        as_rows = as_rows if isinstance(as_rows, bool) else False
        # Filter on the columns
        df_rtn = Commons.filter_columns(df=other, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                        regex=regex, re_ignore_case=re_ignore_case, copy=False)
        if shuffle:
            df_rtn.sample(frac=1, random_state=_seed).reset_index(drop=True)
        if canonical.shape[0] <= df_rtn.shape[0]:
            df_rtn = df_rtn.iloc[:canonical.shape[0]]
        axis = 'index' if as_rows else 'columns'
        ignore_index = True if axis == 'index' else False
        return pd.concat([canonical, df_rtn], axis=axis, join='outer', ignore_index=ignore_index)

    def _model_dict_column(self, canonical: Any, header: str, convert_str: bool=None, replace_null: Any=None,
                           seed: int=None) -> pd.DataFrame:
        """ takes a column that contains dict and expands them into columns. Note, the column must be a flat dictionary.
        Complex structures will not work.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header of the column to be convert
        :param convert_str: (optional) if the header has the dict as a string convert to dict using ast.literal_eval()
        :param replace_null: (optional) after conversion, replace null values with this value
        :param seed: (optional) this is a place holder, here for compatibility across methods
        :return: pd.DataFrame
        """
        canonical = self._get_canonical(canonical)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        _seed = self._seed() if seed is None else seed
        convert_str = convert_str if isinstance(convert_str, bool) else False
        # replace NaN with '{}' if the column is strings, otherwise replace with {}
        if convert_str:
            canonical[header] = canonical[header].fillna('{}').apply(ast.literal_eval)
        else:
            canonical[header] = canonical[header].fillna({i: {} for i in canonical.index})
        # convert the key/values into columns (this is the fasted code)
        result = pd.json_normalize(canonical[header])
        if isinstance(replace_null, (int, float, str)):
            result.replace(np.nan, replace_null, inplace=True)
        return canonical.join(result).drop(columns=[header])

    def _model_explode(self, canonical: Any, header: str, seed: int=None) -> pd.DataFrame:
        """ takes a single column of list values and explodes the DataFrame so row is represented by each elements
        in the row list

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header of the column to be exploded
        :param seed: (optional) this is a placeholder, here for compatibility across methods
        :return: a pd.DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:
        """
        canonical = self._get_canonical(canonical)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        _seed = self._seed() if seed is None else seed
        return canonical.explode(column=header, ignore_index=True)

    def _model_sample(self, canonical: Any, other: Any, headers: list, replace: bool=None,
                      relative_freq: list=None, seed: int=None) -> pd.DataFrame:
        """ Takes a target dataset and samples from that target to the size of the canonical

        :param canonical: a pd.DataFrame as the reference dataframe
        :param other: a direct or generated pd.DataFrame. see context notes below
        :param headers: the headers to be selected from the other DataFrame
        :param replace: assuming other is bigger than canonical, selects without replacement when True
        :param relative_freq: (optional) a weighting pattern that does not have to add to 1
        :param seed: (optional) a seed value for the random function: default to None
        :return: a pd.DataFrame
        """
        canonical = self._get_canonical(canonical)
        other = self._get_canonical(other)
        headers = headers if isinstance(headers, list) else list(other.columns)
        replace = replace if isinstance(replace, bool) else True
        # build the distribution sizes
        if isinstance(relative_freq, list) and len(relative_freq) > 1:
            relative_freq = self._freq_dist_size(relative_freq=relative_freq, size=other.shape[0], seed=seed)
        else:
            relative_freq = None
        seed = self._seed() if seed is None else seed
        other = Commons.filter_columns(other, headers=headers)
        other = other.sample(n=canonical.shape[0], weights=relative_freq, random_state=seed, ignore_index=True,
                             replace=replace)
        return pd.concat([canonical, other], axis=1)

    # def _model_sample_data(self, canonical: Any, other: Any, seed: int=None) -> pd.DataFrame:
    #     """"""
    #     canonical = self._get_canonical(canonical)
    #     other = self._get_canonical(other)
    #     seed = self._seed() if seed is None else seed
    #     rng = np.random.default_rng()
    #     for header in other.columns:
    #         if other['header'].dtype.name == 'category' or (other['header'].dtype.name == 'object' and
    #                                                         other['header'].nunique() < 99):
    #             vc = other['header'].value_counts()
    #             result = self._get_category(selection=vc.index.to_list(),
    #                                         relative_freq=vc.to_list(),
    #                                         seed=seed, size=canonical.shape[0])
    #         if str(other['header'].dtype).startswith('float') or str(other['header'].dtype).startswith('int'):
    #             sample = rng.choice(other['header'].dropna(), 30, replace=False)
    #             precision = max([Commons.precision_scale(x)[1] for x in sample])
    #             choice = rng.choice(other['header'], canonical.shape[0]-int(canonical.shape[0] * 0.1))
    #             jitter_results =
    #
    #
    #             corr_result = self._correlate_numbers(other, header='header', precision=precision, jitter=0.5)
    #             diff_size = canonical.shape[0] - len(corr_result)
    #             noise = rng.normal()
    #             pd.concat([corr_result, result], axis=0)

    # def _model_analysis(self, canonical: Any, other: Any, columns_list: list=None, exclude_associate: list=None,
    #                     detail_numeric: bool=None, strict_typing: bool=None, category_limit: int=None,
    #                     seed: int=None) -> pd.DataFrame:
    #     """ builds a set of columns based on an other (see analyse_association)
    #     if a reference DataFrame is passed then as the analysis is run if the column already exists the row
    #     value will be taken as the reference to the sub category and not the random value. This allows already
    #     constructed association to be used as reference for a sub category.
    #
    #     :param canonical: a pd.DataFrame as the reference dataframe
    #     :param other: a direct or generated pd.DataFrame. see context notes below
    #     :param columns_list: (optional) a list structure of columns to select for association
    #     :param exclude_associate: (optional) a list of dot separated tree of items to exclude from iteration
    #             (e.g. ['age.gender.salary']
    #     :param detail_numeric: (optional) as a default, if numeric columns should have detail stats, slowing analysis
    #     :param strict_typing: (optional) stops objects and string types being seen as categories
    #     :param category_limit: (optional) a global cap on categories captured. zero value returns no limits
    #     :param seed: seed: (optional) a seed value for the random function: default to None
    #     :return: a DataFrame
    #
    #     The other is a pd.DataFrame, a pd.Series, int or list, a connector contract str reference or a set of
    #     parameter instructions on how to generate a pd.Dataframe. the description of each is:
    #
    #     - pd.Dataframe -> a deep copy of the pd.DataFrame
    #     - pd.Series or list -> creates a pd.DataFrame of one column with the 'header' name or 'default' if not given
    #     - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
    #     - int -> generates an empty pd.Dataframe with an index size of the int passed.
    #     - dict -> use canonical2dict(...) to help construct a dict with a 'method' to build a pd.DataFrame
    #         methods:
    #             - model_*(...) -> one of the SyntheticBuilder model methods and parameters
    #             - @empty -> generates an empty pd.DataFrame where size and headers can be passed
    #                 :size sets the index size of the dataframe
    #                 :headers any initial headers for the dataframe
    #             - @generate -> generate a synthetic file from a remote Domain Contract
    #                 :task_name the name of the SyntheticBuilder task to run
    #                 :repo_uri the location of the Domain Product
    #                 :size (optional) a size to generate
    #                 :seed (optional) if a seed should be applied
    #                 :run_book (optional) if specific intent should be run only
    #
    #     """
    #
    #     def get_level(analysis: dict, sample_size: int, _seed: int=None):
    #         _seed = self._seed(seed=_seed, increment=True)
    #         for name, values in analysis.items():
    #             if row_dict.get(name) is None:
    #                 row_dict[name] = list()
    #             _analysis = DataAnalytics(analysis=values.get('insight', {}))
    #             result_type = object
    #             if str(_analysis.intent.dtype).startswith('cat'):
    #                 result_type = 'category'
    #                 result = self._get_category(selection=_analysis.intent.categories,
    #                                             relative_freq=_analysis.patterns.get('relative_freq', None),
    #                                             seed=_seed, size=sample_size)
    #             elif str(_analysis.intent.dtype).startswith('num'):
    #                 result_type = 'int' if _analysis.params.precision == 0 else 'float'
    #                 result = self._get_intervals(intervals=[tuple(x) for x in _analysis.intent.intervals],
    #                                              relative_freq=_analysis.patterns.get('relative_freq', None),
    #                                              precision=_analysis.params.get('precision', None),
    #                                              seed=_seed, size=sample_size)
    #             elif str(_analysis.intent.dtype).startswith('date'):
    #                 result_type = 'object' if _analysis.params.is_element('data_format') else 'date'
    #                 result = self._get_datetime(start=_analysis.stats.lowest,
    #                                             until=_analysis.stats.highest,
    #                                             relative_freq=_analysis.patterns.get('relative_freq', None),
    #                                             date_format=_analysis.params.get('data_format', None),
    #                                             day_first=_analysis.params.get('day_first', None),
    #                                             year_first=_analysis.params.get('year_first', None),
    #                                             seed=_seed, size=sample_size)
    #             else:
    #                 result = []
    #             # if the analysis was done with excluding dominance then se if they should be added back
    #             if _analysis.patterns.is_element('dominant_excluded'):
    #                 _dom_percent = _analysis.patterns.dominant_percent/100
    #                 _dom_values = _analysis.patterns.dominant_excluded
    #                 if len(_dom_values) > 0:
    #                     s_values = pd.Series(result, dtype=result_type)
    #                     non_zero = s_values[~s_values.isin(_dom_values)].index
    #                     choice_size = int((s_values.size * _dom_percent) - (s_values.size - len(non_zero)))
    #                     if choice_size > 0:
    #                         generator = np.random.default_rng(_seed)
    #                         _dom_choice = generator.choice(_dom_values, size=choice_size)
    #                         s_values.iloc[generator.choice(non_zero, size=choice_size, replace=False)] = _dom_choice
    #                         result = s_values.to_list()
    #             # now add the result to the row_dict
    #             row_dict[name] += result
    #             if sum(_analysis.patterns.relative_freq) == 0:
    #                 unit = 0
    #             else:
    #                 unit = sample_size / sum(_analysis.patterns.relative_freq)
    #             if values.get('sub_category'):
    #                 leaves = values.get('branch', {}).get('leaves', {})
    #                 for idx in range(len(leaves)):
    #                     section_size = int(round(_analysis.patterns.relative_freq[idx] * unit, 0)) + 1
    #                     next_item = values.get('sub_category').get(leaves[idx])
    #                     get_level(next_item, section_size, _seed)
    #         return
    #
    #     canonical = self._get_canonical(canonical)
    #     other = self._get_canonical(other)
    #     columns_list = columns_list if isinstance(columns_list, list) else other.columns.to_list()
    #     blob = DataDiscovery.analyse_association(other, columns_list=columns_list, exclude_associate=exclude_associate,
    #                                              detail_numeric=detail_numeric, strict_typing=strict_typing,
    #                                              category_limit=category_limit)
    #     row_dict = dict()
    #     seed = self._seed() if seed is None else seed
    #     size = canonical.shape[0]
    #     get_level(blob, sample_size=size, _seed=seed)
    #     for key in row_dict.keys():
    #         row_dict[key] = row_dict[key][:size]
    #     return pd.concat([canonical, pd.DataFrame.from_dict(data=row_dict)], axis=1)

    def _model_missing_cca(self, canonical: Any, threshold: float=None, seed: int=None) -> pd.DataFrame:
        """ Applies Complete Case Analysis to the canonical. Complete-case analysis (CCA), also called "list-wise
        deletion" of cases, consists of discarding observations with any missing values. In other words, we only keep
        observations with data on all the variables. CCA works well when the data is missing completely at random.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param threshold: (optional) a null threshold between 0 and 1 where 1 is all nulls. Default to 0.5
        :param seed: (optional) a placeholder
        :return: pd.DataFrame
        """
        # intend code block on the canonical
        canonical = self._get_canonical(canonical)
        threshold = threshold if isinstance(threshold, float) and 0 < threshold < 1 else 0.5
        _ = self._seed() if seed is None else seed
        vars_cca = [var for var in canonical if 0.0 < canonical[var].isnull().mean() < threshold]
        return canonical.dropna(subset=vars_cca)

    def _model_drop_outliers(self, canonical: Any, header: str, method: str=None, measure: [int, float]=None,
                             seed: int=None):
        """ Drops rows in the canonical where the values are deemed outliers based on the method and measure.
        There are two selectable methods of choice, interquartile or empirical, of which interquartile
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
        :return: list
        """
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        _seed = seed if isinstance(seed, int) else self._seed()
        measure = measure if isinstance(measure, (int, float)) else 1.5
        method = method if isinstance(method, str) else 'interquartile'
        if method.startswith('emp'):
            result_idx = DataDiscovery.empirical_outliers(values=s_values, std_width=measure)
        elif method.startswith('int') or method.startswith('irq'):
            result_idx = DataDiscovery.interquartile_outliers(values=s_values, k_factor=measure)
        else:
            raise ValueError(f"The method '{method}' is not recognised. Please use one of interquartile or empirical")
        canonical.drop(result_idx[0] + result_idx[1], inplace=True)
        return canonical.reset_index(drop=True)

    # convert objects to categories
    def _model_to_category(self, canonical: Any, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                           exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None, seed: int=None):
        """ converts columns to categories

        :param canonical: a pd.DataFrame as the reference dataframe
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param seed: (optional) a placeholder
        :return: pandas.DataFrame.
        """
        # Code block for intent
        canonical = self._get_canonical(canonical)
        _ = self._seed() if seed is None else seed
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        obj_cols = Commons.filter_headers(canonical, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                          regex=regex, re_ignore_case=re_ignore_case)
        for c in obj_cols:
            if not all(canonical[c].astype(str).str.isnumeric()):
                canonical[c] = canonical[c].astype(str).str.strip()
            canonical[c] = canonical[c].astype('category')
        return canonical

    # convert objects to categories
    def _model_to_numeric(self, canonical: Any, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                          exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None, precision: int=None,
                          seed: int=None):
        """ converts columns to numeric form

        :param canonical: a pd.DataFrame as the reference dataframe
        :param headers: (optional) a list of headers to drop or filter on type
        :param drop: (optional) to drop or not drop the headers
        :param dtype: (optional) the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: (optional) to exclude or include the dtypes
        :param regex: (optional) a regular expression to search the headers
        :param re_ignore_case: (optional) true if the regex should ignore case. Default is False
        :param precision: (optional) an int value of the precision for the float
        :param seed: (optional) a placeholder
        :return: pandas.DataFrame.
        """
        # Code block for intent
        canonical = self._get_canonical(canonical)
        precision = precision if isinstance(precision, int) else 15
        _ = self._seed() if seed is None else seed
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        obj_cols = Commons.filter_headers(canonical, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                          regex=regex, re_ignore_case=re_ignore_case)
        for c in obj_cols:
            if canonical[c].dtype == bool:
                canonical[c] = canonical[c].replace({True: 1, False: 0})
            canonical[c] = pd.to_numeric(canonical[c], errors='coerce').round(precision)
        return canonical

    def _model_encode_integer(self, canonical: Any, headers: [str, list], ranking: list=None, prefix :str=None,
                              seed: int=None):
        """ Integer encoding replaces the categories by digits from 1 to n, where n is the number of distinct
        categories of the variable. Integer encoding can be either nominal or orinal.

        Nominal data is categorical variables without any particular order between categories. This means that
        the categories cannot be sorted and there is no natural order between them.

        Ordinal data represents categories with a natural, ordered relationship between each category. This means
        that the categories can be sorted in either ascending or descending order. In order to encode integers as
        ordinal, a ranking must be provided.

        If ranking is given, the return will be ordinal values based on the ranking order of the list. If a
        categorical value is not found in the list it is grouped with other missing values and given the last
        ranking.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param headers: the header(s) to apply the encoding
        :param ranking: (optional) if used, ranks the categorical values to the list given
        :param prefix: (optional) a str to prefix the column
        :param seed: seed: (optional) a seed value for the random function: default to None
        :return: a pd.Dataframe
        """
        # intend code block on the canonical
        canonical = self._get_canonical(canonical)
        headers = Commons.list_formatter(headers)
        _ = self._seed() if seed is None else seed
        for header in headers:
            rank = ranking if isinstance(ranking, list) else canonical[header].unique().tolist()
            missing = Commons.list_diff(canonical[header].unique().tolist(), rank, symmetric=False)
            full_rank = rank + missing
            values = np.arange(len(rank)).tolist()
            values = values + ([len(rank)] * (len(full_rank) - len(rank)))
            mapper = dict(zip(full_rank, values))
            canonical[header] =  canonical[header].replace(mapper)
            if isinstance(prefix, str):
                canonical[header].rename(f"{prefix}{header}")
        return canonical

    def _model_encode_count(self, canonical: Any, headers: [str, list], prefix :str=None, seed: int=None):
        """ encodes categorical data types, In count encoding we replace the categories by the count of the
        observations that show that category in the dataset. This techniques capture's the representation of each label
        in a dataset, but the encoding may not necessarily be predictive of the outcome.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param headers: the header(s) to apply the encoding
        :param prefix: a str to prefix the column
        :param seed: seed: (optional) a seed value for the random function: default to None
        :return: a pd.Dataframe
        """
        # intend code block on the canonical
        canonical = self._get_canonical(canonical)
        headers = Commons.list_formatter(headers)
        _ = self._seed() if seed is None else seed
        for header in headers:
            count_map = canonical[header].value_counts().to_dict()
            canonical[header] = canonical[header].map(count_map)
            canonical[header] = canonical[header].fillna(-1)
            canonical[header] = canonical[header].astype(int)
            if isinstance(prefix, str):
                canonical[header].rename(f"{prefix}{header}")
        return canonical

    def _model_encode_one_hot(self, canonical: Any, headers: [str, list], prefix=None, dtype: Any=None,
                              prefix_sep: str=None, dummy_na: bool=False, drop_first: bool=False, seed: int=None):
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
        :return: a pd.Dataframe
        """
        # intend code block on the canonical
        canonical = self._get_canonical(canonical)
        headers = Commons.list_formatter(headers)
        _ = self._seed() if seed is None else seed
        prefix_sep = prefix_sep if isinstance(prefix_sep, str) else "_"
        dummy_na = dummy_na if isinstance(dummy_na, bool) else False
        drop_first = drop_first if isinstance(drop_first, bool) else False
        dtype = dtype if dtype else np.uint8
        return pd.get_dummies(canonical, columns=headers, prefix=prefix, prefix_sep=prefix_sep,
                              dummy_na=dummy_na, drop_first=drop_first, dtype=dtype)

    # def _model_encode_woe(self, canonical: Any, headers: [str, list], target: str=None, prefix=None, seed: int=None):
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
    #     :return: a pd.Dataframe
    #     """
    #     # intend code block on the canonical
    #     canonical = self._get_canonical(canonical)
    #     headers = Commons.list_formatter(headers)
    #     seed = self._seed() if seed is None else seed
    #     for header in headers:
    #         # total survivors
    #         total_survived = canonical[target].sum()
    #         # percentage of passenges who survived, from total survivors per category of cabin
    #         survived = canonical.groupby([header])[target].sum() / total_survived
    #         # total passengers who did not survive
    #         total_non_survived = len(canonical) - canonical[target].sum()
    #         # let's create a flag for passenges who did not survive
    #         canonical['non_target'] = np.where(canonical[target] == 1, 0, 1)
    #         # now let's calculate the % of passengers who did not survive per category of cabin
    #         non_survived = canonical.groupby([header])['non_target'].sum() / total_non_survived
    #         # let's concatenate the series in a dataframe
    #         prob_df = pd.concat([survived, non_survived], axis=1)
    #         # let's calculate the Weight of Evidence
    #         prob_df.replace(to_replace=0, value=1, inplace=True)
    #         prob_df['woe'] = np.log(prob_df[target] / prob_df['non_target'])
    #         print(prob_df)
    #         # and now let's capture the woe in a dictionary
    #         ordered_labels = prob_df['woe'].to_dict()
    #         # now, we replace the labels with the woe
    #         canonical[header] = canonical[header].map(ordered_labels)
    #
    #         canonical = canonical.drop('non_target', axis=1)
    #
    #         if isinstance(prefix, str):
    #             canonical[header].rename(f"{prefix}{header}")
    #     return canonical
