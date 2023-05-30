import ast

import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Any
import pandas.api.types as ptypes
from aistac.components.aistac_commons import DataAnalytics
from aistac.handlers.abstract_handlers import HandlerFactory

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
                         re_ignore_case: bool=None, connector_name: str=None, seed: int=None, **kwargs):
        """ Data profiling provides, analyzing, and creating useful summaries of data. The process yields a high-level
        overview which aids in the discovery of data quality issues, risks, and overall trends. It can be used to
        identify any errors, anomalies, or patterns that may exist within the data. There are three types of data
        profiling available 'dictionary', 'schema' or 'quality'

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param profiling: The profiling name. Options are 'dictionary', 'schema' or 'quality'
        :param headers: (optional) a filter of headers from the 'other' dataset
        :param drop: (optional) to drop or not drop the headers if specified
        :param dtype: (optional) a filter on data type for the 'other' dataset. int, float, bool, object
        :param exclude: (optional) to exclude or include the data types if specified
        :param regex: (optional) a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt'
        :param re_ignore_case: (optional) true if the regex should ignore case. Default is False
        :param connector_name::(optional) a connector name where the outcome is sent
        :param seed:(optional) this is a placeholder, here for compatibility across methods
        :param kwargs: if using connector_name, any kwargs to pass to the handler
        :return: pd.DataFrame
        """
        canonical = self._get_canonical(canonical)
        columns = Commons.filter_headers(canonical, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                        regex=regex, re_ignore_case=re_ignore_case)
        _seed = self._seed() if seed is None else seed
        if profiling == 'dictionary':
            pkg = HandlerFactory.get_module('ds_discovery')
            cleaned = pkg.Transition.from_memory().tools.auto_transition(canonical)
            result =  DataDiscovery.data_dictionary(df=cleaned, stylise=False, inc_next_dom=True)
        elif profiling == 'schema':
            blob = DataDiscovery.analyse_association(df=canonical, columns_list=columns)
            report = pd.DataFrame(columns=['root', 'section', 'element', 'value'])
            root_list = DataAnalytics.get_tree_roots(analytics_blob=blob)
            for root_items in root_list:
                data_analysis = DataAnalytics.from_root(analytics_blob=blob, root=root_items)
                for section in data_analysis.section_names:
                    for element, value in data_analysis.get(section).items():
                        to_append = [root_items, section, element, value]
                        a_series = pd.Series(to_append, index=report.columns)
                        report = pd.concat([report, a_series.to_frame().transpose()], ignore_index=True)
            result = report
        elif profiling == 'quality':
            result =  DataDiscovery.data_quality(df=canonical)
        else:
            raise ValueError(f"The report name '{profiling}' is not recognised. Use 'dictionary', 'schema' or 'quality'")
        if isinstance(connector_name, str):
            if self._pm.has_connector(connector_name):
                handler = self._pm.get_connector_handler(connector_name)
                handler.persist_canonical(result, **kwargs)
                return canonical
            raise ValueError(f"The connector name {connector_name} has been given but no Connect Contract added")
        # set the index
        if profiling == 'dictionary':
           result = result.set_index([result.columns[0]])
        elif profiling == 'schema':
            result = result.set_index(['root', 'section', 'element'])
        elif profiling == 'quality':
            result = result.set_index(['sections', 'elements'])
        return result

    def _model_difference(self, canonical: Any, other: Any, on_key: [str, list], drop_zero_sum: bool=None,
                          summary_connector: bool=None, flagged_connector: str=None, detail_connector: str=None,
                          unmatched_connector: str=None, seed: int=None, **kwargs):
        """returns the difference between two canonicals, joined on a common and unique key.
        The ``on_key`` parameter can be a direct reference to the canonical column header or to an environment
        variable. If the environment variable is used ``on_key`` should be set to ``"${<<YOUR_ENVIRON>>}"`` where
        <<YOUR_ENVIRON>> is the environment variable name.

        If the ``flagged connector`` parameter is used, a report flagging mismatched left data with right data
        is produced for this connector where 1 indicate a difference and 0 they are the same. By default this method
        returns this report but if this parameter is set the original canonical returned. This allows a canonical
        pipeline to continue through the component while outputting the difference report.

        If the ``detail connector`` parameter is used, a detail report of the difference where the left and right
        values that differ are shown.

        If the ``unmatched connector`` parameter is used, the on_key's that don't match between left and right are
        reported

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param other: a direct or generated pd.DataFrame. to concatenate
        :param on_key: The name of the key that uniquely joins the canonical to others
        :param drop_zero_sum: (optional) drops rows and columns which has a total sum of zero differences
        :param summary_connector: (optional) a connector name where the summary report is sent
        :param flagged_connector: (optional) a connector name where the differences are flagged
        :param detail_connector: (optional) a connector name where the differences are shown
        :param unmatched_connector: (optional) a connector name where the unmatched keys are shown
        :param seed: (optional) this is a placeholder, here for compatibility across methods
        :param kwargs: additional parameters for the connector contracts

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
        drop_zero_sum = drop_zero_sum if isinstance(drop_zero_sum, bool) else False
        flagged_connector = self._extract_value(flagged_connector)
        summary_connector = self._extract_value(summary_connector)
        detail_connector = self._extract_value(detail_connector)
        unmatched_connector = self._extract_value(unmatched_connector)
        on_key = Commons.list_formatter(self._extract_value(on_key))
        # remove not matching columns
        left_diff = Commons.list_diff(canonical.columns.to_list(), other.columns.to_list(), symmetric=False)
        right_diff = Commons.list_diff(other.columns.to_list(), canonical.columns.to_list(), symmetric=False)
        _canonical = canonical.copy().drop(left_diff, axis=1)
        _other = other.copy().drop(right_diff, axis=1)
        # sort
        _canonical.sort_values(on_key, inplace=True)
        _other.sort_values(on_key, inplace=True)
        _other = _other.loc[:, _canonical.columns.to_list()]

        # unmatched report
        if isinstance(unmatched_connector, str):
            if self._pm.has_connector(unmatched_connector):
                left_merge = pd.merge(canonical, other, on=on_key, how='left', suffixes=('', '_y'), indicator=True)
                left_merge = left_merge[left_merge['_merge'] == 'left_only']
                left_merge = left_merge[left_merge.columns[~left_merge.columns.str.endswith('_y')]]
                right_merge = pd.merge(canonical, other, on=on_key, how='right', suffixes=('_y', ''), indicator=True)
                right_merge = right_merge[right_merge['_merge'] == 'right_only']
                right_merge = right_merge[right_merge.columns[~right_merge.columns.str.endswith('_y')]]
                unmatched = pd.concat([left_merge, right_merge], axis=0, ignore_index=True)
                unmatched = unmatched.set_index(on_key, drop=True).reset_index()
                unmatched.insert(0, 'found_in', unmatched.pop('_merge'))
                handler = self._pm.get_connector_handler(unmatched_connector)
                handler.persist_canonical(unmatched, **kwargs)
            else:
                raise ValueError(f"The connector name {unmatched_connector} has been given but no Connect Contract added")

        # remove non-matching rows
        df = pd.merge(_canonical, _other, on=on_key, how='inner', suffixes=('_x', '_y'))
        df_x = df.filter(regex='(_x$)', axis=1)
        df_y = df.filter(regex='(_y$)', axis=1)
        df_x.columns = df_x.columns.str.removesuffix('_x')
        df_y.columns = df_y.columns.str.removesuffix('_y')
        # flag the differences
        diff = df_x.ne(df_y).astype(int)
        if drop_zero_sum:
            diff = diff.loc[(diff != 0).any(axis=1),(diff != 0).any(axis=0)]
        # add back the keys
        for n in range(len(on_key)):
            diff.insert(n, on_key[n], df[on_key[n]].iloc[diff.index])

        # detailed report
        if isinstance(detail_connector, str):
            if self._pm.has_connector(detail_connector):
                diff_comp = df_x.astype(str).compare(df_y.astype(str)).fillna('-')
                for n in range(len(on_key)):
                    diff_comp.insert(n, on_key[n], df[on_key[n]].iloc[diff_comp.index])
                diff_comp.columns = ['_'.join(col) for col in diff_comp.columns.values]
                diff_comp.columns = diff_comp.columns.str.replace(r'_self$', '_x', regex=True)
                diff_comp.columns = diff_comp.columns.str.replace(r'_other$', '_y', regex=True)
                diff_comp.columns = diff_comp.columns.str.replace(r'_$', '', regex=True)
                diff_comp = diff_comp.sort_values(on_key)
                diff_comp = diff_comp.reset_index(drop=True)
                handler = self._pm.get_connector_handler(detail_connector)
                handler.persist_canonical(diff_comp, **kwargs)
            else:
                raise ValueError(f"The connector name {detail_connector} has been given but no Connect Contract added")

        # summary report
        if isinstance(summary_connector, str):
            if self._pm.has_connector(summary_connector):
                summary = diff.drop(on_key, axis=1).sum().reset_index()
                summary.columns = ['Attribute', 'Summary']
                summary = summary.sort_values(['Attribute'])
                indicator = pd.merge(canonical[on_key], other[on_key], on=on_key, how='outer', indicator=True)
                count = indicator['_merge'].value_counts().to_frame().reset_index().replace('both', 'matching')
                count.columns = ['Attribute', 'Summary']
                summary = pd.concat([count, summary], axis=0)
                handler = self._pm.get_connector_handler(summary_connector)
                handler.persist_canonical(summary, **kwargs)
            else:
                raise ValueError(f"The connector name {summary_connector} has been given but no Connect Contract added")

        # flagged report
        if isinstance(flagged_connector, str):
            if self._pm.has_connector(flagged_connector):
                diff = diff.sort_values(on_key)
                diff = diff.reset_index(drop=True)
                handler = self._pm.get_connector_handler(flagged_connector)
                handler.persist_canonical(diff, **kwargs)
                return canonical
            raise ValueError(f"The connector name {flagged_connector} has been given but no Connect Contract added")

        if drop_zero_sum:
            diff = diff.sort_values(on_key)
            diff = diff.reset_index(drop=True)

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
        :param other: a direct or generated pd.DataFrame.
        :param headers: the headers to be selected from the other DataFrame
        :param replace:  (optional) assuming other is bigger than canonical, selects without replacement when True
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
