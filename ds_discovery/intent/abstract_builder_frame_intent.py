import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import Any
from ds_discovery.components.commons import Commons
from ds_discovery.intent.abstract_common_intent import AbstractCommonsIntentModel

__author__ = 'Darryl Oatridge'


class AbstractBuilderFrameIntent(AbstractCommonsIntentModel):

    @abstractmethod
    def run_intent_pipeline(self, *args, **kwargs) -> [None, tuple]:
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.
        """

    def _frame_starter(self, canonical: Any, selection: list=None, choice: int=None, headers: [str, list]=None,
                       drop: bool=None, dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                       re_ignore_case: bool=None, rename_map: dict=None, default_size: int=None,
                       seed: int=None) -> pd.DataFrame:
        """ Selects rows and/or columns changing the shape of the DatFrame. This is always run first in a pipeline
        Rows are filtered before columns are filtered so columns can be referenced even though they might not be
        included in the final column list.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param selection: a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
        :param choice: a number of rows to select, randomly selected from the index
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclusive. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt' columns
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param rename_map: a from: to dictionary of headers to rename
        :param default_size: if the canonical fails return an empty dataframe with the default index size
        :param seed: this is a place holder, here for compatibility across methods
        :return: pd.DataFrame

        The starter is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
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
        canonical = self._get_canonical(canonical, size=default_size)
        # not used but in place form method consistency
        seed = self._seed() if seed is None else seed
        if isinstance(selection, list):
            selection = selection.copy()
            # run the select logic
            select_idx = self._selection_index(canonical=canonical, selection=selection)
            canonical = canonical.iloc[select_idx].reset_index(drop=True)
        if isinstance(choice, int) and 0 < choice < canonical.shape[0]:
            gen = np.random.default_rng(seed)
            choice_idx = gen.choice(np.arange(canonical.shape[0]), replace=False, size=choice)
            canonical = canonical.iloc[choice_idx].reset_index(drop=True)
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        rtn_frame = Commons.filter_columns(canonical, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                           regex=regex, re_ignore_case=re_ignore_case)
        if isinstance(rename_map, dict):
            rtn_frame.rename(mapper=rename_map, axis='columns', inplace=True)
        return rtn_frame

    def _frame_selection(self, canonical: Any, selection: list=None, choice: int=None, headers: [str, list]=None,
                         drop: bool=None, dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                         re_ignore_case: bool=None, seed: int=None) -> pd.DataFrame:
        """ This method always runs at the end of the pipeline, unless ordered otherwise, trimming the final
        pd.DataFrame outcome.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param selection: a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
        :param choice: a number of rows to select, randomly selected from the index
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclusive. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$)' excludes '_amt' columns
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param seed: this is a place holder, here for compatibility across methods
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
        return self._frame_starter(canonical=canonical, selection=selection, choice=choice, headers=headers, drop=drop,
                                   dtype=dtype, exclude=exclude, regex=regex, re_ignore_case=re_ignore_case, seed=seed)
