from abc import abstractmethod
from copy import deepcopy
from typing import Any
import numpy as np
import pandas as pd
from aistac.handlers.abstract_handlers import HandlerFactory
from aistac.intent.abstract_intent import AbstractIntentModel
from ds_discovery.components.commons import Commons

__author__ = 'Darryl Oatridge'


class AbstractCommonsIntentModel(AbstractIntentModel):

    @classmethod
    def __dir__(cls):
        """returns the list of available methods associated with the parameterized intent"""
        rtn_list = []
        for m in dir(cls):
            if m.startswith('_get'):
                rtn_list.append(m)
            elif not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @abstractmethod
    def run_intent_pipeline(self, *args, **kwargs) -> [None, tuple]:
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.
        """

    @staticmethod
    def select2dict(column: str, condition: str, expect: str=None, logic: str=None, date_format: str=None,
                    offset: int=None) -> dict:
        """ a utility method to help build feature conditions by aligning method parameters with dictionary format.

        :param column: the column name to apply the condition to
        :param condition: the condition string (special conditions are 'date.now' for current date
        :param expect: (optional) the data type to expect. If None then the data type is assumed from the dtype
        :param logic: (optional) the logic to provide, see below for options
        :param date_format: (optional) a format of the date if only a specific part of the date and time is required
        :param offset: (optional) a time delta in days (+/-) from the current date and time (minutes not supported)
        :return: dictionary of the parameters

        logic:
            AND: the intersect of the current state with the condition result (common to both)
            NAND: outside the intersect of the current state with the condition result (not common to both)
            OR: the union of the current state with the condition result (everything in both)
            NOR: outside the union of the current state with the condition result (everything not in both)
            NOT: the difference between the current state and the condition result
            XOR: the difference between the union and the intersect current state with the condition result
        extra logic:
            ALL: the intersect of the whole index with the condition result irrelevant of level or current state index
            ANY: the intersect of the level index with the condition result irrelevant of current state index
        """
        return Commons.param2dict(**locals())

    @staticmethod
    def action2dict(method: Any, **kwargs) -> dict:
        """ a utility method to help build feature conditions by aligning method parameters with dictionary format.

        :param method: the method to execute
        :param kwargs: name value pairs associated with the method
        :return: dictionary of the parameters

        Special method values
            @header: use a column as the value reference, expects the 'header' key
            @constant: use a value constant, expects the key 'value'
            @sample: use to get sample values, expected 'name' of the Sample method, optional 'shuffle' boolean
            @eval: evaluate a code string, expects the key 'code_str' and any locals() required


        """
        return Commons.param2dict(method=method, **kwargs)

    @staticmethod
    def canonical2dict(method: Any, **kwargs) -> dict:
        """ a utility method to help build feature conditions by aligning method parameters with dictionary format.
        The method parameter can be wither a 'model_*' or 'frame_*' method with two special reserved options

        Special reserved method values
            @empty: returns an empty dataframe, optionally the key values size: int and headers: list
            @generate: generates a dataframe either from_env(task_name) o from a remote repo uri. params are
                task_name: the task name of the generator
                repo_uri: (optional) a remote repo to retrieve the the domain contract
                size: (optional) the generated sample size
                seed: (optional) if seeding should be applied the seed value
                run_book: (optional) a domain contract runbook to execute as part of the pipeline

        :param method: the method to execute
        :param kwargs: name value pairs associated with the method
        :return: dictionary of the parameters
        """
        return Commons.param2dict(method=method, **kwargs)

    """
        PRIVATE METHODS SECTION
    """

    def _apply_action(self, canonical: pd.DataFrame, action: Any, select_idx: pd.Int64Index=None, seed: int=None):
        """ applies an action returning an indexed Series
        Special method values
            @header: use a column as the value reference, expects the 'header' key
            @constant: use a value constant, expects the key 'value'
            @sample: use to get sample values, expected 'name' of the Sample method, optional 'shuffle' boolean
            @eval: evaluate a code string, expects the key 'code_str' and any locals() required

        :param canonical: a reference canonical
        :param action: the action dictionary
        :param select_idx: (optional) the index selection of the return Series. if None then canonical index taken
        :param seed: a seed to apply to any action
        :return: pandas Series with passed index
        """
        if not isinstance(select_idx, pd.Int64Index):
            select_idx = canonical.index
        if isinstance(action, dict):
            action = action.copy()
            method = action.pop('method', None)
            if method is None:
                raise ValueError(f"The action dictionary has no 'method' key.")
            if f"{method}" in self.__dir__() or f"_{method}" in self.__dir__():
                if isinstance(seed, int) and 'seed' not in action.keys():
                    action.update({'seed': seed})
                if str(method).startswith('get_'):
                    if f"{method}" in self.__dir__():
                        action.update({'size': select_idx.size, 'save_intent': False})
                        result = eval(f"self.{method}(**action)", globals(), locals())
                    else:
                        action.update({'size': select_idx.size})
                        result = eval(f"self._{method}(**action)", globals(), locals())
                elif str(method).startswith('correlate_'):
                    action.update({'canonical': canonical.iloc[select_idx], 'save_intent': False})
                    result = eval(f"self.{method}(**action)", globals(), locals())
                else:
                    raise NotImplementedError(f"The method {method} is not implemented as part of the actions")
                dtype = 'object'
                if any(isinstance(x, int) for x in result):
                    dtype = 'int'
                elif any(isinstance(x, float) for x in result):
                    dtype = 'float'
                return pd.Series(data=result, index=select_idx, dtype=dtype)
            elif str(method).startswith('@header'):
                header = action.pop('header', None)
                if header is None:
                    raise ValueError(f"The action '@header' requires a 'header' key.")
                if header not in canonical.columns:
                    raise ValueError(f"When executing the action '@header', the header {header} was not found")
                return canonical[header].iloc[select_idx]
            elif str(method).startswith('@choice'):
                header = action.pop('header', None)
                size = action.pop('size', 1)
                size = action.pop('seed', 1)
                if header is None:
                    raise ValueError(f"The action '@choice' requires a 'header' key")
                if header not in canonical.columns:
                    raise ValueError(f"When executing the action '@choice', the header {header} was not found")
                generator = np.random.default_rng(seed=seed)
                return pd.Series([np.random.choice(x) for x in canonical[header]],
                                 dtype=canonical[header].dtype.name).iloc[select_idx]
            elif str(method).startswith('@eval'):
                code_str = action.pop('code_str', None)
                if code_str is None:
                    raise ValueError(f"The action '@eval' requires a 'code_str' key.")
                e_value = eval(code_str, globals(), action)
                return pd.Series(data=([e_value] * select_idx.size), index=select_idx)
            elif str(method).startswith('@constant'):
                constant = action.pop('value', None)
                if constant is None:
                    raise ValueError(f"The action '@constant' requires a 'value' key.")
                return pd.Series(data=([constant] * select_idx.size), index=select_idx)
            else:
                raise ValueError(f"The 'method' key {method} is not a recognised intent method")
        return pd.Series(data=([action] * select_idx.size), index=select_idx)

    def _selection_index(self, canonical: pd.DataFrame, selection: list, select_idx: pd.Index=None):
        """ private method to iterate a list of selections and return the resulting index

        :param canonical: a pandas DataFrame to select from
        :param selection: the selection list of dictionaries
        :param select_idx: a starting index, if None then canonical index used
        :return:
        """
        select_idx = select_idx if isinstance(select_idx, pd.Index) else canonical.index
        sub_select_idx = select_idx
        state_idx = None
        for condition in selection:
            if isinstance(condition, str):
                condition = {'logic': condition}
            if isinstance(condition, dict):
                if not isinstance(state_idx, pd.Index):
                    state_idx = sub_select_idx
                if len(condition) == 1 and 'logic' in condition.keys():
                    if condition.get('logic') == 'ALL':
                        condition_idx = canonical.index
                    elif condition.get('logic') == 'ANY':
                        condition_idx = sub_select_idx
                    elif condition.get('logic') == 'NOT':
                        condition_idx = state_idx
                        state_idx = sub_select_idx
                    else:
                        condition_idx = state_idx
                else:
                    condition_idx = self._condition_index(canonical=canonical.iloc[sub_select_idx], condition=condition,
                                                          select_idx=sub_select_idx)
                logic = condition.get('logic', 'AND')
                state_idx = self._condition_logic(base_idx=canonical.index, sub_select_idx=sub_select_idx,
                                                  state_idx=state_idx, condition_idx=condition_idx, logic=logic)
            elif isinstance(condition, list):
                if not isinstance(state_idx, pd.Index) or len(state_idx) == 0:
                    state_idx = sub_select_idx
                state_idx = self._selection_index(canonical=canonical, selection=condition, select_idx=state_idx)
            else:
                raise ValueError(f"The subsection of the selection list {condition} is neither a dict or a list")
        return state_idx

    @staticmethod
    def _condition_logic(base_idx: pd.Index, sub_select_idx: pd.Index, state_idx: pd.Index, condition_idx: pd.Index,
                         logic: str) -> pd.Index:
        if str(logic).upper() == 'ALL':
            return base_idx.intersection(condition_idx).sort_values()
        elif str(logic).upper() == 'ANY':
            return sub_select_idx.intersection(condition_idx).sort_values()
        elif str(logic).upper() == 'AND':
            return state_idx.intersection(condition_idx).sort_values()
        elif str(logic).upper() == 'NAND':
            return sub_select_idx.drop(state_idx.intersection(condition_idx)).sort_values()
        elif str(logic).upper() == 'OR':
            return state_idx.append(state_idx.union(condition_idx)).drop_duplicates().sort_values()
        elif str(logic).upper() == 'NOR':
            result = state_idx.append(state_idx.union(condition_idx)).drop_duplicates().sort_values()
            return sub_select_idx.drop(result)
        elif str(logic).upper() == 'NOT':
            return state_idx.difference(condition_idx)
        elif str(logic).upper() == 'XOR':
            return state_idx.union(condition_idx).difference(state_idx.intersection(condition_idx))
        raise ValueError(f"The logic '{logic}' must be AND, NAND, OR, NOR, NOT, XOR ANY or ALL")

    @staticmethod
    def _condition_index(canonical: pd.DataFrame, condition: dict, select_idx: pd.Index) -> pd.Index:
        _column = condition.get('column')
        _condition = condition.get('condition')
        if _column == '@':
            _condition = str(_condition).replace("@", "canonical.iloc[select_idx]")
        elif _column not in canonical.columns:
            raise ValueError(f"The column name '{_column}' can not be found in the canonical headers.")
        else:
            _condition = str(_condition).replace("@", f"canonical['{_column}']")
        # find the selection index
        return eval(f"canonical[{_condition}].index", globals(), locals())

    def _get_canonical(self, data: [pd.DataFrame, pd.Series, list, str, dict, int], header: str=None, size: int=None,
                       deep_copy: bool=None) -> pd.DataFrame:
        """ Used to return or generate a pandas Dataframe from a number of different methods.
        The following can be passed and their returns
        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrame of one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - int -> generates an empty pd.Dataframe with an index size of the int passed.
        - dict -> use the canonical2dict(...) method to construct a dict with a method and related parameters
            methods:
                - model_*(...) -> one of the builder model methods and paramters
                - *_selection(...) -> one of the builder selection methods (get_, correlate_, frame_) and paramters
                - @empty -> generates an empty pd.DataFrame where size and headers can be passed
                    :size sets the index size of the dataframe
                    :headers any initial headers for the dataframe
                - @generate -> generate a synthetic file from a remote Domain Contract
                    :task_name the name of the SyntheticBuilder task to run
                    :repo_uri the location of the Domain Product
                    :size (optional) a size to generate
                    :seed (optional) if a seed should be applied
                    :run_book (optional) if specific intent should be run only

        :param data: a dataframe or action event to generate a dataframe
        :param header: (optional) header for pd.Series or list
        :param size: (optional) a size parameter for @empty of @generate
        :param header: (optional) used in conjunction with lists or pd.Series to give a header reference
        :return: a pd.Dataframe
        """
        deep_copy = deep_copy if isinstance(deep_copy, bool) else True
        if isinstance(data, pd.DataFrame):
            if deep_copy:
                return deepcopy(data)
            return data
        if isinstance(data, dict):
            data = data.copy()
            method = data.pop('method', None)
            if method is None:
                try:
                    return pd.DataFrame.from_dict(data=data)
                except ValueError:
                    raise ValueError("The canonical data passed was of type 'dict' but did not contain a 'method' key "
                                     "or was not convertible to Dataframe")
            if method in self.__dir__():
                if str(method).startswith('model_') or method == 'frame_selection':
                    data.update({'save_intent': False})
                    return eval(f"self.{method}(**data)", globals(), locals())
                if str(method).endswith('_selection'):
                    if not isinstance(header, str):
                        raise ValueError(f"The canonical type 'dict' method '{method}' must have a header parameter.")
                    data.update({'save_intent': False})
                    if method == 'get_selection':
                        if not isinstance(size, int):
                            raise ValueError(f"The canonical type 'dict' method '{method}' must have a size parameter.")
                        data.update({'size': size})
                    return pd.DataFrame(data=eval(f"self.{method}(**data)", globals(), locals()), columns=[header])
            elif str(method).startswith('@generate'):
                task_name = data.pop('task_name', None)
                if task_name is None:
                    raise ValueError(f"The data method '@generate' requires a 'task_name' key.")
                uri_pm_repo = data.pop('repo_uri', None)
                module = HandlerFactory.get_module(module_name='ds_discovery')
                inst = module.SyntheticBuilder.from_env(task_name=task_name, uri_pm_repo=uri_pm_repo,
                                                        default_save=False)
                size = size if isinstance(size, int) and 'size' not in data.keys() else data.pop('size', None)
                seed = data.get('seed', None)
                run_book = data.pop('run_book', None)
                result = inst.tools.run_intent_pipeline(size=size, columns=run_book, seed=seed)
                return inst.tools.frame_selection(canonical=result, save_intent=False, **data)
            elif str(method).startswith('@empty'):
                size = size if isinstance(size, int) and 'size' not in data.keys() else data.pop('size', None)
                headers = data.pop('headers', None)
                size = range(size) if size else None
                return pd.DataFrame(index=size, columns=headers)
            else:
                raise ValueError(f"The data 'method' key {method} is not a recognised intent method")
        elif isinstance(data, (list, pd.Series)):
            header = header if isinstance(header, str) else 'default'
            if deep_copy:
                data = deepcopy(data)
            return pd.DataFrame(data=data, columns=[header])
        elif isinstance(data, str):
            if not self._pm.has_connector(connector_name=data):
                if isinstance(size, int):
                    return pd.DataFrame(index=range(size))
                raise ValueError(f"The data connector name '{data}' is not in the connectors catalog")
            handler = self._pm.get_connector_handler(data)
            canonical = handler.load_canonical()
            if isinstance(canonical, dict):
                canonical = pd.DataFrame.from_dict(data=canonical)
            return canonical
        elif isinstance(data, int):
            return pd.DataFrame(index=range(data)) if data > 0 else pd.DataFrame()
        elif not data:
            return pd.DataFrame()
        raise ValueError(f"The canonical format is not recognised, pd.DataFrame, pd.Series, "
                         f"str, list or dict expected, {type(data)} passed")
