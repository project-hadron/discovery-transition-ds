from abc import abstractmethod
from copy import deepcopy
import pandas as pd
from aistac.handlers.abstract_handlers import HandlerFactory
from aistac.intent.abstract_intent import AbstractIntentModel

__author__ = 'Darryl Oatridge'


class CommonsIntent(AbstractIntentModel):

    @abstractmethod
    def run_intent_pipeline(self, *args, **kwargs) -> [None, tuple]:
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.
        """

    def _get_canonical(self, data: [pd.DataFrame, pd.Series, list, str, dict], header: str=None, size: int=None,
                       deep_copy: bool=None) -> pd.DataFrame:
        """ Used to return or generate a pandas Dataframe from a number of different methods.
        The following can be passed and their returns
        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
        - dict -> use the canonical2dict(...) method to construct a dict with a method and related parameters
            methods:
                - model_*(...) -> one of the SyntheticBuilder model methods and paramters
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
            method = data.pop('method', None)
            if method is None:
                raise ValueError(f"The data dictionary has no 'method' key.")
            if str(method).startswith('@generate'):
                task_name = data.pop('task_name', None)
                if task_name is None:
                    raise ValueError(f"The data method '@generate' requires a 'task_name' key.")
                uri_pm_repo = data.pop('uri_pm_repo', None)
                module = HandlerFactory.get_module(module_name='ds_behavioral')
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
            if data == '@empty':
                return pd.DataFrame()
            if not self._pm.has_connector(connector_name=data):
                raise ValueError(f"The data connector name '{data}' is not in the connectors catalog")
            handler = self._pm.get_connector_handler(data)
            canonical = handler.load_canonical()
            if isinstance(canonical, dict):
                canonical = pd.DataFrame.from_dict(data=canonical, orient='columns')
            return canonical
        raise ValueError(f"The canonical format is not recognised, pd.DataFrame, pd.Series"
                         f"str, list or dict expected, {type(data)} passed")
