from typing import Any
import pandas as pd

class Recomender:
    def _model_iterator(self, canonical: Any, marker_col: str = None, starting_frame: str = None,
                        selection: list = None,
                        default_action: dict = None, iteration_actions: dict = None, iter_start: int = None,
                        iter_stop: int = None, seed: int = None) -> pd.DataFrame:
        """ This method allows one to model repeating data subset that has some form of action applied per iteration.
        The optional marker column must be included in order to apply actions or apply an iteration marker
        An example of use might be a recommender generator where a cohort of unique users need to be selected, for
        different recommendation strategies but users can be repeated across recommendation strategy

        :param canonical: a pd.DataFrame as the reference dataframe
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
        :return: pd.DataFrame

        The starting_frame can be a pd.DataFrame, a pd.Series, int or list, a connector contract str reference or a
        set of parameter instructions on how to generate a pd.Dataframe. the description of each is:

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
        canonical = self._get_canonical(canonical)
        rtn_frame = self._get_canonical(starting_frame)
        _seed = self._seed() if seed is None else seed
        iter_start = iter_start if isinstance(iter_start, int) else 0
        iter_stop = iter_stop if isinstance(iter_stop, int) and iter_stop > iter_start else iter_start + 1
        default_action = default_action if isinstance(default_action, dict) else 0
        iteration_actions = iteration_actions if isinstance(iteration_actions, dict) else {}
        for counter in range(iter_start, iter_stop):
            df_count = canonical.copy()
            # selection
            df_count = self._frame_selection(df_count, selection=selection, seed=_seed)
            # actions
            if isinstance(marker_col, str):
                if counter in iteration_actions.keys():
                    _action = iteration_actions.get(counter, None)
                    df_count[marker_col] = self._apply_action(df_count, action=_action, seed=_seed)
                else:
                    default_action = default_action if isinstance(default_action, dict) else counter
                    df_count[marker_col] = self._apply_action(df_count, action=default_action, seed=_seed)
            rtn_frame = pd.concat([rtn_frame, df_count], ignore_index=True)
        return rtn_frame

