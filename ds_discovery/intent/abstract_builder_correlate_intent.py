import ast
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Any
import pandas.api.types as ptypes
from scipy import stats
from sklearn.impute import KNNImputer
from ds_discovery.components.commons import Commons
from ds_discovery.components.discovery import DataDiscovery
from ds_discovery.intent.abstract_common_intent import AbstractCommonsIntentModel

__author__ = 'Darryl Oatridge'


class AbstractBuilderCorrelateIntent(AbstractCommonsIntentModel):

    @abstractmethod
    def run_intent_pipeline(self, *args, **kwargs) -> [None, tuple]:
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.
        """

    def _correlate_selection(self, canonical: Any, selection: list, action: [str, int, float, dict],
                             default_action: [str, int, float, dict]=None, seed: int=None, rtn_type: str=None):
        """ returns a value set based on the selection list and the action enacted on that selection. If
        the selection criteria is not fulfilled then the default_action is taken if specified, else null value.

        If a DataFrame is not passed, the values column is referenced by the header '_default'

        :param canonical: a pd.DataFrame as the reference dataframe
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
        canonical = self._get_canonical(canonical)
        if len(canonical) == 0:
            raise TypeError("The canonical given is empty")
        if not isinstance(selection, list):
            raise ValueError("The 'selection' parameter must be a 'list' of 'dict' types")
        if not isinstance(action, (str, int, float, dict)) or (isinstance(action, dict) and len(action) == 0):
            raise TypeError("The 'action' parameter is not of an accepted format or is empty")
        _seed = seed if isinstance(seed, int) else self._seed()
        # run the logic
        select_idx = self._selection_index(canonical=canonical, selection=selection)
        if not isinstance(default_action, (str, int, float, dict)):
            default_action = None
        rtn_values = self._apply_action(canonical, action=default_action, seed=_seed)
        # deal with categories
        is_category = False
        if rtn_values.dtype.name == 'category':
            is_category = True
            rtn_values = rtn_values.astype('object')
        rtn_values.update(self._apply_action(canonical, action=action, select_idx=select_idx, seed=_seed))
        if is_category:
            rtn_values = rtn_values.astype('category')
        if isinstance(rtn_type, str):
            if rtn_type in ['category', 'object'] or rtn_type.startswith('int') or rtn_type.startswith('float'):
                rtn_values = rtn_values.astype(rtn_type)
            return rtn_values
        return rtn_values.to_list()

    def _correlate_date_diff(self, canonical: Any, first_date: str, second_date: str, units: str=None,
                             precision: int=None, seed: int=None) -> list:
        """ returns a column for the difference between a primary and secondary date where the primary is an early date
        than the secondary.

        :param canonical: the DataFrame containing the column headers
        :param first_date: the primary or older date field
        :param second_date: the secondary or newer date field
        :param units: (optional) The Timedelta units e.g. 'D', 'W', 'M', 'Y'. default is 'D'
        :param precision: (optional) the precision of the result
        :param seed:  (optional) aplace holder
        :return: the DataFrame with the extra column
        """
        # Code block for intent
        if second_date not in canonical.columns:
            raise ValueError(f"The column header '{second_date}' is not in the canonical DataFrame")
        if first_date not in canonical.columns:
            raise ValueError(f"The column header '{first_date}' is not in the canonical DataFrame")
        canonical = self._get_canonical(canonical)
        _seed = seed if isinstance(seed, int) else self._seed()
        precision = precision if isinstance(precision, int) else 0
        units = units if isinstance(units, str) else 'D'
        selected = canonical[[first_date, second_date]]
        rename = (selected[second_date].sub(selected[first_date], axis=0) / np.timedelta64(1, units))
        return [np.round(v, precision) for v in rename]

    def _correlate_custom(self, canonical: Any, code_str: str, seed: int=None, **kwargs):
        """ Commonly used for custom list comprehension, takes code string that when evaluated returns a list of values
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
        rtn_values = eval(code_str, globals(), local_kwargs)
        if rtn_values is None:
            return [np.nan] * canonical.shape[0]
        return rtn_values

    def _correlate_aggregate(self, canonical: Any, headers: list, agg: str, seed: int=None, precision: int=None,
                             rtn_type: str=None):
        """ correlate two or more columns with each other through a finite set of aggregation functions. The
        aggregation function names are limited to 'sum', 'prod', 'count', 'min', 'max' and 'mean' for numeric columns
        and a special 'list' function name to combine the columns as a list

        :param canonical: a pd.DataFrame as the reference dataframe
        :param headers: a list of headers to correlate
        :param agg: the aggregation function name to enact. The available functions are:
                        'sum', 'prod', 'count', 'min', 'max', 'mean' and 'list' which combines the columns as a list
        :param precision: the value precision of the return values
        :param seed: (optional) a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :return: a list of equal length to the one passed
        """
        canonical = self._get_canonical(canonical)
        if not isinstance(headers, list) or len(headers) < 2:
            raise ValueError("The headers value must be a list of at least two header str")
        agg_choice = ['sum', 'prod', 'count', 'min', 'max', 'mean']
        if agg not in ['sum', 'prod', 'count', 'min', 'max', 'mean', 'list']:
            raise ValueError("The only allowed func values are 'sum', 'prod', 'count', 'min', 'max', 'mean', 'list'")
        # Code block for intent
        _seed = seed if isinstance(seed, int) else self._seed()
        precision = precision if isinstance(precision, int) else 3
        if agg == 'list':
            return canonical.loc[:, headers].values.tolist()
        rtn_values = eval(f"canonical.loc[:, headers].{agg}(axis=1)", globals(), locals()).round(precision)
        if isinstance(rtn_type, str):
            if rtn_type in ['category', 'object'] or rtn_type.startswith('int') or rtn_type.startswith('float'):
                rtn_values = rtn_values.astype(rtn_type)
            return rtn_values
        return rtn_values.to_list()

    def _correlate_list_element(self, canonical: Any, header: str, list_size: int=None, random_choice: bool=None,
                                replace: bool=None, shuffle: bool=None, convert_str: bool=None, seed: int=None,
                                rtn_type: str=None):
        """ correlate a column where the elements of the columns contains a list, and a choice is taken from that list.
        if the list_size == 1 then a single value is correlated otherwise a list is correlated

        Null values are passed through but all other elements must be a list with at least 1 value in.

        if 'random' is true then all returned values will be a random selection from the list and of equal length.
        if 'random' is false then each list will not exceed the 'list_size'

        Also if 'random' is true and 'replace' is False then all lists must have more elements than the list_size.
        By default 'replace' is True and 'shuffle' is False.

        In addition 'convert_str' allows lists that have been formatted as a string can be converted from a string
        to a list using 'ast.literal_eval(x)'

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: The header containing a list to chose from.
        :param list_size: (optional) the number of elements to return, if more than 1 then list
        :param random_choice: (optional) if the choice should be a random choice.
        :param replace: (optional) if the choice selection should be replaced or selected only once
        :param shuffle: (optional) if the final list should be shuffled
        :param convert_str: if the header has the list as a string convert to list using ast.literal_eval()
        :param seed: (optional) a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :return: a list of equal length to the one passed
        """
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        # Code block for intent
        list_size = list_size if isinstance(list_size, int) else 1
        random_choice = random_choice if isinstance(random_choice, bool) else False
        convert_str = convert_str if isinstance(convert_str, bool) else False
        replace = replace if isinstance(replace, bool) else True
        shuffle = shuffle if isinstance(shuffle, bool) else False
        _seed = seed if isinstance(seed, int) else self._seed()
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        s_idx = s_values.where(~s_values.isna()).dropna().index
        if convert_str:
            s_values.iloc[s_idx] = [ast.literal_eval(x) if isinstance(x, str) else x for x in s_values.iloc[s_idx]]
        s_values.iloc[s_idx] = Commons.list_formatter(s_values.iloc[s_idx])
        generator = np.random.default_rng(seed=_seed)
        if random_choice:
            try:
                s_values.iloc[s_idx] = [generator.choice(x, size=list_size, replace=replace, shuffle=shuffle)
                                        for x in s_values.iloc[s_idx]]
            except ValueError:
                raise ValueError(f"Unable to make a choice. Ensure {header} has all appropriate values for the method")
            s_values.iloc[s_idx] = [x[0] if list_size == 1 else list(x) for x in s_values.iloc[s_idx]]
        else:
            s_values.iloc[s_idx] = [x[:list_size] if list_size > 1 else x[0] for x in s_values.iloc[s_idx]]
        if isinstance(rtn_type, str):
            if rtn_type in ['category', 'object'] or rtn_type.startswith('int') or rtn_type.startswith('float'):
                s_values = s_values.astype(rtn_type)
            return s_values
        return s_values.to_list()

    def _correlate_join(self, canonical: Any, header: str, action: [str, dict], sep: str=None, seed: int=None,
                        rtn_type: str=None):
        """ correlate a column and join it with the result of the action, This allows for composite values to be
        build from. an example might be to take a forename and add the surname with a space separator to create a
        composite name field, of to join two primary keys to create a single composite key.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: an ordered list of columns to join
        :param action: (optional) a string or a single action whose outcome will be joined to the header value
        :param sep: (optional) a separator between the values
        :param seed: (optional) a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
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
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(action, (dict, str)):
            raise ValueError(f"The action must be a dictionary of a single action or a string value")
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        # Code block for intent
        _seed = seed if isinstance(seed, int) else self._seed()
        sep = sep if isinstance(sep, str) else ''
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        null_idx = s_values[s_values.isna()].index
        s_values.to_string()
        result = self._apply_action(canonical, action=action, seed=_seed)
        s_values = pd.Series([f"{a}{sep}{b}" for (a, b) in zip(s_values, result)], dtype='object')
        if null_idx.size > 0:
            s_values.iloc[null_idx] = np.nan
        if isinstance(rtn_type, str):
            if rtn_type in ['category', 'object'] or rtn_type.startswith('int') or rtn_type.startswith('float'):
                s_values = s_values.astype(rtn_type)
            return s_values
        return s_values.to_list()

    def _correlate_activation(self, canonical: Any, header: str, activation: str=None, precision: int=None, seed: int=None):
        """Activation functions play a crucial role in the backpropagation algorithm, which is the primary
        algorithm used for training neural networks. During backpropagation, the error of the output is
        propagated backwards through the network, and the weights of the network are updated based on this
        error. The activation function is used to introduce non-linearity into the output of a neural network
        layer.

        Logistic Sigmoid a.k.a logit, tmaps any input value to a value between 0 and 1, making it useful for
        binary classification problems and is defined as f(x) = 1/(1+exp(-x))

        Tangent Hyperbolic (tanh) function is a shifted and stretched version of the Sigmoid function but maps
        the input values to a range between -1 and 1. and is defined as f(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))

        Rectified Linear Unit (ReLU) function. is the most popular activation function, which replaces negative
        values with zero and keeps the positive values unchanged. and is defined as f(x) = x * (x > 0)

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param activation: (optional) the name of the activation function. Options 'sigmoid', 'tanh' and 'relu'
        :param precision: (optional) how many decimal places. default to 3
        :param seed: (optional) the random seed. defaults to current datetime
        :return: an equal length list of correlated values
        """
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        _seed = seed if isinstance(seed, int) else self._seed()
        precision = precision if isinstance(precision, int) else 5
        activation = activation.lower() if isinstance(activation, str) else 'relu'
        null_idx = s_values[s_values.isna()].index
        s_values = s_values.fillna(0)
        if activation.startswith('sigmoid'):
            rtn_values = np.round(1 / (1 + np.exp(-s_values)), precision)
        elif activation.startswith('tanh'):
            rtn_values = np.round((np.exp(s_values)-np.exp(-s_values))/(np.exp(s_values)+np.exp(-s_values)), precision)
        elif activation.startswith('relu'):
            rtn_values = np.round(s_values * (s_values > 0), precision)
        else:
            raise ValueError(f"The activation function '{activation}' is not supported. Current available options "
                             f"are 'sigmoid', 'tanh' and 'relu'")
        if null_idx.size > 0:
            rtn_values.iloc[null_idx] = np.nan
        return rtn_values.to_list()

    def _correlate_polynomial(self, canonical: Any, header: str, coefficient: list, seed: int=None,
                              rtn_type: str=None, keep_zero: bool=None) -> list:
        """ creates a polynomial using the reference header values and apply the coefficients where the
        index of the list represents the degree of the term in reverse order.

                  e.g  [6, -2, 0, 4] => f(x) = 4x**3 - 2x + 6

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param coefficient: the reverse list of term coefficients
        :param seed: (optional) the random seed. defaults to current datetime
        :param keep_zero: (optional) if True then zeros passed remain zero, Default is False
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :return: an equal length list of correlated values
        """
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        keep_zero = keep_zero if isinstance(keep_zero, bool) else False
        _seed = seed if isinstance(seed, int) else self._seed()

        def _calc_polynomial(x, _coefficient):
            if keep_zero and x == 0:
                return 0
            res = 0
            for index, coeff in enumerate(_coefficient):
                res += coeff * x ** index
            return res

        rtn_values = s_values.apply(lambda x: _calc_polynomial(x, coefficient))
        if isinstance(rtn_type, str):
            if rtn_type in ['category', 'object'] or rtn_type.startswith('int') or rtn_type.startswith('float'):
                rtn_values = rtn_values.astype(rtn_type)
            return rtn_values
        return rtn_values.to_list()

    def _correlate_flag_outliers(self, canonical: Any, header: str, method: str=None, measure: [int, float]=None,
                                 seed: int=None):
        """ returns a list of markers or flags identifying outliers in a dataset where 1 represents a suggested outlier.
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
        :param method: (optional) A method to run to identify outliers. interquartile (default), empirical
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
        elif method.startswith('int') or method.startswith('qua'):
            result_idx = DataDiscovery.interquartile_outliers(values=s_values, k_factor=measure)
        else:
            raise ValueError(f"The method '{method}' is not recognised. Please use one of interquartile or empirical")
        rtn_values = pd.Series(data=0, index=np.arange(canonical.shape[0]))
        rtn_values.loc[result_idx[0]+result_idx[1]] = 1
        return rtn_values.to_list()

    def _correlate_missing(self, canonical: Any, header: str, method: str=None, constant: Any=None, weights: str=None,
                           precision: int=None, seed: int=None):
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
         :param method: (optional) 'mean', 'median', 'mode', 'constant', 'random', neighbour, 'indicator'
         :param constant: (optional) a value to us when the method is constant
         :param weights: (optional) Weight function used in prediction of nearest neighbour if used as method. Options
                    ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                    ‘distance’ : weight points by the inverse of their distance.
         :param precision: (optional) if numeric, the precision of the outcome, by default set to 3.
         :param seed: (optional) the random seed. defaults to current datetime
         :return: list
         """
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        _seed = seed if isinstance(seed, int) else self._seed()
        method = method if isinstance(method, str) else 'median'
        precision = precision if isinstance(precision, int) else 3
        null_idx = s_values[s_values.isna()].index
        if isinstance(method, str) and method in ['median', 'mean']:
            if not ptypes.is_numeric_dtype(s_values.dropna()):
                raise ValueError(f"The header '{header}' is not numeric and thus not compatible with median or mean")
            if method.startswith('median'):
                s_values.iloc[null_idx] = s_values.median()
            else:
                s_values.iloc[null_idx] = s_values.mean()
            s_values = s_values.round(precision)
            if precision == 0:
                s_values = s_values.astype(int)
        elif isinstance(method, str) and method.startswith('mode'):
            s_values.iloc[null_idx] = s_values.mode()
        elif isinstance(method, str) and method.startswith('random'):
            rng = np.random.default_rng(seed)
            s_values.iloc[null_idx] = rng.choice(s_values.dropna(), size=null_idx.shape[0])
        elif isinstance(method, str) and method.startswith('neighbour'):
            weights = weights if isinstance(weights, str) and weights in ['uniform', 'distance'] else 'uniform'
            model = KNNImputer(n_neighbors=3, weights=weights)
            np_array = s_values.to_numpy().reshape(-1,1)
            s_values = pd.Series(model.fit_transform(np_array).reshape(1,-1)[0]).round(precision)
            if precision == 0:
                s_values = s_values.astype(int)
        elif isinstance(method, str) and method.startswith('constant'):
            if not isinstance(constant, (int, float, str)):
                raise ValueError("When using the 'constant' method a constant value must be provided")
            if ptypes.is_numeric_dtype(s_values) and isinstance(constant, str):
                raise ValueError(f"The value '{constant}' is a string and column '{header}' expects a numeric value")
            s_values.iloc[null_idx] = constant
        elif isinstance(method, str) and method.startswith('indicator'):
            s_values = pd. Series(data=0, index=range(s_values.shape[0]))
            s_values.iloc[null_idx] = 1
        else:
            raise ValueError(f"The method '{method}' is not recognised. Please use one of 'mean', 'median' or 'mode'")
        return s_values.to_list()

    def _correlate_values(self, canonical: Any, header: str, choice: [int, float, str]=None, choice_header: str=None,
                          jitter: [int, float, str]=None, offset: [int, float, str]=None, code_str: str=None,
                          lower: [int, float]=None, upper: [int, float]=None, precision: int=None, keep_zero: bool=None,
                          seed: int=None):
        """ correlate a list of continuous values adjusting those values, or a subset of those values, with a
        normalised jitter (std from the value) along with a value offset. ``choice``, ``jitter`` and ``offset``
        can accept environment variable string names starting with ``${`` and ending with ``}``.

        If the choice is an int, it represents the number of rows to choose. If the choice is a float it must be
        between 1 and 0 and represent a percentage of rows to choose.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param choice: (optional) The number of values or percentage between 0 and 1 to choose.
        :param choice_header: (optional) those not chosen are given the values of the given header
        :param precision: (optional) to what precision the return values should be
        :param offset: (optional) a fixed value or an environment variable where the name is wrapped with '${' and '}'
        :param code_str: (optional) passing a str lambda function. e.g. 'lambda x: (x - 3) / 2''
        :param jitter: (optional) a perturbation of the value where the jitter is a random normally distributed std
        :param precision: (optional) how many decimal places. default to 3
        :param seed: (optional) the random seed. defaults to current datetime
        :param keep_zero: (optional) if True then zeros passed remain zero despite a change, Default is False
        :param lower: a minimum value not to go below
        :param upper: a max value not to go above
        :return: list
        """
        canonical = self._get_canonical(canonical)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        choice_header = choice_header if isinstance(choice_header, str) and choice_header in canonical.columns else header
        s_others = canonical[choice_header].copy()
        seed = self._seed() if seed is None else seed
        offset = self._extract_value(offset)
        keep_zero = keep_zero if isinstance(keep_zero, bool) else False
        precision = precision if isinstance(precision, int) else 3
        lower = lower if isinstance(lower, (int, float)) else float('-inf')
        upper = upper if isinstance(upper, (int, float)) else float('inf')
        # mark the zeros and nulls
        null_idx = s_values[s_values.isna()].index
        zero_idx = s_values.where(s_values == 0).dropna().index if keep_zero else []
        # choose the items to jitter
        if isinstance(choice, (str, int, float)):
            size = s_values.size
            choice = self._extract_value(choice)
            choice = int(choice * size) if isinstance(choice, float) and 0 <= choice <= 1 else int(choice)
            choice = choice if 0 <= choice < size else size
            gen = np.random.default_rng(seed=seed)
            choice_idx = gen.choice(s_values.index, size=choice, replace=False)
            choice_idx = [choice_idx] if isinstance(choice_idx, int) else choice_idx
            s_values = s_values.iloc[choice_idx]
        if isinstance(jitter, (str, int, float)) and s_values.size > 0:
            jitter = self._extract_value(jitter)
            size = s_values.size
            gen = np.random.default_rng(seed)
            results = gen.normal(loc=0, scale=jitter, size=size)
            s_values = s_values.add(results)
        # set code_str
        if isinstance(code_str, str) and s_values.size > 0:
            if code_str.startswith('lambda'):
                s_values = s_values.transform(eval(code_str))
            else:
                code_str = code_str.replace("@", 'x')
                s_values = s_values.transform(lambda x: eval(code_str))
        # set offset for all values
        if isinstance(offset, (int, float)) and offset != 0 and s_values.size > 0:
            s_values = s_values.add(offset)
        # set the changed values
        if canonical[header].size == s_values.size:
            s_others = s_values
        else:
            s_others.iloc[s_values.index] = s_values
        # max and min caps
        s_others = pd.Series([upper if x > upper else x for x in s_others])
        s_others = pd.Series([lower if x < lower else x for x in s_others])
        if isinstance(keep_zero, bool) and keep_zero:
            if canonical[header].size == zero_idx.size:
                s_others = 0 * zero_idx.size
            else:
                s_others.iloc[zero_idx] = 0
        if canonical[header].size == null_idx.size:
            s_others = np.nan * null_idx.size
        else:
            s_others.iloc[null_idx] = np.nan
        s_others = s_others.round(precision)
        if precision == 0 and not s_others.isnull().any():
            s_others = s_others.astype(int)
        return s_others.to_list()

    def _correlate_numbers(self, canonical: Any, header: str, standardize: bool=None, normalize: bool=None,
                           scalar: tuple=None, transform: str=None, precision: int=None, seed: int=None):
        """ Allows for the scaling transformation of a continuous value set. scaling methods. Thse techniques
        are used to alter the values of a variable so that they are expressed on a common scale. This is often
        done to make it easier to compare different variables or to make it easier to analyze data.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param standardize: (optional) standardise continuous variables with mean 0 and std 1
        :param normalize: (optional) normalize continuous variables between 0 an 1.
        :param scalar: (optional) scales continuous variables between a mix and max value passed in the tuple pair.
        :param transform: (optional) attempts normal distribution of continuous variables.
                            options are log, sqrt, cbrt, boxcox, yeojohnson
        :param precision: (optional) how many decimal places. default to 3
        :param seed: (optional) the random seed. defaults to current datetime
        :return: an equal length list of correlated values
        """
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        if not (s_values.dtype.name.startswith('int') or s_values.dtype.name.startswith('float')):
            raise ValueError(f"The header column is of type '{s_values.dtype.name}' and not numeric. "
                             f"Use the 'to_numeric' parameter if appropriate")
        precision = precision if isinstance(precision, int) else 3
        _seed = seed if isinstance(seed, int) else self._seed()
        null_idx = s_values[s_values.isna()].index
        if isinstance(standardize, bool) and standardize:
            s_values = pd.Series(Commons.list_standardize(s_values.to_list()))
        elif isinstance(normalize, bool):
            s_values = pd.Series(Commons.list_normalize(s_values.to_list(), 0, 1))
        elif isinstance(scalar, tuple):
            if scalar[0] >= scalar[1] or len(scalar) != 2:
                raise ValueError("The scalar tuple must be of size two with the first value lower than the second")
            s_values = pd.Series(Commons.list_normalize(s_values.to_list(), scalar[0], scalar[1]))
        elif isinstance(transform, str):
            if transform == 'log':
                s_values = np.log(s_values)
            elif transform == 'sqrt':
                s_values = np.sqrt(s_values)
            elif transform == 'cbrt':
                s_values = np.cbrt(s_values)
            elif transform == 'boxcox' or transform.startswith('box'):
                bc, _ = stats.boxcox(s_values.to_numpy())
                s_values = pd.Series(bc)
            elif transform == 'yeojohnson' or transform.startswith("yeo"):
                yj, _ = stats.yeojohnson(s_values.to_numpy())
                s_values = pd.Series(yj)
            else:
                raise ValueError(f"The transformer {transform} is not recognized. See contacts notes for reference")
        s_values = s_values.round(precision)
        if precision == 0 and not s_values.isnull().any():
            s_values = s_values.astype(int)
        if null_idx.size > 0:
            s_values.iloc[null_idx] = np.nan
        return s_values.to_list()

    def _correlate_dates(self, canonical: Any, header: str, choice: [int, float, str]=None, choice_header: str=None,
                         offset: [int, dict, str]=None, jitter: [int, str]=None, jitter_units: str=None,
                         ignore_time: bool=None, ignore_seconds: bool=None, min_date: str=None, max_date: str=None,
                         now_delta: str=None, date_format: str=None, day_first: bool=None, year_first: bool=None,
                         seed: int=None):
        """ correlate a list of continuous dates adjusting those dates, or a subset of those dates, with a
        normalised jitter along with a value offset. ``choice``, ``jitter`` and ``offset`` can accept environment
        variable string names starting with ``${`` and ending with ``}``.

        When using offset and a dict is passed, the dict should take the form {'days': 1}, where the unit is plural,
        to add 1 day or a singular name {'hour': 3}, where the unit is singular, to replace the current with 3 hours.
        Offsets can be 'years', 'months', 'weeks', 'days', 'hours', 'minutes' or 'seconds'. If an int is passed
        days are assumed.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param choice: (optional) The number of values or percentage between 0 and 1 to choose.
        :param choice_header: (optional) those not chosen are given the values of the given header
        :param offset: (optional) Temporal parameter that add to or replace the offset value. if int then assume 'days'
        :param jitter: (optional) the random jitter or deviation in days
        :param jitter_units: (optional) the units of the jitter, Options: W, D, h, m, s, milli, micro. default 'D'
        :param ignore_time: ignore time elements and only select from Year, Month, Day elements. Default is False
        :param ignore_seconds: ignore second elements and only select from Year to minute elements. Default is False
        :param min_date: (optional)a minimum date not to go below
        :param max_date: (optional)a max date not to go above
        :param now_delta: (optional) returns a delta from now as an int list, Options: 'Y', 'M', 'W', 'D', 'h', 'm', 's'
        :param day_first: (optional) if the dates given are day first format. Default to True
        :param year_first: (optional) if the dates given are year first. Default to False
        :param date_format: (optional) the format of the output
        :param seed: (optional) a seed value for the random function: default to None
        :return: a list of equal size to that given
        """
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        values = canonical[header].copy()
        choice_header = choice_header if isinstance(choice_header, str) and choice_header in canonical.columns else header
        others = canonical[choice_header].copy()

        def _clean(control):
            _unit_type = ['years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds',
                          'year', 'month', 'week', 'day', 'hour', 'minute', 'second']
            _params = {}
            if isinstance(control, int):
                control = {'days': control}
            if isinstance(control, dict):
                for k, v in control.items():
                    if k not in _unit_type:
                        raise ValueError(f"The key '{k}' in 'offset', is not a recognised unit type for pd.DateOffset")
            return control

        seed = self._seed() if seed is None else seed
        ignore_seconds = ignore_seconds if isinstance(ignore_seconds, bool) else False
        ignore_time = ignore_time if isinstance(ignore_time, bool) else False
        offset = _clean(offset) if isinstance(offset, (dict, int)) else None
        if isinstance(now_delta, str) and now_delta not in ['Y', 'M', 'W', 'D', 'h', 'm', 's']:
            raise ValueError(f"the now_delta offset unit '{now_delta}' is not recognised "
                             f"use of of ['Y', 'M', 'W', 'D', 'h', 'm', 's']")
        # set minimum date
        _min_date = pd.to_datetime(min_date, errors='coerce')
        if _min_date is None or _min_date is pd.NaT:
            _min_date = pd.to_datetime(pd.Timestamp.min)
        # set max date
        _max_date = pd.to_datetime(max_date, errors='coerce')
        if _max_date is None or _max_date is pd.NaT:
            _max_date = pd.to_datetime(pd.Timestamp.max)
        if _min_date >= _max_date:
            raise ValueError(f"the min_date {min_date} must be less than max_date {max_date}")
        # convert values into datetime
        s_values = pd.Series(pd.to_datetime(values, errors='coerce', dayfirst=day_first, yearfirst=year_first))
        s_others = pd.Series(pd.to_datetime(others, errors='coerce', dayfirst=day_first, yearfirst=year_first))
        dt_tz = s_values.dt.tz
        # choose the items to jitter
        if isinstance(choice, (str, int, float)):
            size = s_values.size
            choice = self._extract_value(choice)
            choice = int(choice * size) if isinstance(choice, float) and 0 <= choice <= 1 else int(choice)
            choice = choice if 0 <= choice < size else size
            gen = np.random.default_rng(seed=seed)
            choice_idx = gen.choice(s_values.index, size=choice, replace=False)
            choice_idx = [choice_idx] if isinstance(choice_idx, int) else choice_idx
            s_values = s_values.iloc[choice_idx]
        if isinstance(jitter, (str, int)):
            size = s_values.size
            jitter = self._extract_value(jitter)
            jitter_units = self._extract_value(jitter_units)
            units_allowed = ['W', 'D', 'h', 'm', 's', 'milli', 'micro']
            jitter_units = jitter_units if isinstance(jitter_units, str) and jitter_units in units_allowed else 'D'
            # set jitters to time deltas
            jitter = pd.Timedelta(value=jitter, unit=jitter_units) if isinstance(jitter, int) else pd.Timedelta(value=0)
            jitter = int(jitter.to_timedelta64().astype(int) / 10 ** 3)
            gen = np.random.default_rng(seed)
            results = gen.normal(loc=0, scale=jitter, size=size)
            results = pd.Series(pd.to_timedelta(results, unit='micro'), index=s_values.index)
            s_values = s_values.add(results)
        null_idx = s_values[s_values.isna()].index
        if isinstance(offset, dict) and offset:
            s_values = s_values.add(pd.DateOffset(**offset))
        # sort max and min
        if _min_date > pd.to_datetime(pd.Timestamp.min):
            if _min_date > s_values.min():
                min_idx = s_values.dropna().where(s_values < _min_date).dropna().index
                s_values.iloc[min_idx] = _min_date
            else:
                raise ValueError(f"The min value {min_date} is greater than the max result value {s_values.max()}")
        if _max_date < pd.to_datetime(pd.Timestamp.max):
            if _max_date < s_values.max():
                max_idx = s_values.dropna().where(s_values > _max_date).dropna().index
                s_values.iloc[max_idx] = _max_date
            else:
                raise ValueError(f"The max value {max_date} is less than the min result value {s_values.min()}")
        # set the changed values
        if canonical[header].size == s_values.size:
            s_others = s_values
        else:
            s_others.iloc[s_values.index] = s_values
        if now_delta:
            s_others = s_others.dt.tz_convert(None) if s_others.dt.tz else s_others
            s_others = (s_others - pd.Timestamp.now()).abs()
            s_others = (s_others / np.timedelta64(1, now_delta))
            s_others = s_others.round(0) if null_idx.size > 0 else s_others.astype(int)
        else:
            if isinstance(date_format, str):
                s_others = s_others.dt.strftime(date_format)
            else:
                if s_others.dt.tz:
                    s_others = s_others.dt.tz_convert(dt_tz)
                else:
                    s_others = s_others.dt.tz_localize(dt_tz)
        if ignore_time:
            s_others = pd.Series(pd.DatetimeIndex(s_others).normalize())
        elif ignore_seconds:
            s_others = s_others.dt.round('min')
        return s_others.to_list()

    def _correlate_categories(self, canonical: Any, header: str, correlations: list, actions: dict,
                              default_action: [str, int, float, dict]=None, seed: int=None, rtn_type: str=None):
        """ correlation of a set of values to an action, the correlations must map to the dictionary index values.
        Note. to use the current value in the passed values as a parameter value pass an empty dict {} as the keys
        value. If you want the action value to be the current value of the passed value then again pass an empty dict
        action to be the current value
            simple correlation list:
                ['A', 'B', 'C'] # if values is 'A' then action is 0 and so on
            multiple choice correlation:
                [['A','B'], 'C'] # if values is 'A' OR 'B' then action is 0 and so on

            For more complex correlation the selection logic can be used, see notes below.

            for actions also see notes below.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param correlations: a list of categories (can also contain lists for multiple correlations.
        :param actions: the correlated set of categories that should map to the index
        :param default_action: (optional) a default action to take if the selection is not fulfilled
        :param seed: a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :return: a list of equal length to the one passed

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
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        _seed = seed if isinstance(seed, int) else self._seed()
        corr_list = []
        for corr in correlations:
            corr_list.append(Commons.list_formatter(corr))
        if not isinstance(default_action, (str, int, float, dict)):
            default_action = None
        rtn_values = self._apply_action(canonical, action=default_action, seed=_seed)
        # deal with categories
        if rtn_values.dtype.name == 'category':
            rtn_values = rtn_values.astype('object')
        s_values = canonical[header].copy().astype(str)
        for i in range(len(corr_list)):
            action = actions.get(i, actions.get(str(i), -1))
            if action == -1:
                continue
            if isinstance(corr_list[i][0], dict):
                corr_idx = self._selection_index(canonical, selection=corr_list[i])
            else:
                corr_idx = s_values[s_values.isin(map(str, corr_list[i]))].index
            rtn_values.update(self._apply_action(canonical, action=action, select_idx=corr_idx, seed=_seed))
        if isinstance(rtn_type, str):
            if rtn_type in ['category', 'object'] or rtn_type.startswith('int') or rtn_type.startswith('float'):
                rtn_values = rtn_values.astype(rtn_type)
            return rtn_values
        return rtn_values.to_list()

    def _correlate_discrete_intervals(self, canonical: Any, header: str, granularity: [int, float, list]=None,
                                      lower: [int, float]=None, upper: [int, float]=None, categories: list=None,
                                      precision: int=None, seed: int=None) -> list:
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
        :return: a list of equal size to that given
        """
        # exceptions check
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        _seed = seed if isinstance(seed, int) else self._seed()
        # intend code block on the canonical
        granularity = 3 if not isinstance(granularity, (int, float, list)) or granularity == 0 else granularity
        precision = precision if isinstance(precision, int) else 5
        # firstly get the granularity
        lower = canonical[header].min() if not isinstance(lower, (int, float)) else lower
        upper = canonical[header].max() if not isinstance(upper, (int, float)) else upper
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
                boundaries = canonical[header].quantile(quantiles).values
                boundaries.sort()
                granularity = [(boundaries[0], boundaries[1], 'both')]
                granularity += [(boundaries[i - 1], boundaries[i], 'right') for i in range(2, boundaries.size)]
            else:
                granularity = (lower, upper, 'both')

        granularity = [(np.round(p[0], precision), np.round(p[1], precision), p[2]) for p in granularity]
        # now create the categories
        conditions = []
        for interval in granularity:
            lower, upper, closed = interval
            if str.lower(closed) == 'neither':
                conditions.append((canonical[header] > lower) & (canonical[header] < upper))
            elif str.lower(closed) == 'right':
                conditions.append((canonical[header] > lower) & (canonical[header] <= upper))
            elif str.lower(closed) == 'both':
                conditions.append((canonical[header] >= lower) & (canonical[header] <= upper))
            else:
                conditions.append((canonical[header] >= lower) & (canonical[header] < upper))
        if isinstance(categories, list) and len(categories) == len(conditions):
            choices = categories
        else:
            if canonical[header].dtype.name.startswith('int'):
                choices = [f"{int(i[0])}->{int(i[1])}" for i in granularity]
            else:
                choices = [f"{i[0]}->{i[1]}" for i in granularity]
        # noinspection PyTypeChecker
        return np.select(conditions, choices, default="<NA>").tolist()
