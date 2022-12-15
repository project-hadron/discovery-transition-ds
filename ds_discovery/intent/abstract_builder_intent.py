import ast
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Any
from pandas.api.types import is_numeric_dtype
from matplotlib import dates as mdates
from scipy import stats
from sklearn.impute import KNNImputer
from aistac import ConnectorContract
from aistac.components.aistac_commons import DataAnalytics
from ds_discovery.components.commons import Commons
from aistac.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.components.discovery import DataDiscovery
from ds_discovery.intent.abstract_common_intent import AbstractCommonsIntentModel

__author__ = 'Darryl Oatridge'


class AbstractBuilderIntentModel(AbstractCommonsIntentModel):

    _INTENT_PARAMS = ['self', 'save_intent', 'column_name', 'intent_order',
                      'replace_intent', 'remove_duplicates', 'seed']

    def __init__(self, property_manager: AbstractPropertyManager, default_save_intent: bool=None,
                 default_intent_level: [str, int, float]=None, default_intent_order: int=None,
                 default_replace_intent: bool=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param default_intent_level: (optional) the default level intent should be saved at
        :param default_intent_order: (optional) if the default behaviour for the order should be next available order
        :param default_replace_intent: (optional) the default replace existing intent behaviour
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = default_intent_level if isinstance(default_intent_level, (str, int, float)) else 'A'
        default_intent_order = default_intent_order if isinstance(default_intent_order, int) else 0
        intent_param_exclude = ['size']
        intent_type_additions = [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, pd.Timestamp]
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, canonical: Any=None, intent_levels: [str, int, list]=None, run_book: str=None,
                            seed: int=None, simulate: bool=None, **kwargs) -> pd.DataFrame:
        """Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract. The whole run can be seeded though any parameterised seeding in the intent
        contracts will take precedence

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param intent_levels: (optional) a single or list of intent_level to run in order given
        :param run_book: (optional) a preset runbook of intent_level to run in order
        :param seed: (optional) a seed value that will be applied across the run: default to None
        :param simulate: (optional) returns a report of the order of run and return the indexed column order of run
        :return: a pandas dataframe
        """
        simulate = simulate if isinstance(simulate, bool) else False
        col_sim = {"column": [], "order": [], "method": []}
        # legacy
        if 'size' in kwargs.keys():
            canonical = kwargs.pop('size')
        canonical = self._get_canonical(canonical)
        size = canonical.shape[0] if canonical.shape[0] > 0 else 1000
        # test if there is any intent to run
        if self._pm.has_intent():
            # get the list of levels to run
            if isinstance(intent_levels, (str, list)):
                column_names = Commons.list_formatter(intent_levels)
            elif isinstance(run_book, str) and self._pm.has_run_book(book_name=run_book):
                column_names = self._pm.get_run_book(book_name=run_book)
            else:
                # put all the intent in order of model, get, correlate, associate
                _model = []
                _get = []
                _correlate = []
                _frame_start = []
                _frame_end = []
                for column in self._pm.get_intent().keys():
                    for order in self._pm.get(self._pm.join(self._pm.KEY.intent_key, column), {}):
                        for method in self._pm.get(self._pm.join(self._pm.KEY.intent_key, column, order), {}).keys():
                            if str(method).startswith('get_'):
                                if column in _correlate + _frame_start + _frame_end:
                                    continue
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
                                if str(method).startswith('frame_starter'):
                                    _frame_start.append(column)
                                else:
                                    _frame_end.append(column)
                column_names = Commons.list_unique(_frame_start + _get + _model + _correlate + _frame_end)
            for column in column_names:
                level_key = self._pm.join(self._pm.KEY.intent_key, column)
                for order in sorted(self._pm.get(level_key, {})):
                    for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                        try:
                            if method in self.__dir__():
                                if simulate:
                                    col_sim['column'].append(column)
                                    col_sim['order'].append(order)
                                    col_sim['method'].append(method)
                                    continue
                                result = []
                                params.update(params.pop('kwargs', {}))
                                if isinstance(seed, int):
                                    params.update({'seed': seed})
                                _ = params.pop('intent_creator', 'Unknown')
                                if str(method).startswith('get_'):
                                    result = eval(f"self.{method}(size=size, save_intent=False, **params)",
                                                  globals(), locals())
                                elif str(method).startswith('correlate_'):
                                    result = eval(f"self.{method}(canonical=canonical, save_intent=False, **params)",
                                                  globals(), locals())
                                elif str(method).startswith('model_'):
                                    canonical = eval(f"self.{method}(canonical=canonical, save_intent=False, **params)",
                                                     globals(), locals())
                                    continue
                                elif str(method).startswith('frame_starter'):
                                    canonical = self._get_canonical(params.pop('canonical', canonical), deep_copy=False)
                                    size = canonical.shape[0]
                                    canonical = eval(f"self.{method}(canonical=canonical, save_intent=False, **params)",
                                                     globals(), locals())
                                    continue
                                elif str(method).startswith('frame_'):
                                    canonical = eval(f"self.{method}(canonical=canonical, save_intent=False, **params)",
                                                     globals(), locals())
                                    continue
                                if 0 < size != len(result):
                                    raise IndexError(f"The index size of '{column}' is '{len(result)}', "
                                                     f"should be {size}")
                                canonical[column] = result
                        except ValueError as ve:
                            raise ValueError(f"intent '{column}', order '{order}', method '{method}' failed with: {ve}")
                        except TypeError as te:
                            raise TypeError(f"intent '{column}', order '{order}', method '{method}' failed with: {te}")
        if simulate:
            return pd.DataFrame.from_dict(col_sim)
        return canonical

    def _get_number(self, from_value: Any=None, to_value: Any=None, relative_freq: list=None, precision: int=None,
                    ordered: str=None, at_most: int=None, size: int=None, seed: int=None) -> list:
        """ returns a number in the range from_value to to_value. if only one number given from_value is zero

        :param from_value: (signed) integer or float to start from. See below
        :param to_value: optional, (signed) integer or float the number sequence goes to but not include. See below
        :param relative_freq: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param ordered: order the data ascending 'asc' or descending 'dec', values accepted 'asc' or 'des'
        :param at_most: the most times a selection should be chosen
        :param size: the size of the sample
        :param seed: a seed value for the random function: default to None

        The values can be represented by an environment variable with the format '${NAME}' where NAME is the
        environment variable name
        """
        if isinstance(from_value, str) and from_value.startswith('${'):
            from_value = ConnectorContract.parse_environ(from_value)
            if from_value.isnumeric():
                from_value = float(from_value)
            else:
                raise ValueError("The environment variable for to_value is not convertable from string to numeric")
        if isinstance(to_value, str) and to_value.startswith('${'):
            to_value = ConnectorContract.parse_environ(to_value)
            if to_value.isnumeric():
                to_value = float(to_value)
            else:
                raise ValueError("The environment variable for to_value is not convertable from string to numeric")
        if not isinstance(from_value, (int, float)) and not isinstance(to_value, (int, float)):
            raise ValueError(f"either a 'range_value' or a 'range_value' and 'to_value' must be provided")
        if not isinstance(from_value, (float, int)):
            from_value = 0
        if not isinstance(to_value, (float, int)):
            (from_value, to_value) = (0, from_value)
        if to_value <= from_value:
            raise ValueError("The number range must be a positive difference, where to_value <= from_value")
        at_most = 0 if not isinstance(at_most, int) else at_most
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        precision = 3 if not isinstance(precision, int) else precision
        if precision == 0:
            from_value = int(round(from_value, 0))
            to_value = int(round(to_value, 0))
        is_int = True if (isinstance(to_value, int) and isinstance(from_value, int)) else False
        if is_int:
            precision = 0
        # build the distribution sizes
        if isinstance(relative_freq, list) and len(relative_freq) > 1 and sum(relative_freq) > 1:
            freq_dist_size = self._freq_dist_size(relative_freq=relative_freq, size=size, seed=_seed)
        else:
            freq_dist_size = [size]
        # generate the numbers
        rtn_list = []
        generator = np.random.default_rng(seed=_seed)
        d_type = int if is_int else float
        bins = np.linspace(from_value, to_value, len(freq_dist_size) + 1, dtype=d_type)
        for idx in np.arange(1, len(bins)):
            low = bins[idx - 1]
            high = bins[idx]
            if low >= high:
                continue
            elif at_most > 0:
                sample = []
                for _ in np.arange(at_most, dtype=d_type):
                    count_size = freq_dist_size[idx - 1] * generator.integers(2, 4, size=1)[0]
                    sample += list(set(np.linspace(bins[idx - 1], bins[idx], num=count_size, dtype=d_type,
                                                   endpoint=False)))
                if len(sample) < freq_dist_size[idx - 1]:
                    raise ValueError(f"The value range has insufficient samples to choose from when using at_most."
                                     f"Try increasing the range of values to sample.")
                rtn_list += list(generator.choice(sample, size=freq_dist_size[idx - 1], replace=False))
            else:
                if d_type == int:
                    rtn_list += generator.integers(low=low, high=high, size=freq_dist_size[idx - 1]).tolist()
                else:
                    choice = generator.random(size=freq_dist_size[idx - 1], dtype=float)
                    choice = np.round(choice * (high-low)+low, precision).tolist()
                    # make sure the precision
                    choice = [high - 10**(-precision) if x >= high else x for x in choice]
                    rtn_list += choice
        # order or shuffle the return list
        if isinstance(ordered, str) and ordered.lower() in ['asc', 'des']:
            rtn_list.sort(reverse=True if ordered.lower() == 'asc' else False)
        else:
            generator.shuffle(rtn_list)
        return rtn_list

    def _get_noise(self, size: int, ones: bool=None, seed: int=None) -> list:
        """ A noise bias column of ones unless ones is False, then returns zeros

        :param size: size of the list to return
        :param ones: (optional) by default set to True returning a list of ones, else returning a list of zeros
        :param seed: (optional) placeholder for continuity
        :return: a list of ones or zeros
        """
        _seed = self._seed() if seed is None else seed
        ones = ones if isinstance(ones, bool) else True
        if ones:
            return list(np.ones(size))
        return list(np.zeros(size))

    def _get_category(self, selection: list, relative_freq: list=None, at_most: int=None, size: int=None,
                      seed: int=None) -> list:
        """ returns a category from a list. Of particular not is the at_least parameter that allows you to
        control the number of times a selection can be chosen.

        :param selection: a list of items to select from
        :param relative_freq: a weighting pattern that does not have to add to 1
        :param size: an optional size of the return. default to 1
        :param at_most: the most times a selection should be chosen
        :param seed: a seed value for the random function: default to None
        :return: an item or list of items chosen from the list
        """
        _seed = self._seed() if seed is None else seed
        if len(selection) < 1:
            return [None] * size
        select_index = self._get_number(len(selection), relative_freq=relative_freq, at_most=at_most, size=size,
                                        seed=_seed)
        rtn_list = [selection[i] for i in select_index]
        return list(rtn_list)

    def _get_datetime(self, start: Any, until: Any, relative_freq: list=None, at_most: int=None, ordered: str=None,
                      date_format: str=None, as_num: bool=None, ignore_time: bool=None, size: int=None,
                      seed: int=None, day_first: bool=None, year_first: bool=None) -> list:
        """ returns a random date between two date and/or times. weighted patterns can be applied to the overall date
        range.
        if a signed 'int' type is passed to the start and/or until dates, the inferred date will be the current date
        time with the integer being the offset from the current date time in 'days'.
        if a dictionary of time delta name values is passed this is treated as a time delta from the start time.
        for example if start = 0, until = {days=1, hours=3} the date range will be between now and 1 days and 3 hours

        Note: If no patterns are set this will return a linearly random number between the range boundaries.

        :param start: the start boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp or int
        :param until: up until boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp, pd.delta, int
        :param relative_freq: (optional) A pattern across the whole date range.
        :param at_most: the most times a selection should be chosen
        :param ordered: order the data ascending 'asc' or descending 'dec', values accepted 'asc' or 'des'
        :param ignore_time: ignore time elements and only select from Year, Month, Day elements. Default is False
        :param date_format: the string format of the date to be returned. if not set then pd.Timestamp returned
        :param as_num: returns a list of Matplotlib date values as a float. Default is False
        :param size: the size of the sample to return. Default to 1
        :param seed: a seed value for the random function: default to None
        :param year_first: specifies if to parse with the year first
                If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
                If both dayfirst and yearfirst are True, yearfirst is preceded (same as dateutil).
        :param day_first: specifies if to parse with the day first
                If True, parses dates with the day first, eg %d-%m-%Y.
                If False default to the a preferred preference, normally %m-%d-%Y (but not strict)
        :return: a date or size of dates in the format given.
         """
        # pre check
        if start is None or until is None:
            raise ValueError("The start or until parameters cannot be of NoneType")
        # Code block for intent
        as_num = False if not isinstance(as_num, bool) else as_num
        ignore_time = False if not isinstance(ignore_time, bool) else ignore_time
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        if isinstance(start, int):
            start = (pd.Timestamp.now() + pd.Timedelta(days=start))
        if isinstance(until, int):
            until = (pd.Timestamp.now() + pd.Timedelta(days=until))
        if isinstance(until, dict):
            until = (start + pd.Timedelta(**until))
        if start == until:
            rtn_list = [self._convert_date2value(start, day_first=day_first, year_first=year_first)[0]] * size
        else:
            _dt_start = self._convert_date2value(start, day_first=day_first, year_first=year_first)[0]
            _dt_until = self._convert_date2value(until, day_first=day_first, year_first=year_first)[0]
            precision = 15
            if ignore_time:
                _dt_start = int(_dt_start)
                _dt_until = int(_dt_until)
                precision = 0

            rtn_list = self._get_number(from_value=_dt_start, to_value=_dt_until, relative_freq=relative_freq,
                                        at_most=at_most, ordered=ordered, precision=precision, size=size, seed=seed)
        if not as_num:
            rtn_list = mdates.num2date(rtn_list)
            if isinstance(date_format, str):
                rtn_list = pd.Series(rtn_list).dt.strftime(date_format).to_list()
            else:
                rtn_list = pd.Series(rtn_list).dt.tz_convert(None).to_list()
        return rtn_list

    def _get_intervals(self, intervals: list, relative_freq: list=None, precision: int=None, size: int=None,
                       seed: int=None) -> list:
        """ returns a number based on a list selection of tuple(lower, upper) interval

        :param intervals: a list of unique tuple pairs representing the interval lower and upper boundaries
        :param relative_freq: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes float
        :param size: the size of the sample
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        # Code block for intent
        size = 1 if size is None else size
        precision = precision if isinstance(precision, (float, int)) else 3
        _seed = self._seed() if seed is None else seed
        if not all(isinstance(value, tuple) for value in intervals):
            raise ValueError("The intervals list must be a list of tuples")
        interval_list = self._get_category(selection=intervals, relative_freq=relative_freq, size=size, seed=_seed)
        interval_counts = pd.Series(interval_list, dtype='object').value_counts()
        rtn_list = []
        for index in interval_counts.index:
            size = interval_counts[index]
            if size == 0:
                continue
            if len(index) == 2:
                (lower, upper) = index
                if index == 0:
                    closed = 'both'
                else:
                    closed = 'right'
            else:
                (lower, upper, closed) = index
            if lower == upper:
                rtn_list += [round(lower, precision)] * size
                continue
            if precision == 0:
                margin = 1
            else:
                margin = 10**(((-1)*precision)-1)
            if str.lower(closed) == 'neither':
                lower += margin
                upper -= margin
            elif str.lower(closed) == 'right':
                lower += margin
            elif str.lower(closed) == 'both':
                upper += margin
            # correct adjustments
            if lower >= upper:
                upper = lower + margin
            rtn_list += self._get_number(lower, upper, precision=precision, size=size, seed=_seed)
        np.random.default_rng(seed=_seed).shuffle(rtn_list)
        return rtn_list

    def _get_dist_normal(self, mean: float, std: float, precision: int=None, size: int=None, seed: int=None) -> list:
        """A normal (Gaussian) continuous random distribution.

        :param mean: The mean (“centre”) of the distribution.
        :param std: The standard deviation (jitter or “width”) of the distribution. Must be >= 0
        :param precision: The number of decimal points. The default is 3
        :param size: the size of the sample. if a tuple of intervals, size must match the tuple
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        precision = precision if isinstance(precision, int) else 3
        generator = np.random.default_rng(seed=_seed)
        rtn_list = list(generator.normal(loc=mean, scale=std, size=size))
        return list(np.around(rtn_list, precision))

    def _get_dist_choice(self, number: [int, str], size: int=None, seed: int=None) -> list:
        """Creates a list of latent values of 0 or 1 where 1 is randomly selected both upon the number given.

       :param number: The number of true (1) values to randomly chose from the canonical. see below
       :param size: the size of the sample. if a tuple of intervals, size must match the tuple
       :param seed: a seed value for the random function: default to None
       :return: a list of 1 or 0

        As choice is a fixed value, number can be represented by an environment variable with the format '${NAME}'
        where NAME is the environment variable name
        """
        size = size if isinstance(size, int) else 1
        _seed = self._seed() if seed is None else seed
        if isinstance(number, str) and number.startswith('${') and number.endswith('}'):
            number = ConnectorContract.parse_environ(number)
            number = int(number) if number.isnumeric() else 0
        number = number if 0 <= number < size else size
        if isinstance(number, int) and 0 <= number <= size:
            rtn_list = pd.Series(data=[0] * size)
            choice_idx = self._get_number(to_value=size, size=number, at_most=1, precision=0, ordered='asc', seed=_seed)
            rtn_list.iloc[choice_idx] = [1]*number
            return rtn_list.reset_index(drop=True).to_list()
        return pd.Series(data=[0] * size).to_list()

    def _get_dist_bernoulli(self, probability: float, size: int=None, seed: int=None) -> list:
        """A Bernoulli process is a discrete random distribution. Bernoulli trial is a random experiment with exactly
        two possible outcomes, "success" and "failure", in which the probability of success is the same every time
        the experiment is conducted.

        The mathematical formalisation of the Bernoulli trial, this distribution,  is known as the Bernoulli process.
        A Bernoulli process is a finite or infinite sequence of binary random variables. Prosaically, a Bernoulli
        process is a repeated coin flipping, possibly with an unfair coin (but with consistent unfairness).

        :param probability: the probability occurrence
        :param size: the size of the sample
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        rtn_list = list(stats.bernoulli.rvs(p=probability, size=size, random_state=_seed))
        return rtn_list

    def _get_dist_bounded_normal(self, mean: float, std: float, lower: float, upper: float, precision: int=None,
                                 size: int=None, seed: int=None) -> list:
        """A bounded normal continuous random distribution.

        :param mean: the mean of the distribution
        :param std: the standard deviation
        :param lower: the lower limit of the distribution
        :param upper: the upper limit of the distribution
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param size: the size of the sample
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        size = 1 if size is None else size
        precision = precision if isinstance(precision, int) else 3
        _seed = self._seed() if seed is None else seed
        rtn_list = stats.truncnorm((lower-mean)/std, (upper-mean)/std, loc=mean, scale=std).rvs(size).round(precision)
        return rtn_list

    def _get_distribution(self, distribution: str, is_stats: bool=None, precision: int=None, size: int=None,
                          seed: int=None, **kwargs) -> list:
        """returns a number based the distribution type.

        :param distribution: The string name of the distribution function from numpy random Generator class
        :param is_stats: (optional) if the generator is from the stats package and not numpy
        :param precision: (optional) the precision of the returned number
        :param size: (optional) the size of the sample
        :param seed: (optional) a seed value for the random function: default to None
        :return: a random number
        """
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        precision = 3 if precision is None else precision
        is_stats = is_stats if isinstance(is_stats, bool) else False
        if is_stats:
            rtn_list = eval(f"stats.{distribution}.rvs(size=size, random_state=_seed, **kwargs)", globals(), locals())
        else:
            generator = np.random.default_rng(seed=_seed)
            rtn_list = eval(f"generator.{distribution}(size=size, **kwargs)", globals(), locals())
        return list(np.around(rtn_list, precision))

    def _get_selection(self, select_source: Any, column_header: str, relative_freq: list=None, sample_size: int=None,
                       selection_size: int=None, size: int=None, at_most: bool=None, shuffle: bool=None,
                       seed: int=None) -> list:
        """ returns a random list of values where the selection of those values is taken from a connector source.

        :param select_source: the selection source for the reference dataframe
        :param column_header: the name of the column header to correlate
        :param relative_freq: (optional) a weighting pattern of the final selection
        :param selection_size: (optional) the selection to take from the sample size, normally used with shuffle
        :param sample_size: (optional) the size of the sample to take from the reference file
        :param at_most: (optional) the most times a selection should be chosen
        :param shuffle: (optional) if the selection should be shuffled before selection. Default is true
        :param size: (optional) size of the return. default to 1
        :param seed: (optional) a seed value for the random function: default to None
        :return: list

        The select_source is normally a connector contract str reference or a set of parameter instructions on how to
        generate a pd.Dataframe but can be a pd.DataFrame. the description of each is:

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
        canonical = self._get_canonical(select_source)
        _seed = self._seed() if seed is None else seed
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical)
        if column_header not in canonical.columns:
            raise ValueError(f"The column '{column_header}' not found in the canonical")
        _values = canonical[column_header].iloc[:sample_size]
        if isinstance(selection_size, float) and shuffle:
            _values = _values.sample(frac=1, random_state=_seed).reset_index(drop=True)
        if isinstance(selection_size, int) and 0 < selection_size < _values.size:
            _values = _values.iloc[:selection_size]
        return self._get_category(selection=_values.to_list(), relative_freq=relative_freq, size=size, at_most=at_most,
                                  seed=_seed)

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
        _seed = self._seed() if seed is None else seed
        if isinstance(selection, list):
            selection = deepcopy(selection)
            # run the select logic
            select_idx = self._selection_index(canonical=canonical, selection=selection)
            canonical = canonical.iloc[select_idx].reset_index(drop=True)
        if isinstance(choice, int) and 0 < choice < canonical.shape[0]:
            choice_idx = self._get_number(to_value=canonical.shape[0], size=choice, at_most=1, precision=0, ordered='asc')
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

    def _model_custom(self, canonical: Any, code_str: str, seed: int=None, **kwargs):
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

    def _model_group(self, canonical: Any, group_by: [str, list], headers: [str, list]=None, regex: bool=None,
                     aggregator: str=None, list_choice: int=None, list_max: int=None, drop_group_by: bool=False,
                     seed: int=None, include_weighting: bool=False, freq_precision: int=None,
                     remove_weighting_zeros: bool=False, remove_aggregated: bool=False) -> pd.DataFrame:
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
                        modifier: str=None, aggregator: str=None, agg_header: str=None, precision: int=None,
                        seed: int=None) -> pd.DataFrame:
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
        :return: pd.DataFrame
        """
        canonical = self._get_canonical(canonical)
        other = self._get_canonical(other, size=canonical.shape[0])
        _seed = self._seed() if seed is None else seed
        modifier = modifier if isinstance(modifier, str) and modifier in ['add', 'sub', 'mul', 'div'] else 'add'
        precision = precision if isinstance(precision, int) else 3
        targets_header = targets_header if isinstance(targets_header, str) else other.columns[0]
        values_header = values_header if isinstance(values_header, str) else other.columns[1]
        agg_header = agg_header if isinstance(agg_header, str) else 'latent_aggregator'
        target_headers = []
        for index, row in other.iterrows():
            if row.loc[targets_header] in canonical.columns:
                if not is_numeric_dtype(canonical[row.loc[targets_header]]):
                    raise TypeError(f"The column {row.loc[targets_header]} is not of type numeric")
                canonical[row.loc[targets_header]] = eval(f"canonical[row.loc[targets_header]].{modifier}(row.loc["
                                                          f"values_header])", globals(), locals()).round(precision)
                target_headers.append(row.loc[targets_header])
        if isinstance(aggregator, str):
            canonical[agg_header] = self._correlate_aggregate(canonical=canonical, headers=target_headers,
                                                              agg=aggregator, seed=seed, precision=precision)
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
        # Code block for intent
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
        df_rtn = pd.merge(left=canonical, right=other, how=how, left_on=left_on, right_on=right_on, on=on,
                          suffixes=suffixes, indicator=indicator, validate=validate)
        if replace_nulls:
            for column in df_rtn.columns.to_list():
                Commons.fillna(df_rtn[column])
        return df_rtn

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

    def _model_analysis(self, canonical: Any, other: Any, columns_list: list=None, exclude_associate: list=None,
                        detail_numeric: bool=None, strict_typing: bool=None, category_limit: int=None,
                        seed: int=None) -> pd.DataFrame:
        """ builds a set of columns based on an other (see analyse_association)
        if a reference DataFrame is passed then as the analysis is run if the column already exists the row
        value will be taken as the reference to the sub category and not the random value. This allows already
        constructed association to be used as reference for a sub category.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param other: a direct or generated pd.DataFrame. see context notes below
        :param columns_list: (optional) a list structure of columns to select for association
        :param exclude_associate: (optional) a list of dot separated tree of items to exclude from iteration
                (e.g. ['age.gender.salary']
        :param detail_numeric: (optional) as a default, if numeric columns should have detail stats, slowing analysis
        :param strict_typing: (optional) stops objects and string types being seen as categories
        :param category_limit: (optional) a global cap on categories captured. zero value returns no limits
        :param seed: seed: (optional) a seed value for the random function: default to None
        :return: a DataFrame

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

        def get_level(analysis: dict, sample_size: int, _seed: int=None):
            _seed = self._seed(seed=_seed, increment=True)
            for name, values in analysis.items():
                if row_dict.get(name) is None:
                    row_dict[name] = list()
                _analysis = DataAnalytics(analysis=values.get('insight', {}))
                result_type = object
                if str(_analysis.intent.dtype).startswith('cat'):
                    result_type = 'category'
                    result = self._get_category(selection=_analysis.intent.categories,
                                                relative_freq=_analysis.patterns.get('relative_freq', None),
                                                seed=_seed, size=sample_size)
                elif str(_analysis.intent.dtype).startswith('num'):
                    result_type = 'int' if _analysis.params.precision == 0 else 'float'
                    result = self._get_intervals(intervals=[tuple(x) for x in _analysis.intent.intervals],
                                                 relative_freq=_analysis.patterns.get('relative_freq', None),
                                                 precision=_analysis.params.get('precision', None),
                                                 seed=_seed, size=sample_size)
                elif str(_analysis.intent.dtype).startswith('date'):
                    result_type = 'object' if _analysis.params.is_element('data_format') else 'date'
                    result = self._get_datetime(start=_analysis.stats.lowest,
                                                until=_analysis.stats.highest,
                                                relative_freq=_analysis.patterns.get('relative_freq', None),
                                                date_format=_analysis.params.get('data_format', None),
                                                day_first=_analysis.params.get('day_first', None),
                                                year_first=_analysis.params.get('year_first', None),
                                                seed=_seed, size=sample_size)
                else:
                    result = []
                # if the analysis was done with excluding dominance then se if they should be added back
                if _analysis.patterns.is_element('dominant_excluded'):
                    _dom_percent = _analysis.patterns.dominant_percent/100
                    _dom_values = _analysis.patterns.dominant_excluded
                    if len(_dom_values) > 0:
                        s_values = pd.Series(result, dtype=result_type)
                        non_zero = s_values[~s_values.isin(_dom_values)].index
                        choice_size = int((s_values.size * _dom_percent) - (s_values.size - len(non_zero)))
                        if choice_size > 0:
                            generator = np.random.default_rng(_seed)
                            _dom_choice = generator.choice(_dom_values, size=choice_size)
                            s_values.iloc[generator.choice(non_zero, size=choice_size, replace=False)] = _dom_choice
                            result = s_values.to_list()
                # now add the result to the row_dict
                row_dict[name] += result
                if sum(_analysis.patterns.relative_freq) == 0:
                    unit = 0
                else:
                    unit = sample_size / sum(_analysis.patterns.relative_freq)
                if values.get('sub_category'):
                    leaves = values.get('branch', {}).get('leaves', {})
                    for idx in range(len(leaves)):
                        section_size = int(round(_analysis.patterns.relative_freq[idx] * unit, 0)) + 1
                        next_item = values.get('sub_category').get(leaves[idx])
                        get_level(next_item, section_size, _seed)
            return

        canonical = self._get_canonical(canonical)
        other = self._get_canonical(other)
        columns_list = columns_list if isinstance(columns_list, list) else other.columns.to_list()
        blob = DataDiscovery.analyse_association(other, columns_list=columns_list, exclude_associate=exclude_associate,
                                                 detail_numeric=detail_numeric, strict_typing=strict_typing,
                                                 category_limit=category_limit)
        row_dict = dict()
        seed = self._seed() if seed is None else seed
        size = canonical.shape[0]
        get_level(blob, sample_size=size, _seed=seed)
        for key in row_dict.keys():
            row_dict[key] = row_dict[key][:size]
        return pd.concat([canonical, pd.DataFrame.from_dict(data=row_dict)], axis=1)

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
        seed = self._seed() if seed is None else seed
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
        seed = self._seed() if seed is None else seed
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
        seed = self._seed() if seed is None else seed
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

    def _model_encode_ordinal(self, canonical: Any, headers: [str, list], prefix :str=None, seed: int=None):
        """ encodes categorical data types, Ordinal or Integer encoding consist in replacing the categories by digits
        from 1 to n (or 0 to n-1, depending on the implementation), where n is the number of distinct categories of the
        variable. The numbers are assigned arbitrarily. This encoding method allows for quick benchmarking of machine
        learning models.

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
        :return: a pd.Dataframe
        """
        # intend code block on the canonical
        canonical = self._get_canonical(canonical)
        headers = Commons.list_formatter(headers)
        seed = self._seed() if seed is None else seed
        for header in headers:

            select = canonical[header].unique()
            select.sort()
            action = {}
            for i in range(len(select)):
                action.update({i:i+1})
            canonical[header] = self._correlate_categories(canonical=canonical, header=header,
                                                           correlations=select.tolist(), actions=action,
                                                           default_action=0, seed=seed, rtn_type='int')
            canonical[header] = canonical[header].astype(int)
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
        seed = self._seed() if seed is None else seed
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
        seed = self._seed() if seed is None else seed
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
        # prep the values to be a DataFrame if it isn't already
        action = deepcopy(action)
        selection = deepcopy(selection)
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

    def _correlate_element_choice(self, canonical: Any, header: str, list_size: int=None, random_choice: bool=None,
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
        action = deepcopy(action)
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

    def _correlate_sigmoid(self, canonical: Any, header: str, precision: int=None, seed: int=None,
                           rtn_type: str=None):
        """ logistic sigmoid a.k.a logit, takes an array of real numbers and transforms them to a value
        between (0,1) and is defined as
                                        f(x) = 1/(1+exp(-x)

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param precision: (optional) how many decimal places. default to 3
        :param seed: (optional) the random seed. defaults to current datetime
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
        precision = precision if isinstance(precision, int) else 3
        _seed = seed if isinstance(seed, int) else self._seed()
        rtn_values = np.round(1 / (1 + np.exp(-s_values)), precision)
        if isinstance(rtn_type, str):
            if rtn_type in ['category', 'object'] or rtn_type.startswith('int') or rtn_type.startswith('float'):
                rtn_values = rtn_values.astype(rtn_type)
            return rtn_values
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
        elif method.startswith('int'):
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
            if not is_numeric_dtype(s_values.dropna()):
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
            imputer = KNNImputer(n_neighbors=3, weights=weights)
            X = s_values.to_numpy().reshape(-1,1)
            s_values = pd.Series(imputer.fit_transform(X).reshape(1,-1)[0]).round(precision)
            if precision == 0:
                s_values = s_values.astype(int)
        elif isinstance(method, str) and method.startswith('constant'):
            if not isinstance(constant, (int, float, str)):
                raise ValueError("When using the 'constant' method a constant value must be provided")
            if is_numeric_dtype(s_values) and isinstance(constant, str):
                raise ValueError(f"The value '{constant}' is a string and column '{header}' expects a numeric value")
            s_values.iloc[null_idx] = constant
        elif isinstance(method, str) and method.startswith('indicator'):
            s_values = pd. Series(data=0, index=range(s_values.shape[0]))
            s_values.iloc[null_idx] = 1
        else:
            raise ValueError(f"The method '{method}' is not recognised. Please use one of 'mean', 'median' or 'mode'")
        return s_values.to_list()

    def _correlate_missing_weighted(self, canonical: Any, header: str, granularity: [int, float, list]=None,
                                    as_type: str=None, lower: [int, float]=None, upper: [int, float]=None,
                                    exclude_dominant: bool=None, replace_zero: [int, float]=None,
                                    precision: int=None, day_first: bool=None, year_first: bool=None, seed: int=None,
                                    rtn_type: str=None):
        """ imputes missing continuous or discrete data with a weighted distribution based on the analysis of the other
        elements in the column

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param granularity: (optional) the granularity of the analysis across the range. Default is 5
                int passed - represents the number of periods
                float passed - the length of each interval
                list[tuple] - specific interval periods defined by the list of tuples
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
        :return:
        """
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        as_type = as_type if isinstance(as_type, str) else s_values.dtype.name
        _seed = seed if isinstance(seed, int) else self._seed()
        null_idx = s_values[s_values.isna()].index
        if as_type.startswith('int') or as_type.startswith('float') or as_type.startswith('num'):
            _analysis = DataAnalytics(DataDiscovery.analyse_number(s_values, granularity=granularity, lower=lower,
                                                                   upper=upper, detail_stats=False, precision=precision,
                                                                   exclude_dominant=exclude_dominant))
            s_values.iloc[null_idx] = self._get_intervals(intervals=[tuple(x) for x in _analysis.intent.intervals],
                                                          relative_freq=_analysis.patterns.relative_freq,
                                                          precision=_analysis.params.precision,
                                                          seed=_seed, size=len(null_idx))
        elif as_type.startswith('cat'):
            _analysis = DataAnalytics(DataDiscovery.analyse_category(s_values, replace_zero=replace_zero))
            s_values.iloc[null_idx] = self._get_category(selection=_analysis.intent.categories,
                                                         relative_freq=_analysis.patterns.relative_freq,
                                                         seed=_seed, size=len(null_idx))
        elif as_type.startswith('date'):
            _analysis = DataAnalytics(DataDiscovery.analyse_date(s_values, granularity=granularity, lower=lower,
                                                                 upper=upper, day_first=day_first,
                                                                 year_first=year_first))
            s_values.iloc[null_idx] = self._get_datetime(start=_analysis.intent.lowest,
                                                         until=_analysis.intent.highest,
                                                         relative_freq=_analysis.patterns.relative_freq,
                                                         date_format=_analysis.params.data_format,
                                                         day_first=_analysis.params.day_first,
                                                         year_first=_analysis.params.year_first,
                                                         seed=_seed, size=len(null_idx))
        else:
            raise ValueError(f"The data type '{as_type}' is not supported. Try using the 'as_type' parameter")
        if isinstance(rtn_type, str):
            if rtn_type in ['category', 'object'] or rtn_type.startswith('int') or rtn_type.startswith('float'):
                s_values = s_values.astype(rtn_type)
            return s_values
        return s_values.to_list()

    def _correlate_numbers(self, canonical: Any, header: str, to_numeric: bool=None, standardize: bool=None,
                           normalize: tuple=None, scalarize: bool=None, transform: str=None, offset: [int, float, str]=None,
                           jitter: float=None, jitter_freq: list=None, precision: int=None, keep_zero: bool=None,
                           replace_nulls: [int, float]=None, seed: int=None, min_value: [int, float]=None,
                           max_value: [int, float]=None, rtn_type: str=None):
        """ Returns a number that correlates to the value given. The numbers can be standardized, normalize between
        given limits, scalarized or transformed and offers offset and jitters. The jitter is based on a normal
        distribution with the correlated value being the mean and the jitter its  standard deviation from that mean

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param to_numeric: (optional) ensures numeric type. None convertable strings are set to null
        :param standardize: (optional) if the column should be standardized
        :param normalize: (optional) normalize the column between two values. the tuple is the lower and upper bounds
        :param scalarize: (optional) assuming standard normally distributed, removes the mean and scaling
        :param transform: (optional) attempts normal distribution of values.
                            options are log, sqrt, cbrt, boxcox, yeojohnson
        :param offset: (optional) a fixed value to offset or if str an operation to perform using @ as the header value.
        :param jitter: (optional) a perturbation of the value where the jitter is a std. defaults to 0
        :param jitter_freq: (optional)  a relative freq with the pattern mid point the mid point of the jitter
        :param precision: (optional) how many decimal places. default to 3
        :param replace_nulls: (optional) a numeric value to replace nulls
        :param seed: (optional) the random seed. defaults to current datetime
        :param keep_zero: (optional) if True then zeros passed remain zero despite a change, Default is False
        :param min_value: a minimum value not to go below
        :param max_value: a max value not to go above
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :return: an equal length list of correlated values

        The offset can be a numeric offset that is added to the value, e.g. passing 2 will add 2 to all values.
        If a string is passed if format should be a calculation with the '@' character used to represent the column
        value. e.g.
            '1-@' would subtract the column value from 1,
            '@*0.5' would multiply the column value by 0.5
        """
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        s_values = canonical[header].copy()
        if s_values.empty:
            return list()
        if isinstance(to_numeric, bool) and to_numeric:
            s_values = pd.to_numeric(s_values.apply(str).str.replace('[$£€, ]', '', regex=True), errors='coerce')
        if not (s_values.dtype.name.startswith('int') or s_values.dtype.name.startswith('float')):
            raise ValueError(f"The header column is of type '{s_values.dtype.name}' and not numeric. "
                             f"Use the 'to_numeric' parameter if appropriate")
        keep_zero = keep_zero if isinstance(keep_zero, bool) else False
        precision = precision if isinstance(precision, int) else 3
        _seed = seed if isinstance(seed, int) else self._seed()
        if isinstance(replace_nulls, (int, float)):
            s_values[s_values.isna()] = replace_nulls
        null_idx = s_values[s_values.isna()].index
        zero_idx = s_values.where(s_values == 0).dropna().index if keep_zero else []
        if isinstance(offset, (int, float)) and offset != 0:
            s_values = s_values.add(offset)
        elif isinstance(offset, str):
            offset = offset.replace("@", 'x')
            s_values = s_values.apply(lambda x: eval(offset))
        if isinstance(jitter, (int, float)) and jitter != 0:
            sample = self._get_number(-abs(jitter) / 2, abs(jitter) / 2, relative_freq=jitter_freq,
                                      size=s_values.size, seed=_seed)
            s_values = s_values.add(sample)
        if isinstance(min_value, (int, float)):
            if min_value < s_values.max():
                min_idx = s_values.dropna().where(s_values < min_value).dropna().index
                s_values.iloc[min_idx] = min_value
            else:
                raise ValueError(f"The min value {min_value} is greater than the max result value {s_values.max()}")
        if isinstance(max_value, (int, float)):
            if max_value > s_values.min():
                max_idx = s_values.dropna().where(s_values > max_value).dropna().index
                s_values.iloc[max_idx] = max_value
            else:
                raise ValueError(f"The max value {max_value} is less than the min result value {s_values.min()}")
        if isinstance(standardize, bool) and standardize:
            s_values = pd.Series(Commons.list_standardize(s_values.to_list()))
        if isinstance(normalize, tuple):
            if normalize[0] >= normalize[1] or len(normalize) != 2:
                raise ValueError("The normalize tuple must be of size 2 with the first value lower than the second")
            s_values = pd.Series(Commons.list_normalize(s_values.to_list(), normalize[0], normalize[1]))
        if isinstance(transform, str):
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
        if isinstance(scalarize, bool):
            s_mean = s_values.mean()
            s_std = s_values.std()
            s_values = s_values.apply(lambda x: (x - s_mean)/s_std if s_std != 0 else x - s_mean)
        # reset the zero values if any
        s_values.iloc[zero_idx] = 0
        s_values = s_values.round(precision)
        if precision == 0 and not s_values.isnull().any():
            s_values = s_values.astype(int)
        if null_idx.size > 0:
            s_values.iloc[null_idx] = np.nan
        if isinstance(rtn_type, str):
            if rtn_type in ['category', 'object'] or rtn_type.startswith('int') or rtn_type.startswith('float'):
                s_values = s_values.astype(rtn_type)
            return s_values
        return s_values.to_list()

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
        actions = deepcopy(actions)
        correlations = deepcopy(correlations)
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

    def _correlate_dates(self, canonical: Any, header: str, offset: [int, dict]=None, jitter: int=None,
                         jitter_units: str=None, jitter_freq: list=None, now_delta: str=None, date_format: str=None,
                         min_date: str=None, max_date: str=None, fill_nulls: bool=None, day_first: bool=None,
                         year_first: bool=None, seed: int=None, rtn_type: str=None):
        """ correlates dates to an existing date or list of dates. The return is a list of pd

        :param canonical: a pd.DataFrame as the reference dataframe
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
        :param day_first: (optional) if the dates given are day first format. Default to True
        :param year_first: (optional) if the dates given are year first. Default to False
        :param date_format: (optional) the format of the output
        :param seed: (optional) a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :return: a list of equal size to that given
        """
        canonical = self._get_canonical(canonical, header=header)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        values = canonical[header].copy()
        if values.empty:
            return list()

        def _clean(control):
            _unit_type = ['years', 'months', 'weeks', 'days', 'leapdays', 'hours', 'minutes', 'seconds']
            _params = {}
            if isinstance(control, int):
                control = {'days': control}
            if isinstance(control, dict):
                for k, v in control.items():
                    if k not in _unit_type:
                        raise ValueError(f"The key '{k}' in 'offset', is not a recognised unit type for pd.DateOffset")
            return control

        _seed = self._seed() if seed is None else seed
        fill_nulls = False if fill_nulls is None or not isinstance(fill_nulls, bool) else fill_nulls
        offset = _clean(offset) if isinstance(offset, (dict, int)) else None
        if isinstance(now_delta, str) and now_delta not in ['Y', 'M', 'W', 'D', 'h', 'm', 's']:
            raise ValueError(f"the now_delta offset unit '{now_delta}' is not recognised "
                             f"use of of ['Y', 'M', 'W', 'D', 'h', 'm', 's']")
        units_allowed = ['W', 'D', 'h', 'm', 's']
        jitter_units = jitter_units if isinstance(jitter_units, str) and jitter_units in units_allowed else 'D'
        jitter = pd.Timedelta(value=jitter, unit=jitter_units) if isinstance(jitter, int) else None
        # set minimum date
        _min_date = pd.to_datetime(min_date, errors='coerce', infer_datetime_format=True, utc=True)
        if _min_date is None or _min_date is pd.NaT:
            _min_date = pd.to_datetime(pd.Timestamp.min, utc=True)
        # set max date
        _max_date = pd.to_datetime(max_date, errors='coerce', infer_datetime_format=True, utc=True)
        if _max_date is None or _max_date is pd.NaT:
            _max_date = pd.to_datetime(pd.Timestamp.max, utc=True)
        if _min_date >= _max_date:
            raise ValueError(f"the min_date {min_date} must be less than max_date {max_date}")
        # convert values into datetime
        s_values = pd.Series(pd.to_datetime(values.copy(), errors='coerce', infer_datetime_format=True,
                                            dayfirst=day_first, yearfirst=year_first, utc=True))
        if jitter is not None:
            if jitter_units in ['W', 'D']:
                value = jitter.days
                zip_units = 'D'
            else:
                value = int(jitter.to_timedelta64().astype(int) / 1000000000)
                zip_units = 's'
            zip_spread = self._get_number(-abs(value) / 2, (abs(value + 1) / 2), relative_freq=jitter_freq,
                                          precision=0, size=s_values.size, seed=_seed)
            zipped_dt = list(zip(zip_spread, [zip_units]*s_values.size))
            s_values += np.array([pd.Timedelta(x, y).to_timedelta64() for x, y in zipped_dt])
        if fill_nulls:
            generator = np.random.default_rng(seed=_seed)
            s_values = s_values.fillna(generator.choice(s_values.mode()))
        null_idx = s_values[s_values.isna()].index
        if isinstance(offset, dict) and offset:
            s_values = s_values.add(pd.DateOffset(**offset))
        if _min_date > pd.to_datetime(pd.Timestamp.min, utc=True):
            if _min_date > s_values.min():
                min_idx = s_values.dropna().where(s_values < _min_date).dropna().index
                s_values.iloc[min_idx] = _min_date
            else:
                raise ValueError(f"The min value {min_date} is greater than the max result value {s_values.max()}")
        if _max_date < pd.to_datetime(pd.Timestamp.max, utc=True):
            if _max_date < s_values.max():
                max_idx = s_values.dropna().where(s_values > _max_date).dropna().index
                s_values.iloc[max_idx] = _max_date
            else:
                raise ValueError(f"The max value {max_date} is less than the min result value {s_values.min()}")
        if now_delta:
            s_values = (s_values.dt.tz_convert(None) - pd.Timestamp.now()).abs()
            s_values = (s_values / np.timedelta64(1, now_delta))
            s_values = s_values.round(0) if null_idx.size > 0 else s_values.astype(int)
        else:
            if isinstance(date_format, str):
                s_values = s_values.dt.strftime(date_format)
            else:
                s_values = s_values.dt.tz_convert(None)
            if null_idx.size > 0:
                s_values.iloc[null_idx].apply(lambda x: np.nan)
        if isinstance(rtn_type, str):
            if rtn_type in ['category', 'object'] or rtn_type.startswith('int') or rtn_type.startswith('float'):
                s_values = s_values.astype(rtn_type)
            return s_values
        return s_values.to_list()

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

    """
        UTILITY METHODS SECTION
    """

    @staticmethod
    def _convert_date2value(dates: Any, day_first: bool = True, year_first: bool = False):
        values = pd.to_datetime(dates, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                                yearfirst=year_first)
        return mdates.date2num(pd.Series(values)).tolist()

    @staticmethod
    def _convert_value2date(values: Any, date_format: str=None):
        dates = []
        for date in mdates.num2date(values):
            date = pd.Timestamp(date)
            if isinstance(date_format, str):
                date = date.strftime(date_format)
            dates.append(date)
        return dates

    @staticmethod
    def _freq_dist_size(relative_freq: list, size: int, seed: int=None):
        """ utility method taking a list of relative frequencies and based on size returns the size distribution
        of element based on the frequency. The distribution is based upon binomial distributions

        :param relative_freq: a list of int or float values representing a relative distribution frequency
        :param size: the size to be distributed
        :param seed: (optional) a seed value for the random function: default to None
        :return: an integer list of the distribution that sum to the size
        """
        if not isinstance(relative_freq, list) or not all(isinstance(x, (int, float)) for x in relative_freq):
            raise ValueError("The weighted pattern must be an list of numbers")
        seed = seed if isinstance(seed, int) else int(time.time() * np.random.random())
        if sum(relative_freq) != 1:
            relative_freq = np.round(relative_freq / np.sum(relative_freq), 5)
        generator = np.random.default_rng(seed=seed)
        result = list(generator.binomial(n=size, p=relative_freq, size=len(relative_freq)))
        diff = size - sum(result)
        adjust = [0] * len(relative_freq)
        if diff != 0:
            unit = diff / sum(relative_freq)
            for idx in range(len(relative_freq)):
                adjust[idx] = int(round(relative_freq[idx] * unit, 0))
        result = [a + b for (a, b) in zip(result, adjust)]
        # There is a possibility the required size is not fulfilled, therefore add or remove elements based on freq

        def _freq_choice(p: list):
            """returns a single index of the choice of the relative frequency"""
            rnd = generator.random() * sum(p)
            for i, w in enumerate(p):
                rnd -= w
                if rnd < 0:
                    return i

        while sum(result) != size:
            if sum(result) < size:
                result[_freq_choice(relative_freq)] += 1
            else:
                weight_idx = _freq_choice(relative_freq)
                if result[weight_idx] > 0:
                    result[weight_idx] -= 1
        # ensure we have all positive values
        return [0 if x < 0 else x for x in result]
