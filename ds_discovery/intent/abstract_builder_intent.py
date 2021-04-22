import ast
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Any
from matplotlib import dates as mdates
from scipy import stats
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
        if simulate:
            return pd.DataFrame.from_dict(col_sim)
        return canonical

    def _get_number(self, from_value: [int, float]=None, to_value: [int, float]=None, relative_freq: list=None,
                    precision: int=None, ordered: str=None, at_most: int=None, size: int=None,
                    seed: int=None) -> list:
        """ returns a number in the range from_value to to_value. if only to_value given from_value is zero

        :param from_value: (signed) integer to start from
        :param to_value: optional, (signed) integer the number sequence goes to but not include
        :param relative_freq: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param ordered: order the data ascending 'asc' or descending 'dec', values accepted 'asc' or 'des'
        :param at_most: the most times a selection should be chosen
        :param size: the size of the sample
        :param seed: a seed value for the random function: default to None
        """
        if not isinstance(from_value, (int, float)) and not isinstance(to_value, (int, float)):
            raise ValueError(f"either a 'range_value' or a 'range_value' and 'to_value' must be provided")
        if not isinstance(from_value, (float, int)):
            from_value = 0
        if not isinstance(to_value, (float, int)):
            (from_value, to_value) = (0, from_value)
        if to_value <= from_value:
            raise ValueError("The number range must be a positive different, found to_value <= from_value")
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
        if isinstance(relative_freq, list) and len(relative_freq) > 1:
            freq_dist_size = self._freq_dist_size(relative_freq=relative_freq, size=size, seed=_seed)
        else:
            freq_dist_size = [size]
        # generate the numbers
        rtn_list = []
        generator = np.random.default_rng(seed=_seed)
        dtype = int if is_int else float
        bins = np.linspace(from_value, to_value, len(freq_dist_size) + 1, dtype=dtype)
        for idx in np.arange(1, len(bins)):
            low = bins[idx - 1]
            high = bins[idx]
            if low >= high:
                continue
            elif at_most > 0:
                sample = []
                for _ in np.arange(at_most, dtype=dtype):
                    count_size = freq_dist_size[idx - 1] * generator.integers(2, 4, size=1)[0]
                    sample += list(set(np.linspace(bins[idx - 1], bins[idx], num=count_size, dtype=dtype,
                                                   endpoint=False)))
                if len(sample) < freq_dist_size[idx - 1]:
                    raise ValueError(f"The value range has insufficient samples to choose from when using at_most."
                                     f"Try increasing the range of values to sample.")
                rtn_list += list(generator.choice(sample, size=freq_dist_size[idx - 1], replace=False))
            else:
                if dtype == int:
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

    def _get_category(self, selection: list, relative_freq: list=None, size: int=None, at_most: int=None,
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
        if not isinstance(selection, list) or len(selection) == 0:
            return [None]*size
        _seed = self._seed() if seed is None else seed
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
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param size: the size of the sample
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        # Code block for intent
        size = 1 if size is None else size
        if not isinstance(precision, int):
            precision = 0 if all(isinstance(v[0], int) and isinstance(v[1], int) for v in intervals) else 3
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

    def _get_dist_normal(self, mean: float, std: float, size: int=None, seed: int=None) -> list:
        """A normal (Gaussian) continuous random distribution.

        :param mean: The mean (“centre”) of the distribution.
        :param std: The standard deviation (jitter or “width”) of the distribution. Must be >= 0
        :param size: the size of the sample. if a tuple of intervals, size must match the tuple
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=_seed)
        rtn_list = list(generator.normal(loc=mean, scale=std, size=size))
        return rtn_list

    def _get_dist_logistic(self, mean: float, std: float, size: int=None, seed: int=None) -> list:
        """A logistic continuous random distribution.

        :param mean: The mean (“centre”) of the distribution.
        :param std: The standard deviation (jitter or “width”) of the distribution. Must be >= 0
        :param size: the size of the sample. if a tuple of intervals, size must match the tuple
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=_seed)
        rtn_list = list(generator.logistic(loc=mean, scale=std, size=size))
        return rtn_list

    def _get_dist_exponential(self, scale: [int, float], size: int=None, seed: int=None) -> list:
        """An exponential continuous random distribution.

        :param scale: The scale of the distribution.
        :param size: the size of the sample. if a tuple of intervals, size must match the tuple
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=_seed)
        rtn_list = list(generator.exponential(scale=scale, size=size))
        return rtn_list

    def _get_dist_gumbel(self, mean: float, std: float, size: int=None, seed: int=None) -> list:
        """An gumbel continuous random distribution.

        The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme Value Type I) distribution is one of
        a class of Generalized Extreme Value (GEV) distributions used in modeling extreme value problems.
        The Gumbel is a special case of the Extreme Value Type I distribution for maximums from distributions
        with “exponential-like” tails.

        :param mean: The mean (“centre”) of the distribution.
        :param std: The standard deviation (jitter or “width”) of the distribution. Must be >= 0
        :param size: the size of the sample. if a tuple of intervals, size must match the tuple
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=_seed)
        rtn_list = list(generator.gumbel(loc=mean, scale=std, size=size))
        return rtn_list

    def _get_dist_binomial(self, trials: int, probability: float, size: int=None, seed: int=None) -> list:
        """A binomial discrete random distribution. The Binomial Distribution represents the number of
           successes and failures in n independent Bernoulli trials for some given value of n

        :param trials: the number of trials to attempt, must be >= 0.
        :param probability: the probability distribution, >= 0 and <=1.
        :param size: the size of the sample. if a tuple of intervals, size must match the tuple
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=_seed)
        rtn_list = list(generator.binomial(n=trials, p=probability, size=size))
        return rtn_list

    def _get_dist_poisson(self, interval: float, size: int=None, seed: int=None) -> list:
        """A Poisson discrete random distribution

        :param interval: Expectation of interval, must be >= 0.
        :param size: the size of the sample.
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        generator = np.random.default_rng(seed=_seed)
        rtn_list = list(generator.poisson(lam=interval, size=size))
        return rtn_list

    def _get_dist_bernoulli(self, probability: float, size: int=None, seed: int=None) -> list:
        """A Bernoulli discrete random distribution using scipy

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

    def _get_distribution(self, distribution: str, package: str=None, precision: int=None, size: int=None,
                          seed: int=None, **kwargs) -> list:
        """returns a number based the distribution type.

        :param distribution: The string name of the distribution function from numpy random Generator class
        :param package: (optional) The name of the package to use, options are 'numpy' (default) and 'scipy'.
        :param precision: (optional) the precision of the returned number
        :param size: (optional) the size of the sample
        :param seed: (optional) a seed value for the random function: default to None
        :return: a random number
        """
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        precision = 3 if precision is None else precision
        if isinstance(package, str) and package == 'scipy':
            rtn_list = eval(f"stats.{distribution}.rvs(size=size, random_state=_seed, **kwargs)", globals(), locals())
        else:
            generator = np.random.default_rng(seed=_seed)
            rtn_list = eval(f"generator.{distribution}(size=size, **kwargs)", globals(), locals())
        rtn_list = list(rtn_list.round(precision))
        return rtn_list

    def _get_selection(self, canonical: Any, column_header: str, relative_freq: list=None, sample_size: int=None,
                       selection_size: int=None, size: int=None, at_most: bool=None, shuffle: bool=None,
                       seed: int=None) -> list:
        """ returns a random list of values where the selection of those values is taken from a connector source.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param column_header: the name of the column header to correlate
        :param relative_freq: (optional) a weighting pattern of the final selection
        :param selection_size: (optional) the selection to take from the sample size, normally used with shuffle
        :param sample_size: (optional) the size of the sample to take from the reference file
        :param at_most: (optional) the most times a selection should be chosen
        :param shuffle: (optional) if the selection should be shuffled before selection. Default is true
        :param size: (optional) size of the return. default to 1
        :param seed: (optional) a seed value for the random function: default to None
        :return: list

        The canonical is normally a connector contract str reference or a set of parameter instructions on how to
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
        canonical = self._get_canonical(canonical)
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

    def _frame_starter(self, canonical: Any, selection: list=None, headers: [str, list]=None, drop: bool=None,
                       dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None,
                       rename_map: dict=None, default_size: int=None, seed: int=None) -> pd.DataFrame:
        """ Selects rows and/or columns changing the shape of the DatFrame. This is always run last in a pipeline
        Rows are filtered before the column filter so columns can be referenced even though they might not be included
        the final column list.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param selection: a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
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
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        rtn_frame = Commons.filter_columns(canonical, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                           regex=regex, re_ignore_case=re_ignore_case)
        if isinstance(rename_map, dict):
            rtn_frame.rename(mapper=rename_map, axis='columns', inplace=True)
        return rtn_frame

    def _frame_selection(self, canonical: Any, selection: list=None, headers: [str, list]=None,
                         drop: bool=None, dtype: [str, list]=None, exclude: bool=None, regex: [str, list]=None,
                         re_ignore_case: bool=None, seed: int=None) -> pd.DataFrame:
        """ This method always runs at the start of the pipeline, taking a direct or generated pd.DataFrame,
        see context notes below, as the foundation canonical of all subsequent steps of the pipeline.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param selection: a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'genre', 'condition': "=='Comedy'"}]
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
        return self._frame_starter(canonical=canonical, selection=selection, headers=headers, drop=drop, dtype=dtype,
                                   exclude=exclude, regex=regex, re_ignore_case=re_ignore_case, seed=seed)

    def _model_iterator(self, canonical: Any, marker_col: str=None, starting_frame: str=None, selection: list=None,
                        default_action: dict=None, iteration_actions: dict=None, iter_start: int=None,
                        iter_stop: int=None, seed: int=None) -> pd.DataFrame:
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

    def _model_group(self, canonical: Any, headers: [str, list], group_by: [str, list], aggregator: str=None,
                     list_choice: int=None, list_max: int=None, drop_group_by: bool=False, seed: int=None,
                     include_weighting: bool=False, freq_precision: int=None, remove_weighting_zeros: bool=False,
                     remove_aggregated: bool=False) -> pd.DataFrame:
        """ returns the full column values directly from another connector data source. in addition the the
        standard groupby aggregators there is also 'list' and 'set' that returns an aggregated list or set.
        These can be using in conjunction with 'list_choice' and 'list_size' allows control of the return values.
        if list_max is set to 1 then a single value is returned rather than a list of size 1.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param headers: the column headers to apply the aggregation too
        :param group_by: the column headers to group by
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
        headers = Commons.list_formatter(headers)
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

    def _model_merge(self, canonical: Any, other: Any, left_on: str=None, right_on: str=None,
                     on: str=None, how: str=None, headers: list=None, suffixes: tuple=None, indicator: bool=None,
                     validate: str=None, seed: int=None) -> pd.DataFrame:
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
        indicator = indicator if isinstance(indicator, bool) else False
        suffixes = suffixes if isinstance(suffixes, tuple) and len(suffixes) == 2 else ('', '_dup')
        # Filter on the columns
        if isinstance(headers, list):
            headers.append(right_on if isinstance(right_on, str) else on)
            other = Commons.filter_columns(other, headers=headers)
        df_rtn = pd.merge(left=canonical, right=other, how=how, left_on=left_on, right_on=right_on, on=on,
                          suffixes=suffixes, indicator=indicator, validate=validate)
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
        return pd.concat([canonical, df_rtn], axis=axis)

    def _model_explode(self, canonical: Any, header: str, seed: int=None) -> pd.DataFrame:
        """ takes a single column of list values and explodes the DataFrame so row is represented by each elements
        in the row list


        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header of the column to be exploded
        :param seed: (optional) this is a place holder, here for compatibility across methods
        :return: a pd.DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:
        """
        canonical = self._get_canonical(canonical)
        if not isinstance(header, str) or header not in canonical.columns:
            raise ValueError(f"The header '{header}' can't be found in the canonical DataFrame")
        _seed = self._seed() if seed is None else seed
        return canonical.explode(column=header, ignore_index=True)

    def _model_sample(self, canonical: Any, sample: Any, columns_list: list=None, exclude_associate: list=None,
                      detail_numeric: bool=None, strict_typing: bool=None, category_limit: int=None,
                      apply_bias: bool=None, seed: int = None) -> pd.DataFrame:
        """

        :param canonical:
        :param sample:
        :param columns_list:
        :param exclude_associate:
        :param detail_numeric:
        :param strict_typing:
        :param category_limit:
        :param apply_bias:
        :param seed: (optional) this is a place holder, here for compatibility across methods
        :return: a pd.DataFrame
        """
        canonical = self._get_canonical(canonical)
        sample = self._get_canonical(sample)
        columns_list = columns_list if isinstance(columns_list, list) else list(sample.columns)
        blob = DataDiscovery.analyse_association(sample, columns_list=columns_list, exclude_associate=exclude_associate,
                                                 detail_numeric=detail_numeric, strict_typing=strict_typing,
                                                 category_limit=category_limit)
        return self._model_analysis(canonical=canonical, analytics_blob=blob, apply_bias=apply_bias, seed=seed)

    def _model_script(self, canonical: Any, script_contract: str, seed: int = None) -> pd.DataFrame:
        """

        :param canonical:
        :param script_contract:
        :param seed: (optional) this is a place holder, here for compatibility across methods
        :return: a pd.DataFrame
        """
        canonical = self._get_canonical(canonical)
        script = self._get_canonical(script_contract)
        type_options = {'number': '_get_number', 'date': '_get_datetime', 'category': 'get_category',
                        'selection': 'get_selection', 'intervals': 'get_intervals', 'distribution': 'get_distribution'}
        script['params'] = script['params'].replace(['', ' '], np.nan)
        script['params'].loc[script['params'].isna()] = '[]'
        script['params'] = [ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']')
                            else x for x in script['params']]
        # replace all other items with list
        script['params'] = [x if isinstance(x, list) else [x] for x in script['params']]
        script['params'] = script['params'].astype('object')

        for index, row in script.iterrows():
            method = type_options.get(row['type'])
            params = row['params']
            canonical[row['name']] = eval(f"self.{method}(size={canonical.shape[0]}, **params)", globals(), locals())
        return canonical

    def _model_analysis(self, canonical: Any, analytics_blob: dict, apply_bias: bool=None,
                        seed: int=None) -> pd.DataFrame:
        """ builds a set of columns based on an analysis dictionary of weighting (see analyse_association)
        if a reference DataFrame is passed then as the analysis is run if the column already exists the row
        value will be taken as the reference to the sub category and not the random value. This allows already
        constructed association to be used as reference for a sub category.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param analytics_blob: the analytics blob from DataDiscovery.analyse_association(...)
        :param apply_bias: (optional) if dominant values have been excluded, re-include to maintain bias
        :param seed: seed: (optional) a seed value for the random function: default to None
        :return: a DataFrame
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
                if apply_bias and _analysis.patterns.is_element('dominant_excluded'):
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
        apply_bias = apply_bias if isinstance(apply_bias, bool) else True
        row_dict = dict()
        seed = self._seed() if seed is None else seed
        size = canonical.shape[0]
        get_level(analytics_blob, sample_size=size, _seed=seed)
        for key in row_dict.keys():
            row_dict[key] = row_dict[key][:size]
        return pd.concat([canonical, pd.DataFrame.from_dict(data=row_dict)], axis=1)

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

    def _correlate_custom(self, canonical: Any, code_str: str, use_exec: bool=None, seed: int=None,
                          rtn_type: str=None, **kwargs):
        """ enacts an action on a dataFrame, returning the output of the action or the DataFrame if using exec or
        the evaluation returns None. Note that if using the input dataframe in your action, it is internally referenced
        as it's parameter name 'canonical'.

        :param canonical: a pd.DataFrame as the reference dataframe
        :param code_str: an action on those column values
        :param use_exec: (optional) By default the code runs as eval if set to true exec would be used
        :param kwargs: a set of kwargs to include in any executable function
        :param seed: (optional) a seed value for the random function: default to None
        :param rtn_type: (optional) changes the default return of a 'list' to a pd.Series
                other than the int, float, category, string and object, passing 'as-is' will return as is
        :return: a list or pandas.DataFrame
        """
        canonical = self._get_canonical(canonical)
        _seed = seed if isinstance(seed, int) else self._seed()
        use_exec = use_exec if isinstance(use_exec, bool) else False
        local_kwargs = locals().get('kwargs') if 'kwargs' in locals() else dict()
        if 'canonical' not in local_kwargs:
            local_kwargs['canonical'] = canonical

        rtn_values = exec(code_str, globals(), local_kwargs) if use_exec else eval(code_str, globals(), local_kwargs)
        if rtn_values is None:
            return pd.Series([np.nan] * canonical.shape[0])
        if isinstance(rtn_type, str):
            if rtn_type in ['category', 'object'] or rtn_type.startswith('int') or rtn_type.startswith('float'):
                rtn_values = rtn_values.astype(rtn_type)
            return rtn_values
        return rtn_values.to_list()

    def _correlate_aggregate(self, canonical: Any, headers: list, agg: str, seed: int=None, precision: int=None,
                             rtn_type: str=None):
        """ correlate two or more columns with each other through a finite set of aggregation functions. The
        aggregation function names are limited to 'sum', 'prod', 'count', 'min', 'max' and 'mean' for numeric columns
        and a special 'list' function name to combine the columns as a list

        :param canonical: a pd.DataFrame as the reference dataframe
        :param headers: a list of headers to correlate
        :param agg: the aggregation function name enact. The available functions are:
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

    def _correlate_choice(self, canonical: Any, header: str, list_size: int=None, random_choice: bool=None,
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

    def _correlate_missing(self, canonical: Any, header: str, granularity: [int, float]=None,
                           as_type: str=None, lower: [int, float]=None, upper: [int, float]=None, nulls_list: list=None,
                           exclude_dominant: bool=None, replace_zero: [int, float]=None, precision: int=None,
                           day_first: bool=None, year_first: bool=None, seed: int=None,
                           rtn_type: str=None):
        """ imputes missing data with a weighted distribution based on the analysis of the other elements in the
            column

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param granularity: (optional) the granularity of the analysis across the range. Default is 5
                int passed - represents the number of periods
                float passed - the length of each interval
                list[tuple] - specific interval periods e.g []
                list[float] - the percentile or quantities, All should fall between 0 and 1
        :param as_type: (optional) specify the type to analyse
        :param lower: (optional) the lower limit of the number value. Default min()
        :param upper: (optional) the upper limit of the number value. Default max()
        :param nulls_list: (optional) a list of nulls that should be considered null
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
        nulls_list = nulls_list if isinstance(nulls_list, list) else [np.nan, None, 'nan', '', ' ']
        if isinstance(nulls_list, list):
            s_values.replace(nulls_list, np.nan, inplace=True, regex=True)
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

    def _correlate_numbers(self, canonical: Any, header: str, to_numeric: bool=None,
                           offset: [int, float, str]=None, jitter: float=None, jitter_freq: list=None,
                           precision: int=None, replace_nulls: [int, float]=None, seed: int=None, keep_zero: bool=None,
                           min_value: [int, float]=None, max_value: [int, float]=None, rtn_type: str=None):
        """ returns a number that correlates to the value given. The jitter is based on a normal distribution
        with the correlated value being the mean and the jitter its standard deviation from that mean

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param to_numeric: if the column should be converted to a numeric type. strings not convertible are set to null
        :param offset: (optional) a fixed value to offset or if str an operation to perform using @ as the header value.
        :param jitter: (optional) a perturbation of the value where the jitter is a std. defaults to 0
        :param jitter_freq: (optional)  a relative freq with the pattern mid point the mid point of the jitter
        :param precision: (optional) how many decimal places. default to 3
        :param replace_nulls: (optional) a numeric value to replace nulls
        :param seed: (optional) the random seed. defaults to current datetime
        :param keep_zero: (optional) if True then zeros passed remain zero, Default is False
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
            multiple choice correlation
                [['A','B'], 'C'] # if values is 'A' OR 'B' then action is 0 and so on
            actions dictionary where the method is a class method followed by its parameters
                {0: {'method': 'get_numbers', 'from_value': 0, to_value: 27}}
            you can also use the action to specify a specific value:
                {0: 'F', 1: {'method': 'get_numbers', 'from_value': 0, to_value: 27}}

        :param canonical: a pd.DataFrame as the reference dataframe
        :param header: the header in the DataFrame to correlate
        :param correlations: a list of categories (can also contain lists for multiple correlations.
        :param actions: the correlated set of categories that should map to the index
        :param default_action: (optional) a default action to take if the selection is not fulfilled
        :param seed: a seed value for the random function: default to None
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
            s_values = (s_values.dt.tz_convert(None) - pd.Timestamp('now')).abs()
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
        return result

    @staticmethod
    def _seed(seed: int=None, increment: bool=False):
        if not isinstance(seed, int):
            return int(time.time() * np.random.default_rng().random())
        if increment:
            seed += 1
            if seed > 2 ** 31:
                seed = int(time.time() * np.random.default_rng(seed=seed-1).random())
        return seed
