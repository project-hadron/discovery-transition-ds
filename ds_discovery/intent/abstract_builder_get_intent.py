import numpy as np
from abc import abstractmethod
import pandas as pd
from typing import Any

from aistac.components.aistac_commons import DataAnalytics
from scipy import stats

from ds_discovery.components.commons import Commons
from ds_discovery.intent.abstract_common_intent import AbstractCommonsIntentModel

__author__ = 'Darryl Oatridge'


class AbstractBuilderGetIntent(AbstractCommonsIntentModel):

    @abstractmethod
    def run_intent_pipeline(self, *args, **kwargs) -> [None, tuple]:
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.
        """

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
        from_value = self._extract_value(from_value)
        to_value = self._extract_value(to_value)
        if not size or size == 0:
            raise ValueError("size not set. Size must be an int greater than zero")
        if not isinstance(from_value, (int, float)) and not isinstance(to_value, (int, float)):
            raise ValueError(f"either a 'from_value' or a 'from_value' and 'to_value' must be provided")
        if not isinstance(from_value, (float, int)):
            from_value = 0
        if not isinstance(to_value, (float, int)):
            (from_value, to_value) = (0, from_value)
        if to_value <= from_value:
            raise ValueError("The number range must be a positive difference, where to_value <= from_value")
        at_most = 0 if not isinstance(at_most, int) else at_most
#        size = size if isinstance(size, int) else 1
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
                    choice = np.round(choice * (high - low) + low, precision).tolist()
                    # make sure the precision
                    choice = [high - 10 ** (-precision) if x >= high else x for x in choice]
                    rtn_list += choice
        # order or shuffle the return list
        if isinstance(ordered, str) and ordered.lower() in ['asc', 'des']:
            rtn_list.sort(reverse=True if ordered.lower() == 'asc' else False)
        else:
            generator.shuffle(rtn_list)
        return rtn_list

    def _get_category(self, selection: list, size: int, relative_freq: list=None,
                      seed: int=None) -> list:
        """ returns a category from a list. Of particular not is the at_least parameter that allows you to
        control the number of times a selection can be chosen.
    
        :param selection: a list of items to select from
        :param size: size of the return
        :param relative_freq: a weighting pattern that does not have to add to 1
        :param seed: a seed value for the random function: default to None
        :return: an item or list of items chosen from the list
        """
        if len(selection) < 1:
            return [None] * size
        seed = self._seed() if seed is None else seed
        relative_freq = relative_freq if isinstance(relative_freq, list) else [1]*len(selection)
        select_index = self._freq_dist_size(relative_freq=relative_freq, size=size, dist_length=len(selection),
                                                  dist_on='right', seed=seed)
        rtn_list = []
        for idx in range(len(select_index)):
            rtn_list += [selection[idx]]*select_index[idx]
        gen = np.random.default_rng(seed)
        gen.shuffle(rtn_list)
        return rtn_list
    
    def _get_datetime(self, start: Any, until: Any, relative_freq: list=None, at_most: int=None, ordered: str=None,
                      date_format: str=None, as_num: bool=None, ignore_time: bool=None, ignore_seconds: bool=None,
                      size: int=None, seed: int=None, day_first: bool=None, year_first: bool=None) -> list:
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
        :param ignore_seconds: ignore second elements and only select from Year to minute elements. Default is False
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
        if not isinstance(size, int):
            raise ValueError("size not set. Size must be an int greater than zero")
        if start is None or until is None:
            raise ValueError("The start or until parameters cannot be of NoneType")
        # Code block for intent
        as_num = as_num if isinstance(as_num, bool) else False
        ignore_seconds = ignore_seconds if isinstance(ignore_seconds, bool) else False
        ignore_time = ignore_time if isinstance(ignore_time, bool) else False
        size = 1 if size is None else size
        _seed = self._seed() if seed is None else seed
        # start = start.to_pydatetime() if isinstance(start, pd.Timestamp) else start
        # until = until.to_pydatetime() if isinstance(until, pd.Timestamp) else until
        if isinstance(start, int):
            start = (pd.Timestamp.now() + pd.Timedelta(days=start))
        start = pd.to_datetime(start, errors='coerce', dayfirst=day_first,
                               yearfirst=year_first)
        if isinstance(until, int):
            until = (pd.Timestamp.now() + pd.Timedelta(days=until))
        elif isinstance(until, dict):
            until = (start + pd.Timedelta(**until))
        until = pd.to_datetime(until, errors='coerce', dayfirst=day_first,
                               yearfirst=year_first)
        if start == until:
            rtn_list = pd.Series([start] * size)
        else:
            dt_tz = pd.Series(start).dt.tz
            _dt_start = Commons.date2value(start, day_first=day_first, year_first=year_first)[0]
            _dt_until = Commons.date2value(until, day_first=day_first, year_first=year_first)[0]
            precision = 15
            rtn_list = self._get_number(from_value=_dt_start, to_value=_dt_until, relative_freq=relative_freq,
                                        at_most=at_most, ordered=ordered, precision=precision, size=size, seed=seed)
            rtn_list = pd.Series(Commons.value2date(rtn_list, dt_tz=dt_tz))
        if ignore_time:
            rtn_list = pd.Series(pd.DatetimeIndex(rtn_list).normalize())
        if ignore_seconds:
            rtn_list = rtn_list.apply(lambda t: t.replace(second=0, microsecond=0, nanosecond=0))
        if as_num:
            return Commons.date2value(rtn_list)
        if isinstance(date_format, str) and len(rtn_list) > 0:
            rtn_list = rtn_list.dt.strftime(date_format)
        return rtn_list.to_list()
    
    
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
        if not isinstance(size, int):
            raise ValueError("size not set. Size must be an int greater than zero")
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
                margin = 10 ** (((-1) * precision) - 1)
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
        if not isinstance(size, int):
            raise ValueError("size not set. Size must be an int greater than zero")
        _seed = self._seed() if seed is None else seed
        precision = precision if isinstance(precision, int) else 3
        generator = np.random.default_rng(seed=_seed)
        rtn_list = list(generator.normal(loc=mean, scale=std, size=size))
        return list(np.around(rtn_list, precision))
    
    def _get_dist_choice(self, number: [int, str, float], size: int=None, seed: int=None) -> list:
        """Creates a list of latent values of 0 or 1 where 1 is randomly selected both upon the number given.
    
       :param number: The number of true (1) values to randomly chose from the canonical. see below
       :param size: the size of the sample. if a tuple of intervals, size must match the tuple
       :param seed: a seed value for the random function: default to None
       :return: a list of 1 or 0
    
        As choice is a fixed value, number can be represented by an environment variable with the format '${NAME}'
        where NAME is the environment variable name
    
        If number is an int then that number of 1's are chosen. If number is a float between 0 and 1 it is taken as
        a fraction of the total variable count
        """
        if not isinstance(size, int):
            raise ValueError("size not set. Size must be an int greater than zero")
        _seed = self._seed() if seed is None else seed
        number = self._extract_value(number)
        number = int(number * size) if isinstance(number, float) and 0 <= number <= 1 else int(number)
        number = number if 0 <= number < size else size
        if isinstance(number, int) and 0 <= number <= size:
            rtn_list = pd.Series(data=[0] * size)
            choice_idx = self._get_number(to_value=size, size=number, at_most=1, precision=0, ordered='asc', seed=_seed)
            rtn_list.iloc[choice_idx] = [1] * number
            return rtn_list.reset_index(drop=True).to_list()
        return pd.Series(data=[0] * size).to_list()
    
    def _get_dist_bernoulli(self, probability: float, size: int=None, seed: int=None) -> list:
        """A Bernoulli process is a discrete random distribution. Bernoulli trial is a random experiment with exactly
        two possible outcomes, "success" and "failure", in which the probability of success is the same every time
        the experiment is conducted.
    
        The mathematical formalisation of the Bernoulli trial, this distribution,  is known as the Bernoulli process.
        A Bernoulli process is a finite or infinite sequence of binary random variables. Prosaically, a Bernoulli
        process is a repeated coin flipping, possibly with an unfair coin (but with consistent unfairness).

        As probability is a fixed value, probability can be represented by an environment variable with the
        format '${NAME}' where NAME is the environment variable name
    
        :param probability: the probability occurrence of getting a 1 or 0
        :param size: the size of the sample
        :param seed: a seed value for the random function: default to None
        :return: a random number
        """
        if not isinstance(size, int):
            raise ValueError("size not set. Size must be an int greater than zero")
        _seed = self._seed() if seed is None else seed
        probability = self._extract_value(probability)
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
        if not isinstance(size, int):
            raise ValueError("size not set. Size must be an int greater than zero")
        precision = precision if isinstance(precision, int) else 3
        seed = self._seed() if seed is None else seed
        rtn_list = stats.truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
        rtn_list = rtn_list.rvs(size, random_state=seed).round(precision)
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
        if not isinstance(size, int):
            raise ValueError("size not set. Size must be an int greater than zero")
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
                       selection_size: int=None, size: int=None, shuffle: bool=None, seed: int=None) -> list:
        """ returns a random list of values where the selection of those values is taken from a connector source.
    
        :param select_source: the selection source for the reference dataframe
        :param column_header: the name of the column header to correlate
        :param relative_freq: (optional) a weighting pattern of the final selection
        :param selection_size: (optional) the selection to take from the sample size, normally used with shuffle
        :param sample_size: (optional) the size of the sample to take from the reference file
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
        if not isinstance(size, int):
            raise ValueError("size not set. Size must be an int greater than zero")
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
        return self._get_category(selection=_values.to_list(), relative_freq=relative_freq, size=size,
                                  seed=_seed)

    def _get_analysis(self, schema: dict, size: int):

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

        row_dict = dict()
