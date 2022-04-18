import inspect
import re
import string
from uuid import uuid1, uuid3, uuid4, uuid5, UUID
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Any
from ds_discovery.components.commons import Commons
from ds_discovery.intent.wrangle_intent import WrangleIntentModel
from ds_discovery.managers.synthetic_property_manager import SyntheticPropertyManager
from ds_discovery.sample.sample_data import MappedSample, Sample

__author__ = 'Darryl Oatridge'


class SyntheticIntentModel(WrangleIntentModel):
    
    def __init__(self, property_manager: SyntheticPropertyManager, default_save_intent: bool=None,
                 default_intent_level: [str, int, float]=None, order_next_available: bool=None,
                 default_replace_intent: bool=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param default_intent_level: (optional) the default level intent should be saved at
        :param order_next_available: (optional) if the default behaviour for the order should be next available order
        :param default_replace_intent: (optional) the default replace existing intent behaviour
        """
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         default_intent_level=default_intent_level, order_next_available=order_next_available,
                         default_replace_intent=default_replace_intent)

    def get_number(self, from_value: [int, float, str]=None, to_value: [int, float, str]=None, relative_freq: list=None,
                   precision: int=None, ordered: str=None, at_most: int=None, size: int=None, quantity: float=None,
                   seed: int=None, save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                   replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a number in the range from_value to to_value. if only to_value given from_value is zero

        :param from_value: (signed) integer or float to start from. See below for str
        :param to_value: optional, (signed) integer or float the number sequence goes to but not include. See below
        :param relative_freq: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param ordered: order the data ascending 'asc' or descending 'dec', values accepted 'asc' or 'des'
        :param at_most: the most times a selection should be chosen
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a random number

        The values can be represented by an environment variable with the format '${NAME}' where NAME is the
        environment variable name
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_number(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_category(self, selection: list, relative_freq: list=None, quantity: float=None, size: int=None,
                     at_most: int=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                     intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a category from a list. Of particular not is the at_least parameter that allows you to
        control the number of times a selection can be chosen.

        :param selection: a list of items to select from
        :param relative_freq: a weighting pattern that does not have to add to 1
        :param quantity: a number between 0 and 1 representing the percentage quantity of the data
        :param size: an optional size of the return. default to 1
        :param at_most: the most times a selection should be chosen
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: an item or list of items chosen from the list
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_category(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_datetime(self, start: Any, until: Any, relative_freq: list=None, at_most: int=None, ordered: str=None,
                     date_format: str=None, as_num: bool=None, ignore_time: bool=None, size: int=None,
                     quantity: float=None, seed: int=None, day_first: bool=None, year_first: bool=None,
                     save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                     replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a random date between two date and/or times. weighted patterns can be applied to the overall date
        range.
        if a signed 'int' type is passed to the start and/or until dates, the inferred date will be the current date
        time with the integer being the offset from the current date time in 'days'.

        Note: If no patterns are set this will return a linearly random number between the range boundaries.

        :param start: the start boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp or int
        :param until: up until boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp or int
        :param quantity: the quantity of values that are not null. Number between 0 and 1
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
                If False default to the a prefered preference, normally %m-%d-%Y (but not strict)
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a date or size of dates in the format given.
         """
        # pre check
        if start is None or until is None:
            raise ValueError("The start or until parameters cannot be of NoneType")
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_datetime(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    # def get_datetime_pattern(self, start: Any, until: Any, default: Any = None, ordered: bool = None,
    #                          date_pattern: list = None, year_pattern: list = None, month_pattern: list = None,
    #                          weekday_pattern: list = None, hour_pattern: list = None, minute_pattern: list = None,
    #                          quantity: float = None, date_format: str = None, size: int = None, seed: int = None,
    #                          day_first: bool = True, year_first: bool = False, save_intent: bool = None,
    #                          column_name: [int, str] = None, intent_order: int = None, replace_intent: bool = None,
    #                          remove_duplicates: bool = None) -> list:
    #     """ returns a random date between two date and times. weighted patterns can be applied to the overall date
    #     range, the year, month, day-of-week, hours and minutes to create a fully customised random set of dates.
    #     Note: If no patterns are set this will return a linearly random number between the range boundaries.
    #           Also if no patterns are set and a default date is given, that default date will be returnd each time
    #
    #     :param start: the start boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
    #     :param until: then up until boundary of the date range can be str, datetime, pd.datetime, pd.Timestamp
    #     :param default: (optional) a fixed starting date that patterns are applied too.
    #     :param ordered: (optional) if the return list should be date ordered
    #     :param date_pattern: (optional) A pattern across the whole date range.
    #             If set, is the primary pattern with each subsequent pattern overriding this result
    #             If no other pattern is set, this will return a random date based on this pattern
    #     :param year_pattern: (optional) adjusts the year selection to this pattern
    #     :param month_pattern: (optional) adjusts the month selection to this pattern. Must be of length 12
    #     :param weekday_pattern: (optional) adjusts the weekday selection to this pattern. Must be of length 7
    #     :param hour_pattern: (optional) adjusts the hours selection to this pattern. must be of length 24
    #     :param minute_pattern: (optional) adjusts the minutes selection to this pattern
    #     :param quantity: the quantity of values that are not null. Number between 0 and 1
    #     :param date_format: the string format of the date to be returned. if not set then pd.Timestamp returned
    #     :param size: the size of the sample to return. Default to 1
    #     :param seed: a seed value for the random function: default to None
    #     :param year_first: specifies if to parse with the year first
    #             If True parses dates with the year first, eg 10/11/12 is parsed as 2010-11-12.
    #             If both day_first and year_first are True, year_first is preceded (same as dateutil).
    #     :param day_first: specifies if to parse with the day first
    #             If True, parses dates with the day first, eg %d-%m-%Y.
    #             If False default to the a preferred preference, normally %m-%d-%Y (but not strict)
    #     :param save_intent (optional) if the intent contract should be saved to the property manager
    #     :param column_name: (optional) the column name that groups intent to create a column
    #     :param intent_order: (optional) the order in which each intent should run.
    #                     If None: default's to -1
    #                     if -1: added to a level above any current instance of the intent section, level 0 if not found
    #                     if int: added to the level specified, overwriting any that already exist
    #     :param replace_intent: (optional) if the intent method exists at the level, or default level
    #                     True - replaces the current intent method with the new
    #                     False - leaves it untouched, disregarding the new intent
    #     :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
    #     :return: a date or size of dates in the format given.
    #      """
    #     # intent persist options
    #    self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
    #                                column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
    #                                remove_duplicates=remove_duplicates, save_intent=save_intent)
    #     # Code block for intent
    #     ordered = False if not isinstance(ordered, bool) else ordered
    #     if start is None or until is None:
    #         raise ValueError("The start or until parameters cannot be of NoneType")
    #     quantity = self._quantity(quantity)
    #     size = 1 if size is None else size
    #     _seed = self._seed() if seed is None else seed
    #     _dt_start = pd.to_datetime(start, errors='coerce', infer_datetime_format=True,
    #                                dayfirst=day_first, yearfirst=year_first)
    #     _dt_until = pd.to_datetime(until, errors='coerce', infer_datetime_format=True,
    #                                dayfirst=day_first, yearfirst=year_first)
    #     _dt_base = pd.to_datetime(default, errors='coerce', infer_datetime_format=True,
    #                               dayfirst=day_first, yearfirst=year_first)
    #     if _dt_start is pd.NaT or _dt_until is pd.NaT:
    #         raise ValueError("The start or until parameters cannot be converted to a timestamp")
    #
    #     # ### Apply the patterns if any ###
    #     rtn_dates = []
    #     for _ in range(size):
    #         _seed = self._next_seed(_seed, seed)
    #         with warnings.catch_warnings():
    #             warnings.filterwarnings("ignore", message='Discarding nonzero nanoseconds in conversion')
    #             _min_date = (pd.Timestamp.min + pd.DateOffset(years=1)).replace(month=1, day=1, hour=0, minute=0,
    #                                                                             second=0, microsecond=0, nanosecond=0)
    #             _max_date = (pd.Timestamp.max + pd.DateOffset(years=-1)).replace(month=12, day=31, hour=23, minute=59,
    #                                                                            second=59, microsecond=0, nanosecond=0)
    #             # reset the starting base
    #         _dt_default = _dt_base
    #         if not isinstance(_dt_default, pd.Timestamp):
    #             _dt_default = np.random.random() * (_dt_until - _dt_start) + pd.to_timedelta(_dt_start)
    #         # ### date ###
    #         if date_pattern is not None:
    #             _dp_start = self._convert_date2value(_dt_start)[0]
    #             _dp_until = self._convert_date2value(_dt_until)[0]
    #             value = self.get_number(_dp_start, _dp_until, relative_freq=date_pattern, seed=_seed,
    #                                     save_intent=False)
    #             _dt_default = self._convert_value2date(value)[0]
    #         # ### years ###
    #         rand_year = _dt_default.year
    #         if year_pattern is not None:
    #             rand_select = self._date_choice(_dt_start, _dt_until, year_pattern, seed=_seed)
    #             if rand_select is pd.NaT:
    #                 rtn_dates.append(rand_select)
    #                 continue
    #             rand_year = rand_select.year
    #         _max_date = _max_date.replace(year=rand_year)
    #         _min_date = _min_date.replace(year=rand_year)
    #         _dt_default = _dt_default.replace(year=rand_year)
    #         # ### months ###
    #         rand_month = _dt_default.month
    #         rand_day = _dt_default.day
    #         if month_pattern is not None:
    #             month_start = _dt_start if _dt_start.year == _min_date.year else _min_date
    #             month_end = _dt_until if _dt_until.year == _max_date.year else _max_date
    #             rand_select = self._date_choice(month_start, month_end, month_pattern, limits='month', seed=_seed)
    #             if rand_select is pd.NaT:
    #                 rtn_dates.append(rand_select)
    #                 continue
    #             rand_month = rand_select.month
    #             rand_day = _dt_default.day if _dt_default.day <= rand_select.daysinmonth else rand_select.daysinmonth
    #         _max_date = _max_date.replace(month=rand_month, day=rand_day)
    #         _min_date = _min_date.replace(month=rand_month, day=rand_day)
    #         _dt_default = _dt_default.replace(month=rand_month, day=rand_day)
    #         # ### weekday ###
    #         if weekday_pattern is not None:
    #             if not len(weekday_pattern) == 7:
    #                 raise ValueError("The weekday_pattern mut be a list of size 7 with index 0 as Monday")
    #             _weekday = self._weighted_choice(weekday_pattern, seed=_seed)
    #             if _weekday != _min_date.dayofweek:
    #                 if _dt_start <= (_dt_default + Week(weekday=_weekday)) <= _dt_until:
    #                     rand_day = (_dt_default + Week(weekday=_weekday)).day
    #                     rand_month = (_dt_default + Week(weekday=_weekday)).month
    #                 elif _dt_start <= (_dt_default - Week(weekday=_weekday)) <= _dt_until:
    #                     rand_day = (_dt_default - Week(weekday=_weekday)).day
    #                     rand_month = (_dt_default - Week(weekday=_weekday)).month
    #                 else:
    #                     rtn_dates.append(pd.NaT)
    #                     continue
    #         _max_date = _max_date.replace(month=rand_month, day=rand_day)
    #         _min_date = _min_date.replace(month=rand_month, day=rand_day)
    #         _dt_default = _dt_default.replace(month=rand_month, day=rand_day)
    #         # ### hour ###
    #         rand_hour = _dt_default.hour
    #         if hour_pattern is not None:
    #             hour_start = _dt_start if _min_date.strftime('%d%m%Y') == _dt_start.strftime('%d%m%Y') else _min_date
    #             hour_end = _dt_until if _max_date.strftime('%d%m%Y') == _dt_until.strftime('%d%m%Y') else _max_date
    #             rand_select = self._date_choice(hour_start, hour_end, hour_pattern, limits='hour', seed=seed)
    #             if rand_select is pd.NaT:
    #                 rtn_dates.append(rand_select)
    #                 continue
    #             rand_hour = rand_select.hour
    #         _max_date = _max_date.replace(hour=rand_hour)
    #         _min_date = _min_date.replace(hour=rand_hour)
    #         _dt_default = _dt_default.replace(hour=rand_hour)
    #         # ### minutes ###
    #         rand_minute = _dt_default.minute
    #         if minute_pattern is not None:
    #             minute_start = _dt_start \
    #                 if _min_date.strftime('%d%m%Y%H') == _dt_start.strftime('%d%m%Y%H') else _min_date
    #             minute_end = _dt_until \
    #                 if _max_date.strftime('%d%m%Y%H') == _dt_until.strftime('%d%m%Y%H') else _max_date
    #             rand_select = self._date_choice(minute_start, minute_end, minute_pattern, seed=seed)
    #             if rand_select is pd.NaT:
    #                 rtn_dates.append(rand_select)
    #                 continue
    #             rand_minute = rand_select.minute
    #         _max_date = _max_date.replace(minute=rand_minute)
    #         _min_date = _min_date.replace(minute=rand_minute)
    #         _dt_default = _dt_default.replace(minute=rand_minute)
    #         # ### get the date ###
    #         _dt_default = _dt_default.replace(second=np.random.randint(60))
    #         if isinstance(_dt_default, pd.Timestamp):
    #             _dt_default = _dt_default.tz_localize(None)
    #         rtn_dates.append(_dt_default)
    #     if ordered:
    #         rtn_dates = sorted(rtn_dates)
    #     rtn_list = []
    #     if isinstance(date_format, str):
    #         for d in rtn_dates:
    #             if isinstance(d, pd.Timestamp):
    #                 rtn_list.append(d.strftime(date_format))
    #             else:
    #                 rtn_list.append(str(d))
    #     else:
    #         rtn_list = rtn_dates
    #     return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_intervals(self, intervals: list, relative_freq: list=None, precision: int=None, size: int=None,
                      quantity: float=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                      intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a number based on a list selection of tuple(lower, upper) interval

        :param intervals: a list of unique tuple pairs representing the interval lower and upper boundaries
        :param relative_freq: a weighting pattern or probability that does not have to add to 1
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a random number
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_intervals(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_dist_normal(self, mean: float, std: float, precision: int=None, size: int=None, quantity: float=None, seed: int=None,
                        save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """A normal (Gaussian) continuous random distribution.

        :param mean: The mean (“centre”) of the distribution.
        :param std: The standard deviation (jitter or “width”) of the distribution. Must be >= 0
        :param precision: The number of decimal points. The default is 3
        :param size: the size of the sample. if a tuple of intervals, size must match the tuple
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a random number
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_dist_normal(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_dist_choice(self, number: [int, str], size: int=None, quantity: float=None, seed: int=None,
                        save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """Creates a list of latent values of 0 or 1 where 1 is randomly selected both upon the number given.

       :param number: The number of true (1) values to randomly chose from the canonical. see below
       :param size: the size of the sample. if a tuple of intervals, size must match the tuple
       :param quantity: a number between 0 and 1 representing data that isn't null
       :param seed: a seed value for the random function: default to None
       :param save_intent (optional) if the intent contract should be saved to the property manager
       :param column_name: (optional) the column name that groups intent to create a column
       :param intent_order: (optional) the order in which each intent should run.
                       If None: default's to -1
                       if -1: added to a level above any current instance of the intent section, level 0 if not found
                       if int: added to the level specified, overwriting any that already exist
       :param replace_intent: (optional) if the intent method exists at the level, or default level
                       True - replaces the current intent method with the new
                       False - leaves it untouched, disregarding the new intent
       :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
       :return: a list of 1 or 0

        as choice is a fixed value, number can be represented by an environment variable with the format '${NAME}'
        where NAME is the environment variable name
       """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_dist_choice(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_dist_binomial(self, trials: int, probability: float, size: int=None, quantity: float=None, seed: int=None,
                          save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                          replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """A binomial discrete random distribution. The Binomial Distribution represents the number of
           successes and failures in n independent Bernoulli trials for some given value of n

        :param trials: the number of trials to attempt, must be >= 0.
        :param probability: the probability distribution, >= 0 and <=1.
        :param size: the size of the sample. if a tuple of intervals, size must match the tuple
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a random number
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_dist_binomial(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_dist_poisson(self, interval: float, size: int=None, quantity: float=None, seed: int=None,
                         save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                         replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """A Poisson discrete random distribution

        :param interval: Expectation of interval, must be >= 0.
        :param size: the size of the sample.
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a random number
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_dist_poisson(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_dist_bernoulli(self, probability: float, size: int=None, quantity: float=None, seed: int=None,
                           save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                           replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """A Bernoulli discrete random distribution using scipy

        :param probability: the probability occurrence
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a random number
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_dist_bernoulli(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_dist_bounded_normal(self, mean: float, std: float, lower: float, upper: float, precision: int=None,
                                size: int=None, quantity: float=None, seed: int=None, save_intent: bool=None,
                                column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                                remove_duplicates: bool=None) -> list:
        """A bounded normal continuous random distribution.

        :param mean: the mean of the distribution
        :param std: the standard deviation
        :param lower: the lower limit of the distribution
        :param upper: the upper limit of the distribution
        :param precision: the precision of the returned number. if None then assumes int value else float
        :param size: the size of the sample
        :param quantity: a number between 0 and 1 representing data that isn't null
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a random number
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_dist_bounded_normal(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_distribution(self, distribution: str, package: str=None, precision: int=None, size: int=None,
                         quantity: float=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                         intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None,
                         **kwargs) -> list:
        """returns a number based the distribution type.

        :param distribution: The string name of the distribution function from numpy random Generator class
        :param package: (optional) The name of the package to use, options are 'numpy' (default) and 'scipy'.
        :param precision: (optional) the precision of the returned number
        :param size: (optional) the size of the sample
        :param quantity: (optional) a number between 0 and 1 representing data that isn't null
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :param kwargs: the parameters of the method
        :return: a random number
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        params.update(params.pop('kwargs', {}))
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_distribution(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_string_pattern(self, pattern: str, choices: dict=None, quantity: [float, int]=None, size: int=None,
                           choice_only: bool=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                           intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ Returns a random string based on the pattern given. The pattern is made up from the choices passed but
            by default is as follows:
                c = random char [a-z][A-Z]
                d = digit [0-9]
                l = lower case char [a-z]
                U = upper case char [A-Z]
                p = all punctuation
                s = space
            you can also use punctuation in the pattern that will be retained
            A pattern example might be
                    uuddsduu => BA12 2NE or dl-{uu} => 4g-{FY}

            to create your own choices pass a dictionary with a reference char key with a list of choices as a value

        :param pattern: the pattern to create the string from
        :param choices: an optional dictionary of list of choices to replace the default.
        :param quantity: a number between 0 and 1 representing the percentage quantity of the data
        :param size: the size of the return list. if None returns a single value
        :param choice_only: if to only use the choices given or to take not found characters as is
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a string based on the pattern
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        choice_only = False if choice_only is None or not isinstance(choice_only, bool) else choice_only
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        seed = self._seed(seed=seed)
        if choices is None or not isinstance(choices, dict):
            choices = {'c': list(string.ascii_letters),
                       'd': list(string.digits),
                       'l': list(string.ascii_lowercase),
                       'U': list(string.ascii_uppercase),
                       'p': list(string.punctuation),
                       's': [' '],
                       }
            choices.update({p: [p] for p in list(string.punctuation)})
        else:
            for k, v in choices.items():
                if not isinstance(v, list):
                    raise ValueError(
                        "The key '{}' must contain a 'list' of replacements opotions. '{}' found".format(k, type(v)))

        generator = np.random.default_rng(seed=seed)
        rtn_list = []
        for c in list(pattern):
            if c in choices.keys():
                result = generator.choice(choices[c], size=size)
            elif not choice_only:
                result = [c]*size
            else:
                continue
            rtn_list = [i + j for i, j in zip(rtn_list, result)] if len(rtn_list) > 0 else result
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_selection(self, select_source: str, column_header: str, relative_freq: list=None, sample_size: int=None,
                      selection_size: int=None, size: int=None, at_most: bool=None, shuffle: bool=None,
                      quantity: float=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                      intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> list:
        """ returns a random list of values where the selection of those values is taken from a connector source.

        :param select_source: the selection source for the reference dataframe
        :param column_header: the name of the column header to correlate
        :param relative_freq: (optional) a weighting pattern of the final selection
        :param selection_size: (optional) the selection to take from the sample size, normally used with shuffle
        :param sample_size: (optional) the size of the sample to take from the reference file
        :param at_most: (optional) the most times a selection should be chosen
        :param shuffle: (optional) if the selection should be shuffled before selection. Default is true
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param size: (optional) size of the return. default to 1
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: list

        The select_source is normally a connector contract str reference or a set of parameter instructions on how to
        generate a pd.Dataframe but can be a pd.DataFrame. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
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
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        params = locals()
        [params.pop(k) for k in self._INTENT_PARAMS + ['quantity']]
        # set the seed and call the method
        seed = self._seed(seed=seed)
        rtn_list = self._get_selection(seed=seed, **params)
        return self._set_quantity(rtn_list, quantity=self._quantity(quantity), seed=seed)

    def get_sample(self, sample_name: str, sample_size: int=None, shuffle: bool=None, size: int=None,
                   quantity: float=None, seed: int=None, save_intent: bool=None, column_name: [int, str]=None,
                   intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ returns a sample set based on sector and name
        To see the sample sets available use the Sample class __dir__() method:

            > from ds_discovery.sample.sample_data import Sample
            > Sample().__dir__()

        :param sample_name: The name of the Sample method to be used.
        :param sample_size: (optional) the size of the sample to take from the reference file
        :param shuffle: (optional) if the selection should be shuffled before selection. Default is true
        :param quantity: (optional) a number between 0 and 1 representing the percentage quantity of the data
        :param size: (optional) size of the return. default to 1
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a sample list
        """
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        size = 1 if size is None else size
        sample_size = sample_name if isinstance(sample_size, int) else size
        quantity = self._quantity(quantity)
        _seed = self._seed(seed=seed)
        shuffle = shuffle if isinstance(shuffle, bool) else True
        selection = eval(f"Sample.{sample_name}(size={size}, shuffle={shuffle}, seed={_seed})")
        return self._set_quantity(selection, quantity=quantity, seed=_seed)

    def get_uuid(self, version: int=None, as_hex: bool=None, size: int=None, quantity: float=None, seed: int=None,
                 save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                 replace_intent: bool=None, remove_duplicates: bool=None, **kwargs) -> list:
        """ returns a list of UUID's based on the version presented. By default the uuid version is 4. optional
        parameters for the version number UUID generator can be passed as kwargs.
        Version 1: Generate a UUID from a host ID, sequence number, and the current time. Note as uuid1 contains the
                   computers network address it may compromise privacy
                param node: (optional) used instead of getnode() which returns a hardware address
                param clock_seq: (optional) used as a sequence number alternative
        Version 3: Generate a UUID based on the MD5 hash of a namespace identifier and a name
                param namespace: an alternative namespace as a UUID e.g. uuid.NAMESPACE_DNS
                param name: a string name
        Version 4: Generate a random UUID
        Version 5: Generate a UUID based on the SHA-1 hash of a namespace identifier and name
                param namespace: an alternative namespace as a UUID e.g. uuid.NAMESPACE_DNS
                param name: a string name

        :param version: The version of the UUID to use. 1, 3, 4 or 5
        :param as_hex: if the return value is in hex format, else as a string
        :param size: the size of the sample. Must be smaller than the range
        :param quantity: a number between 0 and 1 representing the percentage quantity of the data
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a unique identifer randomly selected from the range
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed(seed=seed)
        as_hex = as_hex if isinstance(as_hex, bool) else False
        version = version if isinstance(version, int) and version in [1, 3, 4, 5] else 4
        kwargs = kwargs if isinstance(kwargs, dict) else {}
        rtn_list = [eval(f'uuid{version}(**{kwargs})', globals(), locals()) for x in range(size)]
        rtn_list = [x.hex if as_hex else str(x) for x in rtn_list]
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def get_tagged_pattern(self, pattern: [str, list], tags: dict, relative_freq: list=None, size: int=None,
                           quantity: [float, int]=None, seed: int=None, save_intent: bool=None,
                           column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                           remove_duplicates: bool=None) -> list:
        """ Returns the pattern with the tags substituted by tag choice
            example ta dictionary:
                { '<slogan>': {'action': '', 'kwargs': {}},
                  '<phone>': {'action': '', 'kwargs': {}}
                }
            where action is a self method name and kwargs are the arguments to pass
            for sample data use get_custom

        :param pattern: a string or list of strings to apply the ta substitution too
        :param tags: a dictionary of tas and actions
        :param relative_freq: a weighting pattern that does not have to add to 1
        :param quantity: a number between 0 and 1 representing the percentage quantity of the data
        :param size: an optional size of the return. default to 1
        :param seed: a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a list of patterns with tas replaced
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        quantity = self._quantity(quantity)
        size = 1 if size is None else size
        _seed = self._seed(seed=seed)
        pattern = self._pm.list_formatter(pattern)
        if not isinstance(tags, dict):
            raise ValueError("The 'tags' parameter must be a dictionary")
        class_methods = self.__dir__

        rtn_list = []
        for _ in range(size):
            _seed = self._seed(seed=_seed, increment=True)
            choice = self.get_category(pattern, relative_freq=relative_freq, seed=_seed, size=1, save_intent=False)[0]
            for tag, action in tags.items():
                method = action.get('action')
                if method in class_methods:
                    kwargs = action.get('kwargs')
                    result = eval(f"self.{method}('save_intent=False, **{kwargs})")[0]
                else:
                    result = method
                choice = re.sub(tag, str(result), str(choice))
            rtn_list.append(choice)
        return self._set_quantity(rtn_list, quantity=quantity, seed=_seed)

    def model_noise(self, canonical: Any, num_columns: int, inc_targets: bool=None, seed: int=None,
                    save_intent: bool=None, column_name: [int, str]=None, intent_order: int=None,
                    replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ Generates multiple columns of noise in your dataset

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param num_columns: the number of columns of noise
        :param inc_targets: (optional) if a predictor target should be included. default is false
        :param seed: seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: a DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
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
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        _seed = self._seed(seed=seed)
        size = canonical.shape[0]
        num_columns = num_columns if isinstance(num_columns, int) else 1
        inc_targets = inc_targets if isinstance(inc_targets, int) else False
        gen = Commons.label_gen()
        df_rtn = pd.DataFrame()
        generator = np.random.default_rng(seed=_seed)
        for _ in range(num_columns):
            _seed = self._seed(seed=_seed, increment=True)
            a = generator.choice(range(1, 6))
            b = generator.choice(range(1, 6))
            df_rtn[next(gen)] = self.get_distribution(distribution='beta', a=a, b=b, precision=3, size=size, seed=_seed,
                                                      save_intent=False)
        if inc_targets:
            result = df_rtn.mean(axis=1)
            df_rtn['target1'] = result.apply(lambda x: 1 if x > 0.5 else 0)
            df_rtn['target2'] = df_rtn.iloc[:, :5].mean(axis=1).round(2)
        return pd.concat([canonical, df_rtn], axis=1)

    def model_sample_map(self, canonical: Any, sample_map: str, selection: list=None, headers: [str, list]=None,
                         shuffle: bool=None, rename_columns: dict=None, seed: int=None, save_intent: bool=None,
                         column_name: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                         remove_duplicates: bool=None, **kwargs) -> pd.DataFrame:
        """ builds a model of a Sample Mapped distribution.
        To see the sample maps available use the MappedSample class __dir__() method:

            > from ds_discovery.sample.sample_data import MappedSample
            > MappedSample().__dir__()

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param sample_map: the sample map name. use MappedSample().__dir__() to get a list of available samples
        :param rename_columns: (optional) rename the columns 'City', 'Zipcode', 'State'
        :param selection: (optional) a list of selections where conditions are filtered on, executed in list order
                An example of a selection with the minimum requirements is: (see 'select2dict(...)')
                [{'column': 'state', 'condition': "isin(['NY', 'TX']"}]
        :param headers: a header or list of headers to filter on
        :param shuffle: (optional) if the selection should be shuffled before selection. Default is true
        :param seed: seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param column_name: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :param kwargs: any additional parameters to pass to the sample map
        :return: a DataFrame

        The canonical is a pd.DataFrame, a pd.Series or list, a connector contract str reference or a set of
        parameter instructions on how to generate a pd.Dataframe. the description of each is:

        - pd.Dataframe -> a deep copy of the pd.DataFrame
        - pd.Series or list -> creates a pd.DataFrameof one column with the 'header' name or 'default' if not given
        - str -> instantiates a connector handler with the connector_name and loads the DataFrame from the connection
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
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   column_name=column_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        _seed = self._seed(seed=seed)
        shuffle = shuffle if isinstance(shuffle, bool) else True
        size = canonical.shape[0] if canonical.shape[0] > 1 else None
        df_rtn = eval(f"MappedSample.{sample_map}(size={size}, shuffle={shuffle}, seed={_seed}, **{kwargs})")
        if isinstance(headers, (list, str)):
            df_rtn = Commons.filter_columns(df_rtn, headers=headers, copy=False)
        if isinstance(selection, list):
            selection = deepcopy(selection)
            # run the select logic
            select_idx = None
            select_idx = self._selection_index(canonical=df_rtn, selection=selection)
            df_rtn = df_rtn.iloc[select_idx]
        if isinstance(rename_columns, dict):
            df_rtn = df_rtn.rename(columns=rename_columns)
        return pd.concat([canonical, df_rtn], axis=1)

    """
        UTILITY METHODS SECTION
    """

    @property
    def sample_lists(self) -> list:
        """A list of sample options"""
        return Sample().__dir__()

    @property
    def sample_maps(self) -> list:
        """A list of sample options"""
        return MappedSample().__dir__()

    """
        PRIVATE METHODS SECTION
    """

    def _set_quantity(self, selection, quantity, seed=None):
        """Returns the quantity percent of good values in selection with the rest fill"""
        if quantity == 1:
            return selection
        if quantity == 0:
            return [np.nan] * len(selection)
        seed = self._seed(seed=seed)
        generator = np.random.default_rng(seed=seed)

        def replace_fill():
            """Used to run through all the possible fill options for the list type"""
            if isinstance(selection[i], float):
                selection[i] = np.nan
            elif isinstance(selection[i], str):
                selection[i] = ''
            else:
                selection[i] = None
            return

        if len(selection) < 100:
            for i in range(len(selection)):
                if generator.random() > quantity:
                    replace_fill()
        else:
            sample_count = int(round(len(selection) * (1 - quantity), 0))
            generator = np.random.default_rng(seed=seed)
            indices = generator.choice(list(range(len(selection))), sample_count)
            for i in indices:
                replace_fill()
        return selection

    @staticmethod
    def _quantity(quantity: [float, int]) -> float:
        """normalises quantity to a percentate float between 0 and 1.0"""
        if not isinstance(quantity, (int, float)) or not 0 <= quantity <= 100:
            return 1.0
        if quantity > 1:
            return round(quantity / 100, 2)
        return float(quantity)
