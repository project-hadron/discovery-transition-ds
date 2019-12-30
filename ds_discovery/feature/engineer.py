import re
from copy import deepcopy
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.dates as mdates

from pandas.api.types import is_datetime64_any_dtype, is_categorical_dtype, is_numeric_dtype, is_string_dtype

from ds_foundation.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.transition.discovery import DataDiscovery as Discovery
from ds_behavioral.generator.data_builder import DataBuilderTools as Tools

__author__ = 'Darryl Oatridge'


class FeatureEngineerTools(object):
    """A set of methods to help engineer features"""

    def __dir__(self):
        rtn_list = []
        for m in dir(FeatureEngineerTools):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def hellinger(p, q):
        """Hellinger distance between distributions (Hoens et al, 2011)"""
        # TODO: convert the entries into pd.Series and have a common output with TVD
        return sum([(np.sqrt(t[0]) - np.sqrt(t[1])) * (np.sqrt(t[0]) - np.sqrt(t[1])) for t in zip(p, q)])/np.sqrt(2.)

    @staticmethod
    def total_variation_distance(a, b):
        """Total Variation Distance (Levin et al, 2008)"""
        # TODO: convert the entries into pd.Series and have a common output with hellinger
        return sum(abs(a - b)) / 2

    @staticmethod
    def replace_missing(dataset: Any, granularity: [int, float]=None, chunk_size: int=None, lower: [int, float]=None,
                        upper: [int, float]=None, nulls_list: [bool, list]=None, replace_zero: [int, float]=None,
                        precision: int=None, day_first: bool=False, year_first: bool=False, date_format: str = None):
        """ imputes missing data with a weighted distribution based on the analysis of the other elements in the
            column

        :param dataset: the series of dataframe to replace missing values in
        :param granularity: (optional) the granularity of the analysis across the range.
                int passed - the number of sections to break the value range into
                pd.Timedelta passed - a frequency time delta
        :param chunk_size: (optional) number of chuncks if you want weighting over the length of the dataset
        :param lower: (optional) the lower limit of the number or date value. Takes min() if not set
        :param upper: (optional) the upper limit of the number or date value. Takes max() if not set
        :param nulls_list: (optional) a list of nulls other than np.nan
        :param replace_zero: (optional) if zero what to replace the weighting value with to avoid zero probability
        :param precision: (optional) by default set to 3.
        :param day_first: if the date provided has day first
        :param year_first: if the date provided has year first
        :param date_format: the format of the output dates, if None then pd.Timestamp
        :return:
        """
        if not isinstance(dataset, (str, int, float, list, pd.Series, pd.DataFrame)):
            raise TypeError("The parameter values is not an accepted type")
        df = dataset
        if isinstance(df, (str, int, float)):
            df = AbstractPropertyManager.list_formatter(df)
        if isinstance(df, (list, pd.Series)):
            tmp = pd.DataFrame()
            tmp['_default'] = df
            df = tmp
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The dataset given is not or could not be convereted to a pandas DataFrame")
        if isinstance(nulls_list, bool) and nulls_list:
            nulls_list = ['NaN', 'nan', 'null', '', 'None', np.inf, -np.inf]
        elif not isinstance(nulls_list, list):
            nulls_list = None
        for c in df.columns:
            col = deepcopy(df[c])
            # replace alternative nulls with pd.nan
            if nulls_list is not None:
                col.replace(nulls_list, np.nan, inplace=True)
            size = len(col[col.isna()])
            if size > 0:
                if is_numeric_dtype(col):
                    result = Discovery.analyse_number(col, granularity=granularity, lower=lower, upper=upper, precision=precision)
                    col[col.isna()] = Tools.get_number(from_value=result.get('lower'), to_value=result.get('upper'), weight_pattern=result.get('weighting'), precision=0, size=size)
                elif is_datetime64_any_dtype(col):
                    result = Discovery.analyse_date(col, granularity=granularity, lower=lower, upper=upper, day_first=day_first, year_first=year_first, date_format=date_format)
                    synthetic = Tools.get_datetime(start=result.get('lower'), until=result.get('upper'), date_pattern=result.get('weighting'), date_format=date_format, day_first=day_first, year_first=year_first, size=size)
                    col = col.apply(lambda x: synthetic.pop() if x is pd.NaT else x)
                else:
                    result = Discovery.analyse_category(col, replace_zero=replace_zero)
                    col[col.isna()] = Tools.get_category(selection=result.get('selection'),
                                                             weight_pattern=result.get('weighting'), size=size)
            df[c] = col
        return df

    @staticmethod
    def apply_substitution(value: str, **kwargs):
        """ regular expression substitution of key value pairs to the value string

        :param value: the value to apply the substitution to
        :param kwargs: a set of keys to replace with the values
        :return: the amended value
        """
        for k, v in kwargs.items():
            value = re.sub(str(k), str(v), value)
        return value

    @staticmethod
    def date_offset_limits(df: pd.DataFrame, date_header: str, offset_header: str):
        """returns the min, mean and max timedelta between the two Pandas Timestamp columns

        :param df: the dataframe to get the columns from
        :param date_header: the base date column name
        :param offset_header: the offset date colun name
        :return: min, mean, max as pd.Timedelta
        """
        diff_list = []
        for index in range(df.shape[0]):
            diff_list.append(df[date_header] - df[offset_header])
        max_diff = max(diff_list)
        min_diff = min(diff_list)
        mean_diff = np.mean(diff_list)
        return min_diff, mean_diff, max_diff

    @staticmethod
    def recommendation(item: [str, int, float], entities: pd.DataFrame, items: pd.DataFrame, recommend: int=None,
                       top: int=None) -> list:
        """ recommendation """
        if entities.columns.equals(items.columns):
            raise ValueError("The entities and items have to have the same column names")
        recommend = 5 if recommend is None else recommend
        top = 3 if top is None else top
        profile = entities.loc[item]
        if profile is None:
            return []
        categories = profile.sort_values(ascending=False).iloc[:top]
        choice = Tools.get_category(selection=categories.index.to_list(), weight_pattern=categories.values.tolist(),
                                    size=recommend)
        item_select = dict()
        for col in categories.index:
            item_select.update({col: items[col].sort_values(ascending=False).iloc[:recommend].index.to_list()})
        rtn_list = []
        for item_choice in choice:
            rtn_list.append(item_select.get(item_choice).pop())
        return rtn_list
