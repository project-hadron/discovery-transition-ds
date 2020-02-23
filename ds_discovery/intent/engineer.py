import re
from copy import deepcopy
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.dates as mdates

from pandas.api.types import is_datetime64_any_dtype, is_categorical_dtype, is_numeric_dtype, is_string_dtype

from aistac.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.transition.discovery import DataDiscovery as Discovery

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
