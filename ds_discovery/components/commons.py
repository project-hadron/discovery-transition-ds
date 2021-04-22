import re
from copy import deepcopy
import pandas as pd
from aistac.components.aistac_commons import AistacCommons

__author__ = 'Darryl Oatridge'


class Commons(AistacCommons):

    @staticmethod
    def report(canonical: pd.DataFrame, index_header: [str, list], bold: [str, list]=None,
               large_font: [str, list]=None):
        """ generates a stylised report

        :param canonical: the DataFrame to report on
        :param index_header: the header to index on
        :param bold: any columns to make bold
        :param large_font: any columns to enlarge
        :return: stylised report DataFrame
        """
        index_header = Commons.list_formatter(index_header)
        pd.set_option('max_colwidth', 200)
        pd.set_option('expand_frame_repr', True)
        bold = Commons.list_formatter(bold)
        bold += index_header
        large_font = Commons.list_formatter(large_font)
        large_font += index_header
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        for header in index_header:
            prev = ''
            for idx in range(len(canonical[header])):
                if canonical[header].iloc[idx] == prev:
                    canonical[header].iloc[idx] = ''
                else:
                    prev = canonical[header].iloc[idx]
            # index = canonical[canonical[header].duplicated()].index.to_list()
            # canonical.loc[index, header] = ''
        canonical = canonical.reset_index(drop=True)
        df_style = canonical.style.set_table_styles(style)
        _ = df_style.set_properties(**{'text-align': 'left'})
        if len(bold) > 0:
            _ = df_style.set_properties(subset=bold, **{'font-weight': 'bold'})
        if len(large_font) > 0:
            _ = df_style.set_properties(subset=large_font, **{'font-size': "120%"})
        return df_style

    @staticmethod
    def fillna(values: pd.Series):
        """replaces NaN values - 0 for int, float, datetime, <NA> for category, False for bool, '' for objects """
        if values.dtype.name.lower().startswith('int') or values.dtype.name.startswith('float'):
            values.fillna(0, inplace=True)
        elif values.dtype.name.lower().startswith('date') or values.dtype.name.lower().startswith('time'):
            values.fillna(0, inplace=True)
        elif values.dtype.name.lower().startswith('bool'):
            values.fillna(False, inplace=True)
        elif values.dtype.name.lower().startswith('category'):
            values = values.cat.add_categories("<NA>").fillna('<NA>')
        else:
            values.fillna('', inplace=True)
        return values

    @staticmethod
    def list_formatter(value) -> list:
        """ Useful utility method to convert any type of str, list, tuple or pd.Series keys() etc into a list"""
        if isinstance(value, pd.Timestamp):
            return [value]
        if isinstance(value, pd.Series):
            return value.tolist()
        return AistacCommons.list_formatter(value=value)

    @staticmethod
    def filter_headers(df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                       exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None) -> list:
        """ returns a list of headers based on the filter criteria

        :param df: the DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None.
                    example: int, float, bool, 'category', 'object', 'number'. 'datetime', 'datetimetz', 'timedelta'
        :param exclude: to exclude or include the dtypes. Default is False
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$' excludes '_amt' headers
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :return: a filtered list of headers

        :raise: TypeError if any of the types are not as expected
        """
        drop = drop if isinstance(drop, bool) else False
        exclude = exclude if isinstance(exclude, bool) else False
        re_ignore_case = re_ignore_case if isinstance(re_ignore_case, bool) else False
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The first function attribute must be a pandas 'DataFrame'")
        _headers = Commons.list_formatter(headers)
        dtype = Commons.list_formatter(dtype)
        regex = Commons.list_formatter(regex)
        _obj_cols = df.columns
        _rtn_cols = set()
        unmodified = True

        if _headers:
            _rtn_cols = set(_obj_cols).difference(_headers) if drop else set(_obj_cols).intersection(_headers)
            unmodified = False

        if regex and regex:
            re_ignore_case = re.I if re_ignore_case else 0
            _regex_cols = list()
            for exp in regex:
                _regex_cols += [s for s in _obj_cols if re.search(exp, s, re_ignore_case)]
            _rtn_cols = _rtn_cols.union(set(_regex_cols))
            unmodified = False

        if unmodified:
            _rtn_cols = set(_obj_cols)

        if dtype and len(dtype) > 0:
            _df_selected = df.loc[:, _rtn_cols]
            _rtn_cols = (_df_selected.select_dtypes(exclude=dtype) if exclude
                         else _df_selected.select_dtypes(include=dtype)).columns

        return [c for c in _rtn_cols]

    @staticmethod
    def filter_columns(df: pd.DataFrame, headers: [str, list]=None, drop: bool=None, dtype: [str, list]=None,
                       exclude: bool=None, regex: [str, list]=None, re_ignore_case: bool=None,
                       copy: bool=None) -> pd.DataFrame:
        """ Returns a subset of columns based on the filter criteria

        :param df: the DataFrame to get the column headers from
        :param headers: a list of headers to drop or filter on type
        :param drop: to drop or not drop the headers
        :param dtype: the column types to include or exclude. Default None else int, float, bool, object, 'number'
        :param exclude: to exclude or include the dtypes
        :param regex: a regular expression to search the headers. example '^((?!_amt).)*$' excludes '_amt' columns
        :param re_ignore_case: true if the regex should ignore case. Default is False
        :param copy: if the passed pandas.DataFrame should be used or a deep copy. Default is True
        :return:
        """
        copy = copy if isinstance(copy, bool) else True
        if copy:
            df = deepcopy(df)
        obj_cols = Commons.filter_headers(df, headers=headers, drop=drop, dtype=dtype, exclude=exclude,
                                          regex=regex, re_ignore_case=re_ignore_case)
        return df.loc[:, obj_cols]

    @staticmethod
    def col_width(df, column, as_value=False) -> tuple:
        """ gets the max and min length or values of a column as a (max, min) tuple

        :param df: the DataFrame
        :param column: the column to find the max and min for
        :param as_value: if the return should be a length or the values. Default is False
        :return: returns a tuple with the (max, min) length or values
        """
        if as_value:
            field_length = df[column].apply(str).str.len()
            return df.loc[field_length.argmax(), column], df.loc[field_length.argmin(), column]
        return df[column].apply(str).str.len().max(), df[column].apply(str).str.len().min()
