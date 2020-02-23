import pandas as pd


__author__ = 'Darryl Oatridge'


class Commons():

    @staticmethod
    def report(canonical: pd.DataFrame, index_header: str, bold: [str, list]=None, large_font: [str, list]=None):
        """ generates a stylised report

        :param canonical
        :param index_header:
        :param bold:
        :param large_font
        :return: stylised report DataFrame
        """
        bold = Commons.list_formatter(bold).append(index_header)
        large_font= Commons.list_formatter(large_font).append(index_header)
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        index = canonical[canonical[index_header].duplicated()].index.to_list()
        canonical.loc[index, index_header] = ''
        canonical = canonical.reset_index(drop=True)
        df_style = canonical.style.set_table_styles(style)
        _ = df_style.set_properties(**{'text-align': 'left'})
        _ = df_style.set_properties(subset=bold, **{'font-weight': 'bold'})
        _ = df_style.set_properties(subset=large_font, **{'font-size': "120%"})
        return df_style

    @staticmethod
    def list_formatter(value) -> [list, None]:
        """ Useful utility method to convert any type of str, list, tuple or pd.Series into a list"""
        if value is None:
            return list()
        if isinstance(value, (int, float, str, pd.Timestamp)):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, dict):
            return list(value.items())
        return None

