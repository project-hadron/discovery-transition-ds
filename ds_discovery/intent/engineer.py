import pandas as pd

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
