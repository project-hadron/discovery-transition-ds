from ds_discovery.components.commons import Commons
from aistac.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'


class WranglePropertyManager(AbstractPropertyManager):

    def __init__(self, task_name: str, username: str):
        # set additional keys
        root_keys = []
        knowledge_keys = ['observations', 'actions']
        super().__init__(task_name=task_name, root_keys=root_keys, knowledge_keys=knowledge_keys, username=username)

    @staticmethod
    def list_formatter(value) -> list:
        """override of the list_formatter to include Pandas types"""
        return Commons.list_formatter(value=value)
