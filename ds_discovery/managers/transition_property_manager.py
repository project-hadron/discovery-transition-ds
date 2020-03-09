from aistac.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'

from ds_discovery.transition.commons import Commons


class TransitionPropertyManager(AbstractPropertyManager):

    def __init__(self, task_name: str):
        # set additional keys
        root_keys = []
        knowledge_keys = ['source', 'observations', 'actions', 'attributes']
        super().__init__(task_name=task_name, root_keys=root_keys, knowledge_keys=knowledge_keys)

    @staticmethod
    def list_formatter(value) -> list:
        """override of the list_formatter to include Pandas types"""
        return Commons.list_formatter(value=value)

