from aistac.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.components.commons import Commons

__author__ = 'Darryl Oatridge'


class ConceptTolerancePropertyManager(AbstractPropertyManager):

    def __init__(self, task_name: str, creator: str):
        # set additional keys
        root_keys = []
        knowledge_keys = ['drift']
        super().__init__(task_name=task_name, root_keys=root_keys, knowledge_keys=knowledge_keys, creator=creator)

    @staticmethod
    def list_formatter(value) -> list:
        """override of the list_formatter to include Pandas types"""
        return Commons.list_formatter(value=value)
