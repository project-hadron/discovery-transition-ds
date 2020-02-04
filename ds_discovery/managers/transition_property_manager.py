import re
from ds_foundation.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'


class TransitionPropertyManager(AbstractPropertyManager):

    def __init__(self, task_name: str):
        # set additional keys
        root_keys = []
        knowledge_keys = ['attributes', 'observations', 'actions']
        super().__init__(task_name=task_name, root_keys=root_keys, knowledge_keys=knowledge_keys)