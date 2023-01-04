from aistac.properties.abstract_properties import AbstractPropertyManager
from ds_discovery.components.commons import Commons

__author__ = 'Darryl Oatridge'


class ModelsPropertyManager(AbstractPropertyManager):

    CONNECTOR_ML_TRAINED = 'ml_trained_connector'

    def __init__(self, task_name: str, creator: str, root_keys: list=None, knowledge_keys: list=None):
        root_keys = root_keys if isinstance(root_keys, list) else []
        knowledge_keys = knowledge_keys if isinstance(knowledge_keys, list) else []
        # set additional keys
        r_extended = Commons.list_unique(root_keys + [])
        k_extended = Commons.list_unique(knowledge_keys + ['features', 'observations', 'models'])
        super().__init__(task_name=task_name, root_keys=r_extended, knowledge_keys=k_extended, creator=creator)

    @staticmethod
    def list_formatter(value) -> list:
        """override of the list_formatter to include Pandas types"""
        return Commons.list_formatter(value=value)
