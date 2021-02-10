from aistac.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'

from ds_discovery.components.commons import Commons


class SyntheticPropertyManager(AbstractPropertyManager):
    """property manager for the Synthetic Data Builder"""

    def __init__(self, task_name: str, username: str):
        """initialises the properties manager.

        :param task_name: the name of the task name within the property manager
        :param username: a username of this instance
        """
        super().__init__(task_name=task_name, root_keys=[], knowledge_keys=['describe'], username=username)

    @staticmethod
    def list_formatter(value) -> list:
        """override of the list_formatter to include Pandas types"""
        return Commons.list_formatter(value=value)
