from ds_foundation.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'


class TransitionPropertyManager(AbstractPropertyManager):

    @classmethod
    def manager_name(cls) -> str:
        """Class method to return the name of the manager and used to uniquely identify reference names."""
        return str(cls.__name__).lower().replace('propertymanager', '')