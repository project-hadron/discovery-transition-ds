from ds_foundation.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'


class TransitionPropertyManager(AbstractPropertyManager):

    def reset_contract_properties(self):
        """resets the data contract properties back to it's original state."""
        super()._reset_abstract_properties()
        return
