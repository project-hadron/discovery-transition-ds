Extending Component Actions
===========================

.. code-block:: python

    def <method>(self, <params>..., save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                 replace_intent: bool=None, remove_duplicates: bool=None):
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # intend code block on the canonical
        ...

