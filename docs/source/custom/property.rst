AbstractPropertyManager
=======================

The ``AbstractPropertiesManager`` facilitates the management of all the contract properties  including that of the
connector handlers, parameterised intent and Augmented Knowledge

.. image:: /images/custom/abs_property.png
   :align: center
   :width: 500

The Class initialisation is abstracted and is the only abstracted method. A concrete implementation of the
overloaded ``__init__`` manages the ``root_key`` and ``knowledge_key`` for this construct. The ``root_key`` adds a key
property reference to the root of the properties and can be referenced directly with ``<name>_key``. Likewise
the ``knowledge_key`` adds a catalog key to the restricted catalog keys.

More complex ``root_key`` constructs, where a grouping of keys might be desirable, passing a dictionary of name
value pairs as part of the list allows a root base to group related next level keys. For example

.. code-block:: python

    root_key = [{base: [primary, secondary}]

would add ``base.primary_key`` and ``base.secondary_key`` to the list of keys.

Here is a default example of an initialisation method:

.. code-block:: python

    def __init__(self, task_name: str, creator: str):
        # set additional keys
        root_keys = []
        knowledge_keys = ['drift']
        super().__init__(task_name=task_name, root_keys=root_keys, knowledge_keys=knowledge_keys, creator=creator)

The property manager is not responsible for persisting the properties but provides the methods to load and persist
its in memory structure.

