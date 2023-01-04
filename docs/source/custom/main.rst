AbstractComponent
=================

The ``AbstractComponent`` class is a foundation capability class when extending or creating a Project Hadron
capability. It provides an encapsulated view of the Property Management and Intent Modeling as a first class object.
In other words the concrete implementation of the ``PropertyManager`` and ``IntentModel`` are instantiated and
accessed through this class object.

Because this package uses pandas DataFrame as its canonical, rather than inherit directly from ``AbstractComponent``
a new parent abstraction has been created, mostly for reporting where it ``dict`` are converted to the local
canonical.

.. image:: /images/custom/abs_component.png
   :align: center
   :width: 600

So when creating a new component class you can either inherit from an existing capability, extending its current
tasks, or create a new capability and define its own tasks by inheriting from the ``AbstractCommonComponent``,
for example:

.. code-block:: python

    class CustomCapability(AbstractCommonComponent):

Initialising a capability requires a complex setup so for convenience there are three factory initialisation methods
available that make this task straight forward, ``from_env(...)``, ``from_memory(...)`` and ``from_uri(...)`` the
first two being concrete methods. and the third abstract.  ``from_env(...)`` and ``from_memory(...)`` are dependants
of ``from_uri(...)``, giving alternative entry options. All three initialises the concrete implementation of
``AbstractPropertyManager`` and ``AbstractIntentModel`` classes and use the parent ``_init_properties(...)``
methods to set the properties connector.

.. figure:: /images/custom/factory_methods.png
   :align: center
   :width: 700

   fig: capability instantiation

When creating the concrete class the ``from_uri(...)`` abstraction must be implemented as part of the extension of a
capability or the implementation of a new capability through the ``AbstractComponent``. The following method can be
used as a template replacing the two ``<<capability_name>>`` with the capability name which should, by convention,
match the name of your class.

.. code-block:: python

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, creator: str, uri_pm_repo: str=None,
                 pm_file_type: str=None, pm_module: str=None, pm_handler: str=None, pm_kwargs: dict=None,
                 default_save=None, reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None, align_connectors: bool=None,
                 default_save_intent: bool=None, default_intent_level: bool=None, order_next_available: bool=None,
                 default_replace_intent: bool=None, has_contract: bool=None):

        pm_file_type = pm_file_type if isinstance(pm_file_type, str) else 'json'
        pm_module = pm_module if isinstance(pm_module, str) else cls.DEFAULT_MODULE
        pm_handler = pm_handler if isinstance(pm_handler, str) else cls.DEFAULT_PERSIST_HANDLER
        # TODO: Replace <<capability_name>> with your class name. This assumes the IntentModel and PropertyManager
        # follow the recommended prefix naming convention
        _pm = <<capability_name>>PropertyManager(task_name=task_name, creator=creator)
        _intent_model = <<capability_name>>IntentModel(property_manager=_pm, default_save_intent=default_save_intent,
                                                       default_intent_level=default_intent_level,
                                                       order_next_available=order_next_available,
                                                       default_replace_intent=default_replace_intent)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, default_save=default_save,
                                 uri_pm_repo=uri_pm_repo, pm_file_type=pm_file_type, pm_module=pm_module,
                                 pm_handler=pm_handler, pm_kwargs=pm_kwargs, has_contract=has_contract)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save,
                   reset_templates=reset_templates, template_path=template_path, template_module=template_module,
                   template_source_handler=template_source_handler, template_persist_handler=template_persist_handler,
                   align_connectors=align_connectors)

Once created, inheritance provides enough functionality and access to utilise this first class method but
dependency requires ``PropertyManager`` and ``IntentModel`` are also extended.

As a reminder, the new component manages connectivity through the ``AbstractConnectorContract`` providing a
communication broker between external data stores and the internal canonical of the component.

.. raw:: html

   <p>To understand more about Connector Contracts and how to write your own, ensure your video quality is set
      to the highest quality and watch the following short video:
   <a href="https://youtu.be/6oUAImzhV5g" target="_blank">View of a Connector Contract</a>

\
