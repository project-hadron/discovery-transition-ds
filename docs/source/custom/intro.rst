Introduction
============

When building a capability we use the abstract foundation package with three key abstracted classes that need to
be extended, ``AbstractPropertyManager``, ``AbstractIntentModel`` and ``AbstractComponent``.

The Abstracted foundation package of Project Hadron is written in pure Python and based on Object Orientated
Design (OOD), not only to reduce its dependency management but to allow application to multiple packages that
are Python based. As an example, the installation of data science tools, NumPy, Pandas, SkiPy, SciKit Learn and
Matplotlib and incorporated them into a new concrete implementation of a component to deal with the pre-processing
of ML model data with seamless integration.

.. image:: /images/custom/abstract_classes.png
   :align: center
   :width: 700

The diagram illustrates the synergy between the conceptual foundation’s abstract parts. It also illustrates the
output of each of these concept classes.

The ``AbstractIntentModel``, holds the intended actions of the capability, defining a finite set of methods, encapsulating
that capability, that can be selected and set, through parameterization, to create a component task. This combination
of finite methods, or actions, and parameter fine tuning is called parameterised intent or just Intent

The ``AbstractPropertyManager`` is a ultra fast in-memory multi-tenant NoSQL data store with each tenant representing a
moment-in-time snapshot of the components state, behavior, and actions. According to Brewer's theorem, focus for the
``AbstractPropertyManager`` focuses on partition tolerance through tenancy and availability through in-memory
accessibility.

The ``AbstractComponent`` class is a foundation abstract class when extending or creating a Project Hadron
capability. It provides an encapsulated view of the Property Management and Intent Modeling as a first class object.
In other words the concrete implementation of the ``AbstractPropertyManager`` and ``AbstractIntentModel`` are
instantiated and accessed by or through this class object. The ``AbstractComponent`` class acts as the entrypoint
to the components functionality.

In addition the ``AbstractComponent`` presents a common set of reporting tools provides a detail view of the
components state, behavior, and actions. This provides transparency and traceability into each aspect of a run-time
component providing visibility, trust and the integration of knowledge to other systems.

Finally the ``AbstractComponent`` manages connectivity through the ``AbstractConnectorContract`` providing a
communication broker between external data stores and the internal canonical of the component.

