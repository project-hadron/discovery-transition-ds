Overview
========

The Abstracted foundation package of Project Hadron is written in pure Object Python, built using advanced Object
Orientated Design (OOD) and Object Oriented Programming (OOP) principles. As well as the many advantages given through
Polymorphism, Abstraction, Encapsulation and Inheritance, it lends itself to a microservice architecture built
from component tasks. The abstraction package gives this foundation.

When building a capability we use the abstract foundation package with three key abstracted classes that need to
be extended, ``AbstractPropertyManager``, ``AbstractIntentModel`` and ``AbstractComponent``.

.. image:: /images/custom/abstract_classes.png
   :align: center
   :width: 700

The conceptual diagram visualises the relationship between the foundation’s abstract classes and illustrates each
classes managed outcome.

The ``AbstractIntentModel``, holds the intended actions of the capability, defining a finite set of methods,
encapsulating that capability, that can be selected and set, through parameterization, to create a component
task. This combination of finite methods, or actions, and parameter fine tuning is called parameterised intent
or just Intent

The ``AbstractPropertyManager`` is a ultra fast in-memory multi-tenant NoSQL data store with each tenant
representing a moment-in-time snapshot of the components state, behavior, and actions.  The
``AbstractPropertyManager`` has responsibility for the Domain Contract for each tenant but is not responsible
for its persistance.

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

.. raw:: html

   <p>To get a full rundown of Connector Contracts, how they work, their class structure and how they are
      implemented, ensure your video quality is set to the highest quality and watch the following short video:
   <a href="https://youtu.be/6oUAImzhV5g" target="_blank">View of a Connector Contract</a>


\

